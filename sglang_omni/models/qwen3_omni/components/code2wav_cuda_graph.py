# SPDX-License-Identifier: Apache-2.0
"""Exact-shape CUDA graphs for the Qwen3-Omni Code2Wav component."""

from __future__ import annotations

import gc
import logging
import math
import os
from collections import Counter
from contextlib import AbstractContextManager
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GraphKey:
    """One exact Code2Wav input shape, excluding fixed quantizer count."""

    batch_size: int
    frames: int


@dataclass(frozen=True, slots=True)
class Code2WavRunResult:
    """Result metadata for either an exact graph replay or eager fallback.

    A ``cuda_graph`` output is a borrowed static buffer. The caller must finish
    all reads, including trim and device-to-host transfer, before the next graph
    replay. It must not retain the tensor or use it concurrently. The current
    scheduler's outer ``_state_lock`` is intended to cover that whole lifetime;
    this runner deliberately does not clone the output.
    """

    output: torch.Tensor
    execution_mode: str
    key: GraphKey | None
    fallback_reason: str | None


@dataclass(slots=True)
class _CapturedGraph:
    graph: Any
    static_input: torch.Tensor
    static_output: torch.Tensor


class _BuildFailure(RuntimeError):
    pass


class _TorchCudaApi:
    """Small injectable boundary around CUDA-only operations."""

    def device_context(self, device: torch.device) -> AbstractContextManager[Any]:
        return torch.cuda.device(device)

    def memory_stats(self, device: torch.device) -> dict[str, int]:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
        }

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def new_static_input(
        self, shape: tuple[int, int, int], *, device: torch.device
    ) -> torch.Tensor:
        return torch.zeros(shape, dtype=torch.long, device=device)

    def new_stream(self, device: torch.device) -> torch.cuda.Stream:
        return torch.cuda.Stream(device=device)

    def warmup(
        self,
        model: Any,
        static_input: torch.Tensor,
        *,
        iterations: int,
        device: torch.device,
        stream: torch.cuda.Stream,
    ) -> None:
        current_stream = torch.cuda.current_stream(device)
        stream.wait_stream(current_stream)
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(iterations):
                model(static_input)
        current_stream.wait_stream(stream)

    def graph_pool_handle(self) -> Any:
        return torch.cuda.graph_pool_handle()

    def capture(
        self,
        model: Any,
        static_input: torch.Tensor,
        *,
        pool: Any,
        stream: torch.cuda.Stream,
    ) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
        current_stream = torch.cuda.current_stream(static_input.device)
        stream.wait_stream(current_stream)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.inference_mode():
                with torch.cuda.graph(
                    graph,
                    pool=pool,
                    stream=stream,
                    capture_error_mode="thread_local",
                ):
                    static_output = model(static_input)
        finally:
            # torch.cuda.graph.__exit__ calls capture_end before restoring its
            # stream context. If capture_end raises, restore explicitly using
            # the original stream's device-aware identity.
            torch.cuda.set_stream(current_stream)
        current_stream.wait_stream(stream)
        return graph, static_output

    def synchronize(self, device: torch.device) -> None:
        torch.cuda.synchronize(device)

    def is_cuda_tensor(self, tensor: torch.Tensor) -> bool:
        return tensor.is_cuda

    def tensor_device_matches(self, tensor: torch.Tensor, device: torch.device) -> bool:
        return tensor.device == device


class Code2WavCudaGraphRunner:
    """Exact-shape CUDA graph runner for ``[B, Q, T]`` long codes.

    One instance is permanently bound to one model, CUDA device, quantizer
    count, ``torch.long`` input dtype, and owner process. ``batch_size == 1``
    keys form an atomic tier with the original semantics: any failure there
    disables the complete runner and leaves no partial matrix published.
    ``batch_size > 1`` keys are best-effort — captured largest-first into a
    separate pool, published as the greedy prefix that fits the remaining
    memory budget, so an oversized batched graph can never take down the
    single-request tier that serving already relies on.
    """

    _WARMUP_ITERATIONS = 3

    def __init__(
        self,
        model: Any,
        *,
        device: str | torch.device,
        num_quantizers: int,
        graph_keys: tuple[GraphKey, ...],
        cuda_api: Any,
    ) -> None:
        self._model = model
        self._device = torch.device(device)
        if self._device.type != "cuda" or self._device.index is None:
            raise ValueError("Code2Wav CUDA graphs require a concrete CUDA device")
        self._num_quantizers = int(num_quantizers)
        if self._num_quantizers <= 0:
            raise ValueError("Code2Wav CUDA graphs require a positive quantizer count")
        self._graph_keys = graph_keys
        self._tier0_keys = tuple(k for k in graph_keys if k.batch_size == 1)
        self._tier1_keys = tuple(k for k in graph_keys if k.batch_size > 1)
        self._owner_pid = os.getpid()
        self._cuda = cuda_api
        self._graphs: dict[GraphKey, _CapturedGraph] = {}
        self._pool: Any | None = None
        self._tier1_pool: Any | None = None
        self._capture_stream: Any | None = None
        self._enabled = False
        self._disable_reason: str | None = None
        self._build_stats: dict[str, Any] = {
            "attempted_graph_count": 0,
            "published_graph_count": 0,
        }
        self._memory_stats: dict[str, Any] = {"total_gpu_memory_fraction": None}
        self._fallback_counts: Counter[str] = Counter()
        self._graph_replays = 0
        self._replay_failures = 0

    @classmethod
    def build(
        cls,
        model: Any,
        *,
        device: str | torch.device,
        num_quantizers: int,
        total_gpu_memory_fraction: float | None,
        graph_keys: tuple[GraphKey, ...],
        cuda_api: Any | None = None,
    ) -> Code2WavCudaGraphRunner:
        """Build the configured serving-reachable serial graphs."""

        runner = cls(
            model,
            device=device,
            num_quantizers=num_quantizers,
            graph_keys=graph_keys,
            cuda_api=_TorchCudaApi() if cuda_api is None else cuda_api,
        )
        runner._build(total_gpu_memory_fraction)
        return runner

    def _build(self, total_gpu_memory_fraction: float | None) -> None:
        fraction = self._valid_fraction(total_gpu_memory_fraction)
        if fraction is None:
            self._disable_reason = "invalid_total_gpu_memory_fraction"
            return
        self._memory_stats["total_gpu_memory_fraction"] = fraction

        temporary: dict[GraphKey, _CapturedGraph] = {}
        pool: Any | None = None
        capture_stream: Any | None = None
        before: dict[str, int] | None = None
        failure_reason: str | None = None
        try:
            with self._cuda.device_context(self._device):
                before = self._cuda.memory_stats(self._device)
                self._memory_stats["before"] = before
                stage_budget = int(before["total_bytes"] * fraction)
                loaded_model_footprint = before["allocated_bytes"]
                graph_budget = max(0, stage_budget - loaded_model_footprint)
                self._memory_stats.update(
                    {
                        "stage_budget_bytes": stage_budget,
                        "loaded_model_footprint_bytes": loaded_model_footprint,
                        "graph_budget_bytes": graph_budget,
                    }
                )

                pool = self._cuda.graph_pool_handle()
                capture_stream = self._cuda.new_stream(self._device)
                for key in self._priority_order(self._tier0_keys):
                    self._build_stats["attempted_graph_count"] += 1
                    temporary[key] = self._capture_graph(
                        key,
                        pool=pool,
                        stream=capture_stream,
                    )

                # Capture, replay and equivalence checks enqueue CUDA work. Do
                # not make the all-or-nothing graph matrix visible until every
                # key has completed on the bound device.
                self._cuda.synchronize(self._device)
                gc.collect()
                self._cuda.empty_cache()
                after = self._cuda.memory_stats(self._device)
                self._memory_stats["after"] = after
                allocated_delta = max(
                    0,
                    after["allocated_bytes"] - before["allocated_bytes"],
                )
                reserved_delta = max(
                    0,
                    after["reserved_bytes"] - before["reserved_bytes"],
                )
                graph_footprint = max(allocated_delta, reserved_delta)
                self._memory_stats["graph_footprint_bytes"] = graph_footprint
                if graph_footprint > graph_budget:
                    raise _BuildFailure(
                        f"memory_budget_exceeded: graph footprint "
                        f"{graph_footprint} exceeds budget "
                        f"{graph_budget}",
                    )

        except Exception as exc:
            failure_reason = (
                str(exc)
                if isinstance(exc, _BuildFailure)
                else f"capture_failed: {type(exc).__name__}: {exc}"
            )

        if failure_reason is not None:
            pool = None
            capture_stream = None
            self._rollback_build(
                temporary=temporary,
                reason=failure_reason,
            )
            return

        self._pool = pool
        self._capture_stream = capture_stream
        published = {key: temporary[key] for key in self._tier0_keys}
        if self._tier1_keys:
            published.update(
                self._build_tier1(
                    before=before,
                    graph_budget=self._memory_stats["graph_budget_bytes"],
                    stream=capture_stream,
                )
            )
        self._graphs = {
            key: published[key] for key in self._graph_keys if key in published
        }
        self._build_stats["published_graph_count"] = len(self._graphs)
        self._enabled = True
        logger.info(
            "Code2Wav CUDA graph runner published %d exact graphs on %s",
            len(self._graphs),
            self._device,
        )

    # Retries re-capture a strictly smaller key set, so this bound is only a
    # backstop against footprint measurements that never stabilize.
    _TIER1_MAX_ATTEMPTS = 6

    def _build_tier1(
        self,
        *,
        before: dict[str, int],
        graph_budget: int,
        stream: Any,
    ) -> dict[GraphKey, _CapturedGraph]:
        """Capture the batched tier as the greedy prefix fitting the budget.

        The keys are attempted largest-first so the pool's peak blocks are laid
        down once and every later capture reuses them. Pool memory is only
        reclaimable as a whole, which makes the pool the retry unit: on a
        budget violation the attempt is dropped and re-captured with the
        violating key (and everything after it) excluded. A violation by the
        very first key excludes its entire batch-size class instead — same
        batch at shorter frames cannot be assumed to fit. Non-capacity capture
        failures abandon the tier outright: shrinking cannot fix a correctness
        problem, and the atomic tier stays published either way.
        """
        info: dict[str, Any] = {
            "attempted_key_count": len(self._tier1_keys),
            "published_key_count": 0,
            "attempts": 0,
            "skipped_keys": [],
            "disable_reason": None,
            "per_key_footprint_bytes": {},
        }
        self._memory_stats["tier1"] = info

        remaining = list(self._priority_order(self._tier1_keys))
        published: dict[GraphKey, _CapturedGraph] = {}
        pool: Any | None = None
        while remaining and info["attempts"] < self._TIER1_MAX_ATTEMPTS:
            info["attempts"] += 1
            temporary: dict[GraphKey, _CapturedGraph] = {}
            pool = self._cuda.graph_pool_handle()
            violation_index: int | None = None
            error_reason: str | None = None
            try:
                with self._cuda.device_context(self._device):
                    previous_footprint = self._footprint_since(before)
                    for index, key in enumerate(remaining):
                        self._build_stats["attempted_graph_count"] += 1
                        temporary[key] = self._capture_graph(
                            key,
                            pool=pool,
                            stream=stream,
                        )
                        self._cuda.synchronize(self._device)
                        footprint = self._footprint_since(before)
                        info["per_key_footprint_bytes"][self._key_name(key)] = (
                            footprint - previous_footprint
                        )
                        if footprint > graph_budget:
                            violation_index = index
                            break
                        previous_footprint = footprint
            except torch.OutOfMemoryError:
                violation_index = len(temporary)
            except Exception as exc:
                error_reason = (
                    str(exc)
                    if isinstance(exc, _BuildFailure)
                    else f"capture_failed: {type(exc).__name__}: {exc}"
                )

            if violation_index is None and error_reason is None:
                published = temporary
                break

            temporary.clear()
            pool = None
            gc.collect()
            try:
                with self._cuda.device_context(self._device):
                    self._cuda.empty_cache()
            except Exception as cleanup_exc:
                logger.warning(
                    "Code2Wav tier-1 graph rollback cleanup failed: %s",
                    cleanup_exc,
                )
            if error_reason is not None:
                info["disable_reason"] = error_reason
                break
            if violation_index > 0:
                remaining = remaining[:violation_index]
            else:
                oversized_batch = remaining[0].batch_size
                remaining = [
                    key for key in remaining if key.batch_size < oversized_batch
                ]

        if published:
            self._tier1_pool = pool
        info["published_key_count"] = len(published)
        info["skipped_keys"] = [
            {"batch_size": key.batch_size, "frames": key.frames}
            for key in self._tier1_keys
            if key not in published
        ]
        if info["skipped_keys"]:
            logger.warning(
                "Code2Wav tier-1 graphs published %d/%d keys; skipped: %s",
                len(published),
                len(self._tier1_keys),
                info["skipped_keys"],
            )
        return published

    def _footprint_since(self, before: dict[str, int]) -> int:
        snapshot = self._cuda.memory_stats(self._device)
        return max(
            0,
            snapshot["allocated_bytes"] - before["allocated_bytes"],
            snapshot["reserved_bytes"] - before["reserved_bytes"],
        )

    @staticmethod
    def _priority_order(keys: tuple[GraphKey, ...]) -> tuple[GraphKey, ...]:
        # Largest first: the biggest graph lays down the pool's peak blocks so
        # later captures reuse them instead of growing the pool.
        return tuple(sorted(keys, key=lambda k: (k.batch_size, k.frames), reverse=True))

    @staticmethod
    def _key_name(key: GraphKey) -> str:
        return f"b{key.batch_size}t{key.frames}"

    def available_batch_sizes(self, frames: int) -> tuple[int, ...]:
        """Batch sizes with a published graph for this window length, largest
        first; the scheduler decomposes coalesced batches against this."""
        return tuple(
            sorted(
                {key.batch_size for key in self._graphs if key.frames == int(frames)},
                reverse=True,
            )
        )

    def _capture_graph(
        self,
        key: GraphKey,
        *,
        pool: Any,
        stream: Any,
    ) -> _CapturedGraph:
        static_input = self._cuda.new_static_input(
            (key.batch_size, self._num_quantizers, key.frames),
            device=self._device,
        )
        self._cuda.warmup(
            self._model,
            static_input,
            iterations=self._WARMUP_ITERATIONS,
            device=self._device,
            stream=stream,
        )
        graph, static_output = self._cuda.capture(
            self._model,
            static_input,
            pool=pool,
            stream=stream,
        )
        with torch.inference_mode():
            eager_output = self._model(static_input).detach().clone()
            graph.replay()
        self._verify_equivalence(
            key=key,
            eager_output=eager_output,
            graph_output=static_output,
        )
        return _CapturedGraph(graph, static_input, static_output)

    @staticmethod
    def _valid_fraction(value: float | None) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            fraction = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            return None
        return fraction

    @staticmethod
    def _verify_equivalence(
        *,
        key: GraphKey,
        eager_output: torch.Tensor,
        graph_output: torch.Tensor,
    ) -> None:
        if not (
            eager_output.shape == graph_output.shape
            and bool(torch.isfinite(eager_output).all().item())
            and bool(torch.isfinite(graph_output).all().item())
            and torch.equal(eager_output, graph_output)
        ):
            raise _BuildFailure(
                f"equivalence_failed: {key}: eager and graph outputs differ"
            )

    def _rollback_build(
        self,
        *,
        temporary: dict[GraphKey, _CapturedGraph],
        reason: str,
    ) -> None:
        if "after" not in self._memory_stats:
            try:
                self._cuda.synchronize(self._device)
            except Exception as synchronize_exc:
                logger.warning(
                    "Code2Wav CUDA graph rollback synchronize failed: %s",
                    synchronize_exc,
                )
            try:
                self._memory_stats["after"] = self._cuda.memory_stats(self._device)
            except Exception as snapshot_exc:
                logger.warning(
                    "Code2Wav CUDA graph rollback snapshot failed: %s",
                    snapshot_exc,
                )
        self._graphs.clear()
        temporary.clear()
        self._pool = None
        self._tier1_pool = None
        self._capture_stream = None
        self._enabled = False
        self._disable_reason = reason
        gc.collect()
        try:
            with self._cuda.device_context(self._device):
                self._cuda.empty_cache()
                self._memory_stats["after_rollback"] = self._cuda.memory_stats(
                    self._device
                )
        except Exception as cleanup_exc:
            logger.warning(
                "Code2Wav CUDA graph rollback cleanup failed: %s",
                cleanup_exc,
            )
        logger.warning("Code2Wav CUDA graph runner disabled: %s", reason)

    def run(
        self,
        codes: torch.Tensor,
        *,
        eligible: bool = True,
    ) -> Code2WavRunResult:
        """Replay an exact graph or eagerly execute with a stable reason.

        Graph outputs are borrowed and valid only until the next graph replay;
        callers must serialize replay through trim and D2H consumption.
        """

        current_pid = os.getpid()
        if current_pid != self._owner_pid:
            raise RuntimeError(
                "Code2Wav CUDA graph runner/model belongs to PID "
                f"{self._owner_pid}, but was used in PID {current_pid}; it must "
                "be rebuilt in a spawned process before inference"
            )
        if not self._enabled:
            return self._eager(codes, key=None, reason="disabled")
        if not eligible:
            return self._eager(codes, key=None, reason="ineligible")
        self._validate_codes(codes)

        key = GraphKey(
            batch_size=int(codes.shape[0]),
            frames=int(codes.shape[2]),
        )
        captured = self._graphs.get(key)
        if captured is None:
            return self._eager(codes, key=key, reason="key_miss")

        try:
            captured.static_input.copy_(codes)
            captured.graph.replay()
        except Exception as exc:
            self._replay_failures += 1
            reason = f"runtime_replay_failed: {type(exc).__name__}: {exc}"
            # Drop the last local graph reference before cleanup releases its pool.
            captured = None
            self._disable_runtime(reason)
            raise
        self._graph_replays += 1
        return Code2WavRunResult(
            output=captured.static_output,
            execution_mode="cuda_graph",
            key=key,
            fallback_reason=None,
        )

    def _validate_codes(self, codes: torch.Tensor) -> None:
        if not self._cuda.is_cuda_tensor(codes):
            raise TypeError("Code2Wav graph input must be a CUDA tensor")
        if codes.dtype != torch.long:
            raise TypeError("Code2Wav graph input must use torch.long")
        if not self._cuda.tensor_device_matches(codes, self._device):
            raise ValueError(f"Code2Wav graph input must be on {self._device}")
        if codes.ndim != 3:
            raise ValueError("Code2Wav graph input must have shape [B, Q, T]")
        if int(codes.shape[1]) != self._num_quantizers:
            raise ValueError(
                f"Code2Wav graph input must contain {self._num_quantizers} quantizers"
            )

    def _eager(
        self,
        codes: torch.Tensor,
        *,
        key: GraphKey | None,
        reason: str,
    ) -> Code2WavRunResult:
        self._fallback_counts[reason] += 1
        with torch.inference_mode():
            output = self._model(codes)
        return Code2WavRunResult(
            output=output,
            execution_mode="eager",
            key=key,
            fallback_reason=reason,
        )

    def _disable_runtime(self, reason: str) -> None:
        self._graphs.clear()
        self._pool = None
        self._tier1_pool = None
        self._capture_stream = None
        self._enabled = False
        self._disable_reason = reason
        gc.collect()
        try:
            with self._cuda.device_context(self._device):
                self._cuda.empty_cache()
        except Exception as cleanup_exc:
            logger.warning(
                "Code2Wav CUDA graph runtime cleanup failed: %s",
                cleanup_exc,
            )
        logger.exception("Code2Wav CUDA graph replay disabled the runner")

    def stats(self) -> dict[str, Any]:
        """Return a strict JSON-safe snapshot of build and runtime state."""

        return {
            "enabled": self._enabled,
            "disable_reason": self._disable_reason,
            "binding": {
                "device": str(self._device),
                "num_quantizers": self._num_quantizers,
                "input_dtype": "torch.long",
                "owner_pid": self._owner_pid,
            },
            "graph_contract": {
                "keys": [
                    {
                        "batch_size": key.batch_size,
                        "frames": key.frames,
                    }
                    for key in self._graph_keys
                ],
            },
            "build": deepcopy(self._build_stats),
            "memory": deepcopy(self._memory_stats),
            "runtime": {
                "graph_replays": self._graph_replays,
                "replay_failures": self._replay_failures,
                "fallback_counts": dict(sorted(self._fallback_counts.items())),
            },
        }


__all__ = [
    "Code2WavCudaGraphRunner",
    "Code2WavRunResult",
    "GraphKey",
]
