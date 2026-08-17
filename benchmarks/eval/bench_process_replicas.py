# SPDX-License-Identifier: Apache-2.0
"""PR #1175 process-replica A/B driver.

Runs the Qwen3-Omni topology comparison and the Higgs TTS frontend-replica
comparison under conditions strict enough that a win can be attributed to
process replication rather than to restart noise.

What the driver enforces:

- **Interleaving.** Arms are run round-major (`for repeat: for arm:`), so a
  drift in machine state hits every arm roughly equally instead of loading
  onto whichever arm ran last.
- **Independent server starts.** Every (arm, repeat) pair gets its own
  ``sgl-omni serve`` process, with a GPU-idle check before it starts and a
  GPU-memory-release wait after it stops.
- **Fixed everything else.** One model path, one dataset revision, one
  concurrency ladder, one sample budget, one GPU set, shared by all arms.
  The resolved values are written into ``provenance.json`` once per suite.
- **Residency evidence.** Each run captures the server log, the coordinator's
  ``bindings=`` admission lines, a per-GPU NVML compute-process snapshot, and
  a request-level profiler report. A replicated arm whose bindings never
  reach every replica id is reported as ``binding_incomplete`` and must not
  be attributed to replication.
- **Failures are kept.** A failed run keeps its directory, its server log and
  its error, and the driver moves on to the next one.

Usage:

    # Qwen3-Omni: three topologies x four concurrencies x three repeats
    python -m benchmarks.eval.bench_process_replicas \
        --suite qwen3-omni \
        --gpus 0,1,2 \
        --out results/pr1175

    # Higgs TTS: single frontend vs two same-GPU frontend replicas
    python -m benchmarks.eval.bench_process_replicas \
        --suite higgs \
        --gpus 0 \
        --out results/pr1175

    # Both, in one session
    python -m benchmarks.eval.bench_process_replicas \
        --suite all --gpus 0,1,2 --out results/pr1175

``--gpus`` takes *physical* device ids and is passed to each server as
``CUDA_VISIBLE_DEVICES``. The arm configs address GPUs logically (0, 1, 2), so
``--gpus 3,5,7`` runs the same topologies on three otherwise-idle cards
without editing any YAML.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
import requests

from benchmarks.benchmarker.utils import (
    REPO_ROOT,
    disable_proxy,
    no_proxy_env,
    start_server_from_cmd,
    stop_server,
    wait_for_gpu_memory_release,
)
from benchmarks.runtime_metrics import ResourceMonitor, collect_benchmark_provenance

SEEDTTS_META = "zhaochenyang20/seed-tts-eval-arrow"
SEEDTTS_EN_TOTAL = 1088

QWEN_MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
HIGGS_MODEL_PATH = "bosonai/higgs-tts-3-4b"

STARTUP_TIMEOUT = 1800
REQUEST_TIMEOUT = 600

# The coordinator logs one of these per admitted request at INFO level.
_BINDINGS_RE = re.compile(r"bindings=(\{.*?\})")
# Replica instance stages are named `<process>@r<id>` once a process is
# replicated; the unsuffixed name is used when it is not.
_REPLICA_INSTANCE_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)@r(\d+)\b")


@dataclass(frozen=True)
class Arm:
    """One deployment topology under test."""

    key: str
    label: str
    config: str
    # Processes expected to be replicated, mapped to their replica count.
    # Empty for control arms.
    replicated: dict[str, int] = field(default_factory=dict)
    # Logical GPU -> the stages that must be resident on it, for the writeup.
    expected_placement: dict[str, int] = field(default_factory=dict)


QWEN_ARMS = (
    Arm(
        key="pair2gpu",
        label="thinker@0, talker_ar+code2wav@1",
        config="examples/configs/qwen3_omni_speech_pair2gpu.yaml",
        expected_placement={"thinker": 0, "talker_ar": 1, "code2wav": 1},
    ),
    Arm(
        key="split3gpu",
        label="thinker@0, talker_ar@1, code2wav@2",
        config="examples/configs/qwen3_omni_speech_split3gpu.yaml",
        expected_placement={"thinker": 0, "talker_ar": 1, "code2wav": 2},
    ),
    Arm(
        key="replica2",
        label="thinker@0, 2x (talker_ar+code2wav) @1/@2",
        config="examples/configs/qwen3_omni_speech_replica2.yaml",
        replicated={"talker_ar": 2, "code2wav": 2},
        expected_placement={"thinker": 0},
    ),
)

HIGGS_ARMS = (
    Arm(
        key="frontend1",
        label="1x tts_frontend @0",
        config="examples/configs/higgs_frontend_single.yaml",
        expected_placement={"audio_encoder": 0, "tts_engine": 0, "vocoder": 0},
    ),
    Arm(
        key="frontend2",
        label="2x tts_frontend @0 (same GPU)",
        config="examples/configs/higgs_frontend_replica2.yaml",
        replicated={"tts_frontend": 2},
        expected_placement={"tts_engine": 0, "vocoder": 0},
    ),
)


# --------------------------------------------------------------------------
# server lifecycle
# --------------------------------------------------------------------------


def _server_env(gpus: list[int]) -> dict[str, str]:
    env = no_proxy_env()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    return env


@contextlib.contextmanager
def _cuda_visible(gpus: list[int]):
    """Scope the GPU-cleanup helper to this run's cards.

    ``wait_for_gpu_memory_release`` refuses to run without
    ``CUDA_VISIBLE_DEVICES`` so concurrent jobs cannot wipe each other's GPUs.
    The driver otherwise keeps the variable unset, because NVML sampling
    resolves its device index through it and would silently remap.
    """
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous


def _serve_cmd(config: str, model_path: str, port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--config",
        str(REPO_ROOT / config),
        "--model-path",
        model_path,
        "--port",
        str(port),
    ]


def _compute_apps_snapshot(gpus: list[int]) -> dict[str, Any]:
    """Per-GPU NVML compute-process table, used as residency evidence."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        uuids = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"available": False, "error": str(exc)}

    uuid_to_index: dict[str, int] = {}
    for line in uuids.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            uuid_to_index[parts[1]] = int(parts[0])

    per_gpu: dict[str, list[dict[str, Any]]] = {str(g): [] for g in gpus}
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        index = uuid_to_index.get(parts[0])
        if index is None or index not in gpus:
            continue
        per_gpu[str(index)].append(
            {"pid": int(parts[1]) if parts[1].isdigit() else parts[1], "mib": parts[2]}
        )
    return {"available": True, "per_gpu": per_gpu}


def _gpu_pids(snapshot: dict[str, Any], gpu: int) -> list[int]:
    if not snapshot.get("available"):
        return []
    return [
        entry["pid"]
        for entry in snapshot.get("per_gpu", {}).get(str(gpu), [])
        if isinstance(entry.get("pid"), int)
    ]


# --------------------------------------------------------------------------
# profiler
# --------------------------------------------------------------------------


def _profile_start(port: int, run_id: str, event_dir: Path) -> str | None:
    try:
        with disable_proxy():
            resp = requests.post(
                f"http://localhost:{port}/start_request_profile",
                json={"run_id": run_id, "event_dir": str(event_dir)},
                timeout=120,
            )
        resp.raise_for_status()
        return None
    except Exception as exc:
        return str(exc)


def _profile_stop(port: int, run_id: str) -> str | None:
    try:
        with disable_proxy():
            resp = requests.post(
                f"http://localhost:{port}/stop_request_profile",
                json={"run_id": run_id},
                timeout=300,
            )
        resp.raise_for_status()
        return None
    except Exception as exc:
        return str(exc)


def _profiler_report(event_dir: Path) -> dict[str, Any]:
    """Stage/hop breakdown for this run, or the reason there isn't one."""
    if not event_dir.exists() or not any(event_dir.iterdir()):
        return {"available": False, "error": "no profiler events were written"}
    try:
        from sglang_omni.profiler.views import build_report

        report = build_report(event_dir)
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "available": True,
        "request_count": report.get("request_count", 0),
        "stage_breakdown": report.get("stage_breakdown", []),
        "hop_breakdown": report.get("hop_breakdown", []),
    }


# --------------------------------------------------------------------------
# residency / binding evidence
# --------------------------------------------------------------------------


def _member_stages(config_path: str) -> dict[str, list[str]]:
    """Process name -> member stage names, read from the arm's own config."""
    from sglang_omni.config.manager import ConfigManager

    config = ConfigManager.from_file(str(REPO_ROOT / config_path)).config
    members: dict[str, list[str]] = {}
    for stage in config.stages:
        members.setdefault(stage.process or stage.name, []).append(stage.name)
    return members


def _binding_evidence(
    log_text: str, arm: Arm, members: dict[str, list[str]]
) -> dict[str, Any]:
    """Did admission actually spread across the replicas the config declares?

    ``assign_replica_bindings`` projects a process's chosen replica onto each
    of its *member stages*, so the logged dict is keyed by stage name, not by
    process name. Those coincide for Qwen3-Omni (the `talker_ar` process holds
    one stage also called `talker_ar`) but not for Higgs, whose `tts_frontend`
    process holds `preprocessing` and `audio_encoder`. Looking the process
    name up in that dict finds nothing and reports a working deployment as
    unbound, so the member stages have to come from the config.
    """
    observed: dict[str, set[int]] = {}
    admissions = 0
    for raw in _BINDINGS_RE.findall(log_text):
        try:
            parsed = json.loads(raw.replace("'", '"'))
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        admissions += 1
        for process, replica_id in parsed.items():
            if isinstance(replica_id, int):
                observed.setdefault(process, set()).add(replica_id)

    instances: dict[str, set[int]] = {}
    for process, replica_id in _REPLICA_INSTANCE_RE.findall(log_text):
        instances.setdefault(process, set()).add(int(replica_id))

    problems: list[str] = []
    checked_stages: dict[str, list[str]] = {}
    for process, count in arm.replicated.items():
        expected = set(range(count))
        stages = members.get(process) or [process]
        checked_stages[process] = stages

        spawned = instances.get(process, set())
        if spawned != expected:
            problems.append(
                f"{process}: spawned replica instances {sorted(spawned)} "
                f"!= configured {sorted(expected)}"
            )
        for stage in stages:
            bound = observed.get(stage, set())
            if bound != expected:
                problems.append(
                    f"{process}.{stage}: admission bound {sorted(bound)} "
                    f"!= configured {sorted(expected)}"
                )
    if not arm.replicated and observed:
        problems.append(
            f"control arm produced replica bindings {sorted(observed)}; "
            "it should produce none"
        )

    return {
        "admission_lines": admissions,
        "bound_replicas": {k: sorted(v) for k, v in sorted(observed.items())},
        "spawned_instances": {k: sorted(v) for k, v in sorted(instances.items())},
        "expected_replicas": dict(sorted(arm.replicated.items())),
        "checked_member_stages": checked_stages,
        "ok": not problems,
        "problems": problems,
    }


# --------------------------------------------------------------------------
# Qwen3-Omni load: streaming, generate-only, Seed-TTS EN
# --------------------------------------------------------------------------


def _run_qwen_load(
    *,
    port: int,
    concurrency: int,
    samples: int,
    max_new_tokens: int,
    voice_clone: bool,
    out_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.eval.benchmark_omni_seedtts",
        "--generate-only",
        "--stream",
        "--meta",
        SEEDTTS_META,
        "--lang",
        "en",
        "--model",
        "qwen3-omni",
        "--host",
        "localhost",
        "--port",
        str(port),
        "--max-concurrency",
        str(concurrency),
        "--max-samples",
        str(samples),
        "--max-new-tokens",
        str(max_new_tokens),
        "--temperature",
        "0.0",
        "--warmup",
        "1",
        "--output-dir",
        str(out_dir),
        "--disable-tqdm",
    ]
    if voice_clone:
        cmd.append("--voice-clone")

    env = os.environ.copy()
    env.update(no_proxy_env())
    with open(log_path, "w") as handle:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    # --generate-only returns before eval_results.json is written; the speed
    # numbers land in speed_results.json via save_speed_results().
    results_path = out_dir / "speed_results.json"
    if proc.returncode != 0 or not results_path.exists():
        return {
            "ok": False,
            "error": (
                f"benchmark_omni_seedtts exit={proc.returncode}; "
                f"results_present={results_path.exists()}"
            ),
            "command": cmd,
        }
    payload = json.loads(results_path.read_text())
    summary = payload.get("summary", {})
    # benchmark_omni_seedtts exits 0 and still writes speed_results.json when
    # every request failed, so the exit code alone is not evidence that this
    # arm produced any work. An arm that failed fast would otherwise be
    # reported as the fastest one.
    completed = int(summary.get("completed_requests", 0) or 0)
    failed = int(summary.get("failed_requests", 0) or 0)
    if completed == 0:
        return {
            "ok": False,
            "error": f"no request completed ({failed} failed); see client.log",
            "summary": summary,
            "command": cmd,
        }
    if failed:
        return {
            "ok": False,
            "error": (
                f"{failed}/{completed + failed} requests failed; an arm with "
                "failures is not comparable, see client.log"
            ),
            "summary": summary,
            "command": cmd,
        }
    return {"ok": True, "summary": summary, "command": cmd}


# --------------------------------------------------------------------------
# Higgs load: duration-driven closed loop over the full Seed-TTS EN split
# --------------------------------------------------------------------------


async def _higgs_closed_loop(
    *,
    port: int,
    concurrency: int,
    warmup_s: float,
    measure_s: float,
    max_new_tokens: int,
    model_name: str,
    samples: list,
) -> dict[str, Any]:
    """Saturate the server for warmup_s + measure_s and keep only the middle.

    Requests are attributed by **completion** time, not issue time. Workers
    stay busy for the whole window, so the requests still in flight when the
    window closes are balanced by the ones that were in flight when it opened
    and complete inside it — the standard closed-loop steady-state argument,
    and unbiased. Attributing by issue time instead would drop every request
    started near the deadline, and would drop *more* of them from the slower
    arm, which is exactly the arm-dependent bias an A/B cannot afford.

    ``truncated`` counts what was still running when the window closed. It is
    reported but not folded into the metrics.
    """
    from benchmarks.metrics.performance import compute_speed_metrics
    from benchmarks.tasks.tts import make_tts_send_fn

    api_url = f"http://localhost:{port}/v1/audio/speech"
    send_fn = make_tts_send_fn(
        model_name,
        api_url,
        response_format="pcm",
        stream=True,
        ref_format="references",
        max_new_tokens=max_new_tokens,
    )

    measured: list = []
    warmup_count = 0
    truncated = 0
    errors: list[str] = []

    started_at = time.perf_counter()
    warmup_end = started_at + warmup_s
    measure_end = warmup_end + measure_s
    cursor = 0
    cursor_lock = asyncio.Lock()

    async def _next_sample():
        nonlocal cursor
        async with cursor_lock:
            sample = samples[cursor % len(samples)]
            cursor += 1
            return sample

    async def _worker(session: aiohttp.ClientSession) -> None:
        nonlocal warmup_count, truncated
        while True:
            if time.perf_counter() >= measure_end:
                return
            sample = await _next_sample()
            result = await send_fn(session, sample)
            finished = time.perf_counter()
            if finished < warmup_end:
                warmup_count += 1
            elif finished <= measure_end:
                measured.append(result)
                if not result.is_success and result.error:
                    errors.append(result.error)
            else:
                truncated += 1

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    connector = aiohttp.TCPConnector(limit=max(concurrency * 2, 32))
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # A worker that dies on an unexpected exception must not cancel the
        # others: the window is fixed, and losing the remaining workers would
        # silently understate this arm's throughput.
        outcomes = await asyncio.gather(
            *(_worker(session) for _ in range(concurrency)),
            return_exceptions=True,
        )
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            errors.append(f"worker died: {type(outcome).__name__}: {outcome}")

    summary = compute_speed_metrics(measured, wall_clock_s=measure_s)
    summary["measured_requests"] = len(measured)
    summary["warmup_requests"] = warmup_count
    summary["truncated_requests"] = truncated
    summary["measure_window_s"] = measure_s
    summary["warmup_window_s"] = warmup_s

    # Same trap as the Qwen path: requests that fail come back fast, so an arm
    # that errored on everything would post the best throughput.
    completed = int(summary.get("completed_requests", 0) or 0)
    failed = int(summary.get("failed_requests", 0) or 0)
    if completed == 0:
        error = f"no request completed in the window ({failed} failed)"
    elif failed:
        error = f"{failed}/{completed + failed} requests failed; arm not comparable"
    else:
        error = None

    return {
        "ok": error is None,
        "error": error,
        "summary": summary,
        "errors": errors[:20],
        "error_count": len(errors),
    }


def _load_higgs_samples(max_samples: int) -> list:
    from benchmarks.dataset.seedtts import load_seedtts_samples

    return load_seedtts_samples(SEEDTTS_META, max_samples, split="en")


# --------------------------------------------------------------------------
# one (arm, repeat) run
# --------------------------------------------------------------------------


def _run_arm(
    *,
    suite: str,
    arm: Arm,
    repeat: int,
    args: argparse.Namespace,
    gpus: list[int],
    model_path: str,
    run_root: Path,
    higgs_samples: list | None,
) -> dict[str, Any]:
    from sglang_omni.utils.connection import find_available_port

    run_dir = run_root / f"r{repeat}" / arm.key
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO_ROOT / arm.config, run_dir / "arm_config.yaml")

    server_log = run_dir / "server.log"
    record: dict[str, Any] = {
        "suite": suite,
        "arm": arm.key,
        "arm_label": arm.label,
        "config": arm.config,
        "repeat": repeat,
        "gpus": gpus,
        "expected_placement": arm.expected_placement,
        "run_dir": str(run_dir),
        "measurements": [],
    }

    print(f"\n[{suite}] repeat={repeat} arm={arm.key} ({arm.label})", flush=True)

    try:
        with _cuda_visible(gpus):
            wait_for_gpu_memory_release()
    except Exception as exc:
        record["gpu_precheck_error"] = str(exc)

    port = find_available_port()
    record["port"] = port
    cmd = _serve_cmd(arm.config, model_path, port)
    record["serve_command"] = cmd

    proc = None
    try:
        started = time.perf_counter()
        proc = start_server_from_cmd(
            cmd,
            server_log,
            port,
            timeout=STARTUP_TIMEOUT,
            env=_server_env(gpus),
            strip_proxy=True,
        )
        record["startup_s"] = round(time.perf_counter() - started, 2)
    except Exception as exc:
        record["ok"] = False
        record["error"] = f"server start failed: {type(exc).__name__}: {exc}"
        print(f"  server start FAILED: {exc}", flush=True)
        (run_dir / "result.json").write_text(json.dumps(record, indent=2))
        return record

    try:
        snapshot = _compute_apps_snapshot(gpus)
        record["compute_apps"] = snapshot

        if suite == "qwen3-omni":
            for concurrency in args.concurrency:
                record["measurements"].append(
                    _measure_qwen(
                        arm=arm,
                        args=args,
                        port=port,
                        concurrency=concurrency,
                        run_dir=run_dir,
                        repeat=repeat,
                    )
                )
        else:
            record["measurements"].append(
                _measure_higgs(
                    arm=arm,
                    args=args,
                    port=port,
                    run_dir=run_dir,
                    repeat=repeat,
                    model_path=model_path,
                    samples=higgs_samples or [],
                    gpu=gpus[0],
                    snapshot=snapshot,
                )
            )
        record["ok"] = all(m.get("ok") for m in record["measurements"])
    except Exception as exc:
        record["ok"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if proc is not None:
            stop_server(proc)
        try:
            with _cuda_visible(gpus):
                wait_for_gpu_memory_release()
        except Exception as exc:
            record["gpu_release_error"] = str(exc)

    log_text = server_log.read_text() if server_log.exists() else ""
    record["binding"] = _binding_evidence(log_text, arm, _member_stages(arm.config))
    if not record["binding"]["ok"]:
        print(f"  BINDING PROBLEM: {record['binding']['problems']}", flush=True)

    (run_dir / "result.json").write_text(json.dumps(record, indent=2))
    return record


def _qwen_sample_count(args: argparse.Namespace, concurrency: int) -> int:
    """Samples for one measurement, scaled so the closed loop reaches steady state.

    A fixed budget starves the high-concurrency rows: 128 samples at
    concurrency 64 is two waves, and ramp-up plus drain dominate what should
    be a steady-state throughput number. Every arm sees the same count at the
    same concurrency, which is what the comparison requires; counts differ
    across concurrency levels, which is fine because those rows are never
    compared against each other.
    """
    return min(SEEDTTS_EN_TOTAL, max(args.qwen_samples, concurrency * args.qwen_waves))


def _measure_qwen(
    *,
    arm: Arm,
    args: argparse.Namespace,
    port: int,
    concurrency: int,
    run_dir: Path,
    repeat: int,
) -> dict[str, Any]:
    tag = f"c{concurrency}"
    out_dir = run_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    event_dir = out_dir / "events"
    event_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{arm.key}-r{repeat}-{tag}"
    samples = _qwen_sample_count(args, concurrency)

    print(f"  concurrency={concurrency} samples={samples} ...", flush=True)
    profile_error = _profile_start(port, run_id, event_dir)
    started = time.perf_counter()
    load = _run_qwen_load(
        port=port,
        concurrency=concurrency,
        samples=samples,
        max_new_tokens=args.qwen_max_new_tokens,
        voice_clone=args.qwen_voice_clone,
        out_dir=out_dir,
        log_path=out_dir / "client.log",
    )
    elapsed = time.perf_counter() - started
    stop_error = _profile_stop(port, run_id)

    measurement = {
        "concurrency": concurrency,
        "samples": samples,
        "ok": load["ok"],
        "wall_clock_s": round(elapsed, 2),
        "summary": load.get("summary", {}),
        "error": load.get("error"),
        "profile_start_error": profile_error,
        "profile_stop_error": stop_error,
        "profiler": _profiler_report(event_dir),
        "artifacts": str(out_dir),
    }
    (out_dir / "measurement.json").write_text(json.dumps(measurement, indent=2))
    summary = measurement["summary"]
    status = "ok" if load["ok"] else f"FAILED ({load.get('error')})"
    print(
        f"    -> {status} in {elapsed:.1f}s "
        f"[completed={summary.get('completed_requests')} "
        f"failed={summary.get('failed_requests')} "
        f"qps={summary.get('throughput_qps')} "
        f"lat_mean={summary.get('latency_mean_s')}]",
        flush=True,
    )
    return measurement


def _measure_higgs(
    *,
    arm: Arm,
    args: argparse.Namespace,
    port: int,
    run_dir: Path,
    repeat: int,
    model_path: str,
    samples: list,
    gpu: int,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    out_dir = run_dir / f"c{args.higgs_concurrency}"
    out_dir.mkdir(parents=True, exist_ok=True)
    event_dir = out_dir / "events"
    event_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{arm.key}-r{repeat}"

    print(
        f"  concurrency={args.higgs_concurrency} "
        f"warmup={args.higgs_warmup_s}s measure={args.higgs_measure_s}s ...",
        flush=True,
    )

    monitor = ResourceMonitor(
        gpu_index=gpu,
        interval_s=0.2,
        gpu_process_pids=_gpu_pids(snapshot, gpu),
    )
    monitor.start()
    profile_error = _profile_start(port, run_id, event_dir)
    started = time.perf_counter()
    try:
        load = asyncio.run(
            _higgs_closed_loop(
                port=port,
                concurrency=args.higgs_concurrency,
                warmup_s=args.higgs_warmup_s,
                measure_s=args.higgs_measure_s,
                max_new_tokens=args.higgs_max_new_tokens,
                model_name=model_path,
                samples=samples,
            )
        )
    except Exception as exc:
        load = {"ok": False, "summary": {}, "error": f"{type(exc).__name__}: {exc}"}
    elapsed = time.perf_counter() - started
    stop_error = _profile_stop(port, run_id)
    resources = monitor.stop()

    measurement = {
        "concurrency": args.higgs_concurrency,
        "ok": load.get("ok", False),
        "wall_clock_s": round(elapsed, 2),
        "summary": load.get("summary", {}),
        "error": load.get("error"),
        "client_errors": load.get("errors", []),
        "client_error_count": load.get("error_count", 0),
        "profile_start_error": profile_error,
        "profile_stop_error": stop_error,
        "profiler": _profiler_report(event_dir),
        "resources": resources,
        "artifacts": str(out_dir),
    }
    (out_dir / "measurement.json").write_text(json.dumps(measurement, indent=2))
    summary = measurement["summary"]
    status = "ok" if measurement["ok"] else f"FAILED ({load.get('error')})"
    print(
        f"    -> {status} in {elapsed:.1f}s "
        f"[completed={summary.get('completed_requests')} "
        f"failed={summary.get('failed_requests')} "
        f"truncated={summary.get('truncated_requests')} "
        f"qps={summary.get('throughput_qps')}]",
        flush=True,
    )
    return measurement


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

_QWEN_METRICS = (
    "throughput_qps",
    "audio_ttfp_mean_s",
    "audio_ttfp_p95_s",
    "rtf_mean",
    "latency_mean_s",
    "latency_p95_s",
    "inter_chunk_mean_s",
    "inter_chunk_p95_s",
    "audio_throughput_s_per_s",
)

_HIGGS_METRICS = (
    "throughput_qps",
    "latency_mean_s",
    "latency_p95_s",
    "audio_throughput_s_per_s",
    "output_throughput_tok_s",
    "output_tok_per_req_s",
    "measured_requests",
)


def _spread(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 4),
        "stdev": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def _aggregate(records: list[dict[str, Any]], metrics: tuple[str, ...]) -> list[dict]:
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in records:
        for measurement in record.get("measurements", []):
            if not measurement.get("ok"):
                continue
            key = (record["arm"], measurement["concurrency"])
            buckets.setdefault(key, []).append(measurement["summary"])

    rows = []
    for (arm, concurrency), summaries in sorted(buckets.items(), key=lambda i: i[0]):
        row: dict[str, Any] = {
            "arm": arm,
            "concurrency": concurrency,
            "repeats": len(summaries),
        }
        for metric in metrics:
            values = [
                float(s[metric])
                for s in summaries
                if isinstance(s.get(metric), (int, float))
            ]
            row[metric] = _spread(values) if values else None
        rows.append(row)
    return rows


def _markdown(rows: list[dict[str, Any]], metrics: tuple[str, ...]) -> str:
    header = ["arm", "concurrency", "repeats", *metrics]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in rows:
        cells = [str(row["arm"]), str(row["concurrency"]), str(row["repeats"])]
        for metric in metrics:
            spread = row.get(metric)
            cells.append(
                "n/a"
                if spread is None
                else f"{spread['mean']:.4g} ±{spread['stdev']:.3g}"
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def _mps_running() -> bool:
    try:
        out = subprocess.run(
            ["pgrep", "-f", "nvidia-cuda-mps-control"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except Exception:
        return False
    return bool(out.stdout.strip())


def _run_suite(
    *,
    suite: str,
    arms: tuple[Arm, ...],
    args: argparse.Namespace,
    gpus: list[int],
    model_path: str,
    out_root: Path,
) -> None:
    run_root = out_root / suite
    run_root.mkdir(parents=True, exist_ok=True)

    provenance = collect_benchmark_provenance(
        model_id=model_path,
        model_revision=None,
        dataset_id=SEEDTTS_META,
        dataset_revision=None,
        launch_command=" ".join(sys.argv),
        server_config={
            "suite": suite,
            "gpus": gpus,
            "arms": [{"key": a.key, "config": a.config, "label": a.label} for a in arms],
            "repeats": args.repeats,
            "concurrency": args.concurrency,
            "qwen_samples_floor": args.qwen_samples,
            "qwen_waves": args.qwen_waves,
            "qwen_samples_resolved": {
                str(c): _qwen_sample_count(args, c) for c in args.concurrency
            },
            "qwen_max_new_tokens": args.qwen_max_new_tokens,
            "qwen_voice_clone": args.qwen_voice_clone,
            "higgs_concurrency": args.higgs_concurrency,
            "higgs_warmup_s": args.higgs_warmup_s,
            "higgs_measure_s": args.higgs_measure_s,
            "higgs_max_new_tokens": args.higgs_max_new_tokens,
            "mps_running": _mps_running(),
        },
    )
    (run_root / "provenance.json").write_text(json.dumps(provenance, indent=2))

    higgs_samples = None
    if suite == "higgs":
        higgs_samples = _load_higgs_samples(args.higgs_samples)
        print(f"[higgs] loaded {len(higgs_samples)} Seed-TTS EN samples", flush=True)

    records: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        for arm in arms:
            records.append(
                _run_arm(
                    suite=suite,
                    arm=arm,
                    repeat=repeat,
                    args=args,
                    gpus=gpus,
                    model_path=model_path,
                    run_root=run_root,
                    higgs_samples=higgs_samples,
                )
            )
            (run_root / "runs.json").write_text(json.dumps(records, indent=2))

    metrics = _QWEN_METRICS if suite == "qwen3-omni" else _HIGGS_METRICS
    rows = _aggregate(records, metrics)
    binding_ok = all(r.get("binding", {}).get("ok", False) for r in records)
    failed = [
        f"r{r['repeat']}/{r['arm']}" for r in records if not r.get("ok")
    ]

    summary = {
        "suite": suite,
        "rows": rows,
        "binding_ok": binding_ok,
        "failed_runs": failed,
        "binding_problems": {
            f"r{r['repeat']}/{r['arm']}": r.get("binding", {}).get("problems", [])
            for r in records
            if not r.get("binding", {}).get("ok", False)
        },
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2))
    (run_root / "summary.md").write_text(
        f"# {suite}\n\n"
        f"- binding/residency verified: **{binding_ok}**\n"
        f"- failed runs: {failed or 'none'}\n\n"
        f"{_markdown(rows, metrics)}\n"
    )

    print(f"\n=== {suite} ===", flush=True)
    print(_markdown(rows, metrics), flush=True)
    print(f"binding/residency verified: {binding_ok}", flush=True)
    if failed:
        print(f"failed runs kept for inspection: {failed}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["qwen3-omni", "higgs", "all"], default="all")
    parser.add_argument(
        "--gpus",
        default="0,1,2",
        help="Physical GPU ids, passed as CUDA_VISIBLE_DEVICES. The arm "
        "configs address these logically as 0,1,2.",
    )
    parser.add_argument("--out", default="results/pr1175")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--concurrency",
        default="1,16,32,64",
        help="Qwen3-Omni concurrency ladder.",
    )
    parser.add_argument("--qwen-model-path", default=QWEN_MODEL_PATH)
    parser.add_argument(
        "--qwen-samples",
        type=int,
        default=128,
        help="Floor on samples per measurement; see --qwen-waves.",
    )
    parser.add_argument(
        "--qwen-waves",
        type=int,
        default=10,
        help="Samples per measurement are max(--qwen-samples, concurrency x "
        "this), capped at the 1088-sample EN split. A closed loop needs "
        "several waves before it reaches steady state: at 128 samples, "
        "concurrency 64 gets two, and ramp-up and drain dominate the result.",
    )
    parser.add_argument("--qwen-max-new-tokens", type=int, default=256)
    parser.add_argument("--qwen-voice-clone", action="store_true")
    parser.add_argument("--higgs-model-path", default=HIGGS_MODEL_PATH)
    parser.add_argument("--higgs-concurrency", type=int, default=96)
    parser.add_argument("--higgs-warmup-s", type=float, default=20.0)
    parser.add_argument("--higgs-measure-s", type=float, default=90.0)
    parser.add_argument("--higgs-max-new-tokens", type=int, default=512)
    parser.add_argument("--higgs-samples", type=int, default=SEEDTTS_EN_TOTAL)
    args = parser.parse_args()

    gpus = [int(token) for token in args.gpus.split(",") if token.strip()]
    args.concurrency = [
        int(token) for token in str(args.concurrency).split(",") if token.strip()
    ]

    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        print(
            "refusing to run with CUDA_VISIBLE_DEVICES set in the driver's own "
            "environment: it would remap the --gpus ids and the NVML sampling "
            "index. Unset it and pass --gpus instead.",
            file=sys.stderr,
        )
        return 2

    if _mps_running():
        print(
            "an MPS control daemon is running; this comparison is specified "
            "with MPS off. Stop it (echo quit | nvidia-cuda-mps-control) and "
            "rerun.",
            file=sys.stderr,
        )
        return 2

    suites: list[tuple[str, tuple[Arm, ...], str, list[int]]] = []
    if args.suite in ("qwen3-omni", "all"):
        if len(gpus) < 3:
            print(
                f"qwen3-omni needs 3 GPUs, got --gpus {args.gpus}", file=sys.stderr
            )
            return 2
        suites.append(("qwen3-omni", QWEN_ARMS, args.qwen_model_path, gpus[:3]))
    if args.suite in ("higgs", "all"):
        suites.append(("higgs", HIGGS_ARMS, args.higgs_model_path, gpus[:1]))

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for suite, arms, model_path, suite_gpus in suites:
        _run_suite(
            suite=suite,
            arms=arms,
            args=args,
            gpus=suite_gpus,
            model_path=model_path,
            out_root=out_root,
        )

    print(f"\nartifacts under {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
