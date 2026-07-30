# SPDX-License-Identifier: Apache-2.0
"""Offline capture-matrix probe for batched Code2Wav CUDA graphs.

Answers the sizing questions behind batched graph capture before any serving
run: how much pool memory each ``(batch, frames)`` graph costs, whether the
shared pool keeps the full matrix near the largest single graph instead of the
sum, where the OOM cliff sits as batch grows, and how replay compares to eager
per shape. The final report ranks batching policies (e.g. "ceiling 2 with the
full graph matrix" vs "ceiling 8 with a partial one") from the measured
per-shape costs and greedily fits the matrix into candidate memory budgets.

Each batch size runs in its own subprocess: a capture OOM leaves the CUDA
context unreliable, so the cliff must not poison later measurements.

Usage (on the GPU host, serving venv active, from the repo root):
    python -m benchmarks.eval.probe_code2wav_graph_memory \
        --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct --device cuda:0

Attach the printed report and probe_results.json to the tracking issue.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_DEFAULT_BATCHES = "1,2,4,8"
_DEFAULT_FRAMES = "10,20,30,35"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--batch-sizes", default=_DEFAULT_BATCHES)
    parser.add_argument("--frames", default=_DEFAULT_FRAMES)
    parser.add_argument("--replay-iters", type=int, default=50)
    parser.add_argument("--eager-iters", type=int, default=20)
    parser.add_argument("--budget-fractions", default="0.01,0.02,0.03,0.05")
    parser.add_argument(
        "--reference-total-gb",
        type=float,
        default=None,
        help="compute the budget-fit table against this total VRAM (GiB) "
        "instead of the local GPU's — e.g. 80 to predict H100 budgets while "
        "probing on an H200; footprints themselves are machine-independent",
    )
    parser.add_argument("--output-dir", default="results/code2wav-graph-probe")
    parser.add_argument(
        "--worker",
        choices=["per-batch", "all-shared"],
        help="internal: run one measurement process and print JSON",
    )
    parser.add_argument(
        "--worker-batches", default="", help="internal: batches for the worker"
    )
    return parser.parse_args()


def _ints(spec: str) -> list[int]:
    return [int(part) for part in spec.split(",") if part.strip()]


# ---------------------------------------------------------------- worker ----


def _bench(fn, iters: int, device) -> float:
    import torch

    for _ in range(3):
        fn()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def _run_worker(args: argparse.Namespace) -> None:
    import torch

    from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
        load_code2wav_model,
    )

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model = load_code2wav_model(args.model_path, device=str(device), dtype=args.dtype)
    num_quantizers = int(model.config.num_quantizers)
    torch.cuda.synchronize(device)
    model_alloc = torch.cuda.memory_allocated(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)

    frames_list = _ints(args.frames)
    batches = _ints(args.worker_batches)
    # Largest first so the pool peak is laid down once and later captures
    # reuse it — the same order the serving runner uses.
    keys = sorted(((b, t) for b in batches for t in frames_list), reverse=True)

    report: dict[str, Any] = {
        "mode": args.worker,
        "batches": batches,
        "model_alloc_bytes": int(model_alloc),
        "total_bytes": int(total_bytes),
        "free_after_load_bytes": int(free_bytes),
        "keys": [],
        "oom": None,
    }

    pool = torch.cuda.graph_pool_handle()
    stream = torch.cuda.Stream(device=device)
    base_alloc = torch.cuda.memory_allocated(device)
    base_reserved = torch.cuda.memory_reserved(device)
    previous = 0
    for batch, frames in keys:
        static_input = torch.zeros(
            (batch, num_quantizers, frames), dtype=torch.long, device=device
        )
        entry: dict[str, Any] = {"batch": batch, "frames": frames}
        try:
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream), torch.inference_mode():
                for _ in range(3):
                    model(static_input)
            torch.cuda.current_stream(device).wait_stream(stream)

            graph = torch.cuda.CUDAGraph()
            started = time.perf_counter()
            with torch.inference_mode():
                with torch.cuda.graph(graph, pool=pool, stream=stream):
                    static_output = model(static_input)
            torch.cuda.synchronize(device)
            entry["capture_s"] = round(time.perf_counter() - started, 3)

            footprint = max(
                torch.cuda.memory_allocated(device) - base_alloc,
                torch.cuda.memory_reserved(device) - base_reserved,
            )
            entry["cumulative_footprint_bytes"] = int(footprint)
            entry["footprint_delta_bytes"] = int(footprint - previous)
            previous = footprint

            entry["replay_ms"] = round(
                _bench(graph.replay, args.replay_iters, device), 3
            )

            def _eager() -> None:
                with torch.inference_mode():
                    model(static_input)

            entry["eager_ms"] = round(_bench(_eager, args.eager_iters, device), 3)
            del static_output
        except torch.OutOfMemoryError:
            entry["oom"] = True
            report["oom"] = {"batch": batch, "frames": frames}
            report["keys"].append(entry)
            break
        report["keys"].append(entry)

    print(json.dumps(report))


# ---------------------------------------------------------------- driver ----


def _spawn(args: argparse.Namespace, mode: str, batches: list[int]) -> dict:
    command = [
        sys.executable,
        "-m",
        "benchmarks.eval.probe_code2wav_graph_memory",
        "--model-path",
        args.model_path,
        "--device",
        args.device,
        "--frames",
        args.frames,
        "--replay-iters",
        str(args.replay_iters),
        "--eager-iters",
        str(args.eager_iters),
        "--worker",
        mode,
        "--worker-batches",
        ",".join(str(b) for b in batches),
    ]
    if args.dtype:
        command += ["--dtype", args.dtype]
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        return {
            "mode": mode,
            "batches": batches,
            "crashed": True,
            "stderr_tail": proc.stderr[-2000:],
        }
    return json.loads(proc.stdout.splitlines()[-1])


def _fmt_mb(value: int | None) -> str:
    return "-" if value is None else f"{value / (1 << 20):8.1f}"


def _report(
    results: dict[str, Any],
    budget_fractions: list[float],
    reference_total_bytes: int | None = None,
) -> str:
    lines: list[str] = []
    per_batch = results["per_batch"]
    lines.append("== per-(B,T) capture footprint and timing ==")
    lines.append(
        f"{'B':>3} {'T':>4} {'cum_MB':>9} {'delta_MB':>9} "
        f"{'replay_ms':>10} {'eager_ms':>9} {'speedup':>8}"
    )
    key_index: dict[tuple[int, int], dict] = {}
    for worker in per_batch:
        for entry in worker.get("keys", []):
            if entry.get("oom"):
                lines.append(
                    f"{entry['batch']:>3} {entry['frames']:>4}  "
                    "** capture OOM — cliff **"
                )
                continue
            key_index[(entry["batch"], entry["frames"])] = entry
            speedup = (
                entry["eager_ms"] / entry["replay_ms"]
                if entry["replay_ms"]
                else float("nan")
            )
            lines.append(
                f"{entry['batch']:>3} {entry['frames']:>4} "
                f"{_fmt_mb(entry['cumulative_footprint_bytes'])} "
                f"{_fmt_mb(entry['footprint_delta_bytes'])} "
                f"{entry['replay_ms']:>10.3f} {entry['eager_ms']:>9.3f} "
                f"{speedup:>7.2f}x"
            )
        if worker.get("crashed"):
            lines.append(
                f"batches {worker['batches']}: worker crashed "
                "(likely hard OOM) — see stderr_tail in JSON"
            )

    shared = results.get("all_shared") or {}
    if shared.get("keys"):
        combined = max(
            entry["cumulative_footprint_bytes"]
            for entry in shared["keys"]
            if not entry.get("oom")
        )
        separate_sum = sum(
            max(
                (
                    e["cumulative_footprint_bytes"]
                    for e in worker.get("keys", [])
                    if not e.get("oom")
                ),
                default=0,
            )
            for worker in per_batch
        )
        lines.append("")
        lines.append("== shared-pool validation ==")
        lines.append(
            f"full matrix in one pool: {_fmt_mb(combined)} MB; "
            f"sum of per-batch pools: {_fmt_mb(separate_sum)} MB"
        )

        total_bytes = reference_total_bytes or shared["total_bytes"]
        model_alloc = shared["model_alloc_bytes"]
        lines.append("")
        lines.append(
            "== greedy fit per budget (largest-first order; "
            f"basis {'reference' if reference_total_bytes else 'local'} "
            f"total {total_bytes / (1 << 30):.0f} GiB) =="
        )
        for fraction in budget_fractions:
            budget = int(total_bytes * fraction) - model_alloc
            fitted: list[str] = []
            for entry in shared["keys"]:
                if entry.get("oom"):
                    break
                if entry["cumulative_footprint_bytes"] > budget:
                    break
                fitted.append(f"b{entry['batch']}t{entry['frames']}")
            lines.append(
                f"fraction {fraction:.2%}: budget {_fmt_mb(budget)} MB -> "
                f"{len(fitted)}/{len(shared['keys'])} keys: "
                f"{' '.join(fitted) if fitted else '(none)'}"
            )

    lines.append("")
    lines.append("== policy cost per 8 due windows (steady window T=max) ==")
    frames_max = max((frames for _batch, frames in key_index), default=None)
    if frames_max is not None:

        def _cost(label: str, terms: list[tuple[int, str]]) -> None:
            total = 0.0
            for batch, mode in terms:
                entry = key_index.get((batch, frames_max))
                if entry is None:
                    lines.append(f"{label:>34}: n/a (b{batch} not measured)")
                    return
                total += entry["replay_ms" if mode == "graph" else "eager_ms"]
            lines.append(f"{label:>34}: {total:8.3f} ms")

        _cost("ceiling 8, B8 graph", [(8, "graph")])
        _cost("ceiling 8, 2 x B4 graph", [(4, "graph"), (4, "graph")])
        _cost("ceiling 8, B8 eager (no graph)", [(8, "eager")])
        _cost("ceiling 2, 4 x B2 graph", [(2, "graph")] * 4)
        _cost("pure B1 graph, 8 replays", [(1, "graph")] * 8)
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    if args.worker:
        _run_worker(args)
        return

    batches = _ints(args.batch_sizes)
    budget_fractions = [
        float(part) for part in args.budget_fractions.split(",") if part.strip()
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {"per_batch": [], "all_shared": None}
    for batch in batches:
        print(f"[probe] batch {batch} worker", flush=True)
        results["per_batch"].append(_spawn(args, "per-batch", [batch]))

    survivors = [
        batch
        for batch, worker in zip(batches, results["per_batch"])
        if not worker.get("crashed") and not worker.get("oom")
    ]
    if survivors:
        print(f"[probe] all-shared worker for batches {survivors}", flush=True)
        results["all_shared"] = _spawn(args, "all-shared", survivors)

    output_path = output_dir / "probe_results.json"
    output_path.write_text(json.dumps(results, indent=2))
    reference_total_bytes = (
        int(args.reference_total_gb * (1 << 30)) if args.reference_total_gb else None
    )
    report = _report(results, budget_fractions, reference_total_bytes)
    (output_dir / "probe_report.txt").write_text(report + "\n")
    print()
    print(report)
    print(f"\nartifacts: {output_path} and probe_report.txt")


if __name__ == "__main__":
    main()
