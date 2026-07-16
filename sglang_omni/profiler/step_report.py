# SPDX-License-Identifier: Apache-2.0
"""Offline aggregation for per-step timing events (issue #1039).

Reads the ``events_*.jsonl`` files a profiling run produced and reports,
per (run_id, stage), the host-side breakdown of decode steps recorded by
``sglang_omni.profiler.step_timing``:

- per-segment p50/p95 and share of total step time, split into
  graph-replay vs eager steps;
- the step gap (host time between consecutive decode steps that is spent
  outside the model runner: scheduler loop, batch prep, output handling);
- ``talker_batch_prep`` durations and deferred-decode counts;
- first-frame predictor spans and the counts that must be zero in healthy
  serving (``predictor_eager_fallback``, ``predictor_graph_capture_failed``).

Usage::

    python -m sglang_omni.profiler.step_report <event-dir> \
        [--stage talker] [--warmup-steps 16] [--format table|json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

STEP_REQUEST_ID = "__step__"

_STEP_EVENTS = {
    "model_step",
    "talker_batch_prep",
    "talker_first_frame_predictor",
    "predictor_eager_fallback",
    "predictor_graph_capture_failed",
}


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return sorted_values[idx]


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    total = sum(ordered)
    return {
        "count": len(ordered),
        "mean_ms": total / len(ordered) if ordered else 0.0,
        "p50_ms": _percentile(ordered, 0.50),
        "p95_ms": _percentile(ordered, 0.95),
        "total_ms": total,
    }


def load_step_events(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """All step-scoped events from the given JSONL files, oldest first."""
    events: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("request_id") != STEP_REQUEST_ID:
                    continue
                if event.get("event_name") not in _STEP_EVENTS:
                    continue
                events.append(event)
    events.sort(key=lambda e: e.get("timestamp_ns", 0))
    return events


def _summarize_steps(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a homogeneous list of model_step events."""
    totals = [e["metadata"]["total_ms"] for e in steps]
    segment_values: dict[str, list[float]] = defaultdict(list)
    for event in steps:
        for name, value in event["metadata"].get("segments_ms", {}).items():
            segment_values[name].append(value)
    grand_total = sum(totals) or 1.0
    segments = {
        name: {**_stats(values), "share_of_step": sum(values) / grand_total}
        for name, values in sorted(segment_values.items())
    }
    return {"step": _stats(totals), "segments": segments}


def _decode_step_gaps(steps: list[dict[str, Any]]) -> list[float]:
    """Host ms between consecutive decode steps of one pid that was spent
    outside the model runner (wall gap minus the later step's own time)."""
    gaps: list[float] = []
    by_pid: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for event in steps:
        by_pid[event.get("pid")].append(event)
    for pid_steps in by_pid.values():
        for prev, cur in zip(pid_steps, pid_steps[1:]):
            wall_ms = (cur["timestamp_ns"] - prev["timestamp_ns"]) / 1e6
            gap = wall_ms - cur["metadata"]["total_ms"]
            if gap >= 0:
                gaps.append(gap)
    return gaps


def build_report(
    events: list[dict[str, Any]],
    *,
    stage: str | None = None,
    warmup_steps: int = 0,
) -> dict[str, Any]:
    groups: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if stage is not None and event.get("stage") != stage:
            continue
        groups[(event.get("run_id"), event.get("stage"))].append(event)

    report: dict[str, Any] = {}
    for (run_id, event_stage), group in sorted(
        groups.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))
    ):
        by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in group:
            by_name[event["event_name"]].append(event)

        decode_steps = [
            e
            for e in by_name.get("model_step", [])
            if not e["metadata"].get("is_prefill")
        ]
        # Warmup exclusion is per pid so multi-process stages drop the same
        # number of leading steps everywhere.
        if warmup_steps > 0:
            kept: list[dict[str, Any]] = []
            seen: dict[Any, int] = defaultdict(int)
            for event in decode_steps:
                seen[event.get("pid")] += 1
                if seen[event.get("pid")] > warmup_steps:
                    kept.append(event)
            decode_steps = kept
        prefill_steps = [
            e for e in by_name.get("model_step", []) if e["metadata"].get("is_prefill")
        ]
        replay_steps = [
            e for e in decode_steps if e["metadata"].get("can_run_cuda_graph")
        ]
        eager_steps = [
            e for e in decode_steps if not e["metadata"].get("can_run_cuda_graph")
        ]

        batch_prep = by_name.get("talker_batch_prep", [])
        first_frame = by_name.get("talker_first_frame_predictor", [])
        entry: dict[str, Any] = {
            "decode_steps_total": len(decode_steps),
            "decode_graph_replay": (
                _summarize_steps(replay_steps) if replay_steps else None
            ),
            "decode_eager": _summarize_steps(eager_steps) if eager_steps else None,
            "decode_step_gap": _stats(_decode_step_gaps(decode_steps)),
            "prefill": _summarize_steps(prefill_steps) if prefill_steps else None,
            "batch_prep": _stats([e["metadata"]["duration_ms"] for e in batch_prep]),
            "batch_prep_deferred": sum(
                1 for e in batch_prep if e["metadata"].get("deferred")
            ),
            "first_frame_predictor": _stats(
                [e["metadata"]["duration_ms"] for e in first_frame]
            ),
            "predictor_eager_fallback": len(
                by_name.get("predictor_eager_fallback", [])
            ),
            "predictor_graph_capture_failed": len(
                by_name.get("predictor_graph_capture_failed", [])
            ),
        }
        report[f"run={run_id} stage={event_stage}"] = entry
    return report


def _format_table(report: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, entry in report.items():
        lines.append(key)
        lines.append("=" * len(key))
        lines.append(
            f"decode steps: {entry['decode_steps_total']} "
            f"(deferred batch preps: {entry['batch_prep_deferred']}, "
            f"eager fallbacks: {entry['predictor_eager_fallback']}, "
            f"capture failures: {entry['predictor_graph_capture_failed']})"
        )
        for label, block_key in (
            ("graph-replay decode", "decode_graph_replay"),
            ("eager decode", "decode_eager"),
            ("prefill", "prefill"),
        ):
            block = entry.get(block_key)
            if not block:
                continue
            step = block["step"]
            lines.append(
                f"\n{label}: n={step['count']} "
                f"p50={step['p50_ms']:.3f}ms p95={step['p95_ms']:.3f}ms"
            )
            lines.append(f"  {'segment':<32}{'p50 ms':>10}{'p95 ms':>10}{'share':>8}")
            for name, seg in sorted(
                block["segments"].items(),
                key=lambda item: -item[1]["share_of_step"],
            ):
                lines.append(
                    f"  {name:<32}{seg['p50_ms']:>10.3f}{seg['p95_ms']:>10.3f}"
                    f"{seg['share_of_step']:>7.1%}"
                )
        for label, stats_key in (
            ("step gap (outside runner)", "decode_step_gap"),
            ("batch prep", "batch_prep"),
            ("first-frame predictor", "first_frame_predictor"),
        ):
            stats = entry[stats_key]
            lines.append(
                f"{label}: n={stats['count']} "
                f"p50={stats['p50_ms']:.3f}ms p95={stats['p95_ms']:.3f}ms"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("event_dir", type=Path, help="Directory of events_*.jsonl")
    parser.add_argument("--stage", default=None, help="Only this stage (e.g. talker)")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Drop this many leading decode steps per process",
    )
    parser.add_argument("--format", choices=("table", "json"), default="table")
    args = parser.parse_args(argv)

    paths = sorted(args.event_dir.glob("events_*.jsonl"))
    if not paths:
        print(f"no events_*.jsonl under {args.event_dir}", file=sys.stderr)
        return 1
    events = load_step_events(paths)
    report = build_report(events, stage=args.stage, warmup_steps=args.warmup_steps)
    if not report:
        print(
            "no step-timing events found (was step timing recording?)", file=sys.stderr
        )
        return 1
    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(_format_table(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
