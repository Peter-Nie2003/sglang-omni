# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang_omni.profiler.step_report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sglang_omni.profiler.step_report import build_report, load_step_events, main

_MS = 1_000_000  # ns per ms


def _step_event(
    ts_ms: float,
    *,
    total_ms: float,
    segments: dict[str, float],
    is_prefill: bool = False,
    can_run_cuda_graph: bool = True,
    pid: int = 100,
    run_id: str = "run_a",
    stage: str = "talker",
) -> dict:
    return {
        "request_id": "__step__",
        "stage": stage,
        "event_name": "model_step",
        "timestamp_ns": int(ts_ms * _MS),
        "run_id": run_id,
        "pid": pid,
        "metadata": {
            "segments_ms": segments,
            "total_ms": total_ms,
            "is_prefill": is_prefill,
            "batch_size": 8,
            "can_run_cuda_graph": can_run_cuda_graph,
        },
    }


def _span_event(
    ts_ms: float,
    name: str,
    *,
    duration_ms: float = 1.0,
    run_id: str = "run_a",
    stage: str = "talker",
    **extra,
) -> dict:
    return {
        "request_id": "__step__",
        "stage": stage,
        "event_name": name,
        "timestamp_ns": int(ts_ms * _MS),
        "run_id": run_id,
        "pid": 100,
        "metadata": {"duration_ms": duration_ms, **extra},
    }


def test_report_splits_replay_and_eager_and_computes_shares() -> None:
    events = [
        _step_event(0.0, total_ms=10.0, segments={"forward": 6.0, "post_decode": 4.0}),
        _step_event(20.0, total_ms=10.0, segments={"forward": 8.0, "post_decode": 2.0}),
        _step_event(
            40.0,
            total_ms=30.0,
            segments={"forward": 30.0},
            can_run_cuda_graph=False,
        ),
    ]
    report = build_report(events)
    entry = report["run=run_a stage=talker"]
    assert entry["decode_steps_total"] == 3

    replay = entry["decode_graph_replay"]
    assert replay["step"]["count"] == 2
    assert replay["segments"]["forward"]["share_of_step"] == pytest.approx(14 / 20)
    assert replay["segments"]["post_decode"]["share_of_step"] == pytest.approx(6 / 20)

    eager = entry["decode_eager"]
    assert eager["step"]["count"] == 1
    assert eager["segments"]["forward"]["share_of_step"] == pytest.approx(1.0)


def test_report_step_gap_subtracts_step_time() -> None:
    # Steps end at t=10 and t=30; step 2 itself took 12ms → 8ms gap.
    events = [
        _step_event(10.0, total_ms=10.0, segments={"forward": 10.0}),
        _step_event(30.0, total_ms=12.0, segments={"forward": 12.0}),
    ]
    report = build_report(events)
    gap = report["run=run_a stage=talker"]["decode_step_gap"]
    assert gap["count"] == 1
    assert gap["p50_ms"] == pytest.approx(8.0)


def test_report_gap_ignores_cross_pid_pairs() -> None:
    events = [
        _step_event(10.0, total_ms=10.0, segments={}, pid=1),
        _step_event(30.0, total_ms=12.0, segments={}, pid=2),
    ]
    report = build_report(events)
    assert report["run=run_a stage=talker"]["decode_step_gap"]["count"] == 0


def test_report_warmup_drops_leading_decode_steps_per_pid() -> None:
    events = [
        _step_event(float(i * 20), total_ms=10.0, segments={"forward": 10.0})
        for i in range(4)
    ]
    report = build_report(events, warmup_steps=2)
    entry = report["run=run_a stage=talker"]
    assert entry["decode_steps_total"] == 2


def test_report_counts_spans_and_health_events() -> None:
    events = [
        _step_event(0.0, total_ms=10.0, segments={"forward": 10.0}),
        _step_event(5.0, total_ms=50.0, segments={"forward": 50.0}, is_prefill=True),
        _span_event(1.0, "talker_batch_prep", duration_ms=0.5, deferred=False),
        _span_event(2.0, "talker_batch_prep", duration_ms=0.7, deferred=True),
        _span_event(3.0, "talker_first_frame_predictor", duration_ms=9.0),
        _span_event(4.0, "predictor_eager_fallback", batch_size=8),
        _span_event(5.0, "predictor_graph_capture_failed", bucket_size=16),
    ]
    report = build_report(events)
    entry = report["run=run_a stage=talker"]
    assert entry["prefill"]["step"]["count"] == 1
    assert entry["batch_prep"]["count"] == 2
    assert entry["batch_prep_deferred"] == 1
    assert entry["first_frame_predictor"]["p50_ms"] == pytest.approx(9.0)
    assert entry["predictor_eager_fallback"] == 1
    assert entry["predictor_graph_capture_failed"] == 1


def test_report_groups_by_run_and_filters_stage() -> None:
    events = [
        _step_event(0.0, total_ms=10.0, segments={}),
        _step_event(20.0, total_ms=10.0, segments={}, run_id="run_b"),
        _step_event(40.0, total_ms=10.0, segments={}, stage="thinker"),
    ]
    report = build_report(events, stage="talker")
    assert set(report) == {"run=run_a stage=talker", "run=run_b stage=talker"}


def test_main_end_to_end(tmp_path: Path, capsys) -> None:
    events = [
        _step_event(0.0, total_ms=10.0, segments={"forward": 9.0, "finalize": 1.0}),
        _step_event(20.0, total_ms=10.0, segments={"forward": 9.5, "finalize": 0.5}),
        # Non-step traffic must be ignored.
        {
            "request_id": "req-1",
            "stage": "talker",
            "event_name": "scheduler_first_emit",
            "timestamp_ns": 5 * _MS,
            "run_id": "run_a",
            "pid": 100,
            "metadata": {},
        },
    ]
    path = tmp_path / "events_talker_100.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    assert load_step_events([path])[0]["event_name"] == "model_step"

    assert main([str(tmp_path), "--stage", "talker", "--format", "json"]) == 0
    parsed = json.loads(capsys.readouterr().out)
    entry = parsed["run=run_a stage=talker"]
    assert entry["decode_steps_total"] == 2
    assert entry["decode_graph_replay"]["segments"]["forward"]["count"] == 2

    assert main([str(tmp_path), "--format", "table"]) == 0
    table = capsys.readouterr().out
    assert "graph-replay decode" in table
    assert "step gap" in table


def test_main_errors_on_missing_events(tmp_path: Path, capsys) -> None:
    assert main([str(tmp_path)]) == 1
    (tmp_path / "events_talker_1.jsonl").write_text(
        json.dumps(
            {
                "request_id": "req-1",
                "stage": "talker",
                "event_name": "scheduler_first_emit",
                "timestamp_ns": 0,
                "run_id": "run_a",
                "pid": 1,
                "metadata": {},
            }
        )
        + "\n"
    )
    assert main([str(tmp_path)]) == 1
