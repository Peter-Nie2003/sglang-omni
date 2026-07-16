# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang_omni.profiler.step_timing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sglang_omni.profiler import step_timing
from sglang_omni.profiler.event_recorder import get_recorder, reset_active_stage


@pytest.fixture(autouse=True)
def _reset_recorder():
    rec = get_recorder()
    if rec.is_active():
        rec.stop()
    reset_active_stage(None)
    step_timing.discard()
    yield
    if rec.is_active():
        rec.stop()
    step_timing.discard()


def _read_events(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def test_begin_is_noop_when_recorder_inactive() -> None:
    assert not step_timing.recording()
    assert step_timing.begin() is None
    assert step_timing.active() is None
    # emit_span must be silent too.
    step_timing.emit_span("talker_first_frame_predictor", 0.001, batch_size=1)


def test_step_timing_emits_segments_and_flags(tmp_path: Path) -> None:
    path = get_recorder().start("run_a", str(tmp_path), "talker")

    timing = step_timing.begin()
    assert timing is not None
    assert step_timing.active() is timing

    timing.mark("build_batch")
    # Nested sub-segments through active(), the way talker before_decode does.
    nested = step_timing.active()
    assert nested is not None
    nested.mark("before_decode.prepare_buffers")
    nested.mark("before_decode.write_feedback")
    timing.mark("before_decode")
    timing.mark("forward")
    timing.flag(is_prefill=False, batch_size=4, can_run_cuda_graph=True)
    step_timing.finish(timing, "model_step")
    assert step_timing.active() is None

    events = _read_events(path)
    assert len(events) == 1
    event = events[0]
    assert event["event_name"] == "model_step"
    assert event["request_id"] == step_timing.STEP_REQUEST_ID
    assert event["stage"] == "talker"
    metadata = event["metadata"]
    assert metadata["is_prefill"] is False
    assert metadata["batch_size"] == 4
    assert metadata["can_run_cuda_graph"] is True
    segments = metadata["segments_ms"]
    assert set(segments) == {
        "build_batch",
        "before_decode.prepare_buffers",
        "before_decode.write_feedback",
        "before_decode",
        "forward",
    }
    assert all(value >= 0.0 for value in segments.values())
    assert metadata["total_ms"] >= max(segments.values())


def test_mark_accumulates_repeated_names(tmp_path: Path) -> None:
    path = get_recorder().start("run_b", str(tmp_path), "talker")
    timing = step_timing.begin()
    assert timing is not None
    timing.mark("forward")
    timing.mark("forward")
    step_timing.finish(timing, "model_step")
    events = _read_events(path)
    assert list(events[0]["metadata"]["segments_ms"]) == ["forward"]


def test_discard_drops_without_emitting(tmp_path: Path) -> None:
    path = get_recorder().start("run_c", str(tmp_path), "talker")
    timing = step_timing.begin()
    assert timing is not None
    step_timing.discard()
    assert step_timing.active() is None
    assert _read_events(path) == []


def test_emit_span_records_duration_and_metadata(tmp_path: Path) -> None:
    path = get_recorder().start("run_d", str(tmp_path), "talker")
    step_timing.emit_span(
        "talker_batch_prep", 0.002, deferred=True, is_decode=True, batch_size=2
    )
    events = _read_events(path)
    assert len(events) == 1
    metadata = events[0]["metadata"]
    assert metadata["duration_ms"] == pytest.approx(2.0)
    assert metadata["deferred"] is True
    assert metadata["is_decode"] is True
    assert metadata["batch_size"] == 2
