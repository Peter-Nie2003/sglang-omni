# SPDX-License-Identifier: Apache-2.0
"""Per-step host-side segment timing for AR decode loops.

Measurement-only instrumentation (issue sgl-project/sglang-omni#1039):
breaks one scheduler step into host segments (batch build, before_decode,
forward launch/replay, post_decode, finalize) and emits a single JSONL
event per step through the request-event recorder. Inert unless the
recorder is active — normal serving pays one ``is_active()`` check per
step and nothing else.

Durations are host wall time (``perf_counter``). On schedulers that run
without overlap (the talker), host segments serialize directly into the
step interval, which is exactly what these events measure; GPU self-time
must come from a torch trace instead.

Aggregate the emitted events with ``python -m
sglang_omni.profiler.step_report <event-dir>``.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from sglang_omni.profiler.event_recorder import get_recorder

# Step events are batch-scoped, not request-scoped; they carry a fixed
# synthetic request id so the per-request views simply group them apart.
STEP_REQUEST_ID = "__step__"

_local = threading.local()


class StepTiming:
    """Segment collector for one scheduler step (single-thread use).

    ``mark(name)`` attributes the time since the previous mark to ``name``;
    nested code may add sub-segments through :func:`active` (dotted names,
    e.g. ``before_decode.write_feedback``), in which case the enclosing
    mark only receives the remainder.
    """

    __slots__ = ("_t0", "_last", "segments", "flags")

    def __init__(self) -> None:
        now = time.perf_counter()
        self._t0 = now
        self._last = now
        self.segments: dict[str, float] = {}
        self.flags: dict[str, Any] = {}

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self.segments[name] = self.segments.get(name, 0.0) + (now - self._last)
        self._last = now

    def flag(self, **flags: Any) -> None:
        self.flags.update(flags)

    def emit(self, event_name: str) -> None:
        total = time.perf_counter() - self._t0
        get_recorder().emit(
            request_id=STEP_REQUEST_ID,
            stage=None,
            event_name=event_name,
            metadata={
                "segments_ms": {
                    name: seconds * 1e3 for name, seconds in self.segments.items()
                },
                "total_ms": total * 1e3,
                **self.flags,
            },
        )


def begin() -> StepTiming | None:
    """Start timing a step. Returns None when the recorder is inactive."""
    if not get_recorder().is_active():
        _local.current = None
        return None
    timing = StepTiming()
    _local.current = timing
    return timing


def active() -> StepTiming | None:
    """The in-flight step's collector, for nested sub-segment marks."""
    return getattr(_local, "current", None)


def finish(timing: StepTiming, event_name: str) -> None:
    """Emit the step event and clear the in-flight collector."""
    _local.current = None
    timing.emit(event_name)


def discard() -> None:
    """Drop the in-flight collector without emitting (empty batch)."""
    _local.current = None


def recording() -> bool:
    return get_recorder().is_active()


def emit_span(event_name: str, duration_s: float, **metadata: Any) -> None:
    """One-shot duration event outside the step collector (first-frame
    predictor call, deferred-batch prep)."""
    recorder = get_recorder()
    if not recorder.is_active():
        return
    recorder.emit(
        request_id=STEP_REQUEST_ID,
        stage=None,
        event_name=event_name,
        metadata={"duration_ms": duration_s * 1e3, **metadata},
    )
