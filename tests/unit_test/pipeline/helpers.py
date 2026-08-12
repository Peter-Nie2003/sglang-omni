# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from contextlib import suppress
from typing import Any

import torch

from sglang_omni.config.placement import build_stage_placement_plan
from sglang_omni.config.schema import PipelineConfig, StageConfig
from sglang_omni.config.topology import (
    ProcessTopologyPlan,
    build_process_topology_plan,
    compile_logical_processes,
)
from sglang_omni.pipeline.replicas import expand_replica_stages
from sglang_omni.pipeline.stage.runtime import Stage
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import IncomingMessage
from tests.unit_test.fixtures.pipeline_fakes import (
    FakeRelay,
    FakeScheduler,
    RecordingStageControlPlane,
    fake_factory_path,
)

FACTORY = fake_factory_path("make_scheduler")


def stage(name: str, **kwargs: Any) -> StageConfig:
    kwargs.setdefault("factory", FACTORY)
    if kwargs.get("tp_size", 1) == 1:
        kwargs.setdefault("process", "pipeline")
    return StageConfig(name=name, **kwargs)


def build_compiled_process_topology(
    config: PipelineConfig,
) -> ProcessTopologyPlan:
    logical_plan, stages = compile_logical_processes(config)
    stages, replica_topology = expand_replica_stages(stages, logical_plan)
    placement = build_stage_placement_plan(
        config,
        stages_cfg=stages,
        replica_instances=replica_topology.replicas,
    )
    return build_process_topology_plan(
        config,
        placement,
        stages_cfg=stages,
    )


def make_stage(
    *,
    name: str = "stage",
    role: str = "single",
    get_next=None,
    endpoints: dict[str, str] | None = None,
    gpu_id: int | None = None,
    scheduler: FakeScheduler | None = None,
    relay: FakeRelay | None = None,
    control_plane: RecordingStageControlPlane | None = None,
    **kwargs: Any,
) -> Stage:
    return Stage(
        name=name,
        role=role,
        get_next=get_next or (lambda request_id, output: None),
        gpu_id=gpu_id,
        endpoints=endpoints or {},
        control_plane=control_plane or RecordingStageControlPlane(),
        relay=relay or FakeRelay(),
        scheduler=scheduler or FakeScheduler(),
        **kwargs,
    )


async def round_trip_payload_over_shm(
    payload: StagePayload,
    *,
    from_stage: str,
    to_stage: str,
) -> StagePayload:
    """Round-trip a payload through the production SHM wire contract."""

    from sglang_omni.comm import stage_io
    from sglang_omni.comm.data_ref import DataRef, TransportKind
    from sglang_omni.pipeline.control_plane import (
        deserialize_message,
        serialize_message,
    )
    from sglang_omni.proto import DataReadyMessage
    from sglang_omni.relay.shm import ShmRelay

    probe_key = "_cross_process_relay_probe"
    wire_payload = StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data={**payload.data, probe_key: torch.tensor([1])},
    )
    relay = ShmRelay(
        engine_id=f"payload-round-trip-{uuid.uuid4().hex}",
        device="cpu",
    )
    operation = None
    operation_completed = False
    try:
        data_ref, operation = await stage_io.write_payload(
            relay,
            wire_payload.request_id,
            wire_payload,
            transport=TransportKind.SHM,
            from_stage=from_stage,
            to_stage=to_stage,
        )
        message = DataReadyMessage(
            request_id=wire_payload.request_id,
            from_stage=from_stage,
            to_stage=to_stage,
            data_ref=data_ref.to_dict(),
        )
        restored_message = deserialize_message(serialize_message(message))
        restored = await stage_io.read_payload(
            relay,
            wire_payload.request_id,
            DataRef.from_dict(restored_message.data_ref),
        )
        operation.mark_receiver_done()
        await operation.wait_for_completion()
        operation_completed = True
        probe = restored.data.pop(probe_key)
        assert probe.tolist() == [1]
        return restored
    finally:
        if operation is not None and not operation_completed:
            with suppress(Exception):
                operation.mark_receiver_failed(
                    RuntimeError("payload round-trip did not reach receiver ack")
                )
            with suppress(Exception):
                await operation.wait_for_completion()
        relay.close()


def run_scheduler(
    scheduler: Any,
    messages: list[IncomingMessage],
    *,
    output_count: int,
    before_collect: Callable[[], None] | None = None,
) -> list[Any]:
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    try:
        for message in messages:
            scheduler.inbox.put(message)
        if before_collect is not None:
            before_collect()
        return [scheduler.outbox.get(timeout=2.0) for _ in range(output_count)]
    finally:
        scheduler.stop()
        thread.join(timeout=2.0)
