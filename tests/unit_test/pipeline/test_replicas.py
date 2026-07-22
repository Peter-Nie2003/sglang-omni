# SPDX-License-Identifier: Apache-2.0
"""Unit tests for runtime-level stage replicas."""

import pytest

from sglang_omni.config.schema import PipelineConfig, StageConfig
from sglang_omni.pipeline.replicas import (
    ReplicaTopology,
    RoundRobinBindingPolicy,
    assign_replica_bindings,
    expand_replica_stages,
    parse_replica_instance_name,
    replica_instance_name,
    split_replica_devices,
)


def _stage(name: str, **kwargs) -> StageConfig:
    defaults = dict(factory="pkg.mod.create", terminal=True, process=name)
    defaults.update(kwargs)
    return StageConfig(name=name, **defaults)


class TestInstanceNaming:
    def test_round_trip(self):
        name = replica_instance_name("talker_ar", 1)
        assert name == "talker_ar@r1"
        assert parse_replica_instance_name(name) == ("talker_ar", 1)

    def test_plain_name_passthrough(self):
        assert parse_replica_instance_name("thinker") == ("thinker", None)

    def test_non_numeric_suffix_is_not_replica(self):
        assert parse_replica_instance_name("stage@rx") == ("stage@rx", None)


class TestSplitReplicaDevices:
    def test_pool_mode_tp1(self):
        assert split_replica_devices(
            "1,2", stage_name="s", num_replicas=2, tp_size=1
        ) == [[1], [2]]

    def test_pool_mode_tp2(self):
        assert split_replica_devices(
            "0,1,2,3", stage_name="s", num_replicas=2, tp_size=2
        ) == [[0, 1], [2, 3]]

    def test_template_mode_tp1(self):
        assert split_replica_devices(
            "0", stage_name="s", num_replicas=4, tp_size=1
        ) == [[0], [1], [2], [3]]

    def test_template_mode_tp2(self):
        assert split_replica_devices(
            "0,1", stage_name="s", num_replicas=2, tp_size=2
        ) == [[0, 1], [2, 3]]

    def test_list_input(self):
        assert split_replica_devices(
            [1, 2], stage_name="s", num_replicas=2, tp_size=1
        ) == [[1], [2]]

    def test_none_for_cpu_stage(self):
        assert split_replica_devices(
            None, stage_name="s", num_replicas=3, tp_size=1
        ) == [None, None, None]

    def test_invalid_length_raises(self):
        with pytest.raises(ValueError, match="replica_devices has 3"):
            split_replica_devices("0,1,2", stage_name="s", num_replicas=2, tp_size=2)


class TestExpandReplicaStages:
    def test_no_replicas_is_identity(self):
        stages = [_stage("a"), _stage("b")]
        expanded, topo = expand_replica_stages(stages)
        assert expanded == stages
        assert not topo
        assert topo.to_dict() == {}

    def test_expansion_names_gpus_processes(self):
        stages = [
            _stage(
                "talker_ar",
                terminal=False,
                next="code2wav",
                stream_to=["code2wav"],
                gpu=1,
                num_replicas=2,
                replica_devices="1,2",
            ),
            _stage("code2wav"),
        ]
        expanded, topo = expand_replica_stages(stages)
        names = [s.name for s in expanded]
        assert names == ["talker_ar@r0", "talker_ar@r1", "code2wav"]
        r0, r1 = expanded[0], expanded[1]
        assert (r0.gpu, r1.gpu) == (1, 2)
        assert r0.process == "talker_ar@r0"
        assert r0.num_replicas == 1 and r0.replica_devices is None
        # Wiring keeps logical names.
        assert r0.next == "code2wav" and r0.stream_to == ["code2wav"]
        assert topo.to_dict() == {"talker_ar": ["talker_ar@r0", "talker_ar@r1"]}

    def test_gpu_field_fallback_as_template(self):
        stages = [_stage("s", gpu=1, num_replicas=2)]
        expanded, _ = expand_replica_stages(stages)
        assert [s.gpu for s in expanded] == [1, 2]


class TestReplicaTopology:
    def _topo(self) -> ReplicaTopology:
        _, topo = expand_replica_stages(
            [
                _stage("talker_ar", num_replicas=2, replica_devices="1,2", gpu=1),
                _stage("code2wav", num_replicas=2, replica_devices="1,2", gpu=1),
                _stage("thinker"),
            ]
        )
        return topo

    def test_resolve_and_logical_name(self):
        topo = self._topo()
        assert topo.resolve("talker_ar", 1) == "talker_ar@r1"
        assert topo.logical_name("talker_ar@r1") == "talker_ar"
        assert topo.logical_name("thinker") == "thinker"

    def test_resolve_out_of_range(self):
        with pytest.raises(ValueError, match="has 2 replicas"):
            self._topo().resolve("talker_ar", 5)

    def test_resolve_unreplicated(self):
        topo = self._topo()
        assert topo.resolve("thinker", 0) == "thinker"
        with pytest.raises(ValueError, match="not replicated"):
            topo.resolve("thinker", 1)

    def test_instances(self):
        topo = self._topo()
        assert topo.instances("code2wav") == ("code2wav@r0", "code2wav@r1")
        assert topo.instances("thinker") == ("thinker",)

    def test_unregistered_suffix_name_is_not_normalized(self):
        assert self._topo().logical_name("other@r0") == "other@r0"

    def test_dict_round_trip(self):
        topo = self._topo()
        restored = ReplicaTopology.from_dict(topo.to_dict())
        assert restored == topo
        assert not ReplicaTopology.from_dict(None)


class TestBinding:
    def test_round_robin_cycles_per_stage(self):
        policy = RoundRobinBindingPolicy()
        picks = [policy.bind("talker_ar", 2, f"req{i}") for i in range(4)]
        assert picks == [0, 1, 0, 1]
        assert policy.bind("code2wav", 3, "reqx") == 0

    def test_assign_bindings(self):
        _, topo = expand_replica_stages(
            [
                _stage("talker_ar", num_replicas=2, replica_devices="1,2", gpu=1),
                _stage("code2wav", num_replicas=2, replica_devices="1,2", gpu=1),
            ]
        )
        policy = RoundRobinBindingPolicy()
        first = assign_replica_bindings(topo, policy, "req0")
        second = assign_replica_bindings(topo, policy, "req1")
        assert first == {"talker_ar": 0, "code2wav": 0}
        assert second == {"talker_ar": 1, "code2wav": 1}

    def test_empty_topology_binds_none(self):
        assert (
            assign_replica_bindings(ReplicaTopology(), RoundRobinBindingPolicy(), "r")
            is None
        )


class TestSchemaValidation:
    def test_num_replicas_must_be_positive(self):
        with pytest.raises(ValueError, match="num_replicas >= 1"):
            _stage("s", num_replicas=0)

    def test_entry_stage_cannot_be_replicated(self):
        with pytest.raises(ValueError, match="cannot be replicated"):
            PipelineConfig(
                model_path="m",
                stages=[
                    _stage("entry", terminal=False, next="sink", num_replicas=2),
                    _stage("sink"),
                ],
            )

    def test_fused_group_cannot_include_replicated_stage(self):
        with pytest.raises(ValueError, match="cannot include replicated"):
            PipelineConfig(
                model_path="m",
                stages=[
                    _stage("a", terminal=False, next="b"),
                    _stage("b", terminal=False, next="c", num_replicas=2),
                    _stage("c"),
                ],
                fused_stages=[["a", "b"]],
            )
