# SPDX-License-Identifier: Apache-2.0
"""Static checks for the PR #1175 process-replica benchmark arms.

Each arm of that comparison is a YAML file, and a typo in one of them only
surfaces after a multi-minute server start on a GPU box. These tests compile
every arm on CPU in well under a second, so a broken topology fails before any
GPU time is spent — and they pin the placements the benchmark writeup claims,
so an arm cannot silently drift into testing something else.
"""

from __future__ import annotations

import pytest

from sglang_omni.config.manager import ConfigManager
from sglang_omni.pipeline.runtime_config import prepare_pipeline_runtime

QWEN_PAIR = "examples/configs/qwen3_omni_speech_pair2gpu.yaml"
QWEN_SPLIT = "examples/configs/qwen3_omni_speech_split3gpu.yaml"
QWEN_REPLICA = "examples/configs/qwen3_omni_speech_replica2.yaml"
HIGGS_SINGLE = "examples/configs/higgs_frontend_single.yaml"
HIGGS_REPLICA = "examples/configs/higgs_frontend_replica2.yaml"

ALL_ARMS = (QWEN_PAIR, QWEN_SPLIT, QWEN_REPLICA, HIGGS_SINGLE, HIGGS_REPLICA)


def _stage_gpus(config_path: str) -> dict[str, int | None]:
    """Compile a config the way the runner does and return stage -> GPU id."""
    config = ConfigManager.from_file(config_path).config
    prep = prepare_pipeline_runtime(config)
    try:
        return {stage.name: stage.gpu for stage in prep.stages_cfg}
    finally:
        prep.runtime_dir.close()


@pytest.mark.parametrize("config_path", ALL_ARMS)
def test_every_benchmark_arm_compiles(config_path: str) -> None:
    assert _stage_gpus(config_path)


def test_pair_arm_colocates_talker_and_code2wav_on_gpu1() -> None:
    gpus = _stage_gpus(QWEN_PAIR)

    # The pipeline default puts code2wav on the thinker GPU; this arm is only
    # meaningful if the override actually moved it next to the talker.
    assert gpus["thinker"] == 0
    assert gpus["talker_ar"] == 1
    assert gpus["code2wav"] == 1


def test_split_arm_gives_code2wav_its_own_gpu() -> None:
    gpus = _stage_gpus(QWEN_SPLIT)

    assert gpus["thinker"] == 0
    assert gpus["talker_ar"] == 1
    assert gpus["code2wav"] == 2


def test_replica_arm_spreads_both_replicated_processes_over_gpu1_and_gpu2() -> None:
    gpus = _stage_gpus(QWEN_REPLICA)

    assert gpus["thinker"] == 0
    assert gpus["talker_ar@r0"] == 1
    assert gpus["code2wav@r0"] == 1
    assert gpus["talker_ar@r1"] == 2
    assert gpus["code2wav@r1"] == 2
    # The unsuffixed names must be gone, otherwise something is running an
    # unreplicated stage alongside the replicas.
    assert "talker_ar" not in gpus
    assert "code2wav" not in gpus


def test_control_arms_declare_no_replica_instances() -> None:
    for config_path in (QWEN_PAIR, QWEN_SPLIT, HIGGS_SINGLE):
        gpus = _stage_gpus(config_path)
        suffixed = [name for name in gpus if "@r" in name]
        assert not suffixed, f"{config_path} unexpectedly replicated {suffixed}"


def test_higgs_arms_differ_only_in_frontend_replication() -> None:
    single = _stage_gpus(HIGGS_SINGLE)
    replicated = _stage_gpus(HIGGS_REPLICA)

    assert single["audio_encoder"] == 0
    assert replicated["audio_encoder@r0"] == 0
    assert replicated["audio_encoder@r1"] == 0
    assert replicated["preprocessing@r0"] is None
    assert replicated["preprocessing@r1"] is None

    # Everything downstream of the frontend must be identical, or the A/B is
    # measuring more than frontend replication.
    for stage in ("tts_engine", "vocoder"):
        assert single[stage] == replicated[stage] == 0


def test_higgs_arms_share_the_engine_and_encoder_memory_budget() -> None:
    single = ConfigManager.from_file(HIGGS_SINGLE).config
    replicated = ConfigManager.from_file(HIGGS_REPLICA).config

    def fractions(config) -> dict[str, float | None]:
        return {
            stage.name: stage.runtime.resources.total_gpu_memory_fraction
            for stage in config.stages
        }

    single_fractions = fractions(single)
    replicated_fractions = fractions(replicated)

    assert single_fractions["audio_encoder"] == pytest.approx(0.0245)
    assert replicated_fractions["audio_encoder"] == pytest.approx(0.0245)
    assert single_fractions["tts_engine"] == pytest.approx(0.85)
    assert replicated_fractions["tts_engine"] == pytest.approx(0.85)


def test_higgs_arms_share_the_scheduler_knobs() -> None:
    expected = {
        "max_new_tokens": 512,
        "max_running_requests": 96,
        "cuda_graph_max_bs": 96,
        "prefill_coalesce_requests": 32,
        "prefill_coalesce_wait_ms": 300,
    }
    for config_path in (HIGGS_SINGLE, HIGGS_REPLICA):
        config = ConfigManager.from_file(config_path).config
        assert config.runtime_overrides["tts_engine"] == expected, config_path
