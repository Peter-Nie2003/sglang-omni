# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from sglang_omni.config import EndpointsConfig, PipelineConfig, StageConfig
from sglang_omni.models.ming_tts.config import (
    PREPROCESSING_STAGE,
    REFERENCE_ENCODE_STAGE,
    TTS_ENGINE_STAGE,
    MingTTSPipelineConfig,
)
from sglang_omni.models.ming_tts.payload_types import (
    MING_TTS_DEFAULT_MAX_DECODE_STEPS,
    MingTTSState,
)
from sglang_omni.models.ming_tts.prompt_builder import build_ming_tts_prompt
from sglang_omni.models.ming_tts.request_builders import preprocess_ming_tts_payload
from sglang_omni.models.ming_tts.tokenizer import (
    AUDIO_PATCH_TOKEN,
    AUDIO_START_TOKEN,
    SPK_END_TOKEN,
    SPK_START_TOKEN,
    MingTTSSpecialTokenIds,
    MingTTSTokenizerBundle,
)
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
from sglang_omni.proto import OmniRequest, StagePayload
from tests.unit_test.pipeline.helpers import build_compiled_process_topology


class _FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text in ("<role>HUMAN</role>", "<role>ASSISTANT</role>"):
            return [1, 2]
        if text == AUDIO_PATCH_TOKEN:
            return [3]
        if text == AUDIO_START_TOKEN:
            return [4]
        if text == SPK_START_TOKEN:
            return [6]
        if text == f"{SPK_END_TOKEN}\n":
            return [7]
        if text:
            return [10]
        return []

    def __len__(self) -> int:
        return 128


def _tokenizer() -> MingTTSTokenizerBundle:
    return MingTTSTokenizerBundle(
        tokenizer=_FakeTokenizer(),
        special=MingTTSSpecialTokenIds(
            bos=8,
            eos=9,
            pad=9,
            role_start=1,
            role_end=2,
            audio_patch=3,
            audio_start=4,
            end_of_audio=5,
            spk_start=6,
            spk_end=7,
        ),
    )


def _make_ming_process_edge_scheduler(*, model_path: str, role: str) -> Any:
    """Run both Ming frontend handoffs through real model adapters."""

    del model_path
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    tokenizer = _tokenizer()
    if role == "preprocessing":

        def compute(payload: StagePayload) -> StagePayload:
            return preprocess_ming_tts_payload(
                payload,
                tokenizer=tokenizer,
                context_length=512,
            )

        return SimpleScheduler(compute)

    if role == "reference_encode":
        from sglang_omni.models.ming_tts.reference_encode import (
            MingTTSReferenceEncoder,
        )

        encoder = MingTTSReferenceEncoder.__new__(MingTTSReferenceEncoder)
        encoder._service = None

        def encode_reference(ref_audio: str) -> dict[str, Any]:
            assert ref_audio == "/tmp/reference.wav"
            return {
                "spk_emb": torch.tensor([[0.25, 0.75]], dtype=torch.float32),
                "prompt_latent": torch.tensor(
                    [[[1.0, 2.0], [3.0, 4.0]]],
                    dtype=torch.float32,
                ),
                "prompt_latent_token_count": 2,
            }

        encoder._encode_reference = encode_reference

        def compute(payload: StagePayload) -> StagePayload:
            return encoder.encode_payload(
                payload,
                tokenizer=tokenizer,
                context_length=512,
            )

        return SimpleScheduler(compute)

    if role == "tts_engine":
        from sglang_omni.models.ming_tts.engine_io import (
            make_ming_tts_scheduler_adapters,
        )

        request_builder, _ = make_ming_tts_scheduler_adapters(
            model=SimpleNamespace(config=SimpleNamespace(vocab_size=128)),
            tokenizer=tokenizer,
            reset_request=lambda request_id: None,
        )

        def compute(payload: StagePayload) -> StagePayload:
            request_data = request_builder(payload)
            state = request_data.state
            payload.data.update(
                {
                    "probe_engine_pid": os.getpid(),
                    "probe_input_ids": request_data.input_ids.tolist(),
                    "probe_projected_prefill": request_data.input_embeds_are_projected,
                    "probe_speaker_embedding": (
                        state.spk_emb.tolist() if state.spk_emb is not None else None
                    ),
                    "probe_prompt_latent": (
                        state.prompt_latent.tolist()
                        if state.prompt_latent is not None
                        else None
                    ),
                    "probe_prompt_text": state.prompt_text,
                }
            )
            return payload

        return SimpleScheduler(compute)

    raise ValueError(f"unknown Ming process-edge probe role: {role!r}")


def _payload(*, params: dict | None = None, tts_params: dict | None = None):
    return StagePayload(
        request_id="req-ming-tts",
        request=OmniRequest(
            inputs="hello",
            params=params or {},
            metadata={"tts_params": tts_params or {}},
        ),
        data={},
    )


@pytest.mark.parametrize(
    ("frontend_stages", "edge"),
    [
        (
            {PREPROCESSING_STAGE},
            (PREPROCESSING_STAGE, REFERENCE_ENCODE_STAGE),
        ),
        (
            {PREPROCESSING_STAGE, REFERENCE_ENCODE_STAGE},
            (REFERENCE_ENCODE_STAGE, TTS_ENGINE_STAGE),
        ),
    ],
)
def test_ming_frontend_edges_remain_process_local_for_compatibility(
    frontend_stages: set[str],
    edge: tuple[str, str],
) -> None:
    config = MingTTSPipelineConfig(model_path="model")
    for stage in config.stages:
        if stage.name in frontend_stages:
            stage.process = "ming_frontend"

    assert config.process_local_edges() == frozenset(
        {
            (PREPROCESSING_STAGE, REFERENCE_ENCODE_STAGE),
            (REFERENCE_ENCODE_STAGE, TTS_ENGINE_STAGE),
        }
    )
    with pytest.raises(ValueError, match="Cross-process edge") as exc_info:
        build_compiled_process_topology(config)
    assert f"{edge[0]!r} -> {edge[1]!r}" in str(exc_info.value)


def test_ming_engine_to_audio_decode_remains_cross_process_safe() -> None:
    config = MingTTSPipelineConfig(model_path="model")
    config.stages[-1].process = "ming_audio_decode"
    fractions = {
        "reference_encode": 0.08,
        "tts_engine": 0.72,
        "audio_decode": 0.12,
    }
    for stage in config.stages:
        if stage.name in fractions:
            stage.runtime.resources.total_gpu_memory_fraction = fractions[stage.name]

    plan = build_compiled_process_topology(config)

    assert plan.stage_to_process["tts_engine"] == "pipeline"
    assert plan.stage_to_process["audio_decode"] == "ming_audio_decode"


def test_ming_tts_prompt_embedding_positions_match_special_tokens() -> None:
    tokenizer = _tokenizer()
    prompt_latent_token_count = 3
    plan = build_ming_tts_prompt(
        MingTTSState(text="target text", prompt="prompt"),
        tokenizer,
        prompt_text="reference text",
        speaker_count=1,
        prompt_latent_token_count=prompt_latent_token_count,
    )

    injection_position = plan.spk_injection_positions[0]
    assert plan.input_ids[injection_position - 1] == tokenizer.special.spk_start
    assert plan.input_ids[injection_position] == tokenizer.special.audio_patch

    latent_start = plan.prompt_latent_start_position
    assert latent_start is not None
    assert plan.input_ids[latent_start - 1] == tokenizer.special.audio_start
    assert (
        plan.input_ids[latent_start : latent_start + prompt_latent_token_count]
        == [tokenizer.special.audio_patch] * prompt_latent_token_count
    )


@pytest.mark.parametrize(
    ("params", "tts_params"),
    [
        ({}, {"seed": 1}),
        ({"seed": 1}, {}),
        ({"stage_params": {"tts_engine": {"seed": 1}}}, {}),
    ],
)
def test_ming_tts_rejects_seed_until_fl_rng_contract_exists(
    params: dict,
    tts_params: dict,
) -> None:
    with pytest.raises(ValueError, match="seed is currently unsupported"):
        preprocess_ming_tts_payload(
            _payload(params=params, tts_params=tts_params),
            tokenizer=_tokenizer(),
            context_length=MING_TTS_DEFAULT_MAX_DECODE_STEPS + 64,
        )


@pytest.mark.parametrize(
    ("params", "tts_params"),
    [
        ({}, {"initial_codec_chunk_frames": 1}),
        ({"initial_codec_chunk_frames": 1}, {}),
        ({"initial_codec_chunk_frames": 0}, {}),
    ],
)
def test_ming_tts_rejects_initial_codec_chunk_frames(
    params: dict,
    tts_params: dict,
) -> None:
    with pytest.raises(
        ValueError,
        match="initial_chunk_patches.*steady_chunk_patches",
    ):
        preprocess_ming_tts_payload(
            _payload(params=params, tts_params=tts_params),
            tokenizer=_tokenizer(),
            context_length=MING_TTS_DEFAULT_MAX_DECODE_STEPS + 64,
        )


@pytest.mark.parametrize("name", ["cfg", "sigma", "temperature"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_ming_tts_rejects_non_finite_sampling_params(name: str, value: float) -> None:
    with pytest.raises(ValueError, match=f"{name} must be a finite number"):
        preprocess_ming_tts_payload(
            _payload(tts_params={name: value}),
            tokenizer=_tokenizer(),
            context_length=MING_TTS_DEFAULT_MAX_DECODE_STEPS + 64,
        )


def _reference_payload(reference: dict) -> StagePayload:
    return StagePayload(
        request_id="req-ming-tts",
        request=OmniRequest(
            inputs={"text": "hello", "references": [reference]},
            params={},
            metadata={"tts_params": {}},
        ),
        data={},
    )


def test_ming_tts_rejects_inline_reference_audio() -> None:
    with pytest.raises(ValueError, match="local file path"):
        preprocess_ming_tts_payload(
            _reference_payload({"data": "AAAA", "media_type": "audio/wav"}),
            tokenizer=_tokenizer(),
            context_length=MING_TTS_DEFAULT_MAX_DECODE_STEPS + 64,
        )


def test_ming_tts_rejects_reference_without_audio_path() -> None:
    with pytest.raises(ValueError, match="local reference audio path"):
        preprocess_ming_tts_payload(
            _reference_payload({"speaker": "a"}),
            tokenizer=_tokenizer(),
            context_length=MING_TTS_DEFAULT_MAX_DECODE_STEPS + 64,
        )


@pytest.mark.asyncio
async def test_ming_frontend_wire_contract_crosses_process_boundaries(
    tmp_path,
) -> None:
    factory = f"{__name__}._make_ming_process_edge_scheduler"
    config = PipelineConfig(
        model_path="model",
        entry_stage="preprocessing",
        stages=[
            StageConfig(
                name="preprocessing",
                process="ming_preprocessing",
                factory=factory,
                factory_args={"role": "preprocessing"},
                next="reference_encode",
            ),
            StageConfig(
                name="reference_encode",
                process="ming_reference_encode",
                factory=factory,
                factory_args={"role": "reference_encode"},
                next="tts_engine",
            ),
            StageConfig(
                name="tts_engine",
                process="ming_tts_engine",
                factory=factory,
                factory_args={"role": "tts_engine"},
                terminal=True,
            ),
        ],
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
    )
    requests = {
        True: OmniRequest(
            inputs={
                "text": "hello",
                "references": [
                    {
                        "audio_path": "/tmp/reference.wav",
                        "text": "reference text",
                    }
                ],
            },
            params={},
            metadata={"tts_params": {}},
        ),
        False: OmniRequest(
            inputs="hello",
            params={},
            metadata={"tts_params": {}},
        ),
    }
    runner = MultiProcessPipelineRunner(config)

    await runner.start(timeout=30.0)
    processes = [process for group in runner._groups for process in group.processes]
    process_pids = {
        group.group_name: group.processes[0].pid for group in runner._groups
    }
    try:
        results = {
            with_reference: await asyncio.wait_for(
                runner.coordinator.submit(
                    f"ming-cross-process-{with_reference}",
                    requests[with_reference],
                ),
                timeout=15.0,
            )
            for with_reference in (False, True)
        }
    finally:
        await runner.stop()

    assert len(set(process_pids.values())) == 3
    for with_reference, result in results.items():
        assert result["text"] == "hello"
        assert result["probe_input_ids"] == result["input_ids"]
        assert result["probe_projected_prefill"] is with_reference
        assert result["max_decode_steps"] == MING_TTS_DEFAULT_MAX_DECODE_STEPS
        assert result["cfg"] == 2.0
        assert result["sigma"] == 0.25
        assert result["temperature"] == 0.0
        assert result["probe_engine_pid"] == process_pids["ming_tts_engine"]

    referenced = results[True]
    assert referenced["ref_audio"] == "/tmp/reference.wav"
    assert referenced["ref_text"] == "reference text"
    assert referenced["probe_speaker_embedding"] == [[0.25, 0.75]]
    assert referenced["probe_prompt_latent"] == [[[1.0, 2.0], [3.0, 4.0]]]
    assert referenced["probe_prompt_text"] == "reference text"
    assert referenced["spk_emb_shape"] == [1, 2]
    assert referenced["spk_emb_dtype"] == "float32"
    assert referenced["prompt_latent_shape"] == [1, 2, 2]
    assert referenced["prompt_latent_dtype"] == "float32"

    unreferenced = results[False]
    assert unreferenced["probe_speaker_embedding"] is None
    assert unreferenced["probe_prompt_latent"] is None
    assert "spk_emb_bytes" not in unreferenced
    assert "prompt_latent_bytes" not in unreferenced
    assert all(not process.is_alive() for process in processes)
    assert [process.exitcode for process in processes] == [0, 0, 0]
    assert list(tmp_path.iterdir()) == []
