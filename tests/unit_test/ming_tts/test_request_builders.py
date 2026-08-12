# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.config.topology import compile_logical_processes
from sglang_omni.models.ming_tts.config import MingTTSPipelineConfig
from sglang_omni.models.ming_tts.engine_io import make_ming_tts_scheduler_adapters
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
from sglang_omni.proto import OmniRequest, StagePayload
from tests.unit_test.pipeline.helpers import round_trip_payload_over_shm


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


def test_ming_frontend_edges_can_cross_process() -> None:
    config = MingTTSPipelineConfig(model_path="model")
    for stage in config.stages[:3]:
        stage.process = stage.name

    plan, _ = compile_logical_processes(config)

    assert plan.stage_to_process["preprocessing"] == "preprocessing"
    assert plan.stage_to_process["reference_encode"] == "reference_encode"
    assert plan.stage_to_process["tts_engine"] == "tts_engine"


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


@pytest.mark.parametrize("with_reference", [False, True])
def test_ming_frontend_payload_survives_cross_process_wire(
    with_reference: bool,
) -> None:
    tokenizer = _tokenizer()
    payload = (
        _reference_payload(
            {"audio_path": "/tmp/reference.wav", "text": "reference text"}
        )
        if with_reference
        else _payload()
    )
    preprocessed = preprocess_ming_tts_payload(
        payload,
        tokenizer=tokenizer,
        context_length=512,
    )
    reference_input = asyncio.run(
        round_trip_payload_over_shm(
            preprocessed,
            from_stage="preprocessing",
            to_stage="reference_encode",
        )
    )

    from sglang_omni.models.ming_tts.reference_encode import MingTTSReferenceEncoder

    speaker_embedding = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    prompt_latent = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]]],
        dtype=torch.float32,
    )

    def encode_reference(ref_audio: str) -> dict:
        assert ref_audio == "/tmp/reference.wav"
        return {
            "spk_emb": speaker_embedding,
            "prompt_latent": prompt_latent,
            "prompt_latent_token_count": 2,
        }

    encoder = MingTTSReferenceEncoder.__new__(MingTTSReferenceEncoder)
    encoder._service = None
    encoder._encode_reference = encode_reference
    engine_input = encoder.encode_payload(
        reference_input,
        tokenizer=tokenizer,
        context_length=512,
    )
    restored = asyncio.run(
        round_trip_payload_over_shm(
            engine_input,
            from_stage="reference_encode",
            to_stage="tts_engine",
        )
    )
    request_builder, _ = make_ming_tts_scheduler_adapters(
        model=SimpleNamespace(config=SimpleNamespace(vocab_size=128)),
        tokenizer=tokenizer,
        reset_request=lambda request_id: None,
    )
    request_data = request_builder(restored)

    assert request_data.state.text == "hello"
    assert request_data.input_ids.tolist() == request_data.state.input_ids
    assert request_data.input_embeds_are_projected is with_reference
    if with_reference:
        assert torch.equal(request_data.state.spk_emb, speaker_embedding)
        assert torch.equal(request_data.state.prompt_latent, prompt_latent)
        assert request_data.state.prompt_text == "reference text"
    else:
        assert request_data.state.spk_emb is None
        assert request_data.state.prompt_latent is None
