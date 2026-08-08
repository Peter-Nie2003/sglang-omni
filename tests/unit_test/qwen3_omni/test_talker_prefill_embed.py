# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_omni.components import talker_prefill
from sglang_omni.models.qwen3_omni.components.talker_prefill import TalkerPrefillBuilder


def _builder(cache: dict[int, torch.Tensor]) -> TalkerPrefillBuilder:
    builder = object.__new__(TalkerPrefillBuilder)
    builder._thinker_embed_cache = cache
    builder._projected_text_cache = {}
    builder._model_path = "unused"
    builder._device = torch.device("cpu")
    builder._dtype = torch.float32
    return builder


class _CountingProjection:
    def __init__(self, weight: torch.Tensor) -> None:
        self.weight = weight
        self.calls = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x @ self.weight


def _projecting_builder(
    cache: dict[int, torch.Tensor], weight: torch.Tensor
) -> tuple[TalkerPrefillBuilder, _CountingProjection]:
    builder = _builder(cache)
    projection = _CountingProjection(weight)
    builder._model = SimpleNamespace(text_projection=projection)
    return builder, projection


def _chunk(token_id: int | None, data: torch.Tensor | None = None) -> SimpleNamespace:
    return SimpleNamespace(metadata={"token_id": token_id}, data=data)


def test_single_token_fast_path_matches_multi_token_path_bitwise() -> None:
    torch.manual_seed(0)
    cache = {7: torch.randn(4), 9: torch.randn(4)}
    builder = _builder(dict(cache))

    fast = builder._load_prompt_token_embeddings(torch.tensor([7], dtype=torch.long))
    general = builder._load_prompt_token_embeddings(
        torch.tensor([7, 9], dtype=torch.long)
    )

    assert fast.shape == (1, 4)
    assert fast.dtype == general.dtype
    assert torch.equal(fast[0], general[0])
    assert torch.equal(fast[0], cache[7])


def test_single_token_fast_path_does_not_alias_cache() -> None:
    cache = {7: torch.zeros(4)}
    builder = _builder(cache)

    fast = builder._load_prompt_token_embeddings(torch.tensor([7], dtype=torch.long))
    fast[0] = 5.0

    assert torch.equal(cache[7], torch.zeros(4))


def test_single_token_fast_path_loads_and_caches_missing_row(monkeypatch) -> None:
    loaded_row = torch.arange(4, dtype=torch.float32)
    calls: list[list[int]] = []

    def _fake_load(model_path: str, row_ids: list[int]) -> torch.Tensor:
        calls.append(row_ids)
        return loaded_row.unsqueeze(0)

    monkeypatch.setattr(talker_prefill, "load_thinker_embedding_rows", _fake_load)
    builder = _builder({})

    out = builder._load_prompt_token_embeddings(torch.tensor([3], dtype=torch.long))
    again = builder._load_prompt_token_embeddings(torch.tensor([3], dtype=torch.long))

    assert calls == [[3]]
    assert torch.equal(out, loaded_row.unsqueeze(0))
    assert torch.equal(again, out)
    assert 3 in builder._thinker_embed_cache


def test_single_token_fast_path_skips_the_general_machinery(monkeypatch) -> None:
    def _boom(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("the single-token path must not call torch.unique")

    monkeypatch.setattr(torch, "unique", _boom)
    builder = _builder({7: torch.randn(4)})

    out = builder._load_prompt_token_embeddings(torch.tensor([7], dtype=torch.long))

    assert out.shape == (1, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a device tensor")
def test_single_token_device_tensor_falls_back_to_general_path() -> None:
    row = torch.randn(4)
    builder = _builder({7: row})

    out = builder._load_prompt_token_embeddings(
        torch.tensor([7], dtype=torch.long, device="cuda")
    )

    assert out.shape == (1, 4)
    assert torch.equal(out[0], row)


def test_projected_text_row_is_computed_once_per_token_id() -> None:
    torch.manual_seed(0)
    builder, projection = _projecting_builder(
        {7: torch.randn(4), 9: torch.randn(4)}, torch.randn(4, 6)
    )

    first = builder.project_assistant_chunk(_chunk(7))
    repeat = builder.project_assistant_chunk(_chunk(7))
    other = builder.project_assistant_chunk(_chunk(9))

    assert projection.calls == 2
    assert first is repeat
    assert other is not first


def test_projected_text_row_matches_an_uncached_projection() -> None:
    torch.manual_seed(0)
    row = torch.randn(4)
    weight = torch.randn(4, 6)
    builder, _ = _projecting_builder({7: row}, weight)

    out = builder.project_assistant_chunk(_chunk(7))

    assert out.shape == (6,)
    assert torch.equal(out, (row.unsqueeze(0) @ weight)[0])


def test_projected_text_row_loads_a_missing_embed_row_once(monkeypatch) -> None:
    loaded_row = torch.arange(4, dtype=torch.float32)
    calls: list[list[int]] = []

    def _fake_load(model_path: str, row_ids: list[int]) -> torch.Tensor:
        calls.append(row_ids)
        return loaded_row.unsqueeze(0)

    monkeypatch.setattr(talker_prefill, "load_thinker_embedding_rows", _fake_load)
    builder, projection = _projecting_builder({}, torch.eye(4))

    first = builder.project_assistant_chunk(_chunk(3))
    repeat = builder.project_assistant_chunk(_chunk(3))

    assert calls == [[3]]
    assert projection.calls == 1
    assert first is repeat
    assert torch.equal(first, loaded_row)
    assert 3 in builder._thinker_embed_cache


def test_chunk_without_token_id_bypasses_the_cache() -> None:
    builder, projection = _projecting_builder({}, torch.eye(4))
    data = torch.arange(4, dtype=torch.float32)

    builder.project_assistant_chunk(_chunk(None, data))
    builder.project_assistant_chunk(_chunk(None, data))

    assert projection.calls == 2
    assert builder._projected_text_cache == {}
