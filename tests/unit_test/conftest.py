# SPDX-License-Identifier: Apache-2.0
"""Shared unit-test fixtures."""

import pytest

from sglang_omni.pipeline import runtime_config


@pytest.fixture(autouse=True)
def _detach_unit_tests_from_host_gpu_count(monkeypatch: pytest.MonkeyPatch) -> None:
    # Note (wenyao): GPU ids in these configs describe topology, not runner
    # hardware; the range check itself is covered in test_replicas.py.
    monkeypatch.setattr(runtime_config, "_visible_device_count", lambda: None)
