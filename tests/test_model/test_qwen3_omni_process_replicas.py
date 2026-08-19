# SPDX-License-Identifier: Apache-2.0
"""Process-replica smoke test for the Qwen3-Omni speech pipeline.

Launches a 2-GPU deployment with thinker on GPU 0, talker_ar on GPU 1,
and one code2wav replica on each GPU. Drives audio requests through it
and asserts both code2wav replicas are spawned, registered, and selected
by admission round-robin.

Requires 2 GPUs.

Usage:
    pytest tests/test_model/test_qwen3_omni_process_replicas.py -s -x
"""

from __future__ import annotations

import ast
import base64
import re
import sys
from pathlib import Path

import pytest
import requests
import torch

from sglang_omni.utils import find_available_port
from tests.utils import (
    disable_proxy,
    server_log_file,
    start_server_from_cmd,
    stop_server,
)

REQUIRED_GPUS = 2

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < REQUIRED_GPUS,
    reason=f"requires {REQUIRED_GPUS} GPUs",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
REPLICA_CONFIG = "examples/configs/qwen3_omni_speech_code2wav_replica2_ci.yaml"
STARTUP_TIMEOUT = 900
REQUEST_TIMEOUT = 300

NUM_REQUESTS = 4
REPLICATED_PROCESSES = ("code2wav",)
REPLICA_INSTANCES = (
    "code2wav@r0",
    "code2wav@r1",
)

PROMPTS = [
    "Please answer briefly: what is the capital of France?",
    "Count from one to five.",
    "Say hello in English.",
    "Name one primary color.",
]


@pytest.fixture(scope="module")
def replica_server(tmp_path_factory: pytest.TempPathFactory):
    port = find_available_port()
    # Note (wenyao): the test greps this log, so it must exist even locally
    # where server_log_file returns None.
    log_file = server_log_file(tmp_path_factory, "stage_replica_logs") or (
        tmp_path_factory.mktemp("stage_replica_logs") / "server.log"
    )
    cmd = [
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--config",
        str(PROJECT_ROOT / REPLICA_CONFIG),
        "--model-path",
        MODEL_PATH,
        "--port",
        str(port),
    ]
    proc = start_server_from_cmd(cmd, log_file, port, timeout=STARTUP_TIMEOUT, tee=True)
    proc.port = port  # type: ignore[attr-defined]
    proc.log_file = log_file  # type: ignore[attr-defined]
    yield proc
    stop_server(proc)


def _post_audio_request(port: int, prompt: str) -> dict:
    payload = {
        "model": MODEL_PATH,
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["text", "audio"],
        "audio": {"format": "wav"},
        "max_tokens": 256,
        "temperature": 0.0,
        "stream": False,
    }
    with disable_proxy():
        response = requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
    response.raise_for_status()
    return response.json()


def test_every_replica_serves_audio(replica_server):
    port: int = replica_server.port
    log_file: Path = replica_server.log_file

    for index in range(NUM_REQUESTS):
        body = _post_audio_request(port, PROMPTS[index % len(PROMPTS)])
        choice = body["choices"][0]
        audio = choice["message"].get("audio") or {}
        audio_b64 = audio.get("data")
        assert audio_b64, f"request {index}: no audio in response: {body}"
        audio_bytes = base64.b64decode(audio_b64)
        assert len(audio_bytes) > 1000, (
            f"request {index}: audio payload suspiciously small "
            f"({len(audio_bytes)} bytes)"
        )

    log_text = log_file.read_text()
    missing = [name for name in REPLICA_INSTANCES if name not in log_text]
    assert not missing, (
        f"replica instances never appeared in server log: {missing}; "
        "expected both code2wav instances to be spawned and registered"
    )

    admitted = re.findall(r"bindings=(\{.*?\})", log_text)
    assert (
        len(admitted) >= NUM_REQUESTS
    ), f"expected at least {NUM_REQUESTS} admission log lines, got {len(admitted)}"
    bound: dict[str, set[int]] = {}
    for raw in admitted:
        for stage, replica_id in ast.literal_eval(raw).items():
            bound.setdefault(stage, set()).add(replica_id)
    for process_name in REPLICATED_PROCESSES:
        assert bound.get(process_name) == {
            0,
            1,
        }, (
            f"{process_name} did not round-robin across both replicas: "
            f"{bound.get(process_name)}"
        )
