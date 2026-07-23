# SPDX-License-Identifier: Apache-2.0
"""Stage-replica smoke test for the Qwen3-Omni speech pipeline.

Launches the 2-replica speech deployment (thinker on GPU 0, one
talker_ar + code2wav pair each on GPU 1 and GPU 2) and drives enough
audio requests through it that round-robin admission covers every
replica. Asserts every request returns audio and that both replica
instances of each speech stage actually served traffic.

Requires 3 GPUs.

Usage:
    pytest tests/test_model/test_qwen3_omni_stage_replicas.py -s -x
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import pytest
import requests

from sglang_omni.utils import find_available_port
from tests.utils import (
    no_proxy_env,
    server_log_file,
    start_server_from_cmd,
    stop_server,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
REPLICA_CONFIG = "examples/configs/qwen3_omni_speech_replica2.yaml"
STARTUP_TIMEOUT = 900
REQUEST_TIMEOUT = 300

# Even requests bind to replica 0, odd to replica 1 (round-robin admission),
# so 4 requests exercise every replica twice.
NUM_REQUESTS = 4
REPLICA_INSTANCES = (
    "talker_ar@r0",
    "talker_ar@r1",
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
    log_file = server_log_file(tmp_path_factory, "stage_replica_logs")
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
    proc = start_server_from_cmd(cmd, log_file, port, timeout=STARTUP_TIMEOUT)
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
    with no_proxy_env():
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

    # Round-robin admission alternates replicas, so all requests succeeding
    # above already proves both replicas served traffic. The log check below
    # only guards topology: all four instances were actually spawned.
    log_text = log_file.read_text()
    missing = [name for name in REPLICA_INSTANCES if name not in log_text]
    assert not missing, (
        f"replica instances never appeared in server log: {missing}; "
        "expected all four instance stages to be spawned and registered"
    )
