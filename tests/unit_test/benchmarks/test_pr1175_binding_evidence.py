# SPDX-License-Identifier: Apache-2.0
"""Replica-binding verdict for the PR #1175 process-replica harness.

This verdict is the gate that decides whether a measured speedup may be
attributed to process replication at all, so it has to be right about a log
it only sees on a GPU box. The case that matters is that
``assign_replica_bindings`` keys its dict by *member stage*, not by process:
those names coincide for the Qwen3-Omni arms and diverge for Higgs, and a
checker that assumes they coincide silently passes one suite while failing a
perfectly healthy run of the other.
"""

from __future__ import annotations

from benchmarks.eval.bench_process_replicas import (
    HIGGS_ARMS,
    QWEN_ARMS,
    _binding_evidence,
    _member_stages,
)

HIGGS_REPLICA_ARM = next(arm for arm in HIGGS_ARMS if arm.key == "frontend2")
HIGGS_CONTROL_ARM = next(arm for arm in HIGGS_ARMS if arm.key == "frontend1")
QWEN_REPLICA_ARM = next(arm for arm in QWEN_ARMS if arm.key == "replica2")
QWEN_CONTROL_ARM = next(arm for arm in QWEN_ARMS if arm.key == "split3gpu")


def _higgs_log(*binding_reprs: str) -> str:
    spawned = "starting tts_frontend@r0\nstarting tts_frontend@r1\n"
    lines = [
        f"Coordinator submitted req=req-{index} to s at e bindings={raw}"
        for index, raw in enumerate(binding_reprs)
    ]
    return spawned + "\n".join(lines)


def test_higgs_process_is_verified_through_its_member_stages() -> None:
    # What the coordinator actually logs for tts_frontend: the member stages,
    # never the process name.
    log = _higgs_log(
        "{'preprocessing': 0, 'audio_encoder': 0}",
        "{'preprocessing': 1, 'audio_encoder': 1}",
    )

    verdict = _binding_evidence(log, HIGGS_REPLICA_ARM, _member_stages(HIGGS_REPLICA_ARM.config))

    assert verdict["ok"], verdict["problems"]
    assert verdict["checked_member_stages"]["tts_frontend"] == [
        "preprocessing",
        "audio_encoder",
    ]


def test_higgs_arm_fails_when_admission_never_reaches_the_second_replica() -> None:
    log = _higgs_log(
        "{'preprocessing': 0, 'audio_encoder': 0}",
        "{'preprocessing': 0, 'audio_encoder': 0}",
    )

    verdict = _binding_evidence(log, HIGGS_REPLICA_ARM, _member_stages(HIGGS_REPLICA_ARM.config))

    assert not verdict["ok"]
    assert any("preprocessing" in problem for problem in verdict["problems"])


def test_higgs_arm_fails_when_a_replica_never_spawned() -> None:
    log = "starting tts_frontend@r0\n" + (
        "Coordinator submitted req=req-0 to s at e "
        "bindings={'preprocessing': 0, 'audio_encoder': 0}"
    )

    verdict = _binding_evidence(log, HIGGS_REPLICA_ARM, _member_stages(HIGGS_REPLICA_ARM.config))

    assert not verdict["ok"]
    assert any("spawned" in problem for problem in verdict["problems"])


def test_qwen_arm_verifies_when_process_and_stage_names_coincide() -> None:
    log = (
        "starting talker_ar@r0\nstarting talker_ar@r1\n"
        "starting code2wav@r0\nstarting code2wav@r1\n"
        "Coordinator submitted req=a to s at e "
        "bindings={'talker_ar': 0, 'code2wav': 0}\n"
        "Coordinator submitted req=b to s at e "
        "bindings={'talker_ar': 1, 'code2wav': 1}\n"
    )

    verdict = _binding_evidence(log, QWEN_REPLICA_ARM, _member_stages(QWEN_REPLICA_ARM.config))

    assert verdict["ok"], verdict["problems"]


def test_control_arms_must_not_produce_bindings() -> None:
    for arm in (QWEN_CONTROL_ARM, HIGGS_CONTROL_ARM):
        clean = _binding_evidence("no bindings here", arm, _member_stages(arm.config))
        assert clean["ok"], clean["problems"]

        leaked = _binding_evidence(
            "Coordinator submitted req=a to s at e bindings={'talker_ar': 1}",
            arm,
            _member_stages(arm.config),
        )
        assert not leaked["ok"]
        assert any("control arm" in problem for problem in leaked["problems"])
