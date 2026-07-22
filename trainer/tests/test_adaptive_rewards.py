from __future__ import annotations

import asyncio
import json

import pytest

from combat_ai.adaptive_rewards import (
    AdaptiveRewardConfig,
    AdaptiveRewardController,
    PROFILE_BOUNDS,
    RewardProfile,
    publish_reward_profile,
)


def _metrics(**overrides: float) -> dict[str, float]:
    values = {
        "reward_transitions": 4_096,
        "reward_nonfinite_transitions": 0,
        "valid_samples": 4_096,
        "max_kl": 0.008,
        "quarantined_sequences": 0,
        "quarantined_samples": 0,
        "skipped_optimizer_steps": 0,
        "decision_round_trip_ms_p95": 80,
        "effective_decisions_hz_per_agent": 8,
        "server_policy_damage_dealt": 4,
        "server_damage_taken": 20,
        "server_self_damage": 5,
        "server_policy_hits_landed": 4,
        "server_policy_crystal_chains_started_events": 2,
        "server_policy_crystal_chains_damaging_events": 0,
        "policy_owned_kill_events": 1,
        "terminal_attached_events": 20,
        "server_inaction_penalty_ticks": 1_000,
        "server_policy_blocks_placed": 1,
        "server_policy_blocks_mined": 0,
    }
    values.update(overrides)
    return values


def test_adaptation_requires_hysteresis_is_bounded_and_resumes(tmp_path) -> None:
    config = AdaptiveRewardConfig(
        required_streak=2,
        cooldown_updates=2,
        max_categories_per_update=5,
        rollback_evaluation_updates=0,
    )
    controller = AdaptiveRewardController(tmp_path, config)

    first = controller.observe(_metrics(), policy_version=1, rollout_generation=1)
    second = controller.observe(_metrics(), policy_version=2, rollout_generation=2)

    assert not first.changed
    assert second.changed
    assert second.generation == 2
    assert all(
        change["to"] <= change["from"] * 1.075 + 1e-9
        for change in second.changes.values()
    )
    restored = AdaptiveRewardController(tmp_path, config)
    assert restored.profile == controller.profile
    assert restored.telemetry()["cooldowns"] == controller.telemetry()["cooldowns"]

    for generation in range(3, 100):
        restored.observe(_metrics(), policy_version=generation, rollout_generation=generation)
    for name, value in restored.profile.multipliers.items():
        lower, upper = PROFILE_BOUNDS[name]
        assert lower <= value <= upper

    audit = [
        json.loads(line)
        for line in (tmp_path / "adaptive-reward-audit.jsonl").read_text().splitlines()
    ]
    assert audit[-1]["event"] == "adaptive_reward_evaluation"
    assert audit[-1]["after"] == restored.profile.multipliers


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"decision_round_trip_ms_p95": 201}, "exceeds 200 ms"),
        ({"effective_decisions_hz_per_agent": 3.9}, "below 4 Hz"),
        ({"max_kl": 0.016}, "exceeds 0.015"),
        ({"reward_nonfinite_transitions": 1}, "non-finite rewards"),
    ],
)
def test_unhealthy_rollouts_freeze_weights_and_streaks(tmp_path, override, fragment) -> None:
    controller = AdaptiveRewardController(
        tmp_path, AdaptiveRewardConfig(required_streak=1, rollback_evaluation_updates=0)
    )

    decision = controller.observe(
        _metrics(**override), policy_version=1, rollout_generation=1
    )

    assert decision.skipped
    assert not decision.changed
    assert all(value == 0 for value in controller.telemetry()["streaks"].values())
    assert any(fragment in reason for reason in decision.health_reasons)


def test_post_change_composite_regression_rolls_profile_back(tmp_path) -> None:
    config = AdaptiveRewardConfig(
        required_streak=1,
        max_categories_per_update=1,
        rollback_evaluation_updates=2,
        rollback_regression_tolerance=0.05,
    )
    controller = AdaptiveRewardController(tmp_path, config)
    baseline = _metrics(
        server_policy_damage_dealt=20,
        server_damage_taken=20,
        server_self_damage=0,
        server_policy_hits_landed=25,
        server_policy_crystal_chains_started_events=5,
        server_policy_crystal_chains_damaging_events=3,
        policy_owned_kill_events=4,
        terminal_attached_events=20,
        server_inaction_penalty_ticks=400,
        server_policy_blocks_placed=0,
        server_policy_blocks_mined=0,
    )
    changed = controller.observe(baseline, policy_version=1, rollout_generation=1)
    assert changed.changed
    assert changed.changes.keys() == {"building"}
    assert changed.multipliers["building"] > 1.0

    regression = _metrics(
        server_policy_damage_dealt=0,
        server_damage_taken=30,
        server_self_damage=10,
        server_policy_hits_landed=0,
        server_policy_crystal_chains_started_events=4,
        server_policy_crystal_chains_damaging_events=0,
        policy_owned_kill_events=0,
        terminal_attached_events=20,
        server_inaction_penalty_ticks=2_000,
        server_policy_blocks_placed=0,
        server_policy_blocks_mined=0,
    )
    waiting = controller.observe(regression, policy_version=2, rollout_generation=2)
    rolled_back = controller.observe(regression, policy_version=3, rollout_generation=3)

    assert not waiting.changed
    assert rolled_back.changed
    assert rolled_back.rollback
    assert rolled_back.generation == changed.generation + 1
    assert rolled_back.multipliers["building"] == 1.0
    assert controller.telemetry()["cooldowns"]["building"] == config.rollback_cooldown_updates


def test_learner_failure_is_audited_without_moving_profile(tmp_path) -> None:
    controller = AdaptiveRewardController(tmp_path)
    before = controller.profile

    controller.note_update_failure(
        "synthetic optimizer failure", policy_version=7, rollout_generation=8
    )

    assert controller.profile == before
    record = json.loads(
        (tmp_path / "adaptive-reward-audit.jsonl").read_text().splitlines()[-1]
    )
    assert record["event"] == "adaptive_reward_health_freeze"
    assert "synthetic optimizer failure" in record["reason"]


def test_publish_profile_ignores_interleaved_arena_events() -> None:
    async def scenario() -> None:
        received: dict = {}

        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            request = json.loads((await reader.readline()).decode())
            received.update(request)
            writer.write(b'{"type":"event","event":"match_started"}\n')
            writer.write(b'{"type":"response","id":999,"ok":true,"payload":{}}\n')
            writer.write((json.dumps({
                "type": "response", "id": request["id"], "ok": True,
                "payload": {"generation": request["payload"]["generation"]},
            }) + "\n").encode())
            await writer.drain()
            writer.close()

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = int(server.sockets[0].getsockname()[1])
        try:
            response = await publish_reward_profile(
                "127.0.0.1", port,
                RewardProfile(3, {
                    "damage": 1.0, "crystal": 1.1, "terminal_speed": 1.2,
                    "activity": 1.0, "building": 0.9,
                }, "test"),
            )
        finally:
            server.close()
            await server.wait_closed()

        assert received["command"] == "set_reward_multipliers"
        assert received["payload"]["generation"] == 3
        assert response == {"generation": 3}

    asyncio.run(scenario())
