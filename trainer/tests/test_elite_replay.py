from __future__ import annotations

import asyncio
import copy
from pathlib import Path

import pytest
import torch

from combat_ai.checkpoint import CheckpointManager, CheckpointState
from combat_ai.config import PPOConfig, ServiceConfig
from combat_ai.elite_replay import EliteReplayBuffer
from combat_ai.imitation import _imitation_head_weights
from combat_ai.model import CombatPolicy
from combat_ai.ppo import _stratified_imitation_batches
from combat_ai.service import PolicyService
from fixtures import observation


HEADS = [
    "forward", "strafe", "jump", "sprint", "sneak", "primary",
    "release_use", "hotbar", "swap_offhand", "camera",
]


def _wire_action(primary: str = "attack") -> dict:
    return {
        "schema_version": 1,
        "forward": 1,
        "strafe": 0,
        "jump": False,
        "sprint": True,
        "sneak": False,
        "yaw_delta": 0.0,
        "pitch_delta": 0.0,
        "primary": primary,
        "release_use": False,
        "hotbar": 0,
        "swap_offhand": False,
    }


def _observation(episode: str, tick: int, lane: str = "combined") -> dict:
    value = observation(episode, tick)
    value["match"].update({"lane": lane, "mode": "combined", "stage": 1})
    return value


def _record_actions(
    replay: EliteReplayBuffer,
    episode: str,
    lane: str,
    count: int,
    *,
    first_action_id: int = 1,
) -> None:
    for offset in range(count):
        replay.record_policy_action(
            agent_id="fighter-a",
            episode_id=episode,
            action_id=first_action_id + offset,
            policy_version=12,
            observation=_observation(episode, offset + 1, lane),
            action=_wire_action(),
        )


def test_replay_admits_only_verified_promotions_and_keeps_recent_window() -> None:
    replay = EliteReplayBuffer(
        capacity=20, trace_capacity=8, kill_window=4, crystal_window=2,
    )
    _record_actions(replay, "episode-a", "combined", 6)

    assert replay.records == []
    assert replay.promote(
        "fighter-a", "episode-a", "damaging_crystal",
        event_token="crystal-1", quality=1.0,
    ) == 2
    assert [record["tick"] for record in replay.records] == [5, 6]
    assert all(record["execution_source"] == "elite_policy" for record in replay.records)
    assert all(record["elite_kind"] == "damaging_crystal" for record in replay.records)

    # Duplicate server delivery of the same cumulative event is idempotent.
    assert replay.promote(
        "fighter-a", "episode-a", "damaging_crystal",
        event_token="crystal-1", quality=1.0,
    ) == 0
    assert len(replay) == 2


def test_replay_eviction_preserves_rare_lanes_and_stronger_events() -> None:
    replay = EliteReplayBuffer(
        capacity=4, trace_capacity=2, kill_window=2, crystal_window=2,
    )
    _record_actions(replay, "easy-slow", "sword_retention", 2)
    replay.promote(
        "fighter-a", "easy-slow", "kill", event_token="terminal", quality=0.25,
    )
    replay.clear_episode("fighter-a", "easy-slow")

    _record_actions(replay, "easy-fast", "sword_retention", 2, first_action_id=10)
    replay.promote(
        "fighter-a", "easy-fast", "kill", event_token="terminal", quality=1.0,
    )
    replay.clear_episode("fighter-a", "easy-fast")

    _record_actions(replay, "rare", "combined_terrain", 2, first_action_id=20)
    replay.promote(
        "fighter-a", "rare", "kill", event_token="terminal", quality=0.5,
    )

    episodes = {record["match_id"] for record in replay.records}
    assert episodes == {"easy-fast", "rare"}
    assert len(replay) == 4
    assert replay.metrics()["elite_replay_buckets"] == 2


def test_restore_filters_nonfinite_records_and_reapplies_capacity() -> None:
    source = EliteReplayBuffer(capacity=8, trace_capacity=4, kill_window=4)
    _record_actions(source, "valid", "combined", 4)
    source.promote("fighter-a", "valid", "kill", event_token="terminal")
    corrupt = copy.deepcopy(source.records[0])
    corrupt["observation"]["self"]["health"] = float("nan")

    restored = EliteReplayBuffer(
        capacity=2, trace_capacity=4, restored_records=[*source.records, corrupt],
    )

    assert len(restored) == 2
    assert restored.rejected_restored_records == 1
    assert all(record["match_id"] == "valid" for record in restored.records)


def test_checkpoint_sidecar_atomically_carries_post_learner_successes(tmp_path: Path) -> None:
    replay = EliteReplayBuffer(capacity=8, trace_capacity=4, kill_window=4)
    _record_actions(replay, "before", "combined", 2)
    replay.promote("fighter-a", "before", "kill", event_token="terminal")
    staged_records = replay.records

    policy = CombatPolicy()
    optimizer = torch.optim.Adam(policy.parameters())
    manager = CheckpointManager(tmp_path, 100_000)
    state = CheckpointState(policy_version=5, rollout_generation=2)
    latest_stage, snapshot_stage, snapshot = manager.stage(
        policy, optimizer, state, PPOConfig(), {}, [], staged_records,
    )

    _record_actions(replay, "during", "combined_terrain", 2, first_action_id=20)
    replay.promote("fighter-a", "during", "kill", event_token="terminal")
    manager.promote_staged(
        latest_stage, snapshot_stage, snapshot, state, {}, replay.records,
    )

    restored_policy = CombatPolicy()
    restored_optimizer = torch.optim.Adam(restored_policy.parameters())
    restored_manager = CheckpointManager(tmp_path, 100_000)
    restored_manager.restore(restored_policy, restored_optimizer, torch.device("cpu"))
    assert {record["match_id"] for record in restored_manager.restored_elite_replay_records} == {
        "before", "during",
    }


def test_service_promotes_policy_crystal_success_but_not_unverified_actions(tmp_path: Path) -> None:
    service = PolicyService(
        PPOConfig(rollout_agent_ticks=64),
        ServiceConfig(checkpoint_dir=tmp_path, deterministic_inference=True, cpu_threads=1),
    )
    service.league.assign_batch = lambda _steps: None
    service.league.assignment_for = lambda _episode: None

    first = asyncio.run(service.handle_message({
        "schema_version": 1, "type": "step_batch", "sequence": 1,
        "steps": [{
            "agent_id": "fighter-a", "observation": _observation("episode", 1),
            "reward": 0.0, "terminated": False, "truncated": False, "info": {},
        }],
    }))["actions"][0]
    assert service.elite_replay.records == []

    second = asyncio.run(service.handle_message({
        "schema_version": 1, "type": "step_batch", "sequence": 2,
        "steps": [{
            "agent_id": "fighter-a", "observation": _observation("episode", 2),
            "reward": 0.1, "terminated": False, "truncated": False,
            "execution": {
                "source": "policy", "action_id": first["action_id"],
                "action": first["action"],
            },
            "info": {"stats": {"policy_crystal_chains_damaging": 0}},
        }],
    }))["actions"][0]
    assert service.elite_replay.records == []

    asyncio.run(service.handle_message({
        "schema_version": 1, "type": "step_batch", "sequence": 3,
        "steps": [{
            "agent_id": "fighter-a", "observation": _observation("episode", 3),
            "reward": 0.1, "terminated": False, "truncated": False,
            "execution": {
                "source": "policy", "action_id": second["action_id"],
                "action": second["action"],
            },
            "info": {"stats": {"policy_crystal_chains_damaging": 1}},
        }],
    }))

    records = service.elite_replay.records
    assert records
    assert {record["elite_kind"] for record in records} == {"damaging_crystal"}
    assert records[-1]["action"] == second["action"]


def test_delayed_policy_kill_promotes_trace_and_negative_terminal_does_not(tmp_path: Path) -> None:
    def run_episode(episode: str, terminal_reward: float, outcome: str) -> list[dict]:
        service = PolicyService(
            PPOConfig(rollout_agent_ticks=64),
            ServiceConfig(
                checkpoint_dir=tmp_path / episode,
                deterministic_inference=True,
                cpu_threads=1,
            ),
        )
        service.league.assign_batch = lambda _steps: None
        service.league.assignment_for = lambda _episode: None

        first = asyncio.run(service.handle_message({
            "schema_version": 1, "type": "step_batch", "sequence": 1,
            "steps": [{
                "agent_id": "fighter-a", "observation": _observation(episode, 1),
                "reward": 0.0, "terminated": False, "truncated": False, "info": {},
            }],
        }))["actions"][0]
        asyncio.run(service.handle_message({
            "schema_version": 1, "type": "step_batch", "sequence": 2,
            "steps": [{
                "agent_id": "fighter-a", "observation": _observation(episode, 2),
                "reward": 0.0, "terminated": False, "truncated": False, "info": {},
                "execution": {
                    "source": "policy", "action_id": first["action_id"],
                    "action": first["action"],
                },
            }],
        }))
        asyncio.run(service.handle_message({
            "schema_version": 1, "type": "step_batch", "sequence": 3,
            "steps": [{
                "agent_id": "fighter-a", "observation": _observation(episode, 3),
                "reward": terminal_reward, "terminated": True, "truncated": False,
                "execution": {"source": "safety", "action": _wire_action("none")},
                "info": {
                    "episode_id": episode,
                    "reward": terminal_reward,
                    "outcome": outcome,
                    "reason": "death",
                    "terminal_source": "policy",
                    "policy_owned_kill": True,
                },
            }],
        }))
        return service.elite_replay.records

    winner = run_episode("winner", 48.0, "win")
    loser = run_episode("loser", -20.0, "loss")
    assert winner and {record["elite_kind"] for record in winner} == {"kill"}
    assert winner[-1]["elite_quality"] == pytest.approx(0.75)
    assert loser == []


def test_imitation_sampling_reserves_diverse_elite_quota_and_scales_quality() -> None:
    crystals = [{
        "execution_source": "teacher_crystal", "teacher_phase": "place",
        "match_id": f"c-{index}", "tick": index, "action": _wire_action("use_main"),
    } for index in range(30)]
    elites = [{
        "execution_source": "elite_policy",
        "elite_bucket": f"kill:lane-{index % 2}",
        "elite_event_id": f"event-{index}",
        "elite_quality": 0.25,
        "match_id": f"e-{index}", "tick": index, "action": _wire_action(),
    } for index in range(30)]
    others = [{
        "execution_source": "teacher_sword", "match_id": f"s-{index}",
        "tick": index, "action": _wire_action(),
    } for index in range(30)]

    batches = _stratified_imitation_batches([*crystals, *elites, *others], 20)
    assert len(batches) == 4
    for index, batch in enumerate(batches):
        elite = [record for record in batch if record["execution_source"] == "elite_policy"]
        assert len(elite) == (5 if index == 0 else 0)
        if elite:
            assert len({record["elite_event_id"] for record in elite}) == 5
            assert {record["elite_bucket"] for record in elite} == {
                "kill:lane-0", "kill:lane-1",
            }

    weights = _imitation_head_weights([elites[0]], HEADS, torch.device("cpu"))
    assert torch.allclose(weights, torch.full_like(weights, 0.25))


def test_elite_rehearsal_uses_one_action_per_event_in_only_one_batch() -> None:
    records = []
    for event in range(3):
        for tick in range(8):
            records.append({
                "execution_source": "elite_policy",
                "elite_bucket": f"kill:lane-{event % 2}",
                "elite_event_id": f"event-{event}",
                "match_id": f"match-{event}",
                "agent_id": "fighter-a",
                "tick": tick,
                "action": _wire_action(),
            })

    batches = _stratified_imitation_batches(records, batch_size=64)

    assert len(batches) == 1
    assert len(batches[0]) == 3
    assert {record["elite_event_id"] for record in batches[0]} == {
        "event-0", "event-1", "event-2",
    }
