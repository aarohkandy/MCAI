from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
import torch

from combat_ai import cli
from combat_ai.buffer import actions_at, features_at, prepare_sequences
from combat_ai.checkpoint import CheckpointManager, CheckpointState
from combat_ai.config import PPOConfig, ServiceConfig
from combat_ai.model import CombatPolicy
from combat_ai.ppo import UpdateMetrics
from combat_ai.distribution import evaluate_actions
from combat_ai.league import EpisodeAssignment
from combat_ai.service import PolicyService, _actions_match, _trainer_ready_payload
from fixtures import observation


def _step(
    tick: int, execution: dict | None = None, *, episode: str = "test-episode",
    terminated: bool = False, truncated: bool = False,
    reward: float = 0.0, info: dict | None = None,
) -> dict:
    value = {
        "agent_id": "fighter-a",
        "observation": observation(episode=episode, tick=tick),
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": info or {},
    }
    if execution is not None:
        value["execution"] = execution
    return value


def _message(
    tick: int, execution: dict | None = None, *, episode: str = "test-episode",
    terminated: bool = False, truncated: bool = False,
    reward: float = 0.0, info: dict | None = None,
) -> dict:
    return {"schema_version": 1, "type": "step_batch", "sequence": tick,
            "steps": [_step(tick, execution, episode=episode,
                            terminated=terminated, truncated=truncated,
                            reward=reward, info=info)]}


def _sword_attack() -> dict:
    return {
        "schema_version": 1,
        "forward": 1,
        "strafe": 0,
        "jump": False,
        "sprint": True,
        "sneak": False,
        "yaw_delta": 0.0,
        "pitch_delta": 0.0,
        "primary": "attack",
        "release_use": False,
        "hotbar": 0,
        "swap_offhand": False,
    }


def _crystal_place() -> dict:
    value = _sword_attack()
    value.update({
        "forward": 0, "sprint": False, "yaw_delta": -0.2,
        "pitch_delta": -0.1, "primary": "use_main", "hotbar": 3,
    })
    return value


def _noop() -> dict:
    value = _sword_attack()
    value.update({
        "forward": 0, "sprint": False, "primary": "none", "hotbar": -1,
    })
    return value


def test_action_v2_execution_matches_resolved_sword_input() -> None:
    obs = observation()
    obs["schema_version"] = 2
    obs["tactical"] = {
        "crystal_candidates": [], "block_candidates": [],
        "survival": {"has_totem": True, "spare_totems": 0, "heal_available": False},
    }
    proposed = {
        **_noop(), "schema_version": 2, "intent": "sword_engage", "target_index": -1,
        # Conditional heads are deliberately ignored for sword intent.
        "primary": "none", "hotbar": -1,
    }
    executed = {**_noop(), "primary": "attack", "hotbar": 0}
    assert _actions_match(executed, proposed, obs)


def test_action_v2_execution_rejects_wrong_resolved_input() -> None:
    obs = observation()
    obs["schema_version"] = 2
    obs["tactical"] = {
        "crystal_candidates": [], "block_candidates": [],
        "survival": {"has_totem": True, "spare_totems": 0, "heal_available": False},
    }
    proposed = {**_noop(), "schema_version": 2, "intent": "sword_engage", "target_index": -1}
    assert not _actions_match(_noop(), proposed, obs)


def _handle(service: PolicyService, message: dict):
    return asyncio.run(service.handle_message(message))


def test_service_reapplies_runtime_learning_rate_after_checkpoint_restore(tmp_path: Path):
    old_policy = CombatPolicy()
    old_optimizer = torch.optim.Adam(old_policy.parameters(), lr=3e-4)
    manager = CheckpointManager(tmp_path, 100_000)
    manager.save(
        old_policy, old_optimizer, CheckpointState(), PPOConfig(learning_rate=3e-4), {}
    )

    service = PolicyService(
        PPOConfig(learning_rate=1e-4),
        ServiceConfig(checkpoint_dir=tmp_path, cpu_threads=1),
    )

    assert {group["lr"] for group in service.trainer.optimizer.param_groups} == {1e-4}


def test_frozen_service_infers_without_rollouts_or_checkpoint_mutation(tmp_path: Path):
    service = PolicyService(
        PPOConfig(rollout_agent_ticks=1),
        ServiceConfig(checkpoint_dir=tmp_path, deterministic_inference=True, cpu_threads=1),
        freeze_policy=True,
    )
    initial_version = service.state.policy_version
    initial_ticks = service.state.total_agent_ticks

    def unexpected_league_call(*_args, **_kwargs):
        raise AssertionError("frozen evaluation must not enter league matchmaking")

    service.league.assign_batch = unexpected_league_call
    service.league.assignment_for = unexpected_league_call
    service.league.opponent_action = unexpected_league_call

    first = _handle(service, _message(1))
    second = _handle(service, _message(2, {
        "source": "policy", "action_id": first["actions"][0]["action_id"],
        "action": first["actions"][0]["action"],
    }))
    hello = _handle(service, {"schema_version": 1, "type": "hello", "sequence": 3})

    assert first["actions"] and second["actions"]
    assert isinstance(first["actions"][0]["action_id"], int)
    assert second["actions"][0]["action_id"] > first["actions"][0]["action_id"]
    assert hello["payload"]["freeze_policy"] is True
    assert len(service.buffer) == 0
    assert service.pending == {}
    assert service.online_imitation_records == []
    assert service.state.policy_version == initial_version
    assert service.state.total_agent_ticks == initial_ticks
    assert not (tmp_path / "latest.pt").exists()
    assert not (tmp_path / "metrics.jsonl").exists()


def test_league_controlled_opponents_skip_main_policy_forward_and_ppo_state(tmp_path: Path):
    service = PolicyService(
        PPOConfig(rollout_agent_ticks=64),
        ServiceConfig(checkpoint_dir=tmp_path, deterministic_inference=True, cpu_threads=1),
    )
    steps = []
    opponent_agents = set()
    for lane in range(4):
        episode = f"league-{lane}"
        first = f"agent-{lane * 2}"
        second = f"agent-{lane * 2 + 1}"
        opponent_agents.add(second)
        service.league.assignments[episode] = EpisodeAssignment(
            episode, "expert_script", second, style="rush",
        )
        steps.extend([
            {"agent_id": first, "observation": observation(episode, 1),
             "reward": 0.0, "terminated": False, "truncated": False, "info": {}},
            {"agent_id": second, "observation": observation(episode, 1),
             "reward": 0.0, "terminated": False, "truncated": False, "info": {}},
        ])

    forward_batch_sizes = []
    original_forward = service.policy.forward

    def recording_forward(features, hidden):
        forward_batch_sizes.append(features.self_state.shape[0])
        return original_forward(features, hidden)

    service.policy.forward = recording_forward  # type: ignore[method-assign]
    response = _handle(service, {
        "schema_version": 1, "type": "step_batch", "sequence": 1, "steps": steps,
    })

    assert forward_batch_sizes == [4]
    assert len(response["actions"]) == 8
    assert set(service.pending).isdisjoint(opponent_agents)
    assert set(service.hidden).isdisjoint(opponent_agents)
    assert all(
        response_action["action_id"] is not None for response_action in response["actions"]
    )


def test_terminal_info_reaches_honest_league_audit(tmp_path: Path):
    service = PolicyService(
        PPOConfig(rollout_agent_ticks=64),
        ServiceConfig(checkpoint_dir=tmp_path, deterministic_inference=True, cpu_threads=1),
    )
    episode = "audited-league-terminal"
    service.league.assignments[episode] = EpisodeAssignment(
        episode, "expert_script", "script-agent", style="rush",
    )
    service.league._record_assignment(service.league.assignments[episode])
    proposed = _handle(service, _message(1, episode=episode))["actions"][0]

    _handle(service, _message(
        2,
        {
            "source": "policy", "action_id": proposed["action_id"],
            "action": proposed["action"],
        },
        episode=episode,
        terminated=True,
        reward=10.0,
        info={
            "episode_id": episode,
            "outcome": "win",
            "reason": "death",
            "policy_owned_kill": True,
        },
    ))

    result = service.league.matchmaking_report()["results_by_style"]["rush"]
    assert result["matches"] == 1
    assert result["policy_owned_kills"] == 1
    assert result["timeouts"] == 0


def test_delayed_action_ids_align_ppo_and_teacher_imitation(tmp_path: Path):
    service = PolicyService(
        PPOConfig(rollout_agent_ticks=8, recurrent_sequence_length=1),
        ServiceConfig(checkpoint_dir=tmp_path, deterministic_inference=True, cpu_threads=1),
    )
    first = _handle(service, _message(1))["actions"][0]

    # An uncorrelated safety override is useful shaping context, but it must not
    # consume the policy action waiting behind a worker-side delay queue.
    second = _handle(service, _message(2, {
        "source": "safety", "action": _sword_attack(),
    }))["actions"][0]
    assert len(service.buffer) == 0
    assert first["action_id"] in service.pending["fighter-a"]
    assert second["action_id"] in service.pending["fighter-a"]

    # The older action can arrive after a newer proposal has already been
    # issued. Its ID, rather than arrival order, determines the PPO transition.
    third = _handle(service, _message(3, {
        "source": "policy", "action_id": first["action_id"], "action": first["action"],
    }))["actions"][0]
    assert len(service.buffer) == 1
    assert first["action_id"] not in service.pending["fighter-a"]
    assert second["action_id"] in service.pending["fighter-a"]

    # A correlated teacher override retires that proposal, supplies its actual
    # control to imitation, and remains excluded from PPO.
    fourth = _handle(service, _message(4, {
        "source": "teacher_sword", "action_id": second["action_id"],
        "action": _sword_attack(),
    }))["actions"][0]
    assert len(service.buffer) == 1
    assert len(service.online_imitation_records) == 1
    assert service.online_imitation_records[0]["action"] == _sword_attack()
    assert service.online_imitation_records[0]["execution_source"] == "teacher_sword"
    assert second["action_id"] not in service.pending["fighter-a"]

    # A worker claiming policy ownership while reporting a different actual
    # control is conservatively treated as an invalid/off-policy transition.
    mismatched = dict(third["action"])
    mismatched["forward"] = -1 if third["action"]["forward"] != -1 else 1
    fifth = _handle(service, _message(5, {
        "source": "policy", "action_id": third["action_id"], "action": mismatched,
    }))["actions"][0]
    assert len(service.buffer) == 1
    assert third["action_id"] not in service.pending["fighter-a"]

    # Missing action IDs remain compatible with an older immediate worker and
    # consume its newest proposal just as the pre-correlation service did.
    _handle(service, _message(6, {
        "source": "policy", "action": fifth["action"],
    }))
    assert len(service.buffer) == 2

    captured: dict = {}

    def fake_update(_batch):
        return UpdateMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2)

    def fake_imitation(records, weight, batch_size=512):
        captured.update(records=records, weight=weight, batch_size=batch_size)
        return 0.5

    service.trainer.update = fake_update
    service.trainer.auxiliary_imitation_update = fake_imitation
    service.checkpoints.save = lambda *args, **kwargs: None
    service.league.prune_pool = lambda: None
    service._train_update()

    assert captured["weight"] > 0.0
    assert any(record.get("execution_source") == "teacher_sword" for record in captured["records"])
    assert service.pending == {}


def test_teacher_gap_splits_service_rollout_recurrent_replay(tmp_path: Path):
    service = PolicyService(
        PPOConfig(rollout_agent_ticks=16, recurrent_sequence_length=32),
        ServiceConfig(checkpoint_dir=tmp_path, deterministic_inference=True, cpu_threads=1),
    )
    first = _handle(service, _message(1))["actions"][0]
    second = _handle(service, _message(2, {
        "source": "policy", "action_id": first["action_id"],
        "action": first["action"],
    }))["actions"][0]
    third = _handle(service, _message(3, {
        "source": "teacher_sword", "action_id": second["action_id"],
        "action": _sword_attack(),
    }))["actions"][0]
    _handle(service, _message(4, {
        "source": "policy", "action_id": third["action_id"],
        "action": third["action"],
    }))

    transitions = service.buffer.transitions
    assert [entry.action_id for entry in transitions] == [
        first["action_id"], third["action_id"],
    ]
    assert transitions[0].recurrent_parent_action_id is None
    assert transitions[1].recurrent_parent_action_id == second["action_id"]

    batch = prepare_sequences(
        transitions, sequence_length=32, gamma=0.995, gae_lambda=0.95, device="cpu"
    )
    assert batch.sequence_count == 2
    assert batch.valid.sum(dim=1).tolist() == [1.0, 1.0]

    # With the gap split, an unchanged policy must reproduce behavior-policy
    # log probabilities exactly before any optimizer step.
    hidden = batch.hidden
    replayed = []
    service.policy.eval()
    with torch.no_grad():
        for time_index in range(batch.sequence_length):
            output = service.policy(features_at(batch.features, time_index), hidden)
            log_probability, _ = evaluate_actions(
                output,
                features_at(batch.features, time_index),
                actions_at(batch.actions, time_index),
            )
            replayed.append(log_probability)
            hidden = output.hidden * (1.0 - batch.done[:, time_index]).view(1, -1, 1)
    replayed_log_probability = torch.stack(replayed, dim=1)
    assert torch.allclose(
        replayed_log_probability[batch.valid.bool()],
        batch.old_log_probability[batch.valid.bool()],
        atol=1e-5,
        rtol=0.0,
    )


def test_uncorrelated_safety_execution_splits_next_accepted_policy_step(tmp_path: Path):
    service = PolicyService(
        PPOConfig(rollout_agent_ticks=16, recurrent_sequence_length=32),
        ServiceConfig(checkpoint_dir=tmp_path, deterministic_inference=True, cpu_threads=1),
    )
    first = _handle(service, _message(1))["actions"][0]
    second = _handle(service, _message(2, {
        "source": "policy", "action_id": first["action_id"],
        "action": first["action"],
    }))["actions"][0]

    # Safety executes without claiming a queued proposal, so `second` remains
    # pending and its proposal parent is still `first`. Environment chronology
    # nevertheless contains an off-policy control and must create a hard split.
    _handle(service, _message(3, {
        "source": "safety", "action": _noop(),
    }))
    _handle(service, _message(4, {
        "source": "policy", "action_id": second["action_id"],
        "action": second["action"],
    }))

    transitions = service.buffer.transitions
    assert [entry.action_id for entry in transitions] == [
        first["action_id"], second["action_id"],
    ]
    assert transitions[1].recurrent_parent_action_id == first["action_id"]
    assert transitions[1].execution_gap_before is True
    batch = prepare_sequences(
        transitions, sequence_length=32, gamma=0.995, gae_lambda=0.95, device="cpu"
    )
    assert batch.sequence_count == 2


def test_terminal_and_episode_change_clear_orphaned_action_ids(tmp_path: Path):
    service = PolicyService(
        PPOConfig(rollout_agent_ticks=16, recurrent_sequence_length=1),
        ServiceConfig(checkpoint_dir=tmp_path, deterministic_inference=True, cpu_threads=1),
    )
    first = _handle(service, _message(1))["actions"][0]
    second = _handle(service, _message(2, {
        "source": "safety", "action": _sword_attack(),
    }))["actions"][0]
    assert set(service.pending["fighter-a"]) == {first["action_id"], second["action_id"]}

    terminal = _handle(service, _message(3, {
        "source": "safety", "action_id": first["action_id"], "action": _sword_attack(),
    }, terminated=True))
    assert service.pending == {}
    assert "fighter-a" not in service.hidden
    assert len(service.buffer) == 0
    assert service.online_imitation_records == []
    assert terminal["actions"] == []

    old_episode = _handle(service, _message(4, episode="old-episode"))["actions"][0]
    new_episode = _handle(service, _message(5, {
        "source": "policy", "action_id": old_episode["action_id"],
        "action": old_episode["action"],
    }, episode="new-episode"))["actions"][0]
    assert set(service.pending["fighter-a"]) == {new_episode["action_id"]}
    assert service.pending["fighter-a"][new_episode["action_id"]].episode_id == "new-episode"


def test_safety_delayed_kill_credits_last_policy_transition_not_noop(tmp_path: Path):
    service = PolicyService(
        PPOConfig(rollout_agent_ticks=16, recurrent_sequence_length=1),
        ServiceConfig(checkpoint_dir=tmp_path, deterministic_inference=True, cpu_threads=1),
    )
    proposal = _handle(service, _message(1))["actions"][0]
    _handle(service, _message(2, {
        "source": "policy", "action_id": proposal["action_id"],
        "action": proposal["action"],
    }, reward=0.03))
    assert len(service.buffer) == 1
    previous_reward = service.buffer.transitions[0].reward

    # Arena death feedback commonly arrives after the policy execution report,
    # when the worker has already reset its per-step execution to safety/no-op.
    terminal = _handle(service, _message(3, {
        "source": "safety", "action": _noop(),
    }, terminated=True, reward=5.0, info={
        "episode_id": "test-episode", "reward": 5.0,
        "outcome": "win", "reason": "death", "terminal_source": "policy",
        "policy_owned_kill": True,
    }))

    assert terminal["actions"] == []
    assert len(service.buffer) == 1
    credited = service.buffer.transitions[0]
    assert credited.reward == pytest.approx(previous_reward + 5.0)
    assert credited.done is True
    assert credited.next_value == 0.0
    assert service.pending == {}
    metrics = service.reward_telemetry.metrics()
    assert metrics["reward_terminal_transitions"] == 1
    assert metrics["terminal_attached_events"] == 1
    assert metrics["policy_owned_kill_events"] == 1


@pytest.mark.parametrize("reason", ["timeout", "disengaged"])
def test_delayed_failure_to_finish_reaches_the_last_policy_transition(
    tmp_path: Path, reason: str,
):
    service = PolicyService(
        PPOConfig(rollout_agent_ticks=16, recurrent_sequence_length=1),
        ServiceConfig(checkpoint_dir=tmp_path, deterministic_inference=True, cpu_threads=1),
    )
    proposal = _handle(service, _message(1))["actions"][0]
    _handle(service, _message(2, {
        "source": "policy", "action_id": proposal["action_id"],
        "action": proposal["action"],
    }))
    baseline = service.buffer.transitions[0].reward

    _handle(service, _message(3, {
        "source": "safety", "action": _noop(),
    }, truncated=True, reward=-25.0, info={
        "episode_id": "test-episode", "reward": -25.0,
        "outcome": "loss", "reason": reason, "terminal_source": "none",
        "policy_owned_kill": False,
    }))

    transition = service.buffer.transitions[0]
    assert transition.reward == pytest.approx(baseline - 25.0)
    assert transition.done is True
    metrics = service.reward_telemetry.metrics()
    assert metrics["terminal_attached_events"] == 1
    assert metrics["rejected_nonpolicy_terminals"] == 0


def test_policy_kill_teaches_the_victim_loss_even_without_owned_kill_flag(tmp_path: Path):
    service = PolicyService(
        PPOConfig(rollout_agent_ticks=16, recurrent_sequence_length=1),
        ServiceConfig(checkpoint_dir=tmp_path, deterministic_inference=True, cpu_threads=1),
    )
    proposal = _handle(service, _message(1))["actions"][0]
    _handle(service, _message(2, {
        "source": "policy", "action_id": proposal["action_id"],
        "action": proposal["action"],
    }))
    baseline = service.buffer.transitions[0].reward

    _handle(service, _message(3, {
        "source": "safety", "action": _noop(),
    }, terminated=True, reward=-20.0, info={
        "episode_id": "test-episode", "reward": -20.0,
        "outcome": "loss", "reason": "death", "terminal_source": "policy",
        "policy_owned_kill": False,
    }))

    assert service.buffer.transitions[0].reward == pytest.approx(baseline - 20.0)
    assert service.reward_telemetry.metrics()["terminal_attached_events"] == 1


@pytest.mark.parametrize(
    "terminal_source", ["teacher_sword", "teacher_crystal", "teacher_block", "safety"],
)
def test_teacher_or_safety_delayed_kill_closes_without_rewarding_policy(
    tmp_path: Path, terminal_source: str,
):
    service = PolicyService(
        PPOConfig(rollout_agent_ticks=16, recurrent_sequence_length=1),
        ServiceConfig(checkpoint_dir=tmp_path, deterministic_inference=True, cpu_threads=1),
    )
    first = _handle(service, _message(1))["actions"][0]
    second = _handle(service, _message(2, {
        "source": "policy", "action_id": first["action_id"], "action": first["action"],
    }))["actions"][0]
    baseline = service.buffer.transitions[0].reward
    _handle(service, _message(3, {
        "source": "teacher_sword", "action_id": second["action_id"],
        "action": _sword_attack(),
    }))

    _handle(service, _message(4, {
        "source": "safety", "action": _noop(),
    }, terminated=True, reward=0.0, info={
        "episode_id": "test-episode", "reward": 0.0,
        "outcome": "draw", "reason": "death", "terminal_source": terminal_source,
        "policy_owned_kill": False,
    }))

    excluded = service.buffer.transitions[0]
    assert excluded.reward == pytest.approx(baseline)
    assert excluded.done is True
    assert excluded.next_value == 0.0
    assert service.reward_telemetry.metrics()["reward_terminal_transitions"] == 0
    assert service.reward_telemetry.metrics()["rejected_nonpolicy_terminals"] == 1


@pytest.mark.parametrize("terminal_source", ["self", "environment", "policy"])
def test_delayed_double_ko_death_keeps_both_fighters_negative(
    tmp_path: Path, terminal_source: str,
):
    service = PolicyService(
        PPOConfig(rollout_agent_ticks=16, recurrent_sequence_length=1),
        ServiceConfig(checkpoint_dir=tmp_path, deterministic_inference=True, cpu_threads=1),
    )
    proposal = _handle(service, _message(1))["actions"][0]
    _handle(service, _message(2, {
        "source": "policy", "action_id": proposal["action_id"],
        "action": proposal["action"],
    }))
    baseline = service.buffer.transitions[0].reward

    _handle(service, _message(3, {
        "source": "safety", "action": _noop(),
    }, terminated=True, reward=-1.0, info={
        "episode_id": "test-episode", "reward": -1.0,
        "outcome": "loss", "reason": "double_ko",
        "terminal_source": terminal_source, "policy_owned_kill": False,
    }))

    transition = service.buffer.transitions[0]
    assert transition.reward == pytest.approx(baseline - 1.0)
    assert transition.done is True
    assert service.reward_telemetry.metrics()["terminal_attached_events"] == 1


@pytest.mark.parametrize("terminal_source", ["self", "policy"])
def test_double_ko_can_never_receive_positive_terminal_bonus(
    tmp_path: Path, terminal_source: str,
):
    service = PolicyService(
        PPOConfig(rollout_agent_ticks=16, recurrent_sequence_length=1),
        ServiceConfig(checkpoint_dir=tmp_path, deterministic_inference=True, cpu_threads=1),
    )
    proposal = _handle(service, _message(1))["actions"][0]
    _handle(service, _message(2, {
        "source": "policy", "action_id": proposal["action_id"],
        "action": proposal["action"],
    }))
    baseline = service.buffer.transitions[0].reward

    _handle(service, _message(3, {
        "source": "safety", "action": _noop(),
    }, terminated=True, reward=5.0, info={
        "episode_id": "test-episode", "reward": 5.0,
        "outcome": "draw", "reason": "double_ko",
        "terminal_source": terminal_source, "policy_owned_kill": False,
    }))

    assert service.buffer.transitions[0].reward == pytest.approx(baseline)
    metrics = service.reward_telemetry.metrics()
    assert metrics["terminal_attached_events"] == 0
    assert metrics["rejected_nonpolicy_terminals"] == 1


def test_ready_rollout_waits_two_batches_for_delayed_kill_before_drain(tmp_path: Path):
    service = PolicyService(
        PPOConfig(rollout_agent_ticks=1, recurrent_sequence_length=1),
        ServiceConfig(checkpoint_dir=tmp_path, deterministic_inference=True, cpu_threads=1),
    )
    proposal = _handle(service, _message(1))["actions"][0]
    _handle(service, _message(2, {
        "source": "policy", "action_id": proposal["action_id"],
        "action": proposal["action"],
    }))
    baseline = service.buffer.transitions[0].reward
    assert service.buffer.ready
    assert service.buffer_ready_grace_remaining == 2

    _handle(service, _message(3, {"source": "safety", "action": _noop()}))
    assert service.buffer.ready
    assert service.buffer_ready_grace_remaining == 1

    captured: dict[str, object] = {}

    def capture_update() -> None:
        captured["rewards"] = [entry.reward for entry in service.buffer.transitions]
        captured["done"] = [entry.done for entry in service.buffer.transitions]
        captured["terminal_events"] = service.reward_telemetry.terminal_attached_events
        service.buffer.transitions.clear()
        service.buffer_ready_grace_remaining = None

    service._train_update = capture_update  # type: ignore[method-assign]
    _handle(service, _message(4, {
        "source": "safety", "action": _noop(),
    }, terminated=True, reward=5.0, info={
        "episode_id": "test-episode", "reward": 5.0,
        "outcome": "win", "reason": "death",
        "terminal_source": "policy", "policy_owned_kill": True,
    }))

    assert captured["rewards"] == pytest.approx([baseline + 5.0])
    assert captured["done"] == [True]
    assert captured["terminal_events"] == 1


def test_policy_owned_kill_attaches_victim_loss_as_well_as_winner_bonus(tmp_path: Path):
    service = PolicyService(
        PPOConfig(rollout_agent_ticks=16, recurrent_sequence_length=1),
        ServiceConfig(checkpoint_dir=tmp_path, deterministic_inference=True, cpu_threads=1),
    )
    proposal = _handle(service, _message(1))["actions"][0]
    _handle(service, _message(2, {
        "source": "policy", "action_id": proposal["action_id"],
        "action": proposal["action"],
    }))
    baseline = service.buffer.transitions[0].reward

    _handle(service, _message(3, {
        "source": "safety", "action": _noop(),
    }, terminated=True, reward=-1.0, info={
        "episode_id": "test-episode", "reward": -1.0,
        "outcome": "loss", "reason": "death",
        "terminal_source": "policy", "policy_owned_kill": True,
    }))

    assert service.buffer.transitions[0].reward == pytest.approx(baseline - 1.0)
    metrics = service.reward_telemetry.metrics()
    assert metrics["terminal_attached_events"] == 1
    assert metrics["policy_owned_kill_events"] == 0


def test_waiting_sentinel_never_enters_training_or_recurrent_state(tmp_path: Path):
    service = PolicyService(
        PPOConfig(rollout_agent_ticks=8, recurrent_sequence_length=1),
        ServiceConfig(checkpoint_dir=tmp_path, deterministic_inference=True, cpu_threads=1),
    )
    proposal = _handle(service, _message(1))["actions"][0]
    assert service.pending and service.hidden

    waiting = _handle(service, _message(2, {
        "source": "teacher_sword", "action_id": proposal["action_id"],
        "action": _sword_attack(),
    }, episode="waiting"))

    assert waiting["actions"] == []
    assert service.pending == {}
    assert "fighter-a" not in service.hidden
    assert len(service.buffer) == 0
    assert service.online_imitation_records == []


def test_reconnect_hello_clears_pending_ids_before_new_steps(tmp_path: Path):
    service = PolicyService(
        PPOConfig(rollout_agent_ticks=8, recurrent_sequence_length=1),
        ServiceConfig(checkpoint_dir=tmp_path, deterministic_inference=True, cpu_threads=1),
    )
    proposal = _handle(service, _message(1))["actions"][0]
    assert proposal["action_id"] in service.pending["fighter-a"]
    assert "fighter-a" in service.hidden

    hello = _handle(service, {
        "schema_version": 1,
        "type": "hello",
        "sequence": 2,
        "worker_id": "reconnected-worker",
        "agents": ["fighter-a"],
        "capabilities": ["action-correlation-v1"],
    })
    assert hello["command"] == "hello_ack"
    assert service.pending == {}
    assert "fighter-a" not in service.hidden

    # A late execution from the old socket has no pending proposal to consume.
    response = _handle(service, _message(3, {
        "source": "policy",
        "action_id": proposal["action_id"],
        "action": proposal["action"],
    }))
    assert response["actions"]
    assert len(service.buffer) == 0


def test_online_imitation_buffer_discards_oldest_records(tmp_path: Path):
    service = PolicyService(
        PPOConfig(), ServiceConfig(checkpoint_dir=tmp_path, cpu_threads=1)
    )
    service.ONLINE_IMITATION_CAPACITY = 2
    service._append_online_imitation({"id": 1})
    service._append_online_imitation({"id": 2})
    service._append_online_imitation({"id": 3})
    assert service.online_imitation_records == [{"id": 2}, {"id": 3}]


def test_crystal_imitation_uses_exact_pre_execution_observation_and_drops_waits(tmp_path: Path):
    service = PolicyService(
        PPOConfig(rollout_agent_ticks=16, recurrent_sequence_length=1),
        ServiceConfig(checkpoint_dir=tmp_path, deterministic_inference=True, cpu_threads=1),
    )
    first = _handle(service, _message(1))["actions"][0]
    teacher_observation = observation(tick=2)
    teacher_observation["match"].update({"mode": "crystal", "lane": "crystal_retention"})
    teacher_observation["self"]["hotbar"][3] = {
        "name": "end_crystal", "count": 64, "durability": 0,
        "max_durability": 0, "enchant_hash": 0,
    }
    second = _handle(service, _message(3, {
        "source": "teacher_crystal", "action_id": first["action_id"],
        "action": _crystal_place(),
        "pre_execution_observation": teacher_observation,
    }))["actions"][0]

    assert len(service.online_imitation_records) == 1
    record = service.online_imitation_records[0]
    assert record["observation"] is teacher_observation
    assert record["tick"] == 2
    assert record["teacher_phase"] == "place"

    _handle(service, _message(4, {
        "source": "teacher_crystal", "action_id": second["action_id"],
        "action": _noop(),
        "pre_execution_observation": observation(tick=4),
    }))
    assert len(service.online_imitation_records) == 1


def test_server_policy_crystal_sequence_counters_are_metrics_not_duplicate_rewards(tmp_path: Path):
    service = PolicyService(
        PPOConfig(), ServiceConfig(checkpoint_dir=tmp_path, cpu_threads=1)
    )
    info = {"stats": {
        "policy_crystal_chains_started": 2,
        "policy_crystal_chains_detonated": 1,
        "policy_crystal_chains_damaging": 1,
        "policy_crystal_chains_popping": 0,
        "rewarded_crystal_combos": 1,
    }}
    service._record_policy_crystal_counters("fighter-a", "episode", info)
    service._record_policy_crystal_counters("fighter-a", "episode", info)
    metrics = service.reward_telemetry.metrics()
    assert metrics["server_policy_crystal_chains_started_events"] == 2
    assert metrics["server_policy_crystal_chains_detonated_events"] == 1
    assert metrics["server_policy_crystal_chains_damaging_events"] == 1
    assert metrics["server_rewarded_crystal_combos_events"] == 1
    # The server reward already contains combo credit; trainer shaping must not
    # add a second policy-crystal component.
    assert not any("policy_crystal_sequence" in key for key in metrics)


def test_cli_passes_freeze_policy_to_service(monkeypatch, tmp_path: Path):
    captured: dict = {}

    async def fake_serve(*args):
        captured["args"] = args

    monkeypatch.setattr(cli, "serve", fake_serve)
    monkeypatch.setattr(sys, "argv", [
        "mcai-trainer", "serve", "--checkpoints", str(tmp_path), "--freeze-policy",
    ])
    cli.main()
    assert captured["args"][-1] is True


def test_trainer_ready_reports_effective_ppo_runtime_config(tmp_path: Path):
    ppo = PPOConfig(
        rollout_agent_ticks=1234,
        recurrent_sequence_length=7,
        learning_rate=2.5e-5,
        minibatch_samples=321,
        optimization_epochs=3,
        learner_cpu_threads=2,
        target_kl=0.012,
    )
    service_config = ServiceConfig(
        host="127.0.0.9", port=9876, checkpoint_dir=tmp_path, cpu_threads=1,
    )
    service = PolicyService(ppo, service_config)

    ready = _trainer_ready_payload(service, service_config)

    assert ready["rollout_agent_ticks"] == 1234
    assert ready["recurrent_sequence_length"] == 7
    assert ready["learning_rate"] == 2.5e-5
    assert ready["minibatch_samples"] == 321
    assert ready["optimization_epochs"] == 3
    assert ready["learner_cpu_threads"] == 2
    assert ready["target_kl"] == 0.012
