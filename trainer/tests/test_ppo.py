import numpy as np
import pytest
import torch

from combat_ai.buffer import Transition, prepare_sequences
from combat_ai.config import PPOConfig
from combat_ai.distribution import actions_from_wire
from combat_ai.features import encode_observation
from combat_ai.model import CombatPolicy
from combat_ai.ppo import (
    PPOTrainer, _clipped_huber_value_error, _stratified_imitation_batches,
)
from fixtures import observation


def test_adaptive_learning_rate_uses_three_low_updates_and_one_high_update():
    trainer = PPOTrainer(CombatPolicy(), PPOConfig(), torch.device("cpu"))
    # 0.006 is above the historical fixed 0.002 floor but still well below
    # the 0.010 target. Three such under-budget updates must restore speed.
    trainer._adapt_learning_rate(0.006)
    trainer._adapt_learning_rate(0.006)
    assert trainer.current_learning_rate == 1e-4
    trainer._adapt_learning_rate(0.006)
    assert trainer.current_learning_rate == pytest.approx(1.5e-4)
    trainer._adapt_learning_rate(0.02)
    assert trainer.current_learning_rate == pytest.approx(7.5e-5)


def test_adaptive_learning_rate_resets_recovery_streak_near_target():
    trainer = PPOTrainer(CombatPolicy(), PPOConfig(), torch.device("cpu"))
    trainer._adapt_learning_rate(0.006)
    trainer._adapt_learning_rate(0.006)
    trainer._adapt_learning_rate(0.009)
    trainer._adapt_learning_rate(0.006)
    assert trainer.current_learning_rate == 1e-4


def test_online_ppo_defaults_are_conservative():
    config = PPOConfig()

    assert config.learning_rate == 1e-4
    assert config.rollout_agent_ticks == 4096
    assert config.minibatch_samples == 1024
    assert config.entropy_coefficient == 0.01
    assert config.value_huber_delta == 1.0
    assert config.optimization_epochs == 4
    assert config.learner_cpu_threads == 2
    assert config.target_kl == 0.01


def test_critic_terminal_outlier_has_bounded_gradient() -> None:
    value = torch.tensor([0.0], requires_grad=True)
    old_value = torch.tensor([0.0])
    returns = torch.tensor([40.0])
    clipped_value = old_value + (value - old_value).clamp(-0.2, 0.2)

    loss = _clipped_huber_value_error(
        value, clipped_value, returns, delta=1.0,
    ).mean()
    loss.backward()

    assert loss.item() == pytest.approx(39.5)
    assert abs(value.grad.item()) <= 1.0


def test_configured_learning_rate_overrides_restored_optimizer_value():
    trainer = PPOTrainer(
        CombatPolicy(), PPOConfig(learning_rate=1e-4), torch.device("cpu")
    )
    for group in trainer.optimizer.param_groups:
        group["lr"] = 3e-4

    trainer.reapply_configured_learning_rate()

    assert {group["lr"] for group in trainer.optimizer.param_groups} == {1e-4}


def test_recurrent_ppo_update_runs_on_padded_sequences():
    policy = CombatPolicy()
    wire = {"schema_version": 1, "forward": 0, "strafe": 0, "jump": False, "sprint": False,
            "sneak": False, "yaw_delta": 0.0, "pitch_delta": 0.0, "primary": "none",
            "release_use": False, "hotbar": -1, "swap_offhand": False}
    action = actions_from_wire([wire], "cpu")
    transitions = []
    for tick in range(5):
        transitions.append(Transition(
            agent_id="a", episode_id="e", policy_version=0,
            features=encode_observation(observation(tick=tick)), hidden=np.zeros(128, dtype=np.float32),
            categorical_action={name: int(value[0]) for name, value in action.categorical.items()},
            camera_action=np.zeros(2, dtype=np.float32), old_log_probability=-5.0, old_value=0.0,
            reward=0.01, done=tick == 4, next_value=0.0,
        ))
    batch = prepare_sequences(transitions, 4, 0.995, 0.95, "cpu")
    trainer = PPOTrainer(policy, PPOConfig(optimization_epochs=1, minibatch_samples=8), torch.device("cpu"))
    metrics = trainer.update(batch)
    assert metrics.valid_samples > 0
    assert np.isfinite(metrics.policy_loss)


def test_ppo_stops_after_target_kl_and_averages_only_completed_updates(monkeypatch):
    policy = CombatPolicy()
    wire = {"schema_version": 1, "forward": 0, "strafe": 0, "jump": False,
            "sprint": False, "sneak": False, "yaw_delta": 0.0, "pitch_delta": 0.0,
            "primary": "none", "release_use": False, "hotbar": -1,
            "swap_offhand": False}
    action = actions_from_wire([wire], "cpu")
    transitions = [
        Transition(
            agent_id="a", episode_id="e", policy_version=0,
            features=encode_observation(observation(tick=tick)),
            hidden=np.zeros(128, dtype=np.float32),
            categorical_action={name: int(value[0]) for name, value in action.categorical.items()},
            camera_action=np.zeros(2, dtype=np.float32), old_log_probability=-5.0,
            old_value=0.0, reward=0.01, done=tick == 3, next_value=0.0,
        )
        for tick in range(4)
    ]
    batch = prepare_sequences(transitions, 2, 0.995, 0.95, "cpu")
    trainer = PPOTrainer(
        policy,
        PPOConfig(
            optimization_epochs=3, recurrent_sequence_length=2,
            minibatch_samples=2, target_kl=0.02,
        ),
        torch.device("cpu"),
    )
    completed = iter((
        (1.0, 2.0, 3.0, 0.01, 0.10, 4.0, 4.0),
        (3.0, 4.0, 5.0, 0.03, 0.30, 6.0, 6.0),
        (100.0, 100.0, 100.0, 1.00, 1.00, 100.0, 100.0),
    ))
    calls = []

    def fake_minibatch(_batch):
        calls.append(1)
        return next(completed)

    monkeypatch.setattr(trainer, "_minibatch", fake_minibatch)

    metrics = trainer.update(batch)

    assert len(calls) == 2
    assert metrics.optimizer_updates == 2
    assert metrics.early_stopped is True
    assert metrics.policy_loss == 2.0
    assert metrics.value_loss == 3.0
    assert metrics.entropy == 4.0
    assert metrics.approximate_kl == 0.02
    assert metrics.clip_fraction == 0.20
    assert metrics.gradient_norm == 5.0
    assert metrics.valid_samples == 5
    assert metrics.rollout_valid_samples == 4
    assert metrics.optimizer_sample_exposures == 10
    assert metrics.rollout_sample_coverage == 1.0
    assert metrics.max_kl == 0.03


def test_online_imitation_builds_four_half_crystal_phase_balanced_batches():
    phases = ("aim", "select", "place", "detonate")
    records = [
        {"execution_source": "teacher_crystal", "teacher_phase": phase,
         "match_id": "crystal", "agent_id": "a", "tick": tick}
        for tick, phase in enumerate(phases * 3)
    ]
    records.extend(
        {"execution_source": "teacher_sword", "match_id": "sword",
         "agent_id": "b", "tick": tick}
        for tick in range(20)
    )

    batches = _stratified_imitation_batches(records, batch_size=16)

    assert len(batches) == 4
    for batch in batches:
        crystals = [record for record in batch if record["execution_source"] == "teacher_crystal"]
        assert len(crystals) == len(batch) // 2
        assert {record["teacher_phase"] for record in crystals} == set(phases)
        assert [record["tick"] for record in batch if record["match_id"] == "crystal"] == sorted(
            record["tick"] for record in batch if record["match_id"] == "crystal"
        )


def test_crystal_stratified_auxiliary_update_runs_four_weighted_steps():
    policy = CombatPolicy()
    trainer = PPOTrainer(policy, PPOConfig(), torch.device("cpu"))
    base_action = {
        "schema_version": 1, "forward": 0, "strafe": 0, "jump": False,
        "sprint": False, "sneak": False, "yaw_delta": 0.0, "pitch_delta": 0.0,
        "primary": "none", "release_use": False, "hotbar": -1,
        "swap_offhand": False,
    }
    records = []
    for tick, phase in enumerate(("aim", "select", "place", "detonate"), 1):
        action = dict(base_action)
        if phase == "aim":
            action["yaw_delta"] = 0.1
        elif phase == "select":
            action["hotbar"] = 0
        elif phase == "place":
            action.update({"primary": "use_main", "hotbar": 0})
        else:
            action.update({"primary": "attack", "hotbar": 0})
        records.append({
            "match_id": "crystal", "agent_id": "a", "tick": tick,
            "observation": observation(tick=tick), "action": action,
            "execution_source": "teacher_crystal", "teacher_phase": phase,
        })

    loss = trainer.auxiliary_imitation_update(records, weight=0.02, batch_size=8)

    assert np.isfinite(loss)
    assert trainer.last_imitation_metrics == {
        "updates": 4, "samples": 16, "crystal_samples": 16, "crystal_fraction": 1.0,
        "elite_samples": 0, "elite_fraction": 0.0, "elite_events_sampled": 0,
    }


def test_elite_auxiliary_telemetry_counts_unique_events_once(monkeypatch):
    import combat_ai.imitation as imitation_module

    trainer = PPOTrainer(CombatPolicy(), PPOConfig(), torch.device("cpu"))
    records = []
    for event in range(3):
        for tick in range(4):
            records.append({
                "execution_source": "elite_policy",
                "elite_bucket": f"kill:lane-{event % 2}",
                "elite_event_id": f"event-{event}",
                "match_id": f"match-{event}",
                "agent_id": "fighter-a",
                "tick": tick,
            })

    def finite_connected_loss(policy, _records, _device):
        return next(policy.parameters()).sum() * 0.0 + 1.0

    monkeypatch.setattr(imitation_module, "imitation_loss", finite_connected_loss)
    loss = trainer.auxiliary_imitation_update(records, weight=0.015, batch_size=64)

    assert loss == pytest.approx(1.0)
    assert trainer.last_imitation_metrics == {
        "updates": 1,
        "samples": 3,
        "crystal_samples": 0,
        "crystal_fraction": 0.0,
        "elite_samples": 3,
        "elite_fraction": 1.0,
        "elite_events_sampled": 3,
    }
