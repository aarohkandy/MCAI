from __future__ import annotations

import math

import pytest

from combat_ai.reward_shaping import TrainerShaping
from combat_ai.service import RewardTelemetry, sanitize_reward


def test_dense_reward_is_bounded_without_weakening_terminal_win() -> None:
    reward = sanitize_reward({
        "reward": 1.8,
        "terminated": True,
        "truncated": False,
        "info": {"reward": 1.0, "outcome": "win"},
    })

    assert reward.raw_shaping_reward == 0.8
    assert reward.shaping_reward == 0.25
    assert reward.terminal_reward == 1.0
    assert reward.training_reward == 1.25
    assert reward.clipped


def test_trainer_shaping_is_added_before_clip_without_weakening_terminal_win() -> None:
    reward = sanitize_reward({
        "reward": 1.03,
        "terminated": True,
        "truncated": False,
        "info": {"reward": 1.0, "outcome": "win"},
    }, trainer_shaping=0.04)

    assert reward.server_reward == pytest.approx(1.03)
    assert reward.trainer_shaping_reward == pytest.approx(0.04)
    assert reward.raw_reward == pytest.approx(1.07)
    assert reward.raw_shaping_reward == pytest.approx(0.07)
    assert reward.shaping_reward == pytest.approx(0.07)
    assert reward.terminal_reward == 1.0
    assert reward.training_reward == pytest.approx(1.07)


def test_large_kill_bonus_survives_dense_shaping_clip() -> None:
    reward = sanitize_reward({
        "reward": 5.7,
        "terminated": True,
        "truncated": False,
        "info": {"reward": 5.0, "outcome": "win", "reason": "death"},
    })

    assert reward.raw_shaping_reward == pytest.approx(0.7)
    assert reward.shaping_reward == pytest.approx(0.25)
    assert reward.terminal_reward == pytest.approx(5.0)
    assert reward.training_reward == pytest.approx(5.25)
    assert reward.clipped


def test_damage_reward_sign_is_preserved_before_any_terminal_outcome() -> None:
    enemy_damage = sanitize_reward({"reward": 0.04})
    self_damage = sanitize_reward({"reward": -0.04})

    assert enemy_damage.training_reward == pytest.approx(0.04)
    assert self_damage.training_reward == pytest.approx(-0.04)


def test_policy_kill_flood_is_preserved_for_ppo() -> None:
    reward = sanitize_reward({
        "reward": 10.18,
        "terminated": True,
        "truncated": False,
        "info": {"reward": 10.0, "outcome": "win", "reason": "death"},
    })
    assert reward.terminal_reward == pytest.approx(10.0)
    assert reward.shaping_reward == pytest.approx(0.18)
    assert reward.training_reward == pytest.approx(10.18)


def test_nonterminal_aggregation_is_clipped_per_consumed_transition() -> None:
    positive = sanitize_reward({"reward": 0.7})
    negative = sanitize_reward({"reward": -0.7})

    assert positive.training_reward == 0.25
    assert negative.training_reward == -0.25


def test_client_local_death_stays_a_full_terminal_loss() -> None:
    reward = sanitize_reward({"reward": -1, "terminated": True, "info": {"reason": "death"}})

    assert reward.shaping_reward == 0
    assert reward.terminal_reward == -1
    assert reward.training_reward == -1


def test_nonfinite_reward_cannot_poison_training() -> None:
    reward = sanitize_reward({"reward": math.nan})

    assert reward.training_reward == 0
    assert reward.nonfinite


def test_reward_telemetry_reports_clipping_and_means() -> None:
    telemetry = RewardTelemetry()
    telemetry.record(sanitize_reward({"reward": 0.01}))
    shaped = TrainerShaping(total=0.01, components={"tactical_mine_place": 0.01})
    telemetry.record(sanitize_reward({"reward": 0.2}, trainer_shaping=shaped.total), shaped)

    metrics = telemetry.metrics()
    assert metrics["reward_transitions"] == 2
    assert metrics["reward_mean_server_raw"] == pytest.approx(0.105)
    assert metrics["reward_mean_trainer_shaping"] == pytest.approx(0.005)
    assert metrics["reward_mean_raw"] == pytest.approx(0.11)
    assert metrics["reward_mean_training"] == pytest.approx(0.11)
    assert metrics["reward_clipped_transitions"] == 0
    assert metrics["reward_nonfinite_transitions"] == 0
    assert metrics["reward_max_abs_raw"] == pytest.approx(0.21)
    assert metrics["trainer_reward_tactical_mine_place_sum"] == pytest.approx(0.01)
    assert metrics["trainer_reward_tactical_mine_place_events"] == 1
