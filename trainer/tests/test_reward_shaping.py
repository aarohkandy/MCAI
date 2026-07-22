from __future__ import annotations

import copy
import math
from typing import Any

import pytest

from combat_ai.reward_shaping import TrainerRewardShaper, tactical_snapshot
from fixtures import observation


def _stats(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "blocks_placed": 0,
        "obsidian_placed": 0,
        "tactical_obsidian_placed": 0,
        "tactical_mine_place_sequences": 0,
        "policy_built_crystal_chains_damaging": 0,
        "blocks_mined": 0,
        "crystals_placed": 0,
        "crystals_destroyed": 0,
        "crystals_exploded": 0,
        "invalid_interactions": 0,
        "spam_attack_swings": 0,
        "missed_attack_swings": 0,
        "point_breakdown": {},
    }
    values.update(overrides)
    return values


def test_tactical_progress_and_signed_visibility_acquisition_are_shaped() -> None:
    before = observation(tick=1)
    before["opponent"]["relative_position"]["z"] = -6.0
    before["opponent"]["line_of_sight"] = False
    before["action_mask"]["combat_attack_ready"] = False
    after = copy.deepcopy(before)
    after["match"]["tick"] = 2
    after["opponent"]["relative_position"]["z"] = -5.7
    after["opponent"]["line_of_sight"] = True
    after["action_mask"]["combat_attack_ready"] = True

    result = TrainerRewardShaper().shape(
        "agent", "episode", tactical_snapshot(before), after, {}, same_episode=True
    )

    assert result.components["approach"] == pytest.approx(0.0003)
    assert result.components["visibility"] == pytest.approx(0.001)
    assert "crosshair" not in result.components
    assert result.total == pytest.approx(0.0013)


def test_generic_player_raycast_cannot_reward_looking_at_spectator() -> None:
    value = observation()
    value["opponent"]["line_of_sight"] = False
    value["action_mask"]["combat_attack_ready"] = False
    value["self"]["raycast"] = {
        "kind": "entity", "distance": 2.0, "block_name": "", "entity_kind": "player",
    }

    result = TrainerRewardShaper().shape(
        "agent", "episode", tactical_snapshot(value), value, {}, same_episode=True
    )

    assert "crosshair" not in result.components
    assert result.total == 0.0


def test_distance_and_visibility_oscillation_has_no_positive_cycle() -> None:
    close = observation(tick=1)
    close["opponent"]["relative_position"]["z"] = -2.8
    close["opponent"]["line_of_sight"] = True
    close["action_mask"]["combat_attack_ready"] = False
    far = copy.deepcopy(close)
    far["match"]["tick"] = 2
    far["opponent"]["relative_position"]["z"] = -3.2
    far["opponent"]["line_of_sight"] = False
    shaper = TrainerRewardShaper()

    outward = shaper.shape("agent", "episode", tactical_snapshot(close), far, {}, same_episode=True)
    inward = shaper.shape("agent", "episode", tactical_snapshot(far), close, {}, same_episode=True)

    assert outward.components["approach"] == pytest.approx(-0.0002)
    assert inward.components["approach"] == pytest.approx(0.0002)
    assert outward.components["visibility"] == pytest.approx(-0.001)
    assert inward.components["visibility"] == pytest.approx(0.001)
    assert outward.total + inward.total == pytest.approx(0.0)


def test_verified_tactical_setup_and_policy_built_combo_reward_once() -> None:
    value = observation()
    previous = tactical_snapshot(value)
    shaper = TrainerRewardShaper()

    blocks = _stats(
        blocks_placed=1, obsidian_placed=1, tactical_obsidian_placed=1,
        tactical_mine_place_sequences=1,
        blocks_mined=1,
        execution={"policy": {
            "blocks_placed": 1, "tactical_obsidian_placed": 1,
            "tactical_mine_place_sequences": 1,
            "policy_built_crystal_chains_damaging": 0,
            "blocks_mined": 1, "crystals_placed": 0,
            "crystals_destroyed": 0, "crystals_exploded": 0,
        }},
        point_breakdown={"mining": 0.2},
    )
    first = shaper.shape(
        "agent", "episode", previous, value, {"stats": blocks}, same_episode=False
    )
    duplicate = shaper.shape(
        "agent", "episode", previous, value, {"stats": blocks}, same_episode=False
    )
    crystals = shaper.shape(
        "agent", "episode", previous, value,
            {"stats": _stats(
                blocks_placed=1, obsidian_placed=1, tactical_obsidian_placed=1,
                tactical_mine_place_sequences=1,
                policy_built_crystal_chains_damaging=1, blocks_mined=1,
                crystals_placed=1, crystals_destroyed=1, crystals_exploded=1,
                execution={"policy": {
                    "blocks_placed": 1, "tactical_obsidian_placed": 1,
                    "tactical_mine_place_sequences": 1,
                    "policy_built_crystal_chains_damaging": 1,
                    "blocks_mined": 1, "crystals_placed": 1,
                    "crystals_destroyed": 1, "crystals_exploded": 1,
                }},
                point_breakdown={
                    "mining": 0.2, "crystal_placement": 2.5,
                "crystal_destruction": 1.2, "crystal_explosion": 3.0,
            },
        )},
        same_episode=False,
    )

    assert first.components == {
        "tactical_obsidian_setup": pytest.approx(0.006),
        "tactical_mine_place": pytest.approx(0.010),
    }
    assert first.total == pytest.approx(0.016)
    assert duplicate.total == 0.0
    assert crystals.components == {
        "policy_built_damaging_combo": pytest.approx(0.020),
        "crystal_place": pytest.approx(0.002),
        "crystal_destroy": pytest.approx(0.002),
        "crystal_detonate": pytest.approx(0.004),
    }
    assert crystals.total == pytest.approx(0.028)


def test_invalid_and_attack_spam_counters_are_penalized() -> None:
    value = observation()
    previous = tactical_snapshot(value)
    shaper = TrainerRewardShaper()
    shaper.shape("agent", "episode", previous, value, {"stats": _stats()}, same_episode=False)

    result = shaper.shape(
        "agent", "episode", previous, value,
        {"stats": _stats(invalid_interactions=2, spam_attack_swings=1, missed_attack_swings=1)},
        same_episode=False,
    )

    assert result.components["useless_spam"] == pytest.approx(-0.00325)
    assert result.total == pytest.approx(-0.00325)


def test_batched_crystal_points_map_to_exact_verified_event_counts() -> None:
    value = observation()
    result = TrainerRewardShaper().shape(
        "agent", "episode", tactical_snapshot(value), value,
        {"stats": _stats(
            crystals_placed=2, crystals_destroyed=2, crystals_exploded=2,
            point_breakdown={
                "crystal_placement": 5.0,
                "crystal_destruction": 2.4,
                "crystal_explosion": 6.0,
            },
        )},
        same_episode=False,
    )

    assert result.components == {
        "crystal_place": pytest.approx(0.004),
        "crystal_destroy": pytest.approx(0.004),
        "crystal_detonate": pytest.approx(0.008),
    }
    assert result.total == pytest.approx(0.016)


def test_unrewarded_mechanic_counters_are_not_mistaken_for_success() -> None:
    value = observation()
    previous = tactical_snapshot(value)
    result = TrainerRewardShaper().shape(
        "agent", "episode", previous, value,
        {"stats": _stats(blocks_placed=1, blocks_mined=1, crystals_placed=1)},
        same_episode=False,
    )

    assert not any(name in result.components for name in (
        "tactical_obsidian_setup", "block_break", "crystal_place",
    ))
    assert result.components["useless_spam"] < 0


def test_passive_teacher_completion_reported_on_policy_step_is_not_spam() -> None:
    value = observation()
    previous = tactical_snapshot(value)
    shaper = TrainerRewardShaper()
    policy_zero = {
        "blocks_placed": 0, "blocks_mined": 0,
        "crystals_placed": 0, "crystals_destroyed": 0, "crystals_exploded": 0,
        "tactical_obsidian_placed": 0,
        "tactical_mine_place_sequences": 0,
        "policy_built_crystal_chains_damaging": 0,
    }
    shaper.shape(
        "agent", "episode", previous, value,
        {"stats": _stats(execution={"policy": policy_zero})}, same_episode=False,
    )

    # A teacher's asynchronous place completion appears in total counters on
    # an ordinary policy step. No policy-attributed counter or point changed.
    teacher_completion = shaper.shape(
        "agent", "episode", previous, value,
        {"stats": _stats(
            blocks_placed=1, obsidian_placed=1, crystals_placed=8,
            execution={"policy": policy_zero}, point_breakdown={},
        )},
        same_episode=False,
    )

    assert teacher_completion.total == 0.0
    assert "useless_spam" not in teacher_completion.components

    # A later generic autonomous placement is observed but is not success.
    policy_completion = shaper.shape(
        "agent", "episode", previous, value,
        {"stats": _stats(
            blocks_placed=2, obsidian_placed=2, crystals_placed=8,
                execution={"policy": {**policy_zero, "blocks_placed": 1}},
                point_breakdown={},
            )},
            same_episode=False,
        )
    assert policy_completion.components == {"useless_spam": pytest.approx(-0.0005)}

    # Only the exact server-attributed tactical counter earns setup credit.
    useful_completion = shaper.shape(
        "agent", "episode", previous, value,
        {"stats": _stats(
            blocks_placed=3, obsidian_placed=3, tactical_obsidian_placed=1,
            crystals_placed=8,
            execution={"policy": {
                **policy_zero, "blocks_placed": 2, "tactical_obsidian_placed": 1,
            }},
            point_breakdown={},
        )},
        same_episode=False,
    )
    assert useful_completion.components == {
        "tactical_obsidian_setup": pytest.approx(0.006),
    }


def test_total_tactical_counter_never_falls_back_without_policy_attribution() -> None:
    value = observation()
    previous = tactical_snapshot(value)
    policy_zero = {
        "blocks_placed": 0, "blocks_mined": 0,
        "crystals_placed": 0, "crystals_destroyed": 0, "crystals_exploded": 0,
        "tactical_obsidian_placed": 0,
        "tactical_mine_place_sequences": 0,
        "policy_built_crystal_chains_damaging": 0,
    }

    result = TrainerRewardShaper().shape(
        "agent", "episode", previous, value,
        {"stats": _stats(
            tactical_obsidian_placed=1,
            tactical_mine_place_sequences=1,
            policy_built_crystal_chains_damaging=1,
            execution={"policy": policy_zero},
        )},
        same_episode=False,
    )

    assert result.total == 0.0
    assert "tactical_obsidian_setup" not in result.components
    assert "tactical_mine_place" not in result.components
    assert "policy_built_damaging_combo" not in result.components


def test_extreme_pitch_is_free_during_grace_then_penalized_and_resets() -> None:
    value = observation()
    value["opponent"] = None
    value["action_mask"]["combat_attack_ready"] = False
    value["self"]["pitch"] = math.radians(70)
    previous = tactical_snapshot(value)
    shaper = TrainerRewardShaper()

    first_twenty = [
        shaper.shape("agent", "episode", previous, value, {}, same_episode=True)
        for _ in range(20)
    ]
    penalized = shaper.shape("agent", "episode", previous, value, {}, same_episode=True)
    value["self"]["pitch"] = 0.0
    reset = shaper.shape("agent", "episode", previous, value, {}, same_episode=True)

    assert all("extreme_pitch" not in result.components for result in first_twenty)
    assert penalized.components["extreme_pitch"] == pytest.approx(-0.00025)
    assert reset.total == 0.0


def test_mechanic_credit_budget_prevents_infinite_block_farming() -> None:
    value = observation()
    previous = tactical_snapshot(value)
    shaper = TrainerRewardShaper()

    stats = _stats()
    for count in (4,):
        stats = _stats(
            blocks_placed=count, obsidian_placed=count, tactical_obsidian_placed=count,
            execution={"policy": {
                "blocks_placed": count, "tactical_obsidian_placed": count,
                "policy_built_crystal_chains_damaging": 0,
                "blocks_mined": 0, "crystals_placed": 0,
                "crystals_destroyed": 0, "crystals_exploded": 0,
            }},
        )
        shaper.shape("agent", "episode", previous, value, {"stats": stats}, same_episode=False)
    exhausted = shaper.shape(
        "agent", "episode", previous, value,
        {"stats": _stats(
            blocks_placed=5, obsidian_placed=5, tactical_obsidian_placed=5,
            execution={"policy": {
                "blocks_placed": 5, "tactical_obsidian_placed": 5,
                "policy_built_crystal_chains_damaging": 0,
                "blocks_mined": 0, "crystals_placed": 0,
                "crystals_destroyed": 0, "crystals_exploded": 0,
            }},
        )}, same_episode=False,
    )

    assert "tactical_obsidian_setup" not in exhausted.components
    assert exhausted.components["useless_spam"] < 0
