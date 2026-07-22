import copy

import pytest

from combat_ai.evaluation import (
    EVALUATION_MODES, build_manifest, final_human_gate, held_out_partition,
    policy_promotion_gate, summarize_baseline, validate_manifest,
)
from combat_ai.league import CRAZY_STYLES


def test_baseline_manifest_has_500_balanced_held_out_matches():
    manifest = build_manifest("baseline", 17)
    validate_manifest(manifest)
    assert manifest["total_matches"] == 500
    for mode in EVALUATION_MODES:
        assert sum(match["mode"] == mode for match in manifest["matches"]) == 125
    assert all(held_out_partition(match["arena_seed"]) for match in manifest["matches"])
    assert manifest == build_manifest("baseline", 17)


def test_final_manifest_is_disjoint_and_has_25_of_each_mode():
    baseline = build_manifest("baseline", 17)
    final = build_manifest("final_human", 17)
    validate_manifest(final)
    assert final["total_matches"] == 100
    assert {match["arena_seed"] for match in baseline["matches"]}.isdisjoint(
        {match["arena_seed"] for match in final["matches"]}
    )
    for mode in EVALUATION_MODES:
        assert sum(match["mode"] == mode for match in final["matches"]) == 25


def test_manifest_rejects_training_partition_seed():
    manifest = build_manifest("final_human", 3)
    broken = copy.deepcopy(manifest)
    seed = 1
    while held_out_partition(seed):
        seed += 1
    broken["matches"][0]["arena_seed"] = seed
    with pytest.raises(ValueError, match="training-partition"):
        validate_manifest(broken)


def test_baseline_summary_records_combat_metrics_by_mode_and_style():
    manifest = build_manifest("baseline", 9)
    results = [_result(match, "win") for match in manifest["matches"]]
    report = summarize_baseline(manifest, results)
    assert report["complete"] is True
    assert report["qualified"] is False
    assert report["overall"]["matches"] == 500
    assert report["overall"]["damage_efficiency"] == pytest.approx(0.8)
    assert report["overall"]["crystal_conversion"] == pytest.approx(0.5)
    assert report["overall"]["retotem_within_two_ticks"] == pytest.approx(0.95)
    assert set(report["modes"]) == set(EVALUATION_MODES)


def test_final_gate_requires_95_wins_and_no_three_loss_streak():
    manifest = build_manifest("final_human", 11)
    outcomes = ["win"] * 95 + ["loss", "win", "loss", "win", "loss"]
    report = final_human_gate(
        manifest,
        [_result(match, outcome) for match, outcome in zip(manifest["matches"], outcomes)],
    )
    assert report["passed"] is True
    assert report["longest_losing_streak"] == 1

    outcomes = ["loss", "loss", "loss"] + ["win"] * 97
    report = final_human_gate(
        manifest,
        [_result(match, outcome) for match, outcome in zip(manifest["matches"], outcomes)],
    )
    assert report["wins"] == 97
    assert report["passed"] is False


def test_final_gate_rejects_teacher_or_unfrozen_result():
    manifest = build_manifest("final_human", 13)
    results = [_result(match, "win") for match in manifest["matches"]]
    results[0]["teachers_enabled"] = True
    results[1]["policy_frozen"] = False
    report = final_human_gate(manifest, results)
    assert report["passed"] is False
    assert report["clean_evaluation_verified"] is False
    assert report["frozen_policy_verified"] is False


def test_policy_promotion_gate_checks_every_population_and_mechanic():
    results = []
    for style in CRAZY_STYLES:
        results.extend(_promotion_result("expert_script", style) for _ in range(100))
    results.extend(_promotion_result("exploiter", "exploiter-a") for _ in range(100))
    results.extend(_promotion_result("historical", "history-mixture") for _ in range(100))
    report = policy_promotion_gate(results, {"sword": 0.98})
    assert report["passed"] is True
    assert all(report["checks"].values())

    for result in results[:11]:
        result["outcome"] = "loss"
    report = policy_promotion_gate(results, {"sword": 1.0})
    assert report["passed"] is False
    assert report["checks"]["scripted_styles"] is False


def _result(match, outcome):
    return {
        "match_id": match["match_id"],
        "outcome": outcome,
        "died": outcome == "loss",
        "timeout": False,
        "opponent_damage": 8,
        "self_damage": 2,
        "avoidable_self_kill": False,
        "first_hit_seconds": 2.0,
        "legal_crystal_chains": 2,
        "damaging_crystal_chains": 1,
        "retotem_attempts": 20,
        "retotem_within_two_ticks": 19,
        "blocks_placed": 2,
        "blocks_mined": 1,
        "policy_frozen": True,
        "teachers_enabled": False,
        "mid_match_adaptation": False,
    }


def _promotion_result(kind, opponent_id):
    return {
        "opponent_kind": kind,
        "opponent_id": opponent_id,
        "mode": "sword",
        "outcome": "win",
        "avoidable_self_kill": False,
        "legal_crystal_chains": 10,
        "damaging_crystal_chains": 8,
        "retotem_attempts": 20,
        "retotem_within_two_ticks": 19,
        "policy_frozen": True,
        "teachers_enabled": False,
    }
