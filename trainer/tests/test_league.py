import json
import random
from pathlib import Path

import torch

from combat_ai.league import (
    BOOTSTRAP_MATCHMAKING_SHARES, CRAZY_STYLES,
    DEVELOPING_MATCHMAKING_SHARES, MATCHMAKING_SHARES, EpisodeAssignment,
    RECOVERY_MATCHMAKING_SHARES, LeagueManager, scripted_action,
)
from combat_ai.model import CombatPolicy
from fixtures import observation


def test_episode_assignment_is_stable_and_in_declared_modes(tmp_path: Path):
    manager = LeagueManager(tmp_path, __import__('torch').device('cpu'))
    steps = [
        {"agent_id": "a", "observation": observation("episode")},
        {"agent_id": "b", "observation": observation("episode")},
    ]
    manager.assign_batch(steps)
    first = manager.assignment_for("episode")
    manager.assign_batch(steps)
    assert manager.assignment_for("episode") == first
    assert first.mode in {"mirror", "historical", "exploiter", "expert_script"}


def test_scripted_teacher_uses_only_action_v1_controls():
    action = scripted_action(observation(), "rush")
    assert action["schema_version"] == 1
    assert action["hotbar"] == -1
    assert -1 <= action["forward"] <= 1
    assert action["primary"] in {"none", "attack", "use_main", "use_offhand"}


def test_scripted_opponent_prefers_correct_body_frame_geometry():
    value = observation()
    value["opponent"].update({
        # The legacy field claims the target is behind/right; the corrected
        # field and explicit bearing say it is directly ahead.
        "relative_position": {"x": 3.0, "y": 0.0, "z": 2.0},
        "body_relative_position": {"x": 0.0, "y": 0.0, "z": -4.0},
        "distance": 4.0,
        "bearing_error": 0.0,
        "pitch_error": 0.0,
    })

    action = scripted_action(value, "rush")

    assert action["forward"] == 1
    assert action["yaw_delta"] == 0.0


def test_five_flat_evaluations_request_an_exploiter(tmp_path: Path):
    manager = LeagueManager(tmp_path, __import__('torch').device('cpu'))
    _pass_crazy_gauntlet(manager)
    outcomes = [manager.note_evaluation(1200 + index * 0.5) for index in range(5)]
    assert outcomes[-1] is True


def test_flat_evaluation_cannot_trigger_specialization_before_broad_wins(tmp_path: Path):
    manager = LeagueManager(tmp_path, torch.device("cpu"))
    outcomes = [manager.note_evaluation(1200 + index * 0.5) for index in range(5)]
    assert outcomes[-1] is False
    assert manager.crazy_report()["ready"] is False


def test_exploiter_forces_frozen_main_and_promotes_atomically(tmp_path: Path):
    target = tmp_path / "main.pt"
    torch.save({"policy": CombatPolicy().state_dict(), "total_agent_ticks": 123}, target)
    run = tmp_path / "run"
    manager = LeagueManager(run, torch.device("cpu"))
    manager.force_frozen_opponent(target)
    steps = [
        {"agent_id": "a", "observation": observation("exploit")},
        {"agent_id": "b", "observation": observation("exploit")},
    ]
    manager.assign_batch(steps)
    assignment = manager.assignment_for("exploit")
    assert assignment.mode == "historical"
    assert assignment.checkpoint == str(target)

    main_pool = LeagueManager(tmp_path / "main-pool", torch.device("cpu"))
    for won in ([True] * 60 + [False] * 40):
        report = main_pool.note_exploiter_result(target, won)
    assert report["promotable"] is True
    promoted = main_pool.add_frozen_checkpoint(target)
    assert promoted.exists()
    assert promoted.name.startswith("policy-exploiter-")


def test_exploiter_cannot_promote_without_60_wins_in_100(tmp_path: Path):
    target = tmp_path / "candidate.pt"
    torch.save({"policy": CombatPolicy().state_dict()}, target)
    manager = LeagueManager(tmp_path / "league", torch.device("cpu"))
    for won in ([True] * 59 + [False] * 41):
        report = manager.note_exploiter_result(target, won)
    assert report["matches"] == 100
    assert report["promotable"] is False
    with __import__('pytest').raises(ValueError, match="100 held-out"):
        manager.add_frozen_checkpoint(target)


def test_severe_bootstrap_matchmaking_uses_recovery_distribution(tmp_path: Path):
    manager = LeagueManager(tmp_path, torch.device("cpu"))
    checkpoint = tmp_path / "policy-000000000001.pt"
    exploiter = tmp_path / "policy-exploiter-000000000002.pt"
    torch.save({"policy": CombatPolicy().state_dict()}, checkpoint)
    torch.save({"policy": CombatPolicy().state_dict()}, exploiter)
    assert manager.competence_stage() == "bootstrap"
    assert manager.recovery_active() is True
    counts = {key: 0 for key in RECOVERY_MATCHMAKING_SHARES}
    styles = set()
    for index in range(4000):
        episode = f"diversity-{index}"
        manager.assign_batch([
            {"agent_id": "a", "observation": observation(episode)},
            {"agent_id": "b", "observation": observation(episode)},
        ])
        assignment = manager.assignment_for(episode)
        counts[assignment.mode] += 1
        if assignment.style:
            styles.add(assignment.style)
    for mode, share in RECOVERY_MATCHMAKING_SHARES.items():
        assert abs(counts[mode] / 4000 - share) < 0.035
    assert styles == set(CRAZY_STYLES)


def test_matchmaking_mix_advances_only_with_script_competence(tmp_path: Path):
    manager = LeagueManager(tmp_path, torch.device("cpu"))
    assert manager.matchmaking_shares() == RECOVERY_MATCHMAKING_SHARES

    _set_crazy_score(manager, 0.20)
    assert manager.competence_stage() == "bootstrap"
    assert manager.recovery_active() is False
    assert manager.matchmaking_shares() == BOOTSTRAP_MATCHMAKING_SHARES

    _set_crazy_score(manager, 0.30)
    assert manager.competence_stage() == "developing"
    assert manager.matchmaking_shares() == DEVELOPING_MATCHMAKING_SHARES

    _set_crazy_score(manager, 0.50)
    assert manager.competence_stage() == "qualified"
    assert manager.matchmaking_shares() == MATCHMAKING_SHARES


def test_severe_recovery_empty_exploiter_bucket_is_reassigned_to_history(
    tmp_path: Path,
):
    manager = LeagueManager(tmp_path, torch.device("cpu"))
    checkpoint = tmp_path / "policy-000000000001.pt"
    torch.save({"policy": CombatPolicy().state_dict()}, checkpoint)
    counts = {"historical": 0, "expert_script": 0, "mirror": 0}
    for index in range(2000):
        episode = f"no-exploiter-{index}"
        manager.assign_batch([
            {"agent_id": "a", "observation": observation(episode)},
            {"agent_id": "b", "observation": observation(episode)},
        ])
        assignment = manager.assignment_for(episode)
        assert assignment.mode != "exploiter"
        counts[assignment.mode] += 1
    assert abs(counts["historical"] / 2000 - 0.10) < 0.04
    assert abs(counts["expert_script"] / 2000 - 0.30) < 0.04
    assert abs(counts["mirror"] / 2000 - 0.60) < 0.04


def test_nonsevere_missing_population_buckets_use_script_frontier_deterministically(
    tmp_path: Path,
):
    manager = LeagueManager(tmp_path, torch.device("cpu"))
    _set_crazy_score(manager, 0.20)
    assert manager.recovery_active() is False

    original_shares = manager.matchmaking_shares
    try:
        manager.matchmaking_shares = lambda: {
            "historical": 1.0, "exploiter": 0.0,
            "expert_script": 0.0, "mirror": 0.0,
        }
        historical_fallback = manager._new_assignment("missing-history", ["a", "b"])
        assert historical_fallback.mode == "expert_script"
        assert historical_fallback.style in CRAZY_STYLES
        assert historical_fallback == manager._new_assignment("missing-history", ["a", "b"])

        manager.matchmaking_shares = lambda: {
            "historical": 0.0, "exploiter": 1.0,
            "expert_script": 0.0, "mirror": 0.0,
        }
        exploiter_fallback = manager._new_assignment("missing-exploiter", ["a", "b"])
        assert exploiter_fallback.mode == "expert_script"
        assert exploiter_fallback.style in CRAZY_STYLES
        assert exploiter_fallback == manager._new_assignment("missing-exploiter", ["a", "b"])
    finally:
        manager.matchmaking_shares = original_shares


def test_nonsevere_empty_population_effective_distributions(tmp_path: Path):
    cases = (
        ("bootstrap", 0.20, BOOTSTRAP_MATCHMAKING_SHARES),
        ("developing", 0.30, DEVELOPING_MATCHMAKING_SHARES),
        ("qualified", 0.50, MATCHMAKING_SHARES),
    )
    sample_count = 4000
    for label, score, shares in cases:
        manager = LeagueManager(tmp_path / label, torch.device("cpu"))
        _set_crazy_score(manager, score)
        assert manager.recovery_active() is False
        counts = {"historical": 0, "exploiter": 0, "expert_script": 0, "mirror": 0}
        first_pass = []
        for index in range(sample_count):
            assignment = manager._new_assignment(f"{label}-empty-{index}", ["a", "b"])
            first_pass.append(assignment)
            counts[assignment.mode] += 1
            if assignment.mode == "expert_script":
                assert assignment.style in CRAZY_STYLES

        # Stable episode seeding makes both the fallback mode and frontier
        # style exactly replayable.
        assert first_pass == [
            manager._new_assignment(f"{label}-empty-{index}", ["a", "b"])
            for index in range(sample_count)
        ]
        expected_script = (
            shares["historical"] + shares["exploiter"] + shares["expert_script"]
        )
        assert counts["historical"] == 0
        assert counts["exploiter"] == 0
        assert abs(counts["expert_script"] / sample_count - expected_script) < 0.03
        assert abs(counts["mirror"] / sample_count - shares["mirror"]) < 0.03


def test_nonsevere_empty_exploiter_effective_distribution_with_history(
    tmp_path: Path,
):
    manager = LeagueManager(tmp_path, torch.device("cpu"))
    _set_crazy_score(manager, 0.20)
    (tmp_path / "policy-history.pt").touch()
    sample_count = 4000
    counts = {"historical": 0, "exploiter": 0, "expert_script": 0, "mirror": 0}
    for index in range(sample_count):
        assignment = manager._new_assignment(f"bootstrap-history-{index}", ["a", "b"])
        counts[assignment.mode] += 1

    # Live non-severe bootstrap has history but no promoted exploiter: the
    # effective mix is therefore 15% history, 40% scripts, and 45% mirrors.
    assert counts["exploiter"] == 0
    assert abs(counts["historical"] / sample_count - 0.15) < 0.03
    assert abs(counts["expert_script"] / sample_count - 0.40) < 0.03
    assert abs(counts["mirror"] / sample_count - 0.45) < 0.03


def test_bootstrap_checkpoint_sampling_favors_attainable_frontier_but_keeps_hard_games(
    tmp_path: Path,
):
    manager = LeagueManager(tmp_path, torch.device("cpu"))
    frontier = tmp_path / "policy-frontier.pt"
    hard = tmp_path / "policy-hard.pt"
    too_easy = tmp_path / "policy-too-easy.pt"
    for path in (frontier, hard, too_easy):
        path.touch()
    manager.payoffs[frontier.name] = [1.0] * 45 + [0.0] * 55
    manager.payoffs[hard.name] = [0.0] * 100
    manager.payoffs[too_easy.name] = [1.0] * 100

    counts = {path.name: 0 for path in (frontier, hard, too_easy)}
    randomizer = random.Random(17)
    for _ in range(2000):
        selected = manager._pfsp_checkpoint(randomizer, exploiters=False)
        counts[selected.name] += 1

    assert counts[frontier.name] > counts[hard.name] * 2
    assert counts[hard.name] > 0
    assert counts[frontier.name] > counts[too_easy.name]


def test_bootstrap_script_sampling_keeps_frontier_hard_and_undercovered_exposure(
    tmp_path: Path,
):
    manager = LeagueManager(tmp_path, torch.device("cpu"))
    _set_crazy_score(manager, 0.0, samples=16)
    manager.crazy_results["rush"] = [0.5] * 16
    manager.crazy_results["spacing"] = [1.0] * 16
    manager.crazy_results["erratic"] = []
    assert manager.competence_stage() == "bootstrap"

    counts = {style: 0 for style in CRAZY_STYLES}
    randomizer = random.Random(23)
    for _ in range(4000):
        counts[manager._crazy_style(randomizer)] += 1

    assert counts["rush"] > counts["defensive"] * 2
    assert counts["defensive"] > 0
    assert counts["erratic"] > 0


def test_pfsp_prefers_an_opponent_that_beats_main(tmp_path: Path):
    manager = LeagueManager(tmp_path, torch.device("cpu"))
    _set_crazy_score(manager, 0.50)
    weak = tmp_path / "policy-weak.pt"
    strong = tmp_path / "policy-strong.pt"
    torch.save({"policy": CombatPolicy().state_dict()}, weak)
    torch.save({"policy": CombatPolicy().state_dict()}, strong)
    manager.payoffs[weak.name] = [1.0] * 100
    manager.payoffs[strong.name] = [0.0] * 100
    counts = {weak.name: 0, strong.name: 0}
    randomizer = random.Random(7)
    for _ in range(1000):
        counts[manager._pfsp_checkpoint(randomizer, exploiters=False).name] += 1
    assert counts[strong.name] > counts[weak.name] * 3


def test_curriculum_sampling_is_stable_after_legacy_state_reload(tmp_path: Path):
    checkpoint = tmp_path / "policy-history.pt"
    checkpoint.touch()
    (tmp_path / "league.json").write_text(json.dumps({
        "format_version": 1,
        "main_elo": 425.0,
        "ratings": {checkpoint.name: 900.0},
        "payoffs": {checkpoint.name: [0.1, 0.2, 0.0]},
    }), encoding="utf-8")
    first = LeagueManager(tmp_path, torch.device("cpu"))
    first_assignment = first._new_assignment("persisted-episode", ["a", "b"])
    assert first.competence_stage() == "bootstrap"
    first._save()

    restored = LeagueManager(tmp_path, torch.device("cpu"))
    assert restored._new_assignment("persisted-episode", ["a", "b"]) == first_assignment
    assert restored.matchmaking_shares() == RECOVERY_MATCHMAKING_SHARES


def test_matchmaking_audit_persists_mode_style_timeout_and_policy_kills(tmp_path: Path):
    manager = LeagueManager(tmp_path, torch.device("cpu"))
    assignments = [
        EpisodeAssignment("self-kill", "expert_script", "script", style="rush"),
        EpisodeAssignment("policy-kill", "expert_script", "script", style="rush"),
        EpisodeAssignment(
            "timeout", "historical", "history",
            checkpoint=str(tmp_path / "policy-history.pt"),
        ),
        EpisodeAssignment("mirror-kill", "mirror"),
    ]
    for assignment in assignments:
        manager.assignments[assignment.episode_id] = assignment
        manager._record_assignment(assignment)

    manager.record_result(
        "self-kill", "main", 1.0, "win",
        {"reason": "death", "policy_owned_kill": False},
    )
    manager.record_result(
        "policy-kill", "main", 10.0, "win",
        {"reason": "death", "policy_owned_kill": True},
    )
    manager.record_result(
        "timeout", "main", -2.0, "loss",
        {"reason": "timeout", "policy_owned_kill": False},
    )
    # Both mirror fighters are main-policy actors, so a verified kill belongs
    # to main even when the first delivered terminal row is the losing side.
    manager.record_result(
        "mirror-kill", "main-a", -10.0, "loss",
        {"reason": "death", "policy_owned_kill": True},
    )

    assert manager.crazy_results["rush"] == [0.0, 1.0]
    report = manager.matchmaking_report()
    assert report["assigned_total"] == 4
    assert report["completed_total"] == 4
    assert report["results_by_mode"]["expert_script"]["win_rate"] == 1.0
    assert report["results_by_mode"]["expert_script"]["policy_owned_kills"] == 1
    assert report["results_by_mode"]["historical"]["timeout_rate"] == 1.0
    assert report["results_by_mode"]["mirror"]["policy_owned_kill_rate"] == 1.0
    assert report["results_by_style"]["rush"]["matches"] == 2

    restored = LeagueManager(tmp_path, torch.device("cpu"))
    assert restored.matchmaking_report() == report


def test_payoff_matrix_tracks_both_policy_perspectives(tmp_path: Path):
    manager = LeagueManager(tmp_path, torch.device("cpu"))
    manager.record_payoff("main", "exploiter-a", 0.25)
    assert manager.payoff_matrix["main"]["exploiter-a"] == [0.25]
    assert manager.payoff_matrix["exploiter-a"]["main"] == [0.75]


def test_crystal_builder_and_miner_styles_emit_distinct_legal_actions():
    value = observation()
    value["self"]["hotbar"][1] = {
        "name": "end_crystal", "count": 16, "durability": 0,
        "max_durability": 0, "enchant_hash": 0,
    }
    value["self"]["hotbar"][2] = {
        "name": "obsidian", "count": 16, "durability": 0,
        "max_durability": 0, "enchant_hash": 0,
    }
    value["self"]["hotbar"][3] = {
        "name": "diamond_pickaxe", "count": 1, "durability": 0,
        "max_durability": 1561, "enchant_hash": 0,
    }
    value["action_mask"].update({
        "crystal_place_ready": True,
        "tactical_block_place_ready": True,
        "tactical_block_break_ready": True,
    })
    crystal = scripted_action(value, "crystal_rush")
    builder = scripted_action(value, "obsidian_builder")
    miner = scripted_action(value, "tactical_miner")
    assert (crystal["primary"], crystal["hotbar"]) == ("use_main", 1)
    assert (builder["primary"], builder["hotbar"]) == ("use_main", 2)
    assert (miner["primary"], miner["hotbar"]) == ("attack", 3)


def test_builder_infers_placement_legality_when_optional_mask_is_absent():
    value = observation()
    value["self"]["hotbar"][2] = {
        "name": "obsidian", "count": 16, "durability": 0,
        "max_durability": 0, "enchant_hash": 0,
    }
    value["blocks"] = [{
        "tactical_placement_target": True,
        "within_reach": True,
        "raycastable": True,
    }]
    assert "tactical_block_place_ready" not in value["action_mask"]
    action = scripted_action(value, "obsidian_builder")
    assert (action["primary"], action["hotbar"]) == ("use_main", 2)


def test_strong_style_families_have_signature_controls():
    value = observation(tick=7)
    value["opponent"]["distance"] = 2.5
    value["action_mask"]["crystal_attack_ready"] = True
    assert scripted_action(value, "spacing")["forward"] == -1
    assert scripted_action(value, "sprint_reset")["forward"] == 0
    assert scripted_action(value, "crystal_escape")["forward"] == -1
    high_ground = observation(tick=10)
    high_ground["opponent"]["distance"] = 4.0
    assert scripted_action(high_ground, "high_ground")["jump"] is True


def _pass_crazy_gauntlet(manager: LeagueManager) -> None:
    for style in CRAZY_STYLES:
        for index in range(4):
            episode = f"pass-{style}-{index}"
            manager.assignments[episode] = EpisodeAssignment(
                episode, "expert_script", "opponent", style=style
            )
            manager.record_result(episode, "main", 10.0, "win")
    report = manager.crazy_report()
    assert report["ready"] is True


def _set_crazy_score(
    manager: LeagueManager,
    score: float,
    *,
    samples: int = 4,
) -> None:
    manager.crazy_results = {
        style: [score] * samples for style in CRAZY_STYLES
    }
