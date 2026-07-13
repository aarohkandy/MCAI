from pathlib import Path

import torch

from combat_ai.league import LeagueManager, scripted_action
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
    assert first.mode in {"mirror", "historical", "scripted"}


def test_scripted_teacher_uses_only_action_v1_controls():
    action = scripted_action(observation(), "rush")
    assert action["schema_version"] == 1
    assert action["hotbar"] == -1
    assert -1 <= action["forward"] <= 1
    assert action["primary"] in {"none", "attack", "use_main", "use_offhand"}


def test_five_flat_evaluations_request_an_exploiter(tmp_path: Path):
    manager = LeagueManager(tmp_path, __import__('torch').device('cpu'))
    outcomes = [manager.note_evaluation(1200 + index * 0.5) for index in range(5)]
    assert outcomes[-1] is True


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
    promoted = main_pool.add_frozen_checkpoint(target)
    assert promoted.exists()
    assert promoted.name.startswith("policy-exploiter-")
