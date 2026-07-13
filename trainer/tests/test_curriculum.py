from pathlib import Path

import pytest

from combat_ai.curriculum import CurriculumState, GATES, is_held_out


def test_holdout_split_is_stable_and_validates_fraction():
    first = is_held_out(42, 3, 1)
    assert is_held_out(42, 3, 1) is first
    assert is_held_out(0, 0, 0) is True
    assert is_held_out(42, 3, 1) is False
    with pytest.raises(ValueError):
        is_held_out(1, 0, 0, 1.1)


def test_curriculum_does_not_promote_with_missing_gates(tmp_path: Path):
    state = CurriculumState()
    result = state.evaluate({"stage": "infrastructure", "metrics": {}})
    assert result["promoted"] is False
    assert len(result["failed_gates"]) == len(GATES["infrastructure"])
    state.save(tmp_path / "curriculum.json")
    assert CurriculumState.load(tmp_path / "curriculum.json").current_stage == "infrastructure"


def test_curriculum_promotes_only_after_all_exact_gates_pass():
    state = CurriculumState()
    metrics = {
        gate.metric: {"episodes": gate.minimum_episodes, "rate": gate.minimum_rate or 0.0, "value": gate.minimum_value or 0.0}
        for gate in GATES["infrastructure"]
    }
    result = state.evaluate({"stage": "infrastructure", "metrics": metrics})
    assert result["promoted"] is True
    assert state.current_stage == "sword"
