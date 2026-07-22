from __future__ import annotations

import torch
import pytest

from combat_ai.imitation import _imitation_head_weights


HEADS = [
    "forward", "strafe", "jump", "sprint", "sneak", "primary",
    "release_use", "hotbar", "swap_offhand", "camera",
]


def _action(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "primary": "none", "hotbar": -1,
        "yaw_delta": 0.0, "pitch_delta": 0.0,
    }
    value.update(overrides)
    return value


def test_block_teacher_weights_useful_heads_without_teaching_noop_passivity() -> None:
    records = [{
        "execution_source": "teacher_block",
        "action": _action(hotbar=3, yaw_delta=0.2, pitch_delta=-0.1),
    }]

    weights = _imitation_head_weights(records, HEADS, torch.device("cpu"))[0]
    by_name = dict(zip(HEADS, weights.tolist()))

    for name in (
        "forward", "strafe", "jump", "sprint", "sneak",
        "release_use", "swap_offhand",
    ):
        assert by_name[name] == pytest.approx(0.05)
    assert by_name["hotbar"] == pytest.approx(4.0)
    assert by_name["camera"] == pytest.approx(4.0)
    assert by_name["primary"] == pytest.approx(0.5)


def test_block_teacher_placement_emphasizes_primary_and_hotbar() -> None:
    records = [{
        "execution_source": "teacher_block",
        "action": _action(primary="use_main", hotbar=3),
    }]

    weights = _imitation_head_weights(records, HEADS, torch.device("cpu"))[0]
    by_name = dict(zip(HEADS, weights.tolist()))

    assert by_name["primary"] == pytest.approx(4.0)
    assert by_name["hotbar"] == pytest.approx(4.0)
    assert by_name["camera"] == pytest.approx(0.5)
    assert by_name["forward"] == pytest.approx(0.05)
