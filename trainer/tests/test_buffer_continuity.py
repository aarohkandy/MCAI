from __future__ import annotations

import numpy as np
import pytest

from combat_ai.buffer import Transition, prepare_sequences
from combat_ai.distribution import actions_from_wire
from combat_ai.features import encode_observation
from fixtures import observation


_WIRE_NOOP = {
    "schema_version": 1,
    "forward": 0,
    "strafe": 0,
    "jump": False,
    "sprint": False,
    "sneak": False,
    "yaw_delta": 0.0,
    "pitch_delta": 0.0,
    "primary": "none",
    "release_use": False,
    "hotbar": -1,
    "swap_offhand": False,
}
_ACTION = actions_from_wire([_WIRE_NOOP], "cpu")


def _transition(
    action_id: int,
    recurrent_parent_action_id: int | None,
    *,
    tick: int,
    reward: float = 0.0,
    done: bool = False,
    execution_gap_before: bool = False,
) -> Transition:
    return Transition(
        agent_id="fighter-a",
        episode_id="episode-a",
        policy_version=1,
        action_id=action_id,
        recurrent_parent_action_id=recurrent_parent_action_id,
        execution_gap_before=execution_gap_before,
        features=encode_observation(observation(episode="episode-a", tick=tick)),
        hidden=np.full(128, float(action_id), dtype=np.float32),
        categorical_action={
            name: int(value[0]) for name, value in _ACTION.categorical.items()
        },
        camera_action=np.zeros(2, dtype=np.float32),
        old_log_probability=-1.0,
        old_value=0.0,
        reward=reward,
        done=done,
        next_value=0.0,
    )


def test_contiguous_recurrent_parent_chain_stays_in_one_sequence() -> None:
    transitions = [
        _transition(100, None, tick=1),
        _transition(101, 100, tick=2),
        _transition(102, 101, tick=3, done=True),
    ]

    batch = prepare_sequences(
        transitions, sequence_length=4, gamma=0.995, gae_lambda=0.95, device="cpu"
    )

    assert batch.sequence_count == 1
    assert batch.valid[0].tolist() == [1.0, 1.0, 1.0, 0.0]


def test_missing_recurrent_parent_splits_sequences() -> None:
    transitions = [
        _transition(100, None, tick=1),
        # Action 101 was proposed and advanced the GRU, but a teacher executed
        # instead, so it is deliberately absent from the PPO rollout.
        _transition(102, 101, tick=3),
        _transition(103, 102, tick=4, done=True),
    ]

    batch = prepare_sequences(
        transitions, sequence_length=4, gamma=0.995, gae_lambda=0.95, device="cpu"
    )

    assert batch.sequence_count == 2
    assert batch.valid.sum(dim=1).tolist() == [1.0, 2.0]
    assert batch.hidden[0, :, 0].tolist() == [100.0, 102.0]


def test_gae_does_not_cross_a_recurrent_lineage_gap() -> None:
    transitions = [
        _transition(100, None, tick=1, reward=1.0),
        # If GAE incorrectly crossed this missing-parent boundary, the first
        # return would include gamma * lambda * 10 from the later transition.
        _transition(102, 101, tick=3, reward=10.0, done=True),
    ]

    batch = prepare_sequences(
        transitions, sequence_length=4, gamma=0.995, gae_lambda=0.95, device="cpu"
    )

    assert batch.sequence_count == 2
    assert batch.returns[:, 0].tolist() == pytest.approx([1.0, 10.0])


def test_execution_override_splits_even_with_contiguous_proposal_parent() -> None:
    transitions = [
        _transition(100, None, tick=1, reward=1.0),
        _transition(
            101, 100, tick=2, reward=10.0, done=True,
            execution_gap_before=True,
        ),
    ]

    batch = prepare_sequences(
        transitions, sequence_length=4, gamma=0.995, gae_lambda=0.95, device="cpu"
    )

    assert batch.sequence_count == 2
    assert batch.returns[:, 0].tolist() == pytest.approx([1.0, 10.0])
