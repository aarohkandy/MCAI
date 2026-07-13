from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .distribution import ActionTensor
from .features import FeatureBatch


@dataclass
class Transition:
    agent_id: str
    episode_id: str
    policy_version: int
    features: dict[str, np.ndarray]
    hidden: np.ndarray
    categorical_action: dict[str, int]
    camera_action: np.ndarray
    old_log_probability: float
    old_value: float
    reward: float
    done: bool
    next_value: float
    advantage: float = 0.0
    return_value: float = 0.0


@dataclass
class SequenceBatch:
    features: FeatureBatch
    hidden: torch.Tensor
    actions: ActionTensor
    old_log_probability: torch.Tensor
    old_value: torch.Tensor
    advantage: torch.Tensor
    returns: torch.Tensor
    done: torch.Tensor
    valid: torch.Tensor

    @property
    def sequence_count(self) -> int:
        return self.valid.shape[0]

    @property
    def sequence_length(self) -> int:
        return self.valid.shape[1]

    def index(self, indices: torch.Tensor) -> "SequenceBatch":
        feature_values = {
            name: value[indices] for name, value in vars(self.features).items()
        }
        return SequenceBatch(
            features=FeatureBatch(**feature_values),
            hidden=self.hidden[:, indices],
            actions=ActionTensor(
                categorical={name: value[indices] for name, value in self.actions.categorical.items()},
                camera=self.actions.camera[indices],
            ),
            old_log_probability=self.old_log_probability[indices],
            old_value=self.old_value[indices],
            advantage=self.advantage[indices],
            returns=self.returns[indices],
            done=self.done[indices],
            valid=self.valid[indices],
        )


class RolloutBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.transitions: list[Transition] = []

    def append(self, transition: Transition) -> None:
        self.transitions.append(transition)

    @property
    def ready(self) -> bool:
        return len(self.transitions) >= self.capacity

    def drain(self, policy_version: int) -> list[Transition]:
        accepted = [transition for transition in self.transitions if transition.policy_version == policy_version]
        self.transitions.clear()
        return accepted

    def __len__(self) -> int:
        return len(self.transitions)


def prepare_sequences(
    transitions: list[Transition],
    sequence_length: int,
    gamma: float,
    gae_lambda: float,
    device: torch.device | str,
) -> SequenceBatch:
    if not transitions:
        raise ValueError("cannot prepare an empty rollout")
    grouped: dict[tuple[str, str], list[Transition]] = defaultdict(list)
    for transition in transitions:
        grouped[(transition.agent_id, transition.episode_id)].append(transition)
    for group in grouped.values():
        _calculate_advantages(group, gamma, gae_lambda)
    flat_advantages = np.asarray([transition.advantage for transition in transitions], dtype=np.float32)
    mean = float(flat_advantages.mean())
    standard_deviation = float(flat_advantages.std()) + 1e-8
    for transition in transitions:
        transition.advantage = (transition.advantage - mean) / standard_deviation

    chunks: list[list[Transition]] = []
    for group in grouped.values():
        for start in range(0, len(group), sequence_length):
            chunks.append(group[start:start + sequence_length])

    feature_keys = tuple(transitions[0].features.keys())
    feature_arrays: dict[str, list[np.ndarray]] = {key: [] for key in feature_keys}
    categorical_names = tuple(transitions[0].categorical_action.keys())
    categorical_arrays: dict[str, list[np.ndarray]] = {name: [] for name in categorical_names}
    hidden_values: list[np.ndarray] = []
    camera_values: list[np.ndarray] = []
    scalar_values: dict[str, list[np.ndarray]] = {
        name: [] for name in ("old_log_probability", "old_value", "advantage", "returns", "done", "valid")
    }
    for chunk in chunks:
        padding = sequence_length - len(chunk)
        hidden_values.append(chunk[0].hidden)
        for key in feature_keys:
            values = [entry.features[key] for entry in chunk]
            pad_value = np.zeros_like(values[0])
            if key == "legal":
                pad_value[0] = 1.0
            feature_arrays[key].append(np.stack(values + [pad_value] * padding))
        for name in categorical_names:
            values = [entry.categorical_action[name] for entry in chunk] + [0] * padding
            categorical_arrays[name].append(np.asarray(values, dtype=np.int64))
        cameras = [entry.camera_action for entry in chunk] + [np.zeros(2, dtype=np.float32)] * padding
        camera_values.append(np.stack(cameras))
        valid = [1.0] * len(chunk) + [0.0] * padding
        scalar_values["old_log_probability"].append(_padded_scalar(chunk, "old_log_probability", padding))
        scalar_values["old_value"].append(_padded_scalar(chunk, "old_value", padding))
        scalar_values["advantage"].append(_padded_scalar(chunk, "advantage", padding))
        scalar_values["returns"].append(_padded_scalar(chunk, "return_value", padding))
        scalar_values["done"].append(np.asarray([float(entry.done) for entry in chunk] + [1.0] * padding, dtype=np.float32))
        scalar_values["valid"].append(np.asarray(valid, dtype=np.float32))

    features = FeatureBatch(**{
        key: torch.from_numpy(np.stack(values)).to(device) for key, values in feature_arrays.items()
    })
    actions = ActionTensor(
        categorical={
            name: torch.from_numpy(np.stack(values)).to(device)
            for name, values in categorical_arrays.items()
        },
        camera=torch.from_numpy(np.stack(camera_values)).to(device),
    )
    scalars = {
        name: torch.from_numpy(np.stack(values)).to(device)
        for name, values in scalar_values.items()
    }
    hidden = torch.from_numpy(np.stack(hidden_values)).to(device).unsqueeze(0)
    return SequenceBatch(
        features=features,
        hidden=hidden,
        actions=actions,
        old_log_probability=scalars["old_log_probability"],
        old_value=scalars["old_value"],
        advantage=scalars["advantage"],
        returns=scalars["returns"],
        done=scalars["done"],
        valid=scalars["valid"],
    )


def features_at(features: FeatureBatch, time_index: int) -> FeatureBatch:
    return FeatureBatch(**{name: value[:, time_index] for name, value in vars(features).items()})


def actions_at(actions: ActionTensor, time_index: int) -> ActionTensor:
    return ActionTensor(
        categorical={name: value[:, time_index] for name, value in actions.categorical.items()},
        camera=actions.camera[:, time_index],
    )


def _calculate_advantages(group: list[Transition], gamma: float, gae_lambda: float) -> None:
    following_advantage = 0.0
    for transition in reversed(group):
        non_terminal = 0.0 if transition.done else 1.0
        delta = transition.reward + gamma * transition.next_value * non_terminal - transition.old_value
        transition.advantage = delta + gamma * gae_lambda * non_terminal * following_advantage
        transition.return_value = transition.advantage + transition.old_value
        following_advantage = transition.advantage


def _padded_scalar(chunk: list[Transition], name: str, padding: int) -> np.ndarray:
    return np.asarray([float(getattr(entry, name)) for entry in chunk] + [0.0] * padding, dtype=np.float32)
