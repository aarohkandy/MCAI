from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch.distributions import Categorical, Normal

from .features import FeatureBatch, PRIMARY_NAMES, categorical_masks
from .model import PolicyOutput

CAMERA_SCALE = (math.pi, math.pi / 2)


@dataclass
class ActionTensor:
    categorical: dict[str, torch.Tensor]
    camera: torch.Tensor

    def index(self, indices: torch.Tensor) -> "ActionTensor":
        return ActionTensor(
            categorical={name: value[indices] for name, value in self.categorical.items()},
            camera=self.camera[indices],
        )


def sample_actions(
    output: PolicyOutput,
    features: FeatureBatch,
    deterministic: bool = False,
) -> tuple[list[dict[str, Any]], ActionTensor, torch.Tensor, torch.Tensor]:
    masks = categorical_masks(features)
    categorical: dict[str, torch.Tensor] = {}
    total_log_probability = torch.zeros_like(output.value)
    total_entropy = torch.zeros_like(output.value)
    for name, logits in output.logits.items():
        mask = _conditional_mask(name, masks[name], categorical.get("primary"))
        distribution = Categorical(logits=_masked_logits(logits, mask))
        action = distribution.probs.argmax(dim=-1) if deterministic else distribution.sample()
        categorical[name] = action
        total_log_probability += distribution.log_prob(action)
        total_entropy += distribution.entropy()
    camera, camera_log_probability, camera_entropy = _sample_camera(
        output.camera_mean, output.camera_log_std, deterministic
    )
    total_log_probability += camera_log_probability
    total_entropy += camera_entropy
    tensor = ActionTensor(categorical=categorical, camera=camera)
    return _to_wire_actions(tensor), tensor, total_log_probability, total_entropy


def evaluate_actions(
    output: PolicyOutput,
    features: FeatureBatch,
    actions: ActionTensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    masks = categorical_masks(features)
    total_log_probability = torch.zeros_like(output.value)
    total_entropy = torch.zeros_like(output.value)
    for name, logits in output.logits.items():
        mask = _conditional_mask(name, masks[name], actions.categorical.get("primary"))
        distribution = Categorical(logits=_masked_logits(logits, mask))
        total_log_probability += distribution.log_prob(actions.categorical[name])
        total_entropy += distribution.entropy()
    camera_log_probability, camera_entropy = _camera_log_probability(
        output.camera_mean, output.camera_log_std, actions.camera
    )
    return total_log_probability + camera_log_probability, total_entropy + camera_entropy


def actions_from_wire(actions: list[dict[str, Any]], device: torch.device | str) -> ActionTensor:
    categorical = {
        "forward": torch.tensor([int(a["forward"]) + 1 for a in actions], device=device),
        "strafe": torch.tensor([int(a["strafe"]) + 1 for a in actions], device=device),
        "jump": torch.tensor([int(bool(a["jump"])) for a in actions], device=device),
        "sprint": torch.tensor([int(bool(a["sprint"])) for a in actions], device=device),
        "sneak": torch.tensor([int(bool(a["sneak"])) for a in actions], device=device),
        "primary": torch.tensor([PRIMARY_NAMES.index(a["primary"]) for a in actions], device=device),
        "release_use": torch.tensor([int(bool(a["release_use"])) for a in actions], device=device),
        "hotbar": torch.tensor([int(a["hotbar"]) + 1 for a in actions], device=device),
        "swap_offhand": torch.tensor([int(bool(a["swap_offhand"])) for a in actions], device=device),
    }
    camera = torch.tensor([[float(a["yaw_delta"]), float(a["pitch_delta"])] for a in actions], device=device)
    return ActionTensor(categorical=categorical, camera=camera)


def _sample_camera(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normal = Normal(mean, log_std.exp())
    latent = mean if deterministic else normal.rsample()
    squashed = torch.tanh(latent)
    scale = torch.tensor(CAMERA_SCALE, dtype=mean.dtype, device=mean.device)
    action = squashed * scale
    log_probability = normal.log_prob(latent) - torch.log(scale * (1 - squashed.square()) + 1e-6)
    entropy = -log_probability
    return action, log_probability.sum(-1), entropy.sum(-1)


def _camera_log_probability(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = torch.tensor(CAMERA_SCALE, dtype=mean.dtype, device=mean.device)
    squashed = (action / scale).clamp(-0.999999, 0.999999)
    latent = torch.atanh(squashed)
    normal = Normal(mean, log_std.exp())
    log_probability = normal.log_prob(latent) - torch.log(scale * (1 - squashed.square()) + 1e-6)
    return log_probability.sum(-1), (-log_probability).sum(-1)


def _masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if torch.any(~mask.any(dim=-1)):
        raise ValueError("an action head has no legal choice")
    return logits.masked_fill(~mask, -1e9)


def _conditional_mask(name: str, mask: torch.Tensor, primary: torch.Tensor | None) -> torch.Tensor:
    if name != "release_use" or primary is None:
        return mask
    result = mask.clone()
    result[:, 1] &= primary < 2
    return result


def _to_wire_actions(actions: ActionTensor) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    categorical = {name: value.detach().cpu().tolist() for name, value in actions.categorical.items()}
    camera = actions.camera.detach().cpu().tolist()
    for index in range(len(camera)):
        result.append({
            "schema_version": 1,
            "forward": int(categorical["forward"][index]) - 1,
            "strafe": int(categorical["strafe"][index]) - 1,
            "jump": bool(categorical["jump"][index]),
            "sprint": bool(categorical["sprint"][index]),
            "sneak": bool(categorical["sneak"][index]),
            "yaw_delta": float(camera[index][0]),
            "pitch_delta": float(camera[index][1]),
            "primary": PRIMARY_NAMES[int(categorical["primary"][index])],
            "release_use": bool(categorical["release_use"][index]),
            "hotbar": int(categorical["hotbar"][index]) - 1,
            "swap_offhand": bool(categorical["swap_offhand"][index]),
        })
    return result
