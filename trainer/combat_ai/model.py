from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .features import (
    BLOCK_SIZE,
    ENTITY_SIZE,
    LEGAL_SIZE,
    OPPONENT_SIZE,
    SELF_SIZE,
    FeatureBatch,
)

HIDDEN_SIZE = 128
CATEGORICAL_SIZES = {
    "forward": 3,
    "strafe": 3,
    "jump": 2,
    "sprint": 2,
    "sneak": 2,
    "primary": 4,
    "release_use": 2,
    "hotbar": 10,
    "swap_offhand": 2,
}


@dataclass
class PolicyOutput:
    logits: dict[str, torch.Tensor]
    camera_mean: torch.Tensor
    camera_log_std: torch.Tensor
    value: torch.Tensor
    hidden: torch.Tensor


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, output_size), nn.Tanh(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class CombatPolicy(nn.Module):
    """Small structured-state policy shared by every fighter."""

    def __init__(self):
        super().__init__()
        self.self_encoder = MLP(SELF_SIZE, 128, 96)
        self.opponent_encoder = MLP(OPPONENT_SIZE, 96, 64)
        self.entity_encoder = MLP(ENTITY_SIZE, 64, 64)
        self.block_encoder = MLP(BLOCK_SIZE, 64, 64)
        self.legal_encoder = MLP(LEGAL_SIZE, 32, 24)
        self.fusion = MLP(96 + 64 + 128 + 128 + 24, 192, 192)
        self.memory = nn.GRU(192, HIDDEN_SIZE, batch_first=True)
        self.categorical_heads = nn.ModuleDict({
            f"head_{name}": nn.Linear(HIDDEN_SIZE, size) for name, size in CATEGORICAL_SIZES.items()
        })
        self.camera_mean = nn.Linear(HIDDEN_SIZE, 2)
        self.camera_log_std = nn.Parameter(torch.full((2,), -1.4))
        self.value_head = nn.Linear(HIDDEN_SIZE, 1)

    def initial_hidden(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return torch.zeros((1, batch_size, HIDDEN_SIZE), dtype=torch.float32, device=device)

    def encode(self, features: FeatureBatch) -> torch.Tensor:
        self_features = self.self_encoder(features.self_state)
        opponent = self.opponent_encoder(features.opponent) * features.opponent_mask
        entities = self.entity_encoder(features.entities)
        blocks = self.block_encoder(features.blocks)
        entity_pool = _masked_pool(entities, features.entity_mask)
        block_pool = _masked_pool(blocks, features.block_mask)
        legal = self.legal_encoder(features.legal)
        return self.fusion(torch.cat((self_features, opponent, entity_pool, block_pool, legal), dim=-1))

    def forward(self, features: FeatureBatch, hidden: torch.Tensor) -> PolicyOutput:
        fused = self.encode(features).unsqueeze(1)
        memory, next_hidden = self.memory(fused, hidden)
        state = memory[:, 0]
        return PolicyOutput(
            logits={name: self.categorical_heads[f"head_{name}"](state) for name in CATEGORICAL_SIZES},
            camera_mean=self.camera_mean(state),
            camera_log_std=self.camera_log_std.expand(state.shape[0], -1),
            value=self.value_head(state).squeeze(-1),
            hidden=next_hidden,
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _masked_pool(encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.unsqueeze(-1)
    denominator = expanded.sum(dim=1).clamp_min(1.0)
    mean = (encoded * expanded).sum(dim=1) / denominator
    maximum = encoded.masked_fill(expanded < 0.5, -1e9).max(dim=1).values
    any_present = (mask.sum(dim=1, keepdim=True) > 0).to(encoded.dtype)
    maximum = maximum * any_present
    return torch.cat((mean, maximum), dim=-1)
