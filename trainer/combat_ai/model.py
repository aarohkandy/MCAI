from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .features import (
    BLOCK_SIZE, ENTITY_SIZE, HISTORY_SIZE, LEGAL_SIZE, OPPONENT_SIZE,
    SELF_SIZE, SURVIVAL_SIZE, TACTICAL_CANDIDATE_SIZE, THREAT_SIZE, FeatureBatch,
)

HIDDEN_SIZE = 256
INTENT_NAMES = (
    "sword_engage", "crystal_acquire", "crystal_place", "crystal_detonate",
    "build_pad", "mine_path", "heal_retotem", "disengage", "reposition",
)
CATEGORICAL_SIZES = {
    "intent": len(INTENT_NAMES), "target_index": 17,
    "forward": 3, "strafe": 3, "jump": 2, "sprint": 2, "sneak": 2,
    "primary": 4, "release_use": 2, "hotbar": 10, "swap_offhand": 2,
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


class CandidateAttention(nn.Module):
    def __init__(self, input_size: int, width: int = 128):
        super().__init__()
        self.encoder = MLP(input_size, width, width)
        self.query = nn.Linear(width, width)
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.output = MLP(width, width * 2, width)

    def forward(self, values: torch.Tensor, mask: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(values)
        empty = ~mask.bool().any(dim=1)
        safe_mask = mask.bool().clone()
        safe_mask[empty, 0] = True
        attended, _ = self.attention(
            self.query(query).unsqueeze(1), encoded, encoded,
            key_padding_mask=~safe_mask, need_weights=False,
        )
        result = self.output(attended[:, 0])
        return result * (~empty).to(result.dtype).unsqueeze(-1)


class CombatPolicy(nn.Module):
    """ObservationV2 policy with relational candidate attention and hierarchical actions."""

    architecture_version = 2

    def __init__(self):
        super().__init__()
        self.self_encoder = MLP(SELF_SIZE, 192, 128)
        self.opponent_encoder = MLP(OPPONENT_SIZE, 128, 128)
        self.entity_encoder = MLP(ENTITY_SIZE, 96, 96)
        self.block_encoder = MLP(BLOCK_SIZE, 96, 96)
        self.legal_encoder = MLP(LEGAL_SIZE, 64, 64)
        self.context_encoder = MLP(SURVIVAL_SIZE + THREAT_SIZE, 96, 96)
        self.crystal_attention = CandidateAttention(TACTICAL_CANDIDATE_SIZE)
        self.tactical_block_attention = CandidateAttention(TACTICAL_CANDIDATE_SIZE)
        self.history_attention = CandidateAttention(HISTORY_SIZE)
        # The query is grounded in the fighter/opponent relationship. Candidate
        # attention therefore retains which target has which tactical outcome.
        self.attention_query = MLP(256, 192, 128)
        self.fusion = MLP(128 + 128 + 192 + 192 + 64 + 96 + 384, 384, 384)
        self.memory = nn.GRU(384, HIDDEN_SIZE, batch_first=True)
        self.categorical_heads = nn.ModuleDict({
            f"head_{name}": nn.Linear(HIDDEN_SIZE, size)
            for name, size in CATEGORICAL_SIZES.items()
        })
        self.camera_mean = nn.Linear(HIDDEN_SIZE, 2)
        self.camera_log_std = nn.Parameter(torch.full((2,), -1.4))
        self.value_head = nn.Linear(HIDDEN_SIZE, 1)

    def initial_hidden(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return torch.zeros((1, batch_size, HIDDEN_SIZE), dtype=torch.float32, device=device)

    def encode(self, features: FeatureBatch) -> torch.Tensor:
        self_features = self.self_encoder(features.self_state)
        opponent = self.opponent_encoder(features.opponent) * features.opponent_mask
        query = self.attention_query(torch.cat((self_features, opponent), dim=-1))
        entities = _masked_pool(self.entity_encoder(features.entities), features.entity_mask)
        blocks = _masked_pool(self.block_encoder(features.blocks), features.block_mask)
        legal = self.legal_encoder(features.legal)
        context = self.context_encoder(torch.cat((features.survival, features.threat), dim=-1))
        candidates = torch.cat((
            self.crystal_attention(features.crystal_candidates, features.crystal_candidate_mask, query),
            self.tactical_block_attention(features.tactical_blocks, features.tactical_block_mask, query),
            self.history_attention(features.recent_history, features.recent_history_mask, query),
        ), dim=-1)
        return self.fusion(torch.cat((
            self_features, opponent, entities, blocks, legal, context, candidates,
        ), dim=-1))

    def forward(self, features: FeatureBatch, hidden: torch.Tensor) -> PolicyOutput:
        memory, next_hidden = self.memory(self.encode(features).unsqueeze(1), hidden)
        state = memory[:, 0]
        return PolicyOutput(
            logits={name: self.categorical_heads[f"head_{name}"](state) for name in CATEGORICAL_SIZES},
            camera_mean=self.camera_mean(state),
            camera_log_std=self.camera_log_std.expand(state.shape[0], -1),
            value=self.value_head(state).squeeze(-1), hidden=next_hidden,
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _masked_pool(encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.unsqueeze(-1)
    denominator = expanded.sum(dim=1).clamp_min(1.0)
    mean = (encoded * expanded).sum(dim=1) / denominator
    maximum = encoded.masked_fill(expanded < 0.5, -1e9).max(dim=1).values
    maximum = maximum * (mask.sum(dim=1, keepdim=True) > 0).to(encoded.dtype)
    return torch.cat((mean, maximum), dim=-1)
