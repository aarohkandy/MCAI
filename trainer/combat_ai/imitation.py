from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .distribution import actions_from_wire
from .features import batch_observations, categorical_masks
from .model import CombatPolicy


def load_demonstrations(path: Path) -> dict[str, list[dict[str, Any]]]:
    matches: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "observation" not in record or "action" not in record:
                raise ValueError(f"demonstration line {line_number} lacks observation/action")
            match_id = str(record.get("match_id") or record["observation"]["match"]["episode_id"])
            matches[match_id].append(record)
    return dict(matches)


def split_matches(
    matches: dict[str, list[dict[str, Any]]], seed: int = 7
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    identifiers = sorted(matches)
    random.Random(seed).shuffle(identifiers)
    train_end = max(1, round(len(identifiers) * 0.8))
    validation_end = max(train_end, round(len(identifiers) * 0.9))
    groups = (identifiers[:train_end], identifiers[train_end:validation_end], identifiers[validation_end:])
    return tuple([record for identifier in group for record in matches[identifier]] for group in groups)  # type: ignore[return-value]


def behavior_clone(
    policy: CombatPolicy,
    demonstrations: Path,
    output: Path,
    device: torch.device,
    epochs: int = 30,
    batch_size: int = 256,
    patience: int = 4,
) -> dict[str, float]:
    train, validation, test = split_matches(load_demonstrations(demonstrations))
    if not train or not validation:
        raise ValueError("behavior cloning requires demonstrations from at least two whole matches")
    policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    for _ in range(epochs):
        random.shuffle(train)
        policy.train()
        for start in range(0, len(train), batch_size):
            batch = train[start:start + batch_size]
            loss = imitation_loss(policy, batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
        validation_loss = evaluate_imitation(policy, validation, batch_size, device)
        if validation_loss + 1e-5 < best_loss:
            best_loss = validation_loss
            best_state = {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("behavior cloning produced no checkpoint")
    policy.load_state_dict(best_state)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "policy": best_state, "source": str(demonstrations)}, output)
    return {"validation_loss": best_loss, "test_loss": evaluate_imitation(policy, test or validation, batch_size, device)}


def imitation_loss(policy: CombatPolicy, records: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    features = batch_observations([record["observation"] for record in records], device)
    actions = actions_from_wire([record["action"] for record in records], device)
    output = policy(features, policy.initial_hidden(len(records), device))
    masks = categorical_masks(features)
    losses: list[torch.Tensor] = []
    for name, logits in output.logits.items():
        mask = masks[name]
        if name == "release_use":
            mask = mask.clone()
            mask[:, 1] &= actions.categorical["primary"] < 2
        masked = logits.masked_fill(~mask, -1e9)
        losses.append(nn.functional.cross_entropy(masked, actions.categorical[name]))
    camera_scale = torch.tensor((math.pi, math.pi / 2), device=device)
    target = torch.atanh((actions.camera / camera_scale).clamp(-0.999, 0.999))
    losses.append(nn.functional.smooth_l1_loss(output.camera_mean, target))
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate_imitation(
    policy: CombatPolicy, records: list[dict[str, Any]], batch_size: int, device: torch.device
) -> float:
    policy.eval()
    total = 0.0
    count = 0
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        total += float(imitation_loss(policy, batch, device)) * len(batch)
        count += len(batch)
    return total / max(count, 1)
