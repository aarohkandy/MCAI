from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .distribution import (
    CAMERA_SCALE, _hierarchical_mask, action_logits, actions_from_wire,
    camera_action_mean,
)
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
    names: list[str] = []
    for name, logits in output.logits.items():
        mask = _hierarchical_mask(name, logits, masks, actions.categorical, features)
        masked = action_logits(name, logits, features).masked_fill(~mask, -1e9)
        losses.append(nn.functional.cross_entropy(
            masked, actions.categorical[name], reduction="none"
        ))
        names.append(name)
    camera_scale = torch.tensor(CAMERA_SCALE, device=device)
    target = torch.atanh((actions.camera / camera_scale).clamp(-0.999, 0.999))
    losses.append(nn.functional.smooth_l1_loss(
        camera_action_mean(output.camera_mean, features), target, reduction="none"
    ).mean(dim=-1))
    names.append("camera")
    stacked = torch.stack(losses, dim=1)
    weights = _imitation_head_weights(records, names, device)
    return (stacked * weights).sum() / weights.sum().clamp_min(1.0)


def classify_crystal_teacher_action(action: Any) -> str | None:
    """Return the useful phase represented by a crystal teacher control.

    Passive waits/no-ops are deliberately excluded. Training those abundant
    records taught the policy to pause between placement and detonation.
    """
    if not isinstance(action, dict):
        return None
    primary = str(action.get("primary", "none"))
    if primary == "attack":
        return "detonate"
    if primary in {"use_main", "use_offhand"}:
        return "place"
    try:
        hotbar = int(action.get("hotbar", -1))
        yaw = float(action.get("yaw_delta", 0.0))
        pitch = float(action.get("pitch_delta", 0.0))
    except (TypeError, ValueError, OverflowError):
        return None
    if hotbar >= 0:
        return "select"
    if abs(yaw) > 1e-5 or abs(pitch) > 1e-5:
        return "aim"
    return None


def _imitation_head_weights(
    records: list[dict[str, Any]], names: list[str], device: torch.device,
) -> torch.Tensor:
    weights = torch.ones((len(records), len(names)), dtype=torch.float32, device=device)
    indices = {name: index for index, name in enumerate(names)}
    low_relevance = (
        "forward", "strafe", "jump", "sprint", "sneak", "release_use", "swap_offhand",
    )
    phase_emphasis = {
        "aim": {"camera": 4.0, "primary": 0.5, "hotbar": 0.5},
        "select": {"hotbar": 4.0, "camera": 2.0, "primary": 0.5},
        "place": {"primary": 4.0, "hotbar": 3.0, "camera": 2.0},
        "detonate": {"primary": 4.0, "camera": 3.0, "hotbar": 0.5},
    }
    for row, record in enumerate(records):
        if str(record.get("execution_source", "")) == "elite_policy":
            try:
                quality = float(record.get("elite_quality", 1.0))
            except (TypeError, ValueError, OverflowError):
                quality = 0.25
            if not math.isfinite(quality):
                quality = 0.25
            # Fast terminal wins carry a larger arena reward and therefore a
            # higher replay quality.  Keep slower verified wins useful, but do
            # not let them anchor the actor as strongly as the best sequences.
            weights[row] *= max(0.25, min(1.0, quality))
        action = record.get("action")
        if not isinstance(action, dict) or int(action.get("schema_version", 1)) != 2:
            # A resolved V1 demonstration has no hierarchical intent/target
            # ownership. Keep its legal mechanic controls without inventing a
            # candidate index that was never executed.
            for name in ("intent", "target_index"):
                if name in indices:
                    weights[row, indices[name]] = 0.0
        source = str(record.get("execution_source", ""))
        if source == "teacher_block":
            # A terrain demonstration normally holds every movement/control
            # head at no-op while it aims, selects obsidian, and clicks. Giving
            # all of those no-ops equal imitation weight taught the policy to
            # stand still. Preserve the three useful decisions while making
            # incidental locomotion/release/swap supervision nearly inert.
            for name in low_relevance:
                if name in indices:
                    weights[row, indices[name]] = 0.05
            for name in ("primary", "hotbar", "camera"):
                if name in indices:
                    weights[row, indices[name]] = 0.5
            action = record.get("action")
            action = action if isinstance(action, dict) else {}
            primary = str(action.get("primary", "none"))
            try:
                hotbar = int(action.get("hotbar", -1))
                yaw = float(action.get("yaw_delta", 0.0))
                pitch = float(action.get("pitch_delta", 0.0))
            except (TypeError, ValueError, OverflowError):
                hotbar, yaw, pitch = -1, 0.0, 0.0
            if primary in {"attack", "use_main", "use_offhand"} and "primary" in indices:
                weights[row, indices["primary"]] = 4.0
            if hotbar >= 0 and "hotbar" in indices:
                weights[row, indices["hotbar"]] = 4.0
            if (abs(yaw) > 1e-5 or abs(pitch) > 1e-5) and "camera" in indices:
                weights[row, indices["camera"]] = 4.0
            continue
        if source != "teacher_crystal":
            continue
        phase = str(record.get("teacher_phase") or "")
        if phase not in phase_emphasis:
            phase = classify_crystal_teacher_action(record.get("action")) or ""
        if phase not in phase_emphasis:
            continue
        for name in low_relevance:
            if name in indices:
                weights[row, indices[name]] = 0.2
        for name, value in phase_emphasis[phase].items():
            if name in indices:
                weights[row, indices[name]] = value
    return weights


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
