"""Neural introspection for the combat policy.

Answers one question honestly: is this policy learning, or is it broken?
Every function is pure -- it takes a policy (and optionally observations) and
returns a plain dict that is JSON-serialisable, so the dashboard and the CLI
share exactly the same numbers.

Field spans below mirror the hand-written encoders in ``features.py``. They are
duplicated here rather than derived because the encoders build flat Python
lists with no names; ``test_introspect.py`` pins the spans against the real
encoder output so the two cannot silently drift apart.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch
from torch import nn

from .distribution import CAMERA_SCALE
from .features import (
    BLOCK_SIZE,
    ENTITY_SIZE,
    LEGAL_SIZE,
    OPPONENT_SIZE,
    PRIMARY_NAMES,
    SELF_SIZE,
    FeatureBatch,
    batch_observations,
    categorical_masks,
)
from .model import CATEGORICAL_SIZES, CombatPolicy

# Tanh gradient at |a| = 0.99 is ~0.02, so anything past it is effectively
# gradient-dead and will not move again during training.
SATURATION_THRESHOLD = 0.99
# A unit counts as saturated only when it pins for almost the whole batch;
# occasional saturation on extreme inputs is normal and healthy.
UNIT_SATURATION_FRACTION = 0.9
# Below this spread across a batch (or across time, for the GRU) a unit carries
# no information at all -- it is a constant the downstream layer can absorb.
CONSTANT_UNIT_STD = 1e-3

# Slot layout verified against Arena.equipKit in the Paper plugin. The hotbar
# head has a leading "keep" choice because hotbar -1 means "do not switch".
HOTBAR_CHOICES = (
    "keep", "slot0_sword", "slot1_pickaxe", "slot2_obsidian", "slot3_crystal",
    "slot4_golden_apple", "slot5_bow", "slot6_ender_pearl", "slot7_totem", "slot8_totem",
)
CHOICE_NAMES: dict[str, tuple[str, ...]] = {
    "forward": ("backward", "still", "forward"),
    "strafe": ("strafe_negative", "strafe_zero", "strafe_positive"),
    "jump": ("false", "true"),
    "sprint": ("false", "true"),
    "sneak": ("false", "true"),
    "primary": PRIMARY_NAMES,
    "release_use": ("false", "true"),
    "hotbar": HOTBAR_CHOICES,
    "swap_offhand": ("false", "true"),
}

Span = tuple[str, int, int]

SELF_FIELDS: tuple[Span, ...] = (
    ("health", 0, 1), ("absorption", 1, 2), ("food", 2, 3), ("velocity", 3, 6),
    ("yaw_sin_cos", 6, 8), ("pitch", 8, 9), ("on_ground", 9, 10), ("sprinting", 10, 11),
    ("sneaking", 11, 12), ("hurt_time", 12, 13), ("attack_cooldown", 13, 14),
    ("use_ticks", 14, 15), ("mining_progress", 15, 16), ("selected_hotbar", 16, 17),
    ("active_hand", 17, 20), ("raycast_kind", 20, 23), ("raycast_distance", 23, 24),
    ("offhand_item", 24, 30), ("armor", 30, 42), ("hotbar_items", 42, 78),
    ("unused", 78, SELF_SIZE),
)
OPPONENT_FIELDS: tuple[Span, ...] = (
    ("relative_position", 0, 3), ("relative_velocity", 3, 6), ("yaw_sin_cos", 6, 8),
    ("pitch", 8, 9), ("health", 9, 10), ("health_known", 10, 11), ("hurt_time", 11, 12),
    ("on_ground", 12, 13), ("line_of_sight", 13, 14), ("mainhand_item", 14, 20),
    ("offhand_item", 20, 26), ("armor", 26, 38), ("unused", 38, OPPONENT_SIZE),
)
ENTITY_FIELDS: tuple[Span, ...] = (
    ("kind_one_hot", 0, 5), ("relative_position", 5, 8), ("relative_velocity", 8, 11),
    ("age_ticks", 11, 12), ("distance", 12, 13), ("raycastable", 13, 14),
    ("name_hash", 14, 15), ("unused", 15, ENTITY_SIZE),
)
BLOCK_FIELDS: tuple[Span, ...] = (
    ("relative_position", 0, 3), ("collision", 3, 7), ("hardness", 7, 8),
    ("replaceable", 8, 9), ("break_progress", 9, 10), ("crystal_clearance", 10, 11),
    ("exposed_faces", 11, 12), ("distance", 12, 13), ("within_reach", 13, 14),
    ("raycastable", 14, 15), ("sample_age_ticks", 15, 16), ("obsidian_name", 16, 17),
    ("crystal_base_name", 17, 18), ("name_hash", 18, 19), ("unused", 19, BLOCK_SIZE),
)
LEGAL_FIELDS: tuple[Span, ...] = (
    ("primary_none_bias", 0, 1), ("attack_legal", 1, 2), ("use_main_legal", 2, 3),
    ("use_offhand_legal", 3, 4), ("release_hold_bias", 4, 5), ("release_use_legal", 5, 6),
    ("swap_keep_bias", 6, 7), ("swap_offhand_legal", 7, 8), ("hotbar_legal", 8, 17),
    ("constant_tail", 17, LEGAL_SIZE),
)
GROUP_FIELDS: dict[str, tuple[Span, ...]] = {
    "self": SELF_FIELDS, "opponent": OPPONENT_FIELDS, "entities": ENTITY_FIELDS,
    "blocks": BLOCK_FIELDS, "legal": LEGAL_FIELDS,
}


# --------------------------------------------------------------------------
# checkpoint loading
# --------------------------------------------------------------------------

def load_checkpoint(path: Path | str | None) -> tuple[CombatPolicy, dict[str, Any]]:
    """Load a policy plus its training metadata.

    ``path`` of None yields a freshly initialised policy, which is what the
    tooling reports on before the first checkpoint exists.
    """
    policy = CombatPolicy()
    if path is None:
        policy.eval()
        return policy, {"source": "random-initialisation"}
    # Deliberately not reusing export.load_policy: the sidecar metadata
    # (policy version, agent ticks, last metrics) is the training-progress
    # context for this report, and a second torch.load would double the cost.
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    policy.load_state_dict(payload["policy"])
    policy.eval()
    metadata: dict[str, Any] = {"source": str(path)}
    for key in ("format_version", "policy_version", "total_agent_ticks"):
        if key in payload:
            metadata[key] = payload[key]
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        metadata["metrics"] = {name: _json_safe(value) for name, value in metrics.items()}
    return policy, metadata


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _plain(value) if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, float):
        return _plain(value)
    return str(value)


# --------------------------------------------------------------------------
# canned observations
# --------------------------------------------------------------------------

def _item(name: str, count: int = 1) -> dict[str, Any]:
    return {"name": name, "count": count, "durability": 0, "max_durability": 0, "enchant_hash": 0}


def _crystal_kit() -> list[dict[str, Any]]:
    return [
        _item("diamond_sword"), _item("diamond_pickaxe"), _item("obsidian", 64),
        _item("end_crystal", 64), _item("golden_apple", 16), _item("bow"),
        _item("ender_pearl", 16), _item("totem_of_undying"), _item("totem_of_undying"),
    ]


def _obsidian_block(distance: float, clearance: bool = True) -> dict[str, Any]:
    return {
        "name": "obsidian", "relative_position": {"x": 0.0, "y": -1.0, "z": -distance},
        "collision": "solid", "hardness": 50.0, "replaceable": False, "break_progress": 0.0,
        "crystal_clearance": clearance, "exposed_faces": 5, "distance": distance,
        "within_reach": distance <= 4.0, "raycastable": True, "sample_age_ticks": 0,
    }


def _stone_floor(count: int = 4) -> list[dict[str, Any]]:
    return [{
        "name": "stone", "relative_position": {"x": float(index - 2), "y": -1.0, "z": -1.0},
        "collision": "solid", "hardness": 1.5, "replaceable": False, "break_progress": 0.0,
        "crystal_clearance": False, "exposed_faces": 1, "distance": 1.0 + index,
        "within_reach": True, "raycastable": True, "sample_age_ticks": 0,
    } for index in range(count)]


def make_observation(
    *,
    tick: int = 1,
    health: float = 20.0,
    opponent_distance: float | None = 2.6,
    opponent_health: float = 20.0,
    attack_cooldown: float = 1.0,
    attack_legal: bool = True,
    blocks: Sequence[dict[str, Any]] | None = None,
    entities: Sequence[dict[str, Any]] | None = None,
    sprinting: bool = False,
) -> dict[str, Any]:
    """Build a schema-v1 observation for probing. Egocentric frame: forward is -z,
    so an opponent straight ahead sits at (0, 0, -distance)."""
    raycast = {"kind": "none", "distance": 0.0, "block_name": "", "entity_kind": ""}
    opponent = None
    if opponent_distance is not None:
        if opponent_distance <= 3.0:
            raycast = {"kind": "entity", "distance": opponent_distance,
                       "block_name": "", "entity_kind": "player"}
        opponent = {
            "relative_position": {"x": 0.0, "y": 0.0, "z": -float(opponent_distance)},
            "relative_velocity": {"x": 0.0, "y": 0.0, "z": 0.0}, "yaw": math.pi, "pitch": 0.0,
            "health": opponent_health, "hurt_time": 0, "on_ground": True,
            "line_of_sight": True, "mainhand": _item("diamond_sword"),
            "offhand": _item("totem_of_undying"),
            "armor": [_item("diamond_chestplate")] * 4,
        }
    return {
        "schema_version": 1,
        "match": {"episode_id": "introspect", "tick": tick, "policy_version": 0,
                  "arena_seed": 1, "action_delay_ticks": 0, "observation_delay_ticks": 0},
        "self": {
            "health": health, "absorption": 0, "food": 20,
            "position": {"x": 0, "y": 64, "z": 0}, "velocity": {"x": 0, "y": 0, "z": 0},
            "yaw": 0.0, "pitch": 0.0, "on_ground": True, "sprinting": sprinting,
            "sneaking": False, "hurt_time": 0, "attack_cooldown": attack_cooldown,
            "active_hand": "none", "use_ticks": 0, "mining_progress": 0,
            "selected_hotbar": 0, "hotbar": _crystal_kit(),
            "offhand": _item("totem_of_undying"), "armor": [_item("diamond_helmet")] * 4,
            "raycast": raycast,
        },
        "opponent": opponent,
        "entities": list(entities or []),
        "blocks": list(blocks or _stone_floor()),
        "action_mask": {"attack": attack_legal, "use_main": True, "use_offhand": True,
                        "release_use": False, "swap_offhand": True, "hotbar": [True] * 9},
    }


def probe_observations() -> dict[str, dict[str, Any]]:
    """Named scenarios whose correct response is unambiguous to a human."""
    crystal_entity = {
        "kind": "end_crystal", "relative_position": {"x": 0.0, "y": 0.0, "z": -2.0},
        "relative_velocity": {"x": 0.0, "y": 0.0, "z": 0.0}, "age_ticks": 4,
        "distance": 2.0, "raycastable": True,
    }
    return {
        "opponent_in_reach": make_observation(opponent_distance=2.6),
        "opponent_far": make_observation(opponent_distance=14.0, sprinting=True),
        "no_opponent": make_observation(opponent_distance=None),
        "attack_illegal_in_reach": make_observation(opponent_distance=2.6, attack_legal=False),
        "cooldown_not_ready": make_observation(opponent_distance=2.6, attack_cooldown=0.15),
        "crystal_spot_ready": make_observation(
            opponent_distance=3.0, blocks=[_obsidian_block(1.5), *_stone_floor()]),
        "crystal_placed_live": make_observation(
            opponent_distance=3.0, blocks=[_obsidian_block(1.5), *_stone_floor()],
            entities=[crystal_entity]),
        "low_health": make_observation(health=3.0, opponent_distance=2.6, opponent_health=18.0),
    }


def default_observations() -> list[dict[str, Any]]:
    """The probe scenarios plus filler that keeps entity and block slots
    populated, so per-unit batch statistics are computed over real samples
    instead of a single occupied slot."""
    arrows = [{
        "kind": "arrow", "relative_position": {"x": float(index) - 1.0, "y": 1.0, "z": -6.0},
        "relative_velocity": {"x": 0.0, "y": -0.1, "z": 1.4}, "age_ticks": 3 * index + 1,
        "distance": 6.0 + index, "raycastable": True,
    } for index in range(3)]
    crystals = [{
        "kind": "end_crystal", "relative_position": {"x": float(index) - 0.5, "y": 0.0, "z": -2.5},
        "relative_velocity": {"x": 0.0, "y": 0.0, "z": 0.0}, "age_ticks": 10 * index,
        "distance": 2.5 + index, "raycastable": index == 0,
    } for index in range(2)]
    return [
        *probe_observations().values(),
        make_observation(opponent_distance=7.0, entities=[*arrows, *crystals],
                         blocks=[_obsidian_block(2.0), _obsidian_block(3.5, clearance=False),
                                 *_stone_floor(8)]),
        make_observation(health=11.0, opponent_distance=4.5, opponent_health=6.0,
                         entities=crystals, attack_cooldown=0.5,
                         blocks=[_obsidian_block(1.3), *_stone_floor(6)]),
    ]


def approach_sequence(length: int = 24) -> list[dict[str, Any]]:
    """A temporal trajectory: the opponent closes from 16 blocks to melee while
    an obsidian pillar comes into reach. Only a policy that uses its recurrent
    state can distinguish early from late steps here."""
    result = []
    for step in range(length):
        progress = step / max(length - 1, 1)
        distance = 16.0 - 13.5 * progress
        result.append(make_observation(
            tick=step + 1,
            health=20.0 - 8.0 * progress,
            opponent_distance=distance,
            opponent_health=20.0 - 6.0 * progress,
            attack_cooldown=float(step % 5) / 4.0,
            blocks=[_obsidian_block(max(1.2, distance - 1.0)), *_stone_floor()],
            sprinting=distance > 5.0,
        ))
    return result


# --------------------------------------------------------------------------
# 1. layer activation health
# --------------------------------------------------------------------------

def _encoder_modules(policy: CombatPolicy) -> dict[str, nn.Module]:
    return {
        "self_encoder": policy.self_encoder, "opponent_encoder": policy.opponent_encoder,
        "entity_encoder": policy.entity_encoder, "block_encoder": policy.block_encoder,
        "legal_encoder": policy.legal_encoder, "fusion": policy.fusion,
    }


def activation_report(policy: CombatPolicy, features: FeatureBatch) -> dict[str, Any]:
    """Per-Tanh-layer activation statistics.

    Captures the raw encoder output, before the opponent mask and before the
    entity/block pooling, so a saturated encoder cannot hide behind a mask.
    Padded entity/block slots are dropped from the statistics.
    """
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def _hook(name: str) -> Callable[..., None]:
        def capture(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
            captured[name] = output.detach()
        return capture

    for encoder_name, module in _encoder_modules(policy).items():
        for index, layer in enumerate(module.layers):
            if isinstance(layer, nn.Tanh):
                label = "hidden" if index == 1 else "output"
                handles.append(layer.register_forward_hook(_hook(f"{encoder_name}.{label}")))
    try:
        with torch.no_grad():
            policy.encode(features)
    finally:
        for handle in handles:
            handle.remove()

    selectors = {"entity_encoder": features.entity_mask, "block_encoder": features.block_mask}
    layers: dict[str, Any] = {}
    for name, tensor in captured.items():
        mask = selectors.get(name.split(".")[0])
        flat = tensor[mask > 0.5] if mask is not None else tensor
        layers[name] = _activation_statistics(flat)
    saturated = sum(entry["saturated_units"] for entry in layers.values())
    dead = sum(entry["constant_units"] for entry in layers.values())
    units = sum(entry["units"] for entry in layers.values())
    return {
        "layers": layers,
        "total_units": units,
        "total_saturated_units": saturated,
        "total_constant_units": dead,
        "saturated_unit_fraction": _plain(saturated / units) if units else 0.0,
        "constant_unit_fraction": _plain(dead / units) if units else 0.0,
        "saturation_threshold": SATURATION_THRESHOLD,
    }


def _activation_statistics(activations: torch.Tensor) -> dict[str, Any]:
    units = int(activations.shape[-1])
    samples = int(activations.numel() // max(units, 1))
    if samples == 0:
        return {"samples": 0, "units": units, "mean": 0.0, "std": 0.0,
                "absolute_mean": 0.0, "saturated_fraction": 0.0,
                "saturated_units": 0, "constant_units": 0}
    flat = activations.reshape(samples, units)
    saturated = (flat.abs() > SATURATION_THRESHOLD).float()
    per_unit_std = flat.std(dim=0, unbiased=False)
    return {
        "samples": samples,
        "units": units,
        "mean": _plain(flat.mean()),
        "std": _plain(flat.std(unbiased=False)),
        "absolute_mean": _plain(flat.abs().mean()),
        "saturated_fraction": _plain(saturated.mean()),
        "saturated_units": int((saturated.mean(dim=0) > UNIT_SATURATION_FRACTION).sum()),
        # With a single occupied slot every unit is trivially "constant", so
        # the check is only meaningful once there are at least two samples.
        "constant_units": int((per_unit_std < CONSTANT_UNIT_STD).sum()) if samples > 1 else 0,
    }


# --------------------------------------------------------------------------
# 2. GRU memory usage
# --------------------------------------------------------------------------

def memory_report(policy: CombatPolicy, observations: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Roll a trajectory through the GRU and measure whether the hidden state
    carries anything. A collapsed GRU makes the policy memoryless, which is the
    difference between reacting and anticipating."""
    sequence = list(observations) if observations else approach_sequence()
    hidden = policy.initial_hidden(1, "cpu")
    trajectory: list[torch.Tensor] = []
    divergence: list[float] = []
    with torch.no_grad():
        for step in sequence:
            features = batch_observations([step])
            carried = policy(features, hidden)
            reset = policy(features, policy.initial_hidden(1, "cpu"))
            divergence.append(_probability_distance(carried, reset, features))
            hidden = carried.hidden
            trajectory.append(hidden.reshape(-1))
    states = torch.stack(trajectory)
    per_unit_std = states.std(dim=0, unbiased=False)
    return {
        "steps": len(sequence),
        "hidden_size": int(states.shape[1]),
        "mean_absolute_state": _plain(states.abs().mean()),
        "temporal_std_mean": _plain(per_unit_std.mean()),
        "temporal_std_max": _plain(per_unit_std.max()),
        "constant_unit_fraction": _plain((per_unit_std < CONSTANT_UNIT_STD).float().mean()),
        "saturated_unit_fraction": _plain((states.abs() > SATURATION_THRESHOLD).float().mean()),
        "effective_rank": _effective_rank(states),
        # How much the carried state changes the decision versus starting from
        # zeros on the same observation: 0 means the memory is decorative.
        "memory_influence": _plain(sum(divergence) / max(len(divergence), 1)),
    }


def _effective_rank(states: torch.Tensor) -> float:
    """Participation ratio of the PCA spectrum of the hidden trajectory:
    (sum eigenvalue)^2 / sum(eigenvalue^2). 1.0 means the state moves along a
    single direction; higher means the memory encodes more distinct things."""
    centred = states - states.mean(dim=0, keepdim=True)
    if centred.shape[0] < 2:
        return 0.0
    spectrum = torch.linalg.svdvals(centred).square()
    total = spectrum.sum()
    if float(total) <= 1e-12:
        return 0.0
    return _plain(total.square() / spectrum.square().sum())


def _probability_distance(first: Any, second: Any, features: FeatureBatch) -> float:
    masks = categorical_masks(features)
    distances = []
    for name in CATEGORICAL_SIZES:
        left = _masked_probabilities(first.logits[name], masks[name])
        right = _masked_probabilities(second.logits[name], masks[name])
        distances.append(float(0.5 * (left - right).abs().sum(dim=-1).mean()))
    return sum(distances) / len(distances)


# --------------------------------------------------------------------------
# 3. input saliency
# --------------------------------------------------------------------------

def saliency_report(policy: CombatPolicy, features: FeatureBatch) -> dict[str, Any]:
    """Input x gradient attribution, aggregated into feature groups and named
    sub-fields. Targets are the logit of the action each head currently
    prefers, plus the attack logit specifically and the value output."""
    inputs = FeatureBatch(**{
        name: (value.clone().detach().requires_grad_(True)
               if name in ("self_state", "opponent", "entities", "blocks", "legal")
               else value)
        for name, value in vars(features).items()
    })
    tensors = {
        "self": inputs.self_state, "opponent": inputs.opponent,
        "entities": inputs.entities, "blocks": inputs.blocks, "legal": inputs.legal,
    }
    slot_masks = {"entities": inputs.entity_mask, "blocks": inputs.block_mask}
    with torch.enable_grad():
        output = policy(inputs, policy.initial_hidden(inputs.self_state.shape[0], "cpu"))
        targets: dict[str, torch.Tensor] = {
            name: logits.gather(1, logits.argmax(dim=-1, keepdim=True)).sum()
            for name, logits in output.logits.items()
        }
        targets["primary:attack"] = output.logits["primary"][:, PRIMARY_NAMES.index("attack")].sum()
        targets["camera_yaw"] = output.camera_mean[:, 0].sum()
        targets["value"] = output.value.sum()
        result = {}
        names = list(tensors)
        for index, (label, target) in enumerate(targets.items()):
            gradients = torch.autograd.grad(
                target, [tensors[name] for name in names], retain_graph=index < len(targets) - 1
            )
            attribution = {name: (gradient * tensors[name]).detach()
                           for name, gradient in zip(names, gradients)}
            result[label] = _aggregate_attribution(attribution, slot_masks)
    return result


def _aggregate_attribution(
    attribution: dict[str, torch.Tensor],
    slot_masks: dict[str, torch.Tensor],
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    magnitudes: dict[str, float] = {}
    for group, values in attribution.items():
        mask = slot_masks.get(group)
        if mask is not None:
            values = values * mask.unsqueeze(-1)
            values = values.sum(dim=1)  # collapse slots; padded rows contribute zero
        batch = max(values.shape[0], 1)
        fields: dict[str, dict[str, float]] = {}
        for name, start, end in GROUP_FIELDS[group]:
            window = values[:, start:end]
            fields[name] = {
                "width": end - start,
                "attribution": _plain(window.sum() / batch),
                "magnitude": _plain(window.abs().sum() / batch),
            }
        magnitude = float(values.abs().sum() / batch)
        magnitudes[group] = magnitude
        groups[group] = {"magnitude": _plain(magnitude), "fields": fields}
    total = sum(magnitudes.values())
    for group, entry in groups.items():
        entry["share"] = _plain(magnitudes[group] / total) if total > 0 else 0.0
        for field in entry["fields"].values():
            field["share"] = _plain(field["magnitude"] / total) if total > 0 else 0.0
            # A 36-wide span such as self.hotbar_items wins on raw share simply
            # by being wide; the per-dimension share is the fair comparison.
            field["share_per_dimension"] = _plain(field["share"] / field["width"])
    return {
        "total_magnitude": _plain(total),
        "groups": groups,
        "top_fields": _top_fields(groups, "share"),
        "top_fields_per_dimension": _top_fields(groups, "share_per_dimension"),
    }


def _top_fields(groups: dict[str, Any], key: str, count: int = 8) -> list[dict[str, Any]]:
    entries = [
        {"field": f"{group}.{name}", "width": field["width"], "share": field["share"],
         "share_per_dimension": field["share_per_dimension"],
         "attribution": field["attribution"]}
        for group, entry in groups.items() for name, field in entry["fields"].items()
    ]
    entries.sort(key=lambda entry: entry[key], reverse=True)
    return entries[:count]


# --------------------------------------------------------------------------
# 4. per-head entropy and action distribution
# --------------------------------------------------------------------------

def head_report(policy: CombatPolicy, features: FeatureBatch) -> dict[str, Any]:
    """Entropy per head against its own legal-action maximum, so a head that has
    committed is distinguishable from one that is still uniform-random."""
    with torch.no_grad():
        output = policy(features, policy.initial_hidden(features.self_state.shape[0], "cpu"))
    masks = categorical_masks(features)
    heads: dict[str, Any] = {}
    for name in CATEGORICAL_SIZES:
        mask = masks[name]
        probabilities = _masked_probabilities(output.logits[name], mask)
        entropy = -(probabilities.clamp_min(1e-12).log() * probabilities).sum(dim=-1)
        legal = mask.sum(dim=-1).clamp_min(1.0).to(probabilities.dtype)
        maximum = legal.log()
        ratio = torch.where(maximum > 1e-9, entropy / maximum.clamp_min(1e-9),
                            torch.zeros_like(entropy))
        mean_probabilities = probabilities.mean(dim=0)
        heads[name] = {
            "entropy": _plain(entropy.mean()),
            "max_entropy": _plain(maximum.mean()),
            "entropy_ratio": _plain(ratio.mean()),
            "legal_choices": _plain(legal.mean()),
            "probabilities": {
                choice: _plain(mean_probabilities[index])
                for index, choice in enumerate(CHOICE_NAMES[name])
            },
            "top_choice": CHOICE_NAMES[name][int(mean_probabilities.argmax())],
        }
    ratios = [entry["entropy_ratio"] for entry in heads.values()]
    return {
        "heads": heads,
        "camera": _camera_report(output.camera_mean, output.camera_log_std),
        "mean_entropy_ratio": _plain(sum(ratios) / len(ratios)),
        "uniform_heads": [name for name, entry in heads.items() if entry["entropy_ratio"] > 0.995],
        # A head with one legal choice has zero entropy by construction, not by
        # collapse -- release_use is masked to "false" whenever nothing is held.
        "collapsed_heads": [name for name, entry in heads.items()
                            if entry["entropy_ratio"] < 0.05 and entry["legal_choices"] > 1.5],
    }


def _camera_report(mean: torch.Tensor, log_std: torch.Tensor) -> dict[str, Any]:
    standard_deviation = log_std[0].exp()
    gaussian_entropy = (0.5 * math.log(2 * math.pi * math.e) + log_std[0]).sum()
    scale = torch.tensor(CAMERA_SCALE, dtype=mean.dtype)
    # The action is tanh-squashed, so the differential entropy of what is
    # actually emitted needs the log-det of the squash; estimated on a fixed
    # sample so the number is reproducible run to run.
    generator = torch.Generator().manual_seed(0)
    noise = torch.randn((256, mean.shape[0], 2), generator=generator)
    latent = mean.unsqueeze(0) + noise * standard_deviation
    correction = torch.log(scale * (1 - torch.tanh(latent).square()) + 1e-6).sum(-1).mean()
    return {
        "log_std": [_plain(value) for value in log_std[0]],
        "std_radians": [_plain(value) for value in standard_deviation],
        "gaussian_entropy": _plain(gaussian_entropy),
        "squashed_entropy_estimate": _plain(gaussian_entropy + correction),
        "mean_yaw_degrees": _plain(torch.rad2deg(torch.tanh(mean[:, 0]) * scale[0]).mean()),
        "mean_pitch_degrees": _plain(torch.rad2deg(torch.tanh(mean[:, 1]) * scale[1]).mean()),
        "mean_absolute_yaw_degrees": _plain(
            torch.rad2deg(torch.tanh(mean[:, 0]) * scale[0]).abs().mean()),
    }


def _masked_probabilities(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.masked_fill(~mask, -1e9), dim=-1)


# --------------------------------------------------------------------------
# 5. value calibration
# --------------------------------------------------------------------------

def value_report(policy: CombatPolicy, features: FeatureBatch) -> dict[str, Any]:
    with torch.no_grad():
        output = policy(features, policy.initial_hidden(features.self_state.shape[0], "cpu"))
    values = output.value
    spread = float(values.std(unbiased=False))
    return {
        "samples": int(values.shape[0]),
        "mean": _plain(values.mean()),
        "std": _plain(spread),
        "min": _plain(values.min()),
        "max": _plain(values.max()),
        "range": _plain(values.max() - values.min()),
        # A value head that returns the same number for a healthy melee and a
        # dying agent has learned nothing about the state.
        "near_constant": bool(spread < 1e-3),
    }


# --------------------------------------------------------------------------
# 6. behavioural probes
# --------------------------------------------------------------------------

def probe_report(policy: CombatPolicy, scenarios: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Canned situations with obvious human-readable answers. Reported as
    probabilities rather than pass/fail, because an early-training policy is
    supposed to fail these and the trend over checkpoints is the signal."""
    cases = dict(scenarios or probe_observations())
    features = batch_observations(list(cases.values()))
    masks = categorical_masks(features)
    with torch.no_grad():
        output = policy(features, policy.initial_hidden(len(cases), "cpu"))
    probabilities = {name: _masked_probabilities(output.logits[name], masks[name])
                     for name in CATEGORICAL_SIZES}
    results: dict[str, Any] = {}
    for index, name in enumerate(cases):
        results[name] = {
            "value": _plain(output.value[index]),
            "heads": {
                head: {choice: _plain(probabilities[head][index, choice_index])
                       for choice_index, choice in enumerate(CHOICE_NAMES[head])}
                for head in CATEGORICAL_SIZES
            },
            "camera_yaw_degrees": _plain(
                torch.rad2deg(torch.tanh(output.camera_mean[index, 0]) * math.pi)),
        }

    def probability(scenario: str, head: str, choice: str) -> float:
        return results[scenario]["heads"][head][choice] if scenario in results else 0.0

    signals = {
        # Positive means the policy attacks more when the opponent is actually
        # hittable, which is the first thing a sword policy should learn.
        "attack_lift_in_reach": _plain(
            probability("opponent_in_reach", "primary", "attack")
            - probability("opponent_far", "primary", "attack")),
        "forward_lift_when_far": _plain(
            probability("opponent_far", "forward", "forward")
            - probability("opponent_in_reach", "forward", "forward")),
        "attack_lift_off_cooldown": _plain(
            probability("opponent_in_reach", "primary", "attack")
            - probability("cooldown_not_ready", "primary", "attack")),
        "crystal_slot_lift_on_spot": _plain(
            probability("crystal_spot_ready", "hotbar", "slot3_crystal")
            - probability("no_opponent", "hotbar", "slot3_crystal")),
        "attack_lift_on_live_crystal": _plain(
            probability("crystal_placed_live", "primary", "attack")
            - probability("crystal_spot_ready", "primary", "attack")),
        "totem_swap_lift_at_low_health": _plain(
            probability("low_health", "swap_offhand", "true")
            - probability("opponent_in_reach", "swap_offhand", "true")),
        # Hard invariant, not a learned behaviour: the action mask must make
        # attacking impossible when the server says it is illegal.
        "attack_probability_when_illegal": probability(
            "attack_illegal_in_reach", "primary", "attack"),
    }
    return {"scenarios": results, "signals": signals}


# --------------------------------------------------------------------------
# 7. checkpoint comparison
# --------------------------------------------------------------------------

def compare_policies(
    baseline: CombatPolicy,
    candidate: CombatPolicy,
    features: FeatureBatch,
    top: int = 10,
) -> dict[str, Any]:
    """Per-tensor weight movement plus the behavioural shift it produced."""
    left = baseline.state_dict()
    right = candidate.state_dict()
    tensors: list[dict[str, Any]] = []
    total_squared = 0.0
    baseline_squared = 0.0
    for name, value in left.items():
        other = right[name].to(torch.float32)
        value = value.to(torch.float32)
        delta = float(torch.linalg.vector_norm(other - value))
        norm = float(torch.linalg.vector_norm(value))
        total_squared += delta * delta
        baseline_squared += norm * norm
        tensors.append({
            "tensor": name, "l2_delta": _plain(delta), "baseline_l2": _plain(norm),
            "relative_delta": _plain(delta / norm) if norm > 1e-12 else 0.0,
        })
    tensors.sort(key=lambda entry: entry["relative_delta"], reverse=True)
    before = head_report(baseline, features)
    after = head_report(candidate, features)
    head_shift = {}
    for name in CATEGORICAL_SIZES:
        left_probabilities = before["heads"][name]["probabilities"]
        right_probabilities = after["heads"][name]["probabilities"]
        head_shift[name] = {
            "entropy_before": before["heads"][name]["entropy"],
            "entropy_after": after["heads"][name]["entropy"],
            "entropy_delta": _plain(
                after["heads"][name]["entropy"] - before["heads"][name]["entropy"]),
            "total_variation": _plain(0.5 * sum(
                abs(right_probabilities[choice] - left_probabilities[choice])
                for choice in left_probabilities)),
            "probability_delta": {
                choice: _plain(right_probabilities[choice] - left_probabilities[choice])
                for choice in left_probabilities
            },
        }
    return {
        "total_l2_delta": _plain(math.sqrt(total_squared)),
        "baseline_l2_norm": _plain(math.sqrt(baseline_squared)),
        "relative_l2_delta": _plain(
            math.sqrt(total_squared) / math.sqrt(baseline_squared)) if baseline_squared > 0 else 0.0,
        "most_changed_tensors": tensors[:top],
        "unchanged_tensors": [entry["tensor"] for entry in tensors if entry["l2_delta"] == 0.0],
        "head_shift": head_shift,
        "value_before": value_report(baseline, features),
        "value_after": value_report(candidate, features),
    }


# --------------------------------------------------------------------------
# top-level report
# --------------------------------------------------------------------------

def introspect(
    checkpoint: Path | str | None = None,
    compare: Path | str | None = None,
    observations: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    policy, metadata = load_checkpoint(checkpoint)
    batch = list(observations) if observations is not None else default_observations()
    features = batch_observations(batch)
    report: dict[str, Any] = {
        "checkpoint": metadata,
        "parameter_count": policy.parameter_count,
        "batch_size": len(batch),
        "activations": activation_report(policy, features),
        "memory": memory_report(policy),
        "saliency": saliency_report(policy, features),
        "policy_heads": head_report(policy, features),
        "value": value_report(policy, features),
        "probes": probe_report(policy),
    }
    report["diagnosis"] = diagnose(report)
    if compare is not None:
        other, other_metadata = load_checkpoint(compare)
        report["comparison"] = {
            "baseline": other_metadata,
            **compare_policies(other, policy, features),
        }
    return report


def diagnose(report: dict[str, Any]) -> list[dict[str, str]]:
    """Turn the raw numbers into named, actionable failure findings."""
    findings: list[dict[str, str]] = []

    def add(severity: str, code: str, detail: str) -> None:
        findings.append({"severity": severity, "code": code, "detail": detail})

    activations = report["activations"]
    if activations["saturated_unit_fraction"] > 0.5:
        add("error", "activations_saturated",
            f"{activations['saturated_unit_fraction']:.0%} of tanh units are pinned past "
            f"{SATURATION_THRESHOLD}; those units are gradient-dead.")
    elif activations["saturated_unit_fraction"] > 0.2:
        add("warning", "activations_saturating",
            f"{activations['saturated_unit_fraction']:.0%} of tanh units are pinned; "
            "watch for a rising trend across checkpoints.")
    if activations["constant_unit_fraction"] > 0.3:
        add("warning", "activations_constant",
            f"{activations['constant_unit_fraction']:.0%} of units do not vary across the "
            "probe batch and carry no information.")

    memory = report["memory"]
    # A random GRU driven by a smooth approach trajectory lands near 1.2, so
    # only a spectrum that is essentially one direction is a hard failure.
    if memory["effective_rank"] < 1.05:
        add("error", "memory_collapsed",
            f"GRU hidden trajectory has effective rank {memory['effective_rank']:.2f}; "
            "the hidden state moves along a single direction and encodes nothing.")
    elif memory["effective_rank"] < 2.0:
        add("info", "memory_low_rank",
            f"GRU trajectory effective rank is {memory['effective_rank']:.2f}; a fresh "
            "policy sits near 1.2, so this should climb as the memory learns to encode more.")
    if memory["memory_influence"] < 1e-3:
        add("warning", "memory_ignored",
            "Resetting the hidden state barely changes the action distribution "
            f"(mean total variation {memory['memory_influence']:.2e}).")

    if report["value"]["near_constant"]:
        add("error", "value_constant",
            f"Value head outputs a near-constant {report['value']['mean']:.4f}; "
            "it has not learned to distinguish states.")

    heads = report["policy_heads"]
    if heads["uniform_heads"]:
        add("info", "heads_uniform",
            "Heads still at maximum entropy: " + ", ".join(heads["uniform_heads"]))
    if heads["collapsed_heads"]:
        add("warning", "heads_collapsed",
            "Heads with near-zero entropy (no exploration left): "
            + ", ".join(heads["collapsed_heads"]))

    illegal = report["probes"]["signals"]["attack_probability_when_illegal"]
    if illegal > 1e-6:
        add("error", "illegal_action_leak",
            f"Attack retains probability {illegal:.3e} while the action mask forbids it.")

    if not findings:
        add("info", "healthy", "No structural failure detected in this checkpoint.")
    return findings


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def summarize(report: dict[str, Any]) -> str:
    lines: list[str] = []
    metadata = report["checkpoint"]
    lines.append("=" * 72)
    lines.append(f"MCAI policy introspection -- {metadata.get('source', 'unknown')}")
    detail = [f"{key}={metadata[key]}" for key in ("policy_version", "total_agent_ticks")
              if key in metadata]
    detail.append(f"parameters={report['parameter_count']:,}")
    detail.append(f"probe_batch={report['batch_size']}")
    lines.append("  " + "  ".join(detail))
    lines.append("=" * 72)

    lines.append("")
    lines.append("DIAGNOSIS")
    for finding in report["diagnosis"]:
        lines.append(f"  [{finding['severity'].upper():<7}] {finding['code']}: {finding['detail']}")

    activations = report["activations"]
    lines.append("")
    lines.append("1. LAYER ACTIVATION HEALTH  (tanh; |a| > 0.99 is gradient-dead)")
    lines.append(f"  {'layer':<26}{'mean':>9}{'std':>9}{'|a|>.99':>9}{'sat':>7}{'const':>7}{'units':>7}")
    for name, entry in activations["layers"].items():
        lines.append(
            f"  {name:<26}{entry['mean']:>9.3f}{entry['std']:>9.3f}"
            f"{entry['saturated_fraction']:>9.3f}{entry['saturated_units']:>7}"
            f"{entry['constant_units']:>7}{entry['units']:>7}")
    lines.append(f"  totals: {activations['total_saturated_units']} saturated / "
                 f"{activations['total_constant_units']} constant of "
                 f"{activations['total_units']} units")

    memory = report["memory"]
    lines.append("")
    lines.append(f"2. GRU MEMORY  ({memory['steps']}-step approach trajectory, "
                 f"{memory['hidden_size']} units)")
    lines.append(f"  temporal std (mean/max) : {memory['temporal_std_mean']:.4f} / "
                 f"{memory['temporal_std_max']:.4f}")
    lines.append(f"  effectively constant    : {memory['constant_unit_fraction']:.1%} of units")
    lines.append(f"  effective rank (PCA PR) : {memory['effective_rank']:.2f} of "
                 f"{memory['hidden_size']}")
    lines.append(f"  memory influence        : {memory['memory_influence']:.4f} "
                 "(mean TV distance vs a reset hidden state)")

    lines.append("")
    lines.append("3. INPUT SALIENCY  (input x gradient, share of total attribution magnitude)")
    for label, entry in report["saliency"].items():
        shares = "  ".join(f"{group}={entry['groups'][group]['share']:.2f}"
                           for group in GROUP_FIELDS)
        lines.append(f"  {label:<16} {shares}")
        top = "  ".join(f"{field['field']}={field['share']:.2f}"
                        for field in entry["top_fields"][:4])
        lines.append(f"  {'':<16} top: {top}")
        per_dimension = "  ".join(f"{field['field']}={field['share_per_dimension']:.3f}"
                                  for field in entry["top_fields_per_dimension"][:4])
        lines.append(f"  {'':<16} per-dim: {per_dimension}")

    heads = report["policy_heads"]
    lines.append("")
    lines.append("4. HEAD ENTROPY AND ACTION DISTRIBUTION")
    lines.append(f"  {'head':<14}{'entropy':>9}{'max':>8}{'ratio':>8}  top choice")
    for name, entry in heads["heads"].items():
        lines.append(
            f"  {name:<14}{entry['entropy']:>9.3f}{entry['max_entropy']:>8.3f}"
            f"{entry['entropy_ratio']:>8.3f}  {entry['top_choice']} "
            f"({entry['probabilities'][entry['top_choice']]:.2f})")
    camera = heads["camera"]
    lines.append(f"  camera std (rad)  : yaw={camera['std_radians'][0]:.3f} "
                 f"pitch={camera['std_radians'][1]:.3f}")
    lines.append(f"  camera entropy    : gaussian={camera['gaussian_entropy']:.3f} "
                 f"squashed~{camera['squashed_entropy_estimate']:.3f}")
    lines.append(f"  mean |yaw| turn   : {camera['mean_absolute_yaw_degrees']:.1f} deg")

    value = report["value"]
    lines.append("")
    lines.append("5. VALUE FUNCTION")
    lines.append(f"  mean={value['mean']:.4f}  std={value['std']:.4f}  "
                 f"range=[{value['min']:.4f}, {value['max']:.4f}]"
                 + ("  NEAR-CONSTANT" if value["near_constant"] else ""))

    probes = report["probes"]
    lines.append("")
    lines.append("6. BEHAVIOURAL PROBES  (probabilities, not pass/fail)")
    lines.append(f"  {'scenario':<24}{'P(attack)':>10}{'P(fwd)':>9}{'P(use)':>9}"
                 f"{'P(crystal)':>11}{'value':>9}")
    for name, entry in probes["scenarios"].items():
        lines.append(
            f"  {name:<24}{entry['heads']['primary']['attack']:>10.3f}"
            f"{entry['heads']['forward']['forward']:>9.3f}"
            f"{entry['heads']['primary']['use_main']:>9.3f}"
            f"{entry['heads']['hotbar']['slot3_crystal']:>11.3f}{entry['value']:>9.3f}")
    lines.append("  signals (positive = the sensible direction):")
    for name, value_ in probes["signals"].items():
        lines.append(f"    {name:<34}{value_:+.4f}")

    if "comparison" in report:
        comparison = report["comparison"]
        lines.append("")
        lines.append(f"7. COMPARISON vs {comparison['baseline'].get('source', 'baseline')}")
        lines.append(f"  total weight L2 delta   : {comparison['total_l2_delta']:.4f} "
                     f"({comparison['relative_l2_delta']:.2%} of baseline norm)")
        if comparison["unchanged_tensors"]:
            lines.append(f"  UNCHANGED tensors       : "
                         f"{', '.join(comparison['unchanged_tensors'][:6])}")
        lines.append("  most changed tensors:")
        for entry in comparison["most_changed_tensors"][:6]:
            lines.append(f"    {entry['tensor']:<40}{entry['relative_delta']:>9.4f} rel "
                         f"{entry['l2_delta']:>9.4f} abs")
        lines.append("  head shift:")
        for name, entry in comparison["head_shift"].items():
            lines.append(f"    {name:<14}entropy {entry['entropy_before']:.3f} -> "
                         f"{entry['entropy_after']:.3f} "
                         f"({entry['entropy_delta']:+.3f})  TV={entry['total_variation']:.3f}")
    lines.append("")
    return "\n".join(lines)


def _plain(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        value = value.item()
    number = float(value)
    return number if math.isfinite(number) else 0.0


def run_cli(checkpoint: Path | None, compare: Path | None, as_json: bool) -> str:
    report = introspect(checkpoint, compare)
    return json.dumps(report, indent=2) if as_json else summarize(report)
