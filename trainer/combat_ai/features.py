from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch

SCHEMA_VERSION = 1
MAX_ENTITIES = 16
MAX_BLOCKS = 48
SELF_SIZE = 80
OPPONENT_SIZE = 48
ENTITY_SIZE = 18
BLOCK_SIZE = 20
LEGAL_SIZE = 24

PRIMARY_NAMES = ("none", "attack", "use_main", "use_offhand")


@dataclass
class FeatureBatch:
    self_state: torch.Tensor
    opponent: torch.Tensor
    opponent_mask: torch.Tensor
    entities: torch.Tensor
    entity_mask: torch.Tensor
    blocks: torch.Tensor
    block_mask: torch.Tensor
    legal: torch.Tensor

    def to(self, device: torch.device | str) -> "FeatureBatch":
        return FeatureBatch(**{name: value.to(device) for name, value in vars(self).items()})

    def index(self, indices: torch.Tensor) -> "FeatureBatch":
        return FeatureBatch(**{name: value[indices] for name, value in vars(self).items()})

    def as_tuple(self) -> tuple[torch.Tensor, ...]:
        return tuple(vars(self).values())


def encode_observation(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    if int(observation.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported Observation schema_version")
    result = {
        "self_state": np.zeros(SELF_SIZE, dtype=np.float32),
        "opponent": np.zeros(OPPONENT_SIZE, dtype=np.float32),
        "opponent_mask": np.zeros(1, dtype=np.float32),
        "entities": np.zeros((MAX_ENTITIES, ENTITY_SIZE), dtype=np.float32),
        "entity_mask": np.zeros(MAX_ENTITIES, dtype=np.float32),
        "blocks": np.zeros((MAX_BLOCKS, BLOCK_SIZE), dtype=np.float32),
        "block_mask": np.zeros(MAX_BLOCKS, dtype=np.float32),
        "legal": np.zeros(LEGAL_SIZE, dtype=np.float32),
    }
    _encode_self(observation.get("self", {}), result["self_state"])
    opponent = observation.get("opponent")
    if isinstance(opponent, dict):
        result["opponent_mask"][0] = 1.0
        _encode_opponent(opponent, result["opponent"])
    for index, entity in enumerate(observation.get("entities", [])[:MAX_ENTITIES]):
        result["entity_mask"][index] = 1.0
        _encode_entity(entity, result["entities"][index])
    for index, block in enumerate(observation.get("blocks", [])[:MAX_BLOCKS]):
        result["block_mask"][index] = 1.0
        _encode_block(block, result["blocks"][index])
    _encode_legal(observation.get("action_mask", {}), result["legal"])
    return result


def batch_observations(observations: Iterable[dict[str, Any]], device: torch.device | str = "cpu") -> FeatureBatch:
    encoded = [encode_observation(observation) for observation in observations]
    if not encoded:
        raise ValueError("cannot batch zero observations")
    return FeatureBatch(**{
        key: torch.from_numpy(np.stack([entry[key] for entry in encoded])).to(device)
        for key in encoded[0]
    })


def _encode_self(state: dict[str, Any], out: np.ndarray) -> None:
    values: list[float] = [
        _scale(state.get("health"), 20), _scale(state.get("absorption"), 20),
        _scale(state.get("food"), 20), *_vec(state.get("velocity"), 2),
        math.sin(_number(state.get("yaw"))), math.cos(_number(state.get("yaw"))),
        _scale(state.get("pitch"), math.pi / 2), _bool(state.get("on_ground")),
        _bool(state.get("sprinting")), _bool(state.get("sneaking")),
        _scale(state.get("hurt_time"), 10), _number(state.get("attack_cooldown")),
        _scale(state.get("use_ticks"), 32), _number(state.get("mining_progress")),
        _scale(state.get("selected_hotbar"), 8),
    ]
    active = state.get("active_hand", "none")
    values.extend(float(active == item) for item in ("none", "main", "off"))
    raycast = state.get("raycast") or {}
    values.extend(float(raycast.get("kind") == item) for item in ("none", "block", "entity"))
    values.append(_scale(raycast.get("distance"), 6))
    values.extend(_item(state.get("offhand")))
    for item in (state.get("armor") or [])[:4]:
        values.extend(_item(item)[:3])
    hotbar = list(state.get("hotbar") or [])
    for index in range(9):
        values.extend(_item(hotbar[index] if index < len(hotbar) else None)[:4])
    _write(out, values)


def _encode_opponent(state: dict[str, Any], out: np.ndarray) -> None:
    health = state.get("health")
    values = [
        *_vec(state.get("relative_position"), 12), *_vec(state.get("relative_velocity"), 2),
        math.sin(_number(state.get("yaw"))), math.cos(_number(state.get("yaw"))),
        _scale(state.get("pitch"), math.pi / 2),
        _scale(health, 20) if health is not None else 0.0, float(health is not None),
        _scale(state.get("hurt_time"), 10), _bool(state.get("on_ground")),
        _bool(state.get("line_of_sight")),
        *_item(state.get("mainhand")), *_item(state.get("offhand")),
    ]
    for item in (state.get("armor") or [])[:4]:
        values.extend(_item(item)[:3])
    _write(out, values)


def _encode_entity(state: dict[str, Any], out: np.ndarray) -> None:
    kind = str(state.get("kind", ""))
    values = [
        *_one_hot_kind(kind, ("end_crystal", "arrow", "snowball", "egg", "fireball")),
        *_vec(state.get("relative_position"), 12), *_vec(state.get("relative_velocity"), 2),
        _scale(state.get("age_ticks"), 200), _scale(state.get("distance"), 12),
        _bool(state.get("raycastable")), _hash_feature(kind),
    ]
    _write(out, values)


def _encode_block(state: dict[str, Any], out: np.ndarray) -> None:
    collision = str(state.get("collision", "empty"))
    name = str(state.get("name", ""))
    values = [
        *_vec(state.get("relative_position"), 6),
        *(float(collision == item) for item in ("empty", "solid", "liquid", "partial")),
        _scale(state.get("hardness"), 50), _bool(state.get("replaceable")),
        _number(state.get("break_progress")), _bool(state.get("crystal_clearance")),
        _scale(state.get("exposed_faces"), 6), _scale(state.get("distance"), 8),
        _bool(state.get("within_reach")), _bool(state.get("raycastable")),
        _scale(state.get("sample_age_ticks"), 10),
        float("obsidian" in name), float(name in {"bedrock", "obsidian"}), _hash_feature(name),
    ]
    _write(out, values)


def _encode_legal(mask: dict[str, Any], out: np.ndarray) -> None:
    hotbar = list(mask.get("hotbar") or [])
    values = [
        1.0, _bool(mask.get("attack")), _bool(mask.get("use_main")),
        _bool(mask.get("use_offhand")), 1.0, _bool(mask.get("release_use")),
        1.0, _bool(mask.get("swap_offhand")),
    ]
    values.extend(_bool(hotbar[index]) if index < len(hotbar) else 0.0 for index in range(9))
    values.extend([1.0] * 7)
    _write(out, values)


def categorical_masks(batch: FeatureBatch) -> dict[str, torch.Tensor]:
    legal = batch.legal > 0.5
    count = legal.shape[0]
    all_two = torch.ones((count, 2), dtype=torch.bool, device=legal.device)
    all_three = torch.ones((count, 3), dtype=torch.bool, device=legal.device)
    return {
        "forward": all_three,
        "strafe": all_three,
        "jump": all_two,
        "sprint": all_two,
        "sneak": all_two,
        "primary": legal[:, 0:4],
        "release_use": torch.stack((torch.ones(count, dtype=torch.bool, device=legal.device), legal[:, 5]), dim=1),
        "hotbar": torch.cat((torch.ones((count, 1), dtype=torch.bool, device=legal.device), legal[:, 8:17]), dim=1),
        "swap_offhand": torch.stack((torch.ones(count, dtype=torch.bool, device=legal.device), legal[:, 7]), dim=1),
    }


def _item(item: Any) -> list[float]:
    if not isinstance(item, dict):
        return [0.0] * 6
    name = str(item.get("name", ""))
    maximum = max(_number(item.get("max_durability")), 1.0)
    return [
        float(bool(name)), _scale(item.get("count"), 64), _number(item.get("durability")) / maximum,
        _scale(item.get("max_durability"), 2000), _hash_feature(name),
        (_number(item.get("enchant_hash")) % 104729) / 104729.0,
    ]


def _vec(value: Any, scale: float) -> list[float]:
    value = value if isinstance(value, dict) else {}
    return [_scale(value.get(axis), scale) for axis in ("x", "y", "z")]


def _one_hot_kind(value: str, names: tuple[str, ...]) -> list[float]:
    lowered = value.lower()
    return [float(name in lowered) for name in names]


def _hash_feature(value: str) -> float:
    result = 2166136261
    for byte in value.encode("utf-8"):
        result = ((result ^ byte) * 16777619) & 0xFFFFFFFF
    return (result / 0xFFFFFFFF) * 2.0 - 1.0


def _number(value: Any) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _scale(value: Any, scale: float) -> float:
    return max(-4.0, min(4.0, _number(value) / scale))


def _bool(value: Any) -> float:
    return float(bool(value))


def _write(out: np.ndarray, values: list[float]) -> None:
    length = min(len(out), len(values))
    out[:length] = np.asarray(values[:length], dtype=np.float32)
