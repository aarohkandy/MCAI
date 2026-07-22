from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch

SCHEMA_VERSION = 1
SCHEMA_VERSION_V2 = 2
MAX_ENTITIES = 16
MAX_BLOCKS = 48
SELF_SIZE = 80
OPPONENT_SIZE = 48
ENTITY_SIZE = 18
BLOCK_SIZE = 20
LEGAL_SIZE = 24
MAX_TACTICAL_CANDIDATES = 16
MAX_RECENT_HISTORY = 8
TACTICAL_CANDIDATE_SIZE = 16
TACTICAL_DISTANCE_INDEX = 0
TACTICAL_REACHABLE_INDEX = 1
TACTICAL_VISIBLE_INDEX = 2
TACTICAL_LEGAL_INDEX = 3
TACTICAL_OPPONENT_DAMAGE_INDEX = 4
TACTICAL_SELF_DAMAGE_INDEX = 5
TACTICAL_POP_POTENTIAL_INDEX = 6
TACTICAL_ESCAPE_DIRECTION_INDEX = 7
TACTICAL_BODY_X_INDEX = 8
TACTICAL_BODY_Y_INDEX = 9
TACTICAL_BODY_Z_INDEX = 10
TACTICAL_SOURCE_INDEX = 11
# Signed so a present crystal is distinguishable both from a base and from
# zero-filled padding: +1 is a base, -1 is an existing crystal.
TACTICAL_CRYSTAL_KIND_INDEX = 12
TACTICAL_COVER_VALUE_INDEX = 13
TACTICAL_FOLLOWUP_INDEX = 14
TACTICAL_PURPOSE_INDEX = 15
HISTORY_SIZE = 12
SURVIVAL_SIZE = 12
THREAT_SIZE = 12

# These slots were padding in every checkpoint through v261.  Giving them
# structured crystal information keeps every tensor/model shape stable while
# making the pooled entity/block encoders retain the identity of the selected
# target instead of losing it in a mean/max over all samples.
SELF_CRYSTAL_CAPABLE_INDEX = 78
SELF_CRYSTAL_RETENTION_INDEX = 79
ENTITY_CRYSTAL_TARGET_INDEX = 15
ENTITY_BODY_X_INDEX = 16
ENTITY_BODY_Z_INDEX = 17
BLOCK_BODY_BEARING_INDEX = 18
BLOCK_CRYSTAL_TARGET_INDEX = 19
# The existing block encoding already distinguishes a usable crystal base at
# slot 17.  A marked base (17=1, 19=1) is a crystal-placement target; a marked
# non-base support (17=0, 19=1) is a worker-verified tactical obsidian target.
# Reusing those two old padding-derived features preserves every model shape.
BLOCK_CRYSTAL_BASE_INDEX = 17
BLOCK_RAYCASTABLE_INDEX = 14

# Opponent slots 38-47 were zero padding in every existing checkpoint.  They
# now carry the geometry a PvP policy otherwise has to rediscover through a
# legacy yaw transform and several nonlinear layers.
OPPONENT_DISTANCE_INDEX = 38
OPPONENT_HORIZONTAL_DISTANCE_INDEX = 39
OPPONENT_BEARING_SIN_INDEX = 40
OPPONENT_BEARING_COS_INDEX = 41
OPPONENT_PITCH_ERROR_INDEX = 42
OPPONENT_CLOSING_SPEED_INDEX = 43
OPPONENT_MELEE_RANGE_INDEX = 44
OPPONENT_AIM_ALIGNMENT_INDEX = 45
OPPONENT_FACING_SELF_INDEX = 46
OPPONENT_HELD_ITEM_CLASS_INDEX = 47

# The self vector has four scalars per hotbar slot starting at 42.  Preserve
# presence/count/durability and use the former max-durability proxy for a
# stable combat category.  This makes same-durability tools (notably diamond
# swords and pickaxes) distinguishable without changing checkpoint shapes.
SELF_HOTBAR_START_INDEX = 42
HOTBAR_ITEM_FEATURE_SIZE = 4
HOTBAR_SEMANTIC_OFFSET = 3
HOTBAR_SEMANTIC_INDICES = tuple(
    SELF_HOTBAR_START_INDEX + index * HOTBAR_ITEM_FEATURE_SIZE + HOTBAR_SEMANTIC_OFFSET
    for index in range(9)
)
SELF_ARMOR_START_INDEX = 30
OPPONENT_ARMOR_START_INDEX = 26
ARMOR_ITEM_FEATURE_SIZE = 3
ARMOR_SEMANTIC_OFFSET = 1

PVP_ITEM_EMPTY = 0.0
PVP_ITEM_SWORD = 1.0
PVP_ITEM_PICKAXE = 0.8
PVP_ITEM_CRYSTAL = 0.6
PVP_ITEM_OBSIDIAN = 0.4
PVP_ITEM_FOOD = 0.2
PVP_ITEM_TOTEM = -0.2
PVP_ITEM_BLOCK = -0.4
PVP_ITEM_PROJECTILE = -0.6
PVP_ITEM_OTHER = -1.0

LEGAL_ACTION_DELAY_INDEX = 21
LEGAL_OBSERVATION_DELAY_INDEX = 22
LEGAL_TERRAIN_MODE_INDEX = 23

# A crystal can be placed farther away than it can be attacked.  Acquiring a
# base outside this radius commonly produces an orphan crystal, so the trainer
# only advertises bases that leave the follow-up attack inside normal reach.
CRYSTAL_CHAIN_REACH = 3.4
CRYSTAL_EYE_HEIGHT = 1.62
MAX_POLICY_CRYSTAL_TARGET_PITCH = math.pi / 4

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
    crystal_candidates: torch.Tensor
    crystal_candidate_mask: torch.Tensor
    tactical_blocks: torch.Tensor
    tactical_block_mask: torch.Tensor
    recent_history: torch.Tensor
    recent_history_mask: torch.Tensor
    survival: torch.Tensor
    threat: torch.Tensor

    def to(self, device: torch.device | str) -> "FeatureBatch":
        return FeatureBatch(**{name: value.to(device) for name, value in vars(self).items()})

    def index(self, indices: torch.Tensor) -> "FeatureBatch":
        return FeatureBatch(**{name: value[indices] for name, value in vars(self).items()})

    def as_tuple(self) -> tuple[torch.Tensor, ...]:
        return tuple(vars(self).values())


def encode_observation(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    if int(observation.get("schema_version", -1)) not in (SCHEMA_VERSION, SCHEMA_VERSION_V2):
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
        "crystal_candidates": np.zeros((MAX_TACTICAL_CANDIDATES, TACTICAL_CANDIDATE_SIZE), dtype=np.float32),
        "crystal_candidate_mask": np.zeros(MAX_TACTICAL_CANDIDATES, dtype=np.float32),
        "tactical_blocks": np.zeros((MAX_TACTICAL_CANDIDATES, TACTICAL_CANDIDATE_SIZE), dtype=np.float32),
        "tactical_block_mask": np.zeros(MAX_TACTICAL_CANDIDATES, dtype=np.float32),
        "recent_history": np.zeros((MAX_RECENT_HISTORY, HISTORY_SIZE), dtype=np.float32),
        "recent_history_mask": np.zeros(MAX_RECENT_HISTORY, dtype=np.float32),
        "survival": np.zeros(SURVIVAL_SIZE, dtype=np.float32),
        "threat": np.zeros(THREAT_SIZE, dtype=np.float32),
    }
    self_state = observation.get("self", {})
    _encode_self(self_state, result["self_state"])
    crystal_capable, has_crystal = _encode_crystal_context(
        observation, self_state, result["self_state"]
    )
    opponent = observation.get("opponent")
    if isinstance(opponent, dict):
        result["opponent_mask"][0] = 1.0
        _encode_opponent(opponent, self_state, result["opponent"])
    entities = list(observation.get("entities", [])[:MAX_ENTITIES])
    crystal_entity_target = _select_crystal_entity(entities) if crystal_capable else None
    for index, entity in enumerate(entities):
        result["entity_mask"][index] = 1.0
        _encode_entity(
            entity, self_state, result["entities"][index], index == crystal_entity_target
        )
    blocks = list(observation.get("blocks", [])[:MAX_BLOCKS])
    crystal_block_target = (
        _select_crystal_base(blocks, opponent, self_state)
        if crystal_capable and has_crystal else None
    )
    tactical_placement_target = _select_tactical_placement_target(blocks)
    for index, block in enumerate(blocks):
        result["block_mask"][index] = 1.0
        _encode_block(
            block,
            self_state,
            result["blocks"][index],
            crystal_target=index == crystal_block_target,
            tactical_target=index == tactical_placement_target,
        )
    _encode_legal(
        observation.get("action_mask", {}), observation.get("match", {}), result["legal"]
    )
    _encode_tactical(observation.get("tactical"), result)
    return result


def _encode_tactical(value: Any, result: dict[str, np.ndarray]) -> None:
    """Encode ObservationV2's ranked candidates without depending on JS key order."""
    tactical = value if isinstance(value, dict) else {}
    for index, candidate in enumerate(
        (tactical.get("crystal_candidates") or [])[:MAX_TACTICAL_CANDIDATES]
    ):
        if not isinstance(candidate, dict):
            continue
        result["crystal_candidate_mask"][index] = 1.0
        result["crystal_candidates"][index] = _crystal_candidate_vector(candidate)
    for index, candidate in enumerate(
        (tactical.get("block_candidates") or [])[:MAX_TACTICAL_CANDIDATES]
    ):
        if not isinstance(candidate, dict):
            continue
        result["tactical_block_mask"][index] = 1.0
        result["tactical_blocks"][index] = _block_candidate_vector(candidate)
    for index, event in enumerate((tactical.get("recent_history") or [])[-MAX_RECENT_HISTORY:]):
        if isinstance(event, dict):
            result["recent_history_mask"][index] = 1.0
            result["recent_history"][index] = _tactical_vector(event, HISTORY_SIZE)
    result["survival"] = _tactical_vector(tactical.get("survival"), SURVIVAL_SIZE)
    result["threat"] = _tactical_vector(tactical.get("threat"), THREAT_SIZE)


def _tactical_vector(value: Any, size: int) -> np.ndarray:
    out = np.zeros(size, dtype=np.float32)
    if not isinstance(value, dict):
        return out
    preferred = (
        "distance", "reach", "visible", "legal", "opponent_damage", "self_damage",
        "pop_probability", "escape_x", "escape_y", "escape_z", "closing_speed",
        "cover_value", "follow_up_viability", "attack_cooldown", "item_class", "age_ticks",
    )
    for index, key in enumerate(preferred[:size]):
        raw = value.get(key, 0.0)
        if isinstance(raw, bool):
            out[index] = float(raw)
        elif isinstance(raw, (int, float)) and math.isfinite(float(raw)):
            out[index] = max(-10.0, min(10.0, float(raw)))
    return out


def _crystal_candidate_vector(value: dict[str, Any]) -> np.ndarray:
    out = np.zeros(TACTICAL_CANDIDATE_SIZE, dtype=np.float32)
    position = value.get("body_relative_position")
    position = position if isinstance(position, dict) else {}
    kind = str(value.get("kind", "")).lower()
    out[TACTICAL_DISTANCE_INDEX] = _candidate_number(value, "distance")
    out[TACTICAL_REACHABLE_INDEX] = _candidate_bool(value, "reachable", "reach")
    out[TACTICAL_VISIBLE_INDEX] = _candidate_bool(value, "visible", "line_of_sight")
    out[TACTICAL_LEGAL_INDEX] = _candidate_bool(value, "placement_legal", "legal")
    out[TACTICAL_OPPONENT_DAMAGE_INDEX] = _candidate_number(
        value, "estimated_opponent_damage", "opponent_damage"
    )
    out[TACTICAL_SELF_DAMAGE_INDEX] = _candidate_number(
        value, "estimated_self_damage", "self_damage"
    )
    out[TACTICAL_POP_POTENTIAL_INDEX] = _candidate_number(
        value, "pop_potential", "pop_probability"
    )
    out[TACTICAL_ESCAPE_DIRECTION_INDEX] = _candidate_number(
        value, "escape_direction", "escape_x"
    )
    out[TACTICAL_BODY_X_INDEX] = _candidate_number(position, "x")
    out[TACTICAL_BODY_Y_INDEX] = _candidate_number(position, "y")
    out[TACTICAL_BODY_Z_INDEX] = _candidate_number(position, "z")
    out[TACTICAL_SOURCE_INDEX] = _candidate_number(value, "source_index")
    out[TACTICAL_CRYSTAL_KIND_INDEX] = 1.0 if kind == "base" else -1.0 if kind == "crystal" else 0.0
    return out


def _block_candidate_vector(value: dict[str, Any]) -> np.ndarray:
    out = np.zeros(TACTICAL_CANDIDATE_SIZE, dtype=np.float32)
    position = value.get("body_relative_position")
    position = position if isinstance(position, dict) else {}
    purpose = str(value.get("purpose", "")).lower()
    purpose_value = {
        "crystal_base": 1.0, "cover": 0.5, "mine_path": -0.5, "high_ground": -1.0,
    }.get(purpose, 0.0)
    out[TACTICAL_DISTANCE_INDEX] = _candidate_number(value, "distance")
    out[TACTICAL_REACHABLE_INDEX] = _candidate_bool(value, "reachable", "reach")
    out[TACTICAL_VISIBLE_INDEX] = _candidate_bool(value, "visible", "line_of_sight")
    out[TACTICAL_BODY_X_INDEX] = _candidate_number(position, "x")
    out[TACTICAL_BODY_Y_INDEX] = _candidate_number(position, "y")
    out[TACTICAL_BODY_Z_INDEX] = _candidate_number(position, "z")
    out[TACTICAL_SOURCE_INDEX] = _candidate_number(value, "source_index")
    out[TACTICAL_COVER_VALUE_INDEX] = _candidate_number(value, "cover_value")
    out[TACTICAL_FOLLOWUP_INDEX] = _candidate_number(
        value, "followup_crystal_viability", "follow_up_viability"
    )
    out[TACTICAL_PURPOSE_INDEX] = purpose_value
    return out


def _candidate_number(value: dict[str, Any], *names: str) -> float:
    for name in names:
        raw = value.get(name)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(float(raw)):
            return max(-10.0, min(10.0, float(raw)))
    return 0.0


def _candidate_bool(value: dict[str, Any], *names: str) -> float:
    for name in names:
        if name in value:
            return float(bool(value.get(name)))
    return 0.0


def batch_observations(observations: Iterable[dict[str, Any]], device: torch.device | str = "cpu") -> FeatureBatch:
    encoded = [encode_observation(observation) for observation in observations]
    return batch_encoded_observations(encoded, device)


def batch_encoded_observations(
    encoded: Iterable[dict[str, np.ndarray]], device: torch.device | str = "cpu",
) -> FeatureBatch:
    """Stack already encoded observations without repeating Python feature work."""
    encoded = list(encoded)
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
        values.extend(_armor_item(item))
    hotbar = list(state.get("hotbar") or [])
    selected = max(0, min(8, int(_number(state.get("selected_hotbar")))))
    mainhand = state.get("mainhand")
    for index in range(9):
        item = hotbar[index] if index < len(hotbar) else None
        if (
            index == selected and isinstance(mainhand, dict)
            and str(mainhand.get("name", ""))
        ):
            item = mainhand
        values.extend(_hotbar_item(item))
    _write(out, values)


def _encode_opponent(
    state: dict[str, Any], self_state: dict[str, Any], out: np.ndarray,
) -> None:
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
        values.extend(_armor_item(item))
    _write(out, values)
    geometry = _opponent_geometry(state, self_state)
    out[OPPONENT_DISTANCE_INDEX] = _scale(geometry["distance"], 12)
    out[OPPONENT_HORIZONTAL_DISTANCE_INDEX] = _scale(
        geometry["horizontal_distance"], 12
    )
    out[OPPONENT_BEARING_SIN_INDEX] = math.sin(geometry["bearing_error"])
    out[OPPONENT_BEARING_COS_INDEX] = math.cos(geometry["bearing_error"])
    out[OPPONENT_PITCH_ERROR_INDEX] = _scale(
        geometry["pitch_error"], math.pi / 2
    )
    out[OPPONENT_CLOSING_SPEED_INDEX] = _scale(geometry["closing_speed"], 2)
    out[OPPONENT_MELEE_RANGE_INDEX] = geometry["within_melee_reach"]
    out[OPPONENT_AIM_ALIGNMENT_INDEX] = max(
        -1.0, min(1.0, geometry["aim_alignment"])
    )
    out[OPPONENT_FACING_SELF_INDEX] = max(
        -1.0, min(1.0, geometry["facing_toward_self"])
    )
    out[OPPONENT_HELD_ITEM_CLASS_INDEX] = _pvp_item_category(
        (state.get("mainhand") or {}).get("name")
        if isinstance(state.get("mainhand"), dict) else ""
    )


def _encode_entity(
    state: dict[str, Any], self_state: dict[str, Any], out: np.ndarray,
    crystal_target: bool = False,
) -> None:
    kind = str(state.get("kind", ""))
    values = [
        *_one_hot_kind(kind, ("end_crystal", "arrow", "snowball", "egg", "fireball")),
        *_vec(state.get("relative_position"), 12), *_vec(state.get("relative_velocity"), 2),
        _scale(state.get("age_ticks"), 200), _scale(state.get("distance"), 12),
        _bool(state.get("raycastable")), _hash_feature(kind),
    ]
    _write(out, values)
    out[ENTITY_CRYSTAL_TARGET_INDEX] = float(crystal_target)
    body_position = _body_relative_vector(state, self_state, "position")
    out[ENTITY_BODY_X_INDEX] = _scale(body_position[0], 12)
    out[ENTITY_BODY_Z_INDEX] = _scale(body_position[2], 12)


def _encode_block(
    state: dict[str, Any], self_state: dict[str, Any], out: np.ndarray,
    crystal_target: bool = False, tactical_target: bool = False,
) -> None:
    collision = str(state.get("collision", "empty"))
    name = str(state.get("name", ""))
    body_position = _body_relative_vector(state, self_state, "position")
    body_bearing = (
        math.atan2(-body_position[0], -body_position[2])
        if math.hypot(body_position[0], body_position[2]) > 1e-6 else 0.0
    )
    values = [
        *_vec(state.get("relative_position"), 6),
        *(float(collision == item) for item in ("empty", "solid", "liquid", "partial")),
        _scale(state.get("hardness"), 50), _bool(state.get("replaceable")),
        _number(state.get("break_progress")), _bool(state.get("crystal_clearance")),
        _scale(state.get("exposed_faces"), 6), _scale(state.get("distance"), 8),
        _bool(state.get("within_reach")), _bool(state.get("raycastable")),
        _scale(state.get("sample_age_ticks"), 10),
        float("obsidian" in name), float(name in {"bedrock", "obsidian"}),
        _scale(body_bearing, math.pi),
    ]
    _write(out, values)
    out[BLOCK_CRYSTAL_TARGET_INDEX] = float(crystal_target or tactical_target)
    if tactical_target:
        # Slot 17 is an operation discriminator for marked blocks. A tactical
        # support can itself be obsidian, but the intended action is to place a
        # new base on its top face, not to use the support as the crystal base.
        out[BLOCK_CRYSTAL_BASE_INDEX] = 0.0


def _encode_crystal_context(
    observation: dict[str, Any], state: Any, out: np.ndarray,
) -> tuple[bool, bool]:
    state = state if isinstance(state, dict) else {}
    hotbar = state.get("hotbar") or []
    has_crystal = any(
        isinstance(item, dict)
        and _number(item.get("count")) > 0
        and "crystal" in str(item.get("name", "")).lower()
        for item in hotbar
    )
    match = observation.get("match")
    match = match if isinstance(match, dict) else {}
    mode = str(match.get("mode", "")).lower()
    lane = str(match.get("lane", "")).lower()
    # Older workers did not emit mode/lane. Item presence is a safe fallback:
    # the sword-retention kit has no crystals, while all crystal-capable kits do.
    declared_crystal_mode = mode in {"crystal", "combined", "terrain"} or lane in {
        "crystal_retention", "combined", "terrain",
    }
    crystal_capable = declared_crystal_mode or (
        has_crystal and mode != "sword" and lane != "sword_retention"
    )
    crystal_retention = crystal_capable and (
        mode == "crystal" or lane == "crystal_retention"
    )
    out[SELF_CRYSTAL_CAPABLE_INDEX] = float(crystal_capable)
    out[SELF_CRYSTAL_RETENTION_INDEX] = float(crystal_retention)
    return crystal_capable, has_crystal


def _select_tactical_placement_target(blocks: list[Any]) -> int | None:
    """Return the single worker-verified support for an obsidian placement.

    Existing obsidian/bedrock bases keep the positive target marker for the
    crystal path. A raw tactical marker takes precedence and clears encoded
    base slot 17, so the fixed block tensor distinguishes the two operations
    even if the physical support happens to be obsidian.
    """

    for index, block in enumerate(blocks):
        if not isinstance(block, dict) or not bool(block.get("tactical_placement_target")):
            continue
        return index
    return None


def _opponent_geometry(
    state: dict[str, Any], self_state: dict[str, Any],
) -> dict[str, float]:
    position = _body_relative_vector(state, self_state, "position")
    velocity = _body_relative_vector(state, self_state, "velocity")
    distance = _optional_number(state.get("distance"))
    if distance is None:
        distance = math.sqrt(sum(component * component for component in position))
    horizontal_distance = _optional_number(state.get("horizontal_distance"))
    if horizontal_distance is None:
        horizontal_distance = math.hypot(position[0], position[2])
    bearing_error = _optional_number(state.get("bearing_error"))
    if bearing_error is None:
        bearing_error = math.atan2(-position[0], -position[2])
    pitch_error = _optional_number(state.get("pitch_error"))
    if pitch_error is None:
        desired_pitch = math.atan2(position[1], max(horizontal_distance, 1e-6))
        pitch_error = desired_pitch - _number(self_state.get("pitch"))
    closing_speed = _optional_number(state.get("closing_speed"))
    if closing_speed is None:
        closing_speed = -sum(
            position[axis] * velocity[axis] for axis in range(3)
        ) / max(distance, 1e-6)
    within_melee = (
        _bool(state.get("within_melee_reach"))
        if "within_melee_reach" in state else float(distance <= 3.4)
    )
    aim_alignment = _optional_number(state.get("aim_alignment"))
    if aim_alignment is None:
        aim_alignment = math.cos(bearing_error) * math.cos(pitch_error)
    facing_self = _optional_number(state.get("facing_toward_self"))
    if facing_self is None:
        facing_self = _opponent_facing_alignment(state, self_state, position)
    return {
        "distance": max(0.0, distance),
        "horizontal_distance": max(0.0, horizontal_distance),
        "bearing_error": bearing_error,
        "pitch_error": pitch_error,
        "closing_speed": closing_speed,
        "within_melee_reach": within_melee,
        "aim_alignment": aim_alignment,
        "facing_toward_self": facing_self,
    }


def _body_relative_vector(
    state: dict[str, Any], self_state: dict[str, Any], suffix: str,
) -> tuple[float, float, float]:
    explicit = _raw_vec(state.get(f"body_relative_{suffix}"))
    if explicit is not None:
        return explicit
    legacy = _raw_vec(state.get(f"relative_{suffix}")) or (0.0, 0.0, 0.0)
    # Workers predating body_relative_* used the opposite yaw sign. Preserve
    # their wire fields for checkpoint compatibility and expose the corrected
    # vector separately.
    yaw = _number(self_state.get("yaw"))
    sine = math.sin(2.0 * yaw)
    cosine = math.cos(2.0 * yaw)
    return (
        cosine * legacy[0] - sine * legacy[2],
        legacy[1],
        sine * legacy[0] + cosine * legacy[2],
    )


def _opponent_facing_alignment(
    state: dict[str, Any], self_state: dict[str, Any],
    body_position: tuple[float, float, float],
) -> float:
    horizontal = math.hypot(body_position[0], body_position[2])
    if horizontal <= 1e-6:
        return 0.0
    self_yaw = _number(self_state.get("yaw"))
    sine, cosine = math.sin(self_yaw), math.cos(self_yaw)
    world_x = body_position[0] * cosine + body_position[2] * sine
    world_z = -body_position[0] * sine + body_position[2] * cosine
    opponent_yaw = _number(state.get("head_yaw", state.get("yaw")))
    return (
        math.sin(opponent_yaw) * world_x + math.cos(opponent_yaw) * world_z
    ) / horizontal


def _select_crystal_entity(entities: list[Any]) -> int | None:
    candidates: list[tuple[float, int]] = []
    for index, entity in enumerate(entities):
        if not isinstance(entity, dict) or "crystal" not in str(entity.get("kind", "")).lower():
            continue
        distance = max(0.0, _number(entity.get("distance")))
        relative = _raw_vec(entity.get("relative_position"))
        if (
            relative is not None
            and distance <= CRYSTAL_CHAIN_REACH
            and _normal_camera_pitch_reachable(relative, 1.0 - CRYSTAL_EYE_HEIGHT)
        ):
            candidates.append((distance, index))
    return min(candidates)[1] if candidates else None


def _select_crystal_base(blocks: list[Any], opponent: Any, self_state: Any) -> int | None:
    opponent_position = opponent.get("relative_position") if isinstance(opponent, dict) else None
    opponent_vector = _raw_vec(opponent_position)
    self_state = self_state if isinstance(self_state, dict) else {}
    yaw = _number(self_state.get("yaw"))
    yaw_sin, yaw_cos = math.sin(yaw), math.cos(yaw)
    candidates: list[tuple[float, int]] = []
    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        name = str(block.get("name", "")).lower()
        distance = max(0.0, _number(block.get("distance")))
        if (
            name not in {"obsidian", "bedrock"}
            or not bool(block.get("crystal_clearance"))
            or not bool(block.get("within_reach"))
            or distance > CRYSTAL_CHAIN_REACH
        ):
            continue
        relative = _raw_vec(block.get("relative_position"))
        if relative is None:
            continue
        # The ordinary policy camera is intentionally limited to +/-45 degrees.
        # Do not mark a pad almost underneath the fighter: the acquisition
        # residual would otherwise pin pitch at the lower limit and repeatedly
        # click without ever reaching the top face. Teachers can still use the
        # explicit verified-target +/-75 degree path when demonstrating it.
        centered_relative = (
            relative[0] + 0.5 * (yaw_cos + yaw_sin),
            relative[1],
            relative[2] + 0.5 * (yaw_cos - yaw_sin),
        )
        if not _normal_camera_pitch_reachable(
            centered_relative, 1.0 - CRYSTAL_EYE_HEIGHT
        ):
            continue
        # Prefer a damaging setup near the assigned opponent. A sharp
        # point-blank penalty avoids teaching suicidal self-crystals.
        opponent_distance = math.sqrt(sum(
            (relative[axis] - opponent_vector[axis]) ** 2 for axis in range(3)
        )) if opponent_vector is not None else distance
        # The sampler can verify air clearance but not encode entity occupancy.
        # A base essentially underneath either fighter can never become legal.
        if distance < 1.35 or (opponent_vector is not None and opponent_distance < 1.35):
            continue
        close_penalty = 3.0 + (2.0 - distance) * 4.0 if distance < 2.0 else 0.0
        score = opponent_distance + distance * 0.15 + close_penalty
        candidates.append((score, index))
    return min(candidates)[1] if candidates else None


def _raw_vec(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, dict):
        return None
    return tuple(_number(value.get(axis)) for axis in ("x", "y", "z"))


def _normal_camera_pitch_reachable(
    relative: tuple[float, float, float], vertical_adjustment: float,
) -> bool:
    horizontal = math.hypot(relative[0], relative[2])
    if horizontal <= 1e-6:
        return False
    desired_pitch = math.atan2(relative[1] + vertical_adjustment, horizontal)
    return abs(desired_pitch) <= MAX_POLICY_CRYSTAL_TARGET_PITCH


def _encode_legal(mask: dict[str, Any], match: Any, out: np.ndarray) -> None:
    hotbar = list(mask.get("hotbar") or [])
    match = match if isinstance(match, dict) else {}
    mode = str(match.get("mode", "")).lower()
    lane = str(match.get("lane", "")).lower()
    values = [
        1.0, _bool(mask.get("attack")), _bool(mask.get("use_main")),
        _bool(mask.get("use_offhand")), 1.0, _bool(mask.get("release_use")),
        1.0, _bool(mask.get("swap_offhand")),
    ]
    values.extend(_bool(hotbar[index]) if index < len(hotbar) else 0.0 for index in range(9))
    # Slots 17-20 used to be padding. Reusing them preserves LEGAL_SIZE and
    # checkpoint compatibility while distinguishing executable combat and
    # crystal opportunities from the same generic interaction inputs.
    values.extend([
        _bool(mask.get("combat_attack_ready")),
        _bool(mask.get("crystal_place_ready")),
        _bool(mask.get("crystal_attack_ready")),
        _bool(mask.get("tactical_block_break_ready")),
        _scale(match.get("action_delay_ticks"), 5),
        _scale(match.get("observation_delay_ticks"), 5),
        float(mode == "terrain" or lane == "terrain"),
    ])
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


def _hotbar_item(item: Any) -> list[float]:
    if not isinstance(item, dict):
        return [0.0] * HOTBAR_ITEM_FEATURE_SIZE
    maximum = max(_number(item.get("max_durability")), 1.0)
    return [
        float(bool(str(item.get("name", "")))),
        _scale(item.get("count"), 64),
        _number(item.get("durability")) / maximum,
        _pvp_item_category(item.get("name")),
    ]


def _armor_item(item: Any) -> list[float]:
    if not isinstance(item, dict):
        return [0.0, 0.0, 0.0]
    name = str(item.get("name", "")).lower()
    maximum = max(_number(item.get("max_durability")), 1.0)
    return [
        float(bool(name)),
        _armor_category(name),
        _number(item.get("durability")) / maximum,
    ]


def _pvp_item_category(value: Any) -> float:
    name = str(value or "").lower()
    if not name:
        return PVP_ITEM_EMPTY
    if "sword" in name:
        return PVP_ITEM_SWORD
    if "pickaxe" in name:
        return PVP_ITEM_PICKAXE
    if "crystal" in name:
        return PVP_ITEM_CRYSTAL
    if "obsidian" in name or "bedrock" in name:
        return PVP_ITEM_OBSIDIAN
    if "apple" in name or any(food in name for food in ("bread", "carrot", "potato")):
        return PVP_ITEM_FOOD
    if "totem" in name:
        return PVP_ITEM_TOTEM
    if any(projectile in name for projectile in ("bow", "arrow", "pearl", "snowball", "egg")):
        return PVP_ITEM_PROJECTILE
    if any(block in name for block in (
        "stone", "cobblestone", "planks", "dirt", "sand", "netherrack",
    )):
        return PVP_ITEM_BLOCK
    return PVP_ITEM_OTHER


def _armor_category(name: str) -> float:
    if not name:
        return 0.0
    if "netherite" in name:
        return 1.0
    if "diamond" in name:
        return 0.8
    if "iron" in name:
        return 0.6
    if "chainmail" in name or "chain" in name:
        return 0.45
    if "gold" in name:
        return 0.3
    if "leather" in name:
        return 0.15
    return -1.0


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


def _optional_number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _scale(value: Any, scale: float) -> float:
    return max(-4.0, min(4.0, _number(value) / scale))


def _bool(value: Any) -> float:
    return float(bool(value))


def _write(out: np.ndarray, values: list[float]) -> None:
    length = min(len(out), len(values))
    out[:length] = np.asarray(values[:length], dtype=np.float32)
