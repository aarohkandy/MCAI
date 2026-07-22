import math

import numpy as np

from combat_ai.features import (
    BLOCK_CRYSTAL_BASE_INDEX,
    BLOCK_CRYSTAL_TARGET_INDEX,
    BLOCK_BODY_BEARING_INDEX,
    ENTITY_CRYSTAL_TARGET_INDEX,
    ENTITY_BODY_X_INDEX,
    ENTITY_BODY_Z_INDEX,
    HOTBAR_SEMANTIC_INDICES,
    LEGAL_ACTION_DELAY_INDEX,
    LEGAL_OBSERVATION_DELAY_INDEX,
    LEGAL_TERRAIN_MODE_INDEX,
    OPPONENT_AIM_ALIGNMENT_INDEX,
    OPPONENT_BEARING_COS_INDEX,
    OPPONENT_BEARING_SIN_INDEX,
    OPPONENT_CLOSING_SPEED_INDEX,
    OPPONENT_DISTANCE_INDEX,
    OPPONENT_FACING_SELF_INDEX,
    OPPONENT_HELD_ITEM_CLASS_INDEX,
    OPPONENT_HORIZONTAL_DISTANCE_INDEX,
    OPPONENT_MELEE_RANGE_INDEX,
    OPPONENT_PITCH_ERROR_INDEX,
    PVP_ITEM_CRYSTAL,
    PVP_ITEM_OBSIDIAN,
    PVP_ITEM_PICKAXE,
    PVP_ITEM_SWORD,
    SELF_ARMOR_START_INDEX,
    SELF_CRYSTAL_CAPABLE_INDEX,
    SELF_CRYSTAL_RETENTION_INDEX,
    BLOCK_SIZE,
    ENTITY_SIZE,
    LEGAL_SIZE,
    OPPONENT_SIZE,
    SELF_SIZE,
    TACTICAL_BODY_X_INDEX,
    TACTICAL_CRYSTAL_KIND_INDEX,
    TACTICAL_FOLLOWUP_INDEX,
    TACTICAL_LEGAL_INDEX,
    TACTICAL_OPPONENT_DAMAGE_INDEX,
    TACTICAL_POP_POTENTIAL_INDEX,
    TACTICAL_REACHABLE_INDEX,
    TACTICAL_SELF_DAMAGE_INDEX,
    batch_encoded_observations,
    batch_observations,
    encode_observation,
)
from fixtures import observation


def test_preencoded_batch_matches_regular_observation_batch():
    values = [observation("batch-a", 1), observation("batch-b", 2)]
    regular = batch_observations(values)
    preencoded = batch_encoded_observations([encode_observation(value) for value in values])

    for name, expected in vars(regular).items():
        np.testing.assert_array_equal(vars(preencoded)[name].numpy(), expected.numpy())


def test_v2_crystal_candidate_aliases_and_kind_discriminator_are_encoded():
    value = observation()
    value["schema_version"] = 2
    value["tactical"] = {
        "crystal_candidates": [
            {
                "kind": "crystal", "source_index": 3, "distance": 2.5,
                "reach": True, "visible": True, "legal": False,
                "opponent_damage": 7.0, "self_damage": 2.0,
                "pop_probability": 0.75, "escape_x": -1,
                "body_relative_position": {"x": 1.5, "y": 0.5, "z": -2.0},
            },
            {
                "kind": "base", "source_index": 4, "distance": 3.0,
                "reachable": True, "visible": True, "placement_legal": True,
                "estimated_opponent_damage": 9.0, "estimated_self_damage": 1.0,
                "pop_potential": 1.0, "escape_direction": 1,
                "body_relative_position": {"x": -1.0, "y": -1.0, "z": -2.5},
            },
        ],
        "block_candidates": [{
            "source_index": 2, "distance": 2, "reachable": True, "visible": True,
            "purpose": "crystal_base", "cover_value": 0.4,
            "follow_up_viability": 0.9,
            "body_relative_position": {"x": 2, "y": -1, "z": 0},
        }],
        "recent_history": [], "survival": {}, "threat": {},
    }
    encoded = encode_observation(value)
    crystal, base = encoded["crystal_candidates"][:2]
    assert encoded["crystal_candidate_mask"][:2].tolist() == [1.0, 1.0]
    assert crystal[TACTICAL_CRYSTAL_KIND_INDEX] == -1.0
    assert base[TACTICAL_CRYSTAL_KIND_INDEX] == 1.0
    assert crystal[TACTICAL_REACHABLE_INDEX] == 1.0
    assert base[TACTICAL_LEGAL_INDEX] == 1.0
    assert crystal[TACTICAL_OPPONENT_DAMAGE_INDEX] == 7.0
    assert crystal[TACTICAL_SELF_DAMAGE_INDEX] == 2.0
    assert crystal[TACTICAL_POP_POTENTIAL_INDEX] == 0.75
    assert crystal[TACTICAL_BODY_X_INDEX] == 1.5
    assert np.isclose(encoded["tactical_blocks"][0, TACTICAL_FOLLOWUP_INDEX], 0.9)


def test_feature_contract_has_fixed_shapes():
    encoded = encode_observation(observation())
    assert encoded["self_state"].shape == (SELF_SIZE,)
    assert encoded["opponent"].shape == (OPPONENT_SIZE,)
    assert encoded["entities"].shape == (16, ENTITY_SIZE)
    assert encoded["blocks"].shape == (48, BLOCK_SIZE)
    assert encoded["legal"].shape == (LEGAL_SIZE,)
    assert encoded["legal"][17] == 1.0
    assert encoded["legal"][18] == 0.0
    assert encoded["legal"][19] == 0.0
    assert encoded["legal"][20] == 0.0
    assert encoded["opponent_mask"].tolist() == [1.0]
    assert np.isfinite(encoded["self_state"]).all()


def test_missing_slots_are_explicitly_masked():
    value = observation()
    value["opponent"] = None
    encoded = encode_observation(value)
    assert encoded["opponent_mask"].sum() == 0
    assert encoded["entity_mask"].sum() == 0
    assert encoded["block_mask"].sum() == 0


def test_combat_readiness_is_distinct_from_mining_legality():
    value = observation()
    value["action_mask"]["attack"] = True
    value["action_mask"]["combat_attack_ready"] = False
    encoded = encode_observation(value)
    assert encoded["legal"][1] == 1.0
    assert encoded["legal"][17] == 0.0


def test_crystal_readiness_uses_reserved_checkpoint_compatible_slots():
    value = observation()
    value["action_mask"]["crystal_place_ready"] = True
    value["action_mask"]["crystal_attack_ready"] = True
    encoded = encode_observation(value)
    assert encoded["legal"].shape == (LEGAL_SIZE,)
    assert encoded["legal"][18] == 1.0
    assert encoded["legal"][19] == 1.0


def test_block_break_readiness_uses_reserved_checkpoint_compatible_slot():
    value = observation()
    value["action_mask"]["tactical_block_break_ready"] = True
    encoded = encode_observation(value)
    assert encoded["legal"].shape == (LEGAL_SIZE,)
    assert encoded["legal"][20] == 1.0


def test_hotbar_semantics_distinguish_same_durability_tools_and_stack_items():
    value = observation()
    value["self"]["hotbar"][0] = _item("diamond_sword", 1, 1561, 1561)
    value["self"]["hotbar"][1] = _item("diamond_pickaxe", 1, 1561, 1561)
    value["self"]["hotbar"][2] = _item("end_crystal", 64)
    value["self"]["hotbar"][3] = _item("obsidian", 64)

    encoded = encode_observation(value)["self_state"]

    assert encoded[HOTBAR_SEMANTIC_INDICES[0]] == PVP_ITEM_SWORD
    assert encoded[HOTBAR_SEMANTIC_INDICES[1]] == PVP_ITEM_PICKAXE
    assert encoded[HOTBAR_SEMANTIC_INDICES[2]] == PVP_ITEM_CRYSTAL
    assert encoded[HOTBAR_SEMANTIC_INDICES[3]] == PVP_ITEM_OBSIDIAN


def test_explicit_mainhand_and_armor_categories_survive_encoding():
    value = observation()
    value["self"]["selected_hotbar"] = 5
    value["self"]["hotbar"][5] = _item("obsidian", 64)
    value["self"]["mainhand"] = _item("diamond_sword", 1, 1561, 1561)
    value["self"]["armor"][0] = _item("leather_helmet", 1, 55, 55)

    encoded = encode_observation(value)["self_state"]

    assert encoded[HOTBAR_SEMANTIC_INDICES[5]] == PVP_ITEM_SWORD
    assert encoded[SELF_ARMOR_START_INDEX + 1] == np.float32(0.15)


def test_explicit_opponent_pvp_geometry_fills_all_reserved_slots():
    value = observation()
    value["opponent"].update({
        "body_relative_position": {"x": 3.0, "y": 1.0, "z": -4.0},
        "body_relative_velocity": {"x": -0.3, "y": 0.0, "z": 0.4},
        "distance": 5.1,
        "horizontal_distance": 5.0,
        "bearing_error": -0.25,
        "pitch_error": 0.1,
        "closing_speed": 0.5,
        "within_melee_reach": False,
        "aim_alignment": 0.9,
        "facing_toward_self": 0.75,
        "head_yaw": 0.3,
        "mainhand": _item("diamond_pickaxe", 1, 1561, 1561),
    })

    encoded = encode_observation(value)["opponent"]

    assert np.isclose(encoded[OPPONENT_DISTANCE_INDEX], 5.1 / 12)
    assert np.isclose(encoded[OPPONENT_HORIZONTAL_DISTANCE_INDEX], 5.0 / 12)
    assert np.isclose(encoded[OPPONENT_BEARING_SIN_INDEX], math.sin(-0.25))
    assert np.isclose(encoded[OPPONENT_BEARING_COS_INDEX], math.cos(-0.25))
    assert np.isclose(encoded[OPPONENT_PITCH_ERROR_INDEX], 0.1 / (math.pi / 2))
    assert np.isclose(encoded[OPPONENT_CLOSING_SPEED_INDEX], 0.25)
    assert encoded[OPPONENT_MELEE_RANGE_INDEX] == 0.0
    assert np.isclose(encoded[OPPONENT_AIM_ALIGNMENT_INDEX], 0.9)
    assert np.isclose(encoded[OPPONENT_FACING_SELF_INDEX], 0.75)
    assert encoded[OPPONENT_HELD_ITEM_CLASS_INDEX] == PVP_ITEM_PICKAXE


def test_legacy_yaw_geometry_is_corrected_for_opponent_and_entities():
    value = observation()
    yaw = math.pi / 2
    distance = 5.0
    value["self"]["yaw"] = yaw
    legacy = {
        "x": -math.sin(2 * yaw) * distance,
        "y": 0.0,
        "z": -math.cos(2 * yaw) * distance,
    }
    value["opponent"]["relative_position"] = dict(legacy)
    value["entities"] = [{
        "kind": "arrow", "relative_position": dict(legacy),
        "relative_velocity": {"x": 0, "y": 0, "z": 0},
        "age_ticks": 1, "distance": distance, "raycastable": False,
    }]

    encoded = encode_observation(value)

    assert abs(encoded["opponent"][OPPONENT_BEARING_SIN_INDEX]) < 1e-6
    assert encoded["opponent"][OPPONENT_BEARING_COS_INDEX] > 0.999
    assert abs(encoded["entities"][0, ENTITY_BODY_X_INDEX]) < 1e-6
    assert np.isclose(encoded["entities"][0, ENTITY_BODY_Z_INDEX], -distance / 12)


def test_block_body_bearing_prefers_explicit_worker_geometry():
    value = observation()
    block = _block(99.0, -1.0, 99.0, 2.8)
    block["body_relative_position"] = {"x": 2.0, "y": -1.0, "z": -2.0}
    value["blocks"] = [block]

    encoded = encode_observation(value)["blocks"]

    assert np.isclose(encoded[0, BLOCK_BODY_BEARING_INDEX], -0.25)


def test_legacy_block_bearing_is_yaw_invariant_for_front_and_side_targets():
    distance = 3.0
    for yaw in (0.0, math.pi / 2, -math.pi / 2, math.pi / 4):
        sine = math.sin(2 * yaw)
        cosine = math.cos(2 * yaw)

        front = observation()
        front["self"]["yaw"] = yaw
        front["blocks"] = [_block(
            -sine * distance, -1.0, -cosine * distance, distance,
        )]
        front_bearing = encode_observation(front)["blocks"][0, BLOCK_BODY_BEARING_INDEX]
        assert abs(front_bearing) < 1e-6

        right = observation()
        right["self"]["yaw"] = yaw
        # Invert the legacy +2*yaw compatibility rotation for corrected body
        # coordinates x=distance,z=0 (directly to the fighter's right).
        right["blocks"] = [_block(
            cosine * distance, -1.0, -sine * distance, distance,
        )]
        right_bearing = encode_observation(right)["blocks"][0, BLOCK_BODY_BEARING_INDEX]
        assert np.isclose(right_bearing, -0.5)


def test_delay_and_terrain_context_use_former_constant_legal_slots():
    value = observation()
    value["match"].update({
        "action_delay_ticks": 2,
        "observation_delay_ticks": 3,
        "mode": "terrain",
    })
    encoded = encode_observation(value)["legal"]
    assert encoded[LEGAL_ACTION_DELAY_INDEX] == np.float32(0.4)
    assert encoded[LEGAL_OBSERVATION_DELAY_INDEX] == np.float32(0.6)
    assert encoded[LEGAL_TERRAIN_MODE_INDEX] == 1.0


def test_crystal_context_marks_one_reachable_offensive_target_without_changing_shapes():
    value = observation()
    value["match"].update({"mode": "crystal", "lane": "crystal_retention"})
    value["self"]["hotbar"][3] = {
        "name": "end_crystal", "count": 64, "durability": 0,
        "max_durability": 0, "enchant_hash": 0,
    }
    value["opponent"]["relative_position"] = {"x": 1.0, "y": 0.0, "z": -2.5}
    value["blocks"] = [
        _block(0.0, -1.0, -2.5, 2.7),
        _block(-2.5, -1.0, 0.0, 2.7),
        _block(0.0, -1.0, -4.0, 4.1),
    ]
    value["entities"] = [
        {"kind": "end_crystal", "relative_position": {"x": -1.0, "y": 0.0, "z": -2.0},
         "relative_velocity": {"x": 0, "y": 0, "z": 0}, "age_ticks": 2,
         "distance": 2.3, "raycastable": False},
    ]

    encoded = encode_observation(value)

    assert encoded["self_state"][SELF_CRYSTAL_CAPABLE_INDEX] == 1.0
    assert encoded["self_state"][SELF_CRYSTAL_RETENTION_INDEX] == 1.0
    assert encoded["blocks"][:, BLOCK_CRYSTAL_TARGET_INDEX].sum() == 1.0
    assert encoded["blocks"][0, BLOCK_CRYSTAL_TARGET_INDEX] == 1.0
    assert encoded["entities"][:, ENTITY_CRYSTAL_TARGET_INDEX].sum() == 1.0


def test_worker_tactical_support_reuses_target_marker_but_not_crystal_base_bit():
    value = observation()
    value["match"].update({"mode": "terrain", "lane": "terrain"})
    support = _block(1.0, -1.0, -2.0, 2.4)
    support.update({
        "name": "stone",
        "crystal_clearance": False,
        "tactical_placement_target": True,
    })
    value["blocks"] = [support]

    encoded = encode_observation(value)

    assert encoded["blocks"].shape == (48, BLOCK_SIZE)
    assert encoded["blocks"][0, BLOCK_CRYSTAL_TARGET_INDEX] == 1.0
    assert encoded["blocks"][0, BLOCK_CRYSTAL_BASE_INDEX] == 0.0


def test_existing_crystal_base_keeps_base_bit_with_shared_target_marker():
    value = observation()
    value["match"].update({"mode": "terrain", "lane": "terrain"})
    value["self"]["hotbar"][3] = _item("end_crystal", 64)
    value["blocks"] = [_block(1.0, -1.0, -2.0, 2.4)]

    encoded = encode_observation(value)

    assert encoded["blocks"][0, BLOCK_CRYSTAL_TARGET_INDEX] == 1.0
    assert encoded["blocks"][0, BLOCK_CRYSTAL_BASE_INDEX] == 1.0


def test_raw_tactical_marker_overrides_base_bit_for_obsidian_support():
    value = observation()
    value["match"].update({"mode": "terrain", "lane": "terrain"})
    value["self"]["hotbar"][3] = _item("end_crystal", 64)
    support = _block(1.0, -1.0, -2.0, 2.4)
    support["tactical_placement_target"] = True
    value["blocks"] = [support]

    encoded = encode_observation(value)

    assert encoded["blocks"][0, BLOCK_CRYSTAL_TARGET_INDEX] == 1.0
    assert encoded["blocks"][0, BLOCK_CRYSTAL_BASE_INDEX] == 0.0


def test_sword_lane_does_not_activate_crystal_acquisition_even_with_stray_item():
    value = observation()
    value["match"].update({"mode": "sword", "lane": "sword_retention"})
    value["self"]["hotbar"][3] = {
        "name": "end_crystal", "count": 1, "durability": 0,
        "max_durability": 0, "enchant_hash": 0,
    }
    encoded = encode_observation(value)
    assert encoded["self_state"][SELF_CRYSTAL_CAPABLE_INDEX] == 0.0
    assert encoded["self_state"][SELF_CRYSTAL_RETENTION_INDEX] == 0.0


def test_crystal_lane_can_acquire_a_detonation_target_after_inventory_is_empty():
    value = observation()
    value["match"].update({"mode": "crystal", "lane": "crystal_retention"})
    value["entities"] = [{
        "kind": "end_crystal", "relative_position": {"x": 0, "y": 0, "z": -2.5},
        "relative_velocity": {"x": 0, "y": 0, "z": 0}, "age_ticks": 2,
        "distance": 2.5, "raycastable": False,
    }]
    encoded = encode_observation(value)
    assert encoded["self_state"][SELF_CRYSTAL_CAPABLE_INDEX] == 1.0
    assert encoded["entities"][0, ENTITY_CRYSTAL_TARGET_INDEX] == 1.0


def test_crystal_acquisition_ignores_pad_beneath_feet_and_marks_randomized_reachable_pad():
    value = observation()
    value["match"].update({"mode": "crystal", "lane": "crystal_retention"})
    value["self"]["hotbar"][3] = {
        "name": "end_crystal", "count": 64, "durability": 0,
        "max_durability": 0, "enchant_hash": 0,
    }
    value["blocks"] = [
        _block(0.0, -1.0, 0.0, 1.0),
        _block(-2.5, -1.0, 0.5, 2.7),
    ]

    encoded = encode_observation(value)

    assert encoded["blocks"][0, BLOCK_CRYSTAL_TARGET_INDEX] == 0.0
    assert encoded["blocks"][1, BLOCK_CRYSTAL_TARGET_INDEX] == 1.0


def _block(x: float, y: float, z: float, distance: float) -> dict:
    return {
        "name": "obsidian", "relative_position": {"x": x, "y": y, "z": z},
        "collision": "solid", "hardness": 50, "replaceable": False,
        "break_progress": 0, "crystal_clearance": True, "exposed_faces": 5,
        "distance": distance, "within_reach": True, "raycastable": False,
        "sample_age_ticks": 0,
    }


def _item(name: str, count: int, durability: int = 0, maximum: int = 0) -> dict:
    return {
        "name": name, "count": count, "durability": durability,
        "max_durability": maximum, "enchant_hash": 0,
    }
