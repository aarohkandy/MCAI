from __future__ import annotations


def item(name: str = "", count: int = 0, enchant_hash: int = 0) -> dict:
    return {"name": name, "count": count, "durability": 0, "max_durability": 0,
            "enchant_hash": enchant_hash}


def observation(episode: str = "test-episode", tick: int = 1) -> dict:
    empty = item()
    return {
        "schema_version": 1,
        "match": {"episode_id": episode, "tick": tick, "policy_version": 0, "arena_seed": 42,
                  "action_delay_ticks": 0, "observation_delay_ticks": 0},
        "self": {
            "health": 20, "absorption": 0, "food": 20,
            "position": {"x": 0, "y": 64, "z": 0}, "velocity": {"x": 0, "y": 0, "z": 0},
            "yaw": 0, "pitch": 0, "on_ground": True, "sprinting": False, "sneaking": False,
            "hurt_time": 0, "attack_cooldown": 1, "active_hand": "none", "use_ticks": 0,
            "mining_progress": 0, "selected_hotbar": 0,
            "hotbar": [item("diamond_sword", 1), *[empty.copy() for _ in range(8)]],
            "offhand": item("totem_of_undying", 1), "armor": [item("diamond_helmet", 1)] * 4,
            "raycast": {"kind": "entity", "distance": 2.8, "block_name": "", "entity_kind": "player"},
        },
        "opponent": {
            "relative_position": {"x": 0, "y": 0, "z": -2.8},
            "relative_velocity": {"x": 0, "y": 0, "z": 0}, "yaw": 0, "pitch": 0,
            "health": 20, "hurt_time": 0, "on_ground": True, "line_of_sight": True,
            "mainhand": item("diamond_sword", 1), "offhand": item("totem_of_undying", 1),
            "armor": [item("diamond_chestplate", 1)] * 4,
        },
        "entities": [], "blocks": [],
        "action_mask": {"attack": True, "use_main": True, "use_offhand": True,
                        "release_use": False, "swap_offhand": True, "hotbar": [True] * 9},
    }


def parity_observation() -> dict:
    """Observation that exercises the parity-sensitive paths the plain fixture misses:
    negative signed enchant hashes, populated entity slots, and populated block slots.
    Used by scripts/verify_model_parity.py so browser/PyTorch divergence is actually caught."""
    value = observation()
    # Signed 32-bit enchant hashes matching the real kit (worker/src/items.ts hashString),
    # so the JS floored-modulo path is compared against Python's `%` on negatives.
    sword = item("diamond_sword", 1, enchant_hash=-1713569221)
    chestplate = item("diamond_chestplate", 1, enchant_hash=-361629127)
    value["self"]["hotbar"][0] = sword
    value["self"]["offhand"] = item("totem_of_undying", 1, enchant_hash=12345)
    value["opponent"]["mainhand"] = sword
    value["opponent"]["armor"] = [chestplate] * 4
    value["entities"] = [
        {"kind": "end_crystal", "relative_position": {"x": 1.5, "y": 0.0, "z": -2.0},
         "relative_velocity": {"x": 0.0, "y": 0.0, "z": 0.0}, "age_ticks": 12,
         "distance": 2.5, "raycastable": True},
        {"kind": "arrow", "relative_position": {"x": -3.0, "y": 1.0, "z": 4.0},
         "relative_velocity": {"x": 0.2, "y": -0.1, "z": -0.4}, "age_ticks": 3,
         "distance": 5.1, "raycastable": False},
    ]
    value["blocks"] = [
        {"name": "obsidian", "relative_position": {"x": 0.0, "y": -1.0, "z": -1.0},
         "collision": "solid", "hardness": 50.0, "replaceable": False, "break_progress": 0.0,
         "crystal_clearance": True, "exposed_faces": 5, "distance": 1.4, "within_reach": True,
         "raycastable": False, "sample_age_ticks": 0},
        {"name": "stone", "relative_position": {"x": 2.0, "y": 0.0, "z": 1.0},
         "collision": "solid", "hardness": 1.5, "replaceable": False, "break_progress": 0.3,
         "crystal_clearance": False, "exposed_faces": 3, "distance": 2.2, "within_reach": True,
         "raycastable": True, "sample_age_ticks": 2},
    ]
    return value
