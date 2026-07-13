from __future__ import annotations


def item(name: str = "", count: int = 0) -> dict:
    return {"name": name, "count": count, "durability": 0, "max_durability": 0, "enchant_hash": 0}


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
