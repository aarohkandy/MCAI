from __future__ import annotations

import json
from pathlib import Path


SCHEMA_DIR = Path(__file__).resolve().parents[2] / "protocol" / "schema"


def test_protocol_schemas_are_valid_json() -> None:
    for path in SCHEMA_DIR.glob("*.schema.json"):
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(schema, dict)
        assert schema["$id"] == f"https://mc-bot.local/schema/{path.name}"


def test_protocol_documents_tactical_mining_and_reward_stats() -> None:
    observation = json.loads((SCHEMA_DIR / "observation-v1.schema.json").read_text(encoding="utf-8"))
    mask = observation["$defs"]["actionMask"]["properties"]
    assert mask["tactical_block_break_ready"] == {"type": "boolean"}

    messages = json.loads((SCHEMA_DIR / "messages-v1.schema.json").read_text(encoding="utf-8"))
    stats = messages["$defs"]["combatStats"]["properties"]
    for name in (
        "blocks_placed", "blocks_mined", "obsidian_placed",
        "tactical_obsidian_placed", "tactical_mine_place_sequences",
        "policy_built_crystal_chains_damaging", "rewarded_obsidian_combos",
        "crystals_placed", "crystals_destroyed", "crystals_exploded",
        "policy_crystal_chains_started", "policy_crystal_chains_detonated",
        "policy_crystal_chains_damaging", "policy_crystal_chains_popping",
        "rewarded_crystal_combos", "policy_crystal_chain_damage_rate",
        "invalid_interactions", "extreme_pitch_ticks", "spam_attack_swings",
    ):
        assert name in stats
    attributed = messages["$defs"]["attributedEvents"]["properties"]
    assert attributed["tactical_mine_place_sequences"] == {
        "type": "integer", "minimum": 0,
    }


def test_execution_schema_accepts_exact_teacher_observation() -> None:
    messages = json.loads((SCHEMA_DIR / "messages-v1.schema.json").read_text(encoding="utf-8"))
    execution = messages["$defs"]["execution"]

    assert execution["additionalProperties"] is False
    assert execution["properties"]["pre_execution_observation"] == {
        "oneOf": [
            {"$ref": "observation-v1.schema.json"},
            {"$ref": "observation-v2.schema.json"},
        ]
    }


def test_match_schema_carries_active_lane_curriculum_context() -> None:
    observation = json.loads((SCHEMA_DIR / "observation-v1.schema.json").read_text(encoding="utf-8"))
    match = observation["properties"]["match"]["properties"]

    assert match["mode"]["enum"] == ["sword", "crystal", "combined", "terrain"]
    assert match["lane"]["enum"] == [
        "sword_retention", "crystal_retention", "combined", "terrain",
    ]
    assert match["arena_radius"] == {"type": "integer", "minimum": 1}
    assert match["curriculum_stage"] == {"type": "integer", "minimum": 1}
