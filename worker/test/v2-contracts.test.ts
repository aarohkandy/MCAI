import { describe, expect, it } from 'vitest'
import {
  ACTION_V2_SCHEMA_VERSION,
  NOOP_ACTION,
  SCHEMA_VERSION,
  type ActionV2,
  type ObservationV1,
  validateActionV2
} from '../src/contracts.js'
import { observationV2 } from '../src/observation-v2.js'
import { resolveActionV2 } from '../src/action-v2.js'

describe('V2 tactical protocol', () => {
  it('ranks a damaging legal crystal base and exposes survival state', () => {
    const observation = observationV2(fixture())
    expect(observation.schema_version).toBe(2)
    expect(observation.tactical.crystal_candidates[0]).toMatchObject({
      kind: 'base', placement_legal: true, source_index: 0
    })
    expect(observation.tactical.crystal_candidates[0].estimated_opponent_damage).toBeGreaterThan(0)
    expect(observation.tactical.survival.has_totem).toBe(true)
  })

  it('maps crystal placement to one coherent legal use control', () => {
    const observation = observationV2(fixture())
    const action = v2({ intent: 'crystal_place', target_index: 0, primary: 'attack', hotbar: 8 })
    expect(resolveActionV2(action, observation)).toMatchObject({
      schema_version: SCHEMA_VERSION,
      primary: 'use_main',
      hotbar: 1
    })
  })

  it('rejects a cross-family target instead of clicking', () => {
    const observation = observationV2(fixture())
    expect(() => resolveActionV2(v2({ intent: 'crystal_detonate', target_index: 0 }), observation))
      .toThrow(/does not name a crystal/)
  })

  it('validates intent and target bounds independently of V1', () => {
    expect(validateActionV2(v2({ intent: 'reposition', target_index: -1 })).schema_version).toBe(2)
    expect(() => validateActionV2(v2({ target_index: 64 }))).toThrow(/target index/)
  })
})

function v2(overrides: Partial<ActionV2> = {}): ActionV2 {
  return {
    ...NOOP_ACTION,
    schema_version: ACTION_V2_SCHEMA_VERSION,
    intent: 'reposition',
    target_index: -1,
    ...overrides
  }
}

function fixture(): ObservationV1 {
  const item = (name = '', count = 0) => ({ name, count, durability: 0, max_durability: 0, enchant_hash: 0 })
  return {
    schema_version: SCHEMA_VERSION,
    match: { episode_id: 'e', tick: 20, policy_version: 1, arena_seed: 1, action_delay_ticks: 0, observation_delay_ticks: 0, arena_radius: 6 },
    self: {
      health: 20, absorption: 0, food: 20, position: { x: 0, y: 64, z: 0 }, velocity: { x: 0, y: 0, z: 0 },
      yaw: 0, pitch: 0, on_ground: true, sprinting: false, sneaking: false, hurt_time: 0, attack_cooldown: 1,
      active_hand: 'none', use_ticks: 0, mining_progress: 0, selected_hotbar: 0, mainhand: item('diamond_sword', 1),
      hotbar: [item('diamond_sword', 1), item('end_crystal', 16), ...Array.from({ length: 7 }, () => item())],
      offhand: item('totem_of_undying', 1), armor: Array.from({ length: 4 }, () => item('diamond_armor', 1)),
      raycast: { kind: 'block', distance: 2, block_name: 'obsidian', entity_kind: '' }
    },
    opponent: {
      relative_position: { x: 0, y: 0, z: -3 }, relative_velocity: { x: 0, y: 0, z: 0 },
      body_relative_position: { x: 0, y: 0, z: -3 }, body_relative_velocity: { x: 0, y: 0, z: 0 },
      distance: 3, horizontal_distance: 3, bearing_error: 0, pitch_error: 0, closing_speed: 0.1, within_melee_reach: true,
      aim_alignment: 1, facing_toward_self: 1, yaw: 0, head_yaw: 0, pitch: 0, health: 20, absorption: 0,
      server_state_age_ticks: 0, hurt_time: 0, on_ground: true, line_of_sight: true, mainhand: item('diamond_sword', 1),
      offhand: item(), armor: Array.from({ length: 4 }, () => item('diamond_armor', 1))
    },
    entities: [],
    blocks: [{
      name: 'obsidian', relative_position: { x: 0, y: -1, z: -2 }, body_relative_position: { x: 0, y: -1, z: -2 },
      body_relative_velocity: { x: 0, y: 0, z: 0 }, collision: 'solid', hardness: 50, replaceable: false,
      break_progress: 0, crystal_clearance: true, tactical_placement_target: false, exposed_faces: 1,
      distance: 2, within_reach: true, raycastable: true, sample_age_ticks: 0
    }],
    action_mask: {
      attack: true, combat_attack_ready: true, crystal_place_ready: true, crystal_attack_ready: false,
      tactical_block_break_ready: false, tactical_block_place_ready: false, use_main: true, use_offhand: false,
      release_use: false, swap_offhand: false, hotbar: Array(9).fill(true)
    }
  }
}
