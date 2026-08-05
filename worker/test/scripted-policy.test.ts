import { describe, expect, it } from 'vitest'
import { scriptedAction, type ScriptedStyle } from '../src/scripted-policy.js'
import {
  NOOP_ACTION,
  validateAction,
  type ActionV1,
  type BlockSlot,
  type EntitySlot,
  type ItemState,
  type ObservationV1,
  type Vec3Value
} from '../src/contracts.js'

const LEGACY_STYLES: ScriptedStyle[] = ['rush', 'strafe', 'retreat', 'jump_critical', 'defensive', 'erratic']

/**
 * Byte-for-byte copy of the melee-only teacher that shipped before crystal support. SWORD arenas
 * must keep producing exactly these actions, so the test compares against the original code
 * rather than against hand-written expectations that could drift with it.
 */
function legacyScriptedAction(observation: ObservationV1, style: ScriptedStyle): ActionV1 {
  const opponent = observation.opponent
  if (!opponent) return { ...NOOP_ACTION, yaw_delta: 0.15 }
  const relative = opponent.relative_position
  const horizontal = Math.hypot(relative.x, relative.z)
  const yawDelta = clamp(Math.atan2(-relative.x, -relative.z), -0.8, 0.8)
  const pitchDelta = clamp(Math.atan2(relative.y, Math.max(0.1, horizontal)) - observation.self.pitch, -0.4, 0.4)
  const tick = observation.match.tick
  let forward: -1 | 0 | 1 = horizontal > 2.75 ? 1 : 0
  let strafe: -1 | 0 | 1 = tick % 80 < 40 ? -1 : 1
  if (style === 'rush') strafe = 0
  if (style === 'retreat' || style === 'defensive') forward = horizontal < 4.5 ? -1 : 0
  if (style === 'erratic') strafe = tick % 17 < 8 ? -1 : 1
  return {
    ...NOOP_ACTION,
    forward,
    strafe,
    sprint: forward === 1,
    jump: (style === 'erratic' && tick % 23 === 0) || (style === 'jump_critical' && tick % 13 === 0),
    yaw_delta: yawDelta,
    pitch_delta: pitchDelta,
    primary: horizontal <= 3.0 && observation.self.attack_cooldown >= 0.9 ? 'attack' : 'none'
  }
}

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value))
}

function item(name: string, count = 1): ItemState {
  return { name, count, durability: 0, max_durability: 0, enchant_hash: 0 }
}

const EMPTY_ITEM: ItemState = item('', 0)

/** Mirrors Arena.equipKit for the crystal-capable modes. */
function crystalHotbar(): ItemState[] {
  return [
    item('diamond_sword'),
    item('diamond_pickaxe'),
    item('obsidian', 64),
    item('end_crystal', 64),
    item('golden_apple', 16),
    item('bow'),
    item('ender_pearl', 16),
    item('totem_of_undying'),
    item('totem_of_undying')
  ]
}

function swordHotbar(): ItemState[] {
  return [item('diamond_sword'), ...Array.from({ length: 8 }, () => ({ ...EMPTY_ITEM }))]
}

function vec(x: number, y: number, z: number): Vec3Value {
  return { x, y, z }
}

function block(relative: Vec3Value, overrides: Partial<BlockSlot> = {}): BlockSlot {
  const distance = Math.hypot(relative.x, relative.y, relative.z)
  return {
    name: 'obsidian',
    relative_position: relative,
    collision: 'solid',
    hardness: 50,
    replaceable: false,
    break_progress: 0,
    crystal_clearance: false,
    exposed_faces: 1,
    distance,
    within_reach: distance <= 5,
    raycastable: false,
    sample_age_ticks: 0,
    ...overrides
  }
}

function crystal(relative: Vec3Value, overrides: Partial<EntitySlot> = {}): EntitySlot {
  return {
    kind: 'end_crystal',
    relative_position: relative,
    relative_velocity: vec(0, 0, 0),
    age_ticks: 4,
    distance: Math.hypot(relative.x, relative.y, relative.z),
    raycastable: false,
    ...overrides
  }
}

type Scenario = {
  tick?: number
  pitch?: number
  yaw?: number
  health?: number
  cooldown?: number
  selected?: number
  hotbar?: ItemState[]
  offhand?: ItemState
  opponent?: Vec3Value | null
  entities?: EntitySlot[]
  blocks?: BlockSlot[]
}

function observationOf(scenario: Scenario = {}): ObservationV1 {
  const opponentRelative = scenario.opponent === null ? null : scenario.opponent ?? vec(0, 0, -3)
  return {
    schema_version: 1,
    match: {
      episode_id: 'test',
      tick: scenario.tick ?? 20,
      policy_version: 0,
      arena_seed: 0,
      action_delay_ticks: 0,
      observation_delay_ticks: 0
    },
    self: {
      health: scenario.health ?? 20,
      absorption: 0,
      food: 20,
      position: vec(0, 64, 0),
      velocity: vec(0, 0, 0),
      yaw: scenario.yaw ?? 0,
      pitch: scenario.pitch ?? 0,
      on_ground: true,
      sprinting: false,
      sneaking: false,
      hurt_time: 0,
      attack_cooldown: scenario.cooldown ?? 1,
      active_hand: 'none',
      use_ticks: 0,
      mining_progress: 0,
      selected_hotbar: scenario.selected ?? 0,
      hotbar: scenario.hotbar ?? crystalHotbar(),
      offhand: scenario.offhand ?? item('totem_of_undying'),
      armor: [],
      raycast: { kind: 'none', distance: 0, block_name: '', entity_kind: '' }
    },
    opponent: opponentRelative === null ? null : {
      relative_position: opponentRelative,
      relative_velocity: vec(0, 0, 0),
      yaw: 0,
      pitch: 0,
      health: 20,
      hurt_time: 0,
      on_ground: true,
      line_of_sight: true,
      mainhand: item('diamond_sword'),
      offhand: item('totem_of_undying'),
      armor: []
    },
    entities: scenario.entities ?? [],
    blocks: scenario.blocks ?? [],
    action_mask: {
      attack: true,
      use_main: true,
      use_offhand: true,
      release_use: false,
      swap_offhand: true,
      hotbar: Array.from({ length: 9 }, () => true)
    }
  }
}

/** Obsidian with clear air above it, one block to the bot's left and beside the opponent. */
function clearanceSpotLeftOfOpponent(): BlockSlot {
  return block(vec(-1, -1, -4), { crystal_clearance: true })
}

describe('scripted fallback', () => {
  it('attacks an opponent inside reach only after cooldown', () => {
    const observation = {
      match: { tick: 20 },
      self: { pitch: 0, attack_cooldown: 1 },
      opponent: { relative_position: { x: 0, y: 0, z: -2.5 } }
    } as ObservationV1
    expect(scriptedAction(observation).primary).toBe('attack')
  })

  it('idles with a scanning turn when there is no opponent', () => {
    expect(scriptedAction(observationOf({ opponent: null }))).toEqual({ ...NOOP_ACTION, yaw_delta: 0.15 })
  })
})

describe('sword mode no-regression', () => {
  it('reproduces the melee teacher exactly when the kit has no obsidian or crystals', () => {
    const distances = [1.5, 2.5, 3.0, 4.0, 6.0]
    const laterals = [-2, 0, 1.5]
    for (const style of LEGACY_STYLES) {
      for (let tick = 0; tick < 90; tick += 1) {
        for (const distance of distances) {
          for (const lateral of laterals) {
            const observation = observationOf({
              tick,
              hotbar: swordHotbar(),
              offhand: { ...EMPTY_ITEM },
              opponent: vec(lateral, 0.2, -distance),
              cooldown: tick % 3 === 0 ? 1 : 0.4,
              pitch: 0.1,
              // Crystal-shaped context must be ignored outright in SWORD mode.
              entities: [crystal(vec(0, 0, -2))],
              blocks: [clearanceSpotLeftOfOpponent()]
            })
            expect(scriptedAction(observation, style)).toEqual(legacyScriptedAction(observation, style))
          }
        }
      }
    }
  })

  it('stays melee when the kit has crystals but no obsidian to build a base', () => {
    const hotbar = crystalHotbar()
    hotbar[2] = { ...EMPTY_ITEM }
    const observation = observationOf({ hotbar, blocks: [clearanceSpotLeftOfOpponent()] })
    expect(scriptedAction(observation, 'strafe')).toEqual(legacyScriptedAction(observation, 'strafe'))
  })
})

describe('crystal placement', () => {
  it('selects the crystal slot before placing and then places on the pulse tick', () => {
    const blocks = [clearanceSpotLeftOfOpponent()]
    const selecting = scriptedAction(observationOf({ blocks, selected: 0, tick: 20 }), 'strafe')
    expect(selecting.hotbar).toBe(3)
    expect(selecting.primary).toBe('none')

    const placing = scriptedAction(observationOf({ blocks, selected: 3, tick: 20 }), 'strafe')
    // Redundant switches are suppressed: the observation already reports the crystal selected.
    expect(placing.hotbar).toBe(-1)
    expect(placing.primary).toBe('use_main')
    expect(placing.jump).toBe(false)
  })

  it('alternates use_main so the edge-triggered control adapter re-arms', () => {
    const blocks = [clearanceSpotLeftOfOpponent()]
    const odd = scriptedAction(observationOf({ blocks, selected: 3, tick: 21 }), 'strafe')
    expect(odd.primary).toBe('none')
    const even = scriptedAction(observationOf({ blocks, selected: 3, tick: 22 }), 'strafe')
    expect(even.primary).toBe('use_main')
  })

  it('aims left for a spot at negative x and right for a spot at positive x', () => {
    // Forward is -z and the frame is yaw-invariant, so yaw = atan2(-x, -z): a target left of the
    // bot (negative x) needs a positive yaw delta.
    const left = scriptedAction(observationOf({
      blocks: [block(vec(-2, -1, -3), { crystal_clearance: true })],
      selected: 3
    }), 'strafe')
    const right = scriptedAction(observationOf({
      blocks: [block(vec(1, -1, -3), { crystal_clearance: true })],
      selected: 3
    }), 'strafe')
    expect(left.yaw_delta).toBeGreaterThan(0.1)
    expect(right.yaw_delta).toBeLessThan(-0.1)
    // The spot is on the floor, so the teacher must look down (pitch is positive-up).
    expect(left.pitch_delta).toBeLessThan(0)
    expect(right.pitch_delta).toBeLessThan(0)
  })

  it('aims at the block top face measured from the eye, not the feet', () => {
    const observation = observationOf({
      blocks: [block(vec(0, -1, -3), { crystal_clearance: true })],
      opponent: vec(0, 0, -4),
      selected: 3
    })
    const action = scriptedAction(observation, 'strafe')
    // Aim point is the top-face centre of the block, i.e. corner + (0.5, 0.95, 0.5), and the
    // pitch is measured from the eye at 1.62 rather than from the feet.
    const horizontal = Math.hypot(0.5, 2.5)
    expect(action.pitch_delta).toBeCloseTo(Math.atan2(-1 + 0.95 - 1.62, horizontal), 5)
    expect(action.yaw_delta).toBeCloseTo(Math.atan2(-0.5, 2.5), 5)
  })

  it('rotates the block-centre offset with the reported yaw', () => {
    // At yaw = -pi/2 the egocentric frame is rotated, so the world-axis (0.5, 0.5) centre offset
    // has to rotate with it. Without the rotation the aim lands off the block.
    const yaw = -Math.PI / 2
    const observation = observationOf({
      yaw,
      blocks: [block(vec(0, -1, -3), { crystal_clearance: true })],
      opponent: vec(0, 0, -4),
      selected: 3
    })
    const action = scriptedAction(observation, 'strafe')
    const sin = Math.sin(yaw)
    const cos = Math.cos(yaw)
    const offset = { x: 0.5 * cos - 0.5 * sin, z: 0.5 * sin + 0.5 * cos }
    expect(action.yaw_delta).toBeCloseTo(Math.atan2(-offset.x, 3 - offset.z), 5)
  })

  it('never builds a bomb under its own feet', () => {
    const underfoot = block(vec(0, -1, 0), { crystal_clearance: true })
    const observation = observationOf({ blocks: [underfoot], opponent: vec(0, 0, -2), selected: 3 })
    const action = scriptedAction(observation, 'strafe')
    expect(action.primary).not.toBe('use_main')
  })

  it('prefers a spot beside the opponent over one at its own feet', () => {
    const observation = observationOf({
      blocks: [block(vec(0, -1, 0), { crystal_clearance: true }), clearanceSpotLeftOfOpponent()],
      selected: 3,
      tick: 20
    })
    const action = scriptedAction(observation, 'strafe')
    expect(action.primary).toBe('use_main')
    // The chosen spot is the one to the left of the opponent, four blocks out.
    expect(action.yaw_delta).toBeCloseTo(Math.atan2(0.5, 3.5), 5)
  })

  it('skips spots the opponent is nowhere near', () => {
    const observation = observationOf({
      blocks: [block(vec(4, -1, 1), { crystal_clearance: true })],
      opponent: vec(0, 0, -3),
      selected: 3
    })
    expect(scriptedAction(observation, 'strafe').primary).not.toBe('use_main')
  })

  it('skips a spot that already holds a crystal', () => {
    const observation = observationOf({
      blocks: [clearanceSpotLeftOfOpponent()],
      // Same base, crystal already sitting on it, but too far from the opponent's body to pop.
      entities: [crystal(vec(-0.5, 0, -3.5), { distance: 9 })],
      selected: 3,
      tick: 20
    })
    expect(scriptedAction(observation, 'strafe').primary).not.toBe('use_main')
  })
})

describe('obsidian scaffolding', () => {
  const support = block(vec(-1, -1, -3), { name: 'stone', hardness: 1.5 })
  const airAbove = block(vec(-1, 0, -3), { name: 'air', collision: 'empty', replaceable: true, hardness: 0 })

  it('places obsidian beside the opponent when no legal crystal spot exists', () => {
    const selecting = scriptedAction(observationOf({ blocks: [support, airAbove], selected: 0 }), 'strafe')
    expect(selecting.hotbar).toBe(2)
    expect(selecting.primary).toBe('none')

    const placing = scriptedAction(observationOf({ blocks: [support, airAbove], selected: 2, tick: 20 }), 'strafe')
    expect(placing.hotbar).toBe(-1)
    expect(placing.primary).toBe('use_main')
    // Obsidian goes on the top face: aim at (-0.5, 0.95, -2.5) from the eye.
    expect(placing.yaw_delta).toBeCloseTo(Math.atan2(0.5, 2.5), 5)
  })

  it('refuses a support block that is covered', () => {
    // A slab on top leaves no room for obsidian, and being 'partial' it is not itself a base the
    // teacher could stack on, so the only candidate in the observation is unusable.
    const covered = block(vec(-1, 0, -3), { name: 'stone_slab', collision: 'partial' })
    const action = scriptedAction(observationOf({ blocks: [support, covered], selected: 2, tick: 20 }), 'strafe')
    expect(action.primary).not.toBe('use_main')
  })

  it('prefers an existing crystal spot over building a new one', () => {
    const observation = observationOf({
      blocks: [support, airAbove, clearanceSpotLeftOfOpponent()],
      selected: 0
    })
    expect(scriptedAction(observation, 'strafe').hotbar).toBe(3)
  })
})

describe('detonation', () => {
  it('attacks a crystal that is closer to the opponent than to itself', () => {
    const observation = observationOf({
      opponent: vec(0, 0, -3.2),
      entities: [crystal(vec(0, 0, -2.5))],
      selected: 3
    })
    const action = scriptedAction(observation, 'strafe')
    expect(action.primary).toBe('attack')
    expect(action.hotbar).toBe(-1)
    expect(action.jump).toBe(false)
    // Aim a block above the crystal origin, where its two-block hitbox is centred.
    expect(action.pitch_delta).toBeCloseTo(Math.atan2(1 - 1.62, 2.5), 5)
  })

  it('does not detonate a crystal sitting on top of the bot', () => {
    const observation = observationOf({
      opponent: vec(0, 0, -2),
      entities: [crystal(vec(0, 0, -1))],
      blocks: [clearanceSpotLeftOfOpponent()]
    })
    const action = scriptedAction(observation, 'strafe')
    expect(action.primary).not.toBe('attack')
    expect(action.forward).toBe(-1)
    expect(action.jump).toBe(false)
    expect(action.sprint).toBe(false)
  })

  it('does not detonate a crystal that would hurt it more than the opponent', () => {
    const observation = observationOf({
      opponent: vec(0, 0, -6),
      entities: [crystal(vec(0, 0, -2))]
    })
    const action = scriptedAction(observation, 'strafe')
    expect(action.primary).not.toBe('attack')
    expect(action.forward).toBe(-1)
  })

  it('ignores a crystal beyond the adapter attack reach', () => {
    const observation = observationOf({
      opponent: vec(0, 0, -5),
      entities: [crystal(vec(0, 0, -4.2))]
    })
    expect(scriptedAction(observation, 'strafe').primary).not.toBe('attack')
  })

  it('detonates before it places, even with a free spot available', () => {
    const observation = observationOf({
      opponent: vec(0, 0, -3.2),
      entities: [crystal(vec(0, 0, -2.5))],
      blocks: [clearanceSpotLeftOfOpponent()],
      selected: 3,
      tick: 20
    })
    expect(scriptedAction(observation, 'strafe').primary).toBe('attack')
  })
})

describe('melee interleaving', () => {
  it('keeps swinging with melee-first styles when the sword is off cooldown', () => {
    const observation = observationOf({
      opponent: vec(0, 0, -2.5),
      blocks: [clearanceSpotLeftOfOpponent()],
      selected: 0,
      cooldown: 1
    })
    const action = scriptedAction(observation, 'rush')
    expect(action.primary).toBe('attack')
    expect(action.hotbar).toBe(-1)
  })

  it('lets spacing styles crystal through the same window', () => {
    const observation = observationOf({
      opponent: vec(0, 0, -2.5),
      blocks: [clearanceSpotLeftOfOpponent()],
      selected: 3,
      cooldown: 1,
      tick: 20
    })
    expect(scriptedAction(observation, 'strafe').primary).toBe('use_main')
  })

  it('waits for the sword slot before swinging', () => {
    const observation = observationOf({ opponent: vec(0, 0, -2.5), selected: 2, cooldown: 1 })
    const action = scriptedAction(observation, 'rush')
    expect(action.hotbar).toBe(0)
    expect(action.primary).toBe('none')
  })

  it('holds fire while the cooldown is charging', () => {
    const observation = observationOf({ opponent: vec(0, 0, -2.5), selected: 0, cooldown: 0.3 })
    expect(scriptedAction(observation, 'rush').primary).toBe('none')
  })
})

describe('totem management', () => {
  it('swaps a hotbar totem into an empty offhand when low', () => {
    const selecting = scriptedAction(observationOf({
      health: 6,
      offhand: { ...EMPTY_ITEM },
      selected: 0
    }), 'strafe')
    expect(selecting.hotbar).toBe(7)
    expect(selecting.swap_offhand).toBe(false)

    const swapping = scriptedAction(observationOf({
      health: 6,
      offhand: { ...EMPTY_ITEM },
      selected: 7,
      tick: 20
    }), 'strafe')
    expect(swapping.swap_offhand).toBe(true)
    expect(swapping.hotbar).toBe(-1)
  })

  it('leaves a healthy bot or an armed offhand alone', () => {
    const healthy = scriptedAction(observationOf({ health: 20, offhand: { ...EMPTY_ITEM }, selected: 0 }), 'strafe')
    expect(healthy.swap_offhand).toBe(false)
    const armed = scriptedAction(observationOf({ health: 4, selected: 0 }), 'strafe')
    expect(armed.swap_offhand).toBe(false)
  })
})

describe('action legality', () => {
  it('emits schema-valid actions across every style and situation', () => {
    const styles: ScriptedStyle[] = [...LEGACY_STYLES, 'crystal', 'crystal_melee']
    const situations: Scenario[] = [
      {},
      { blocks: [clearanceSpotLeftOfOpponent()] },
      { blocks: [block(vec(-1, -1, -3), { name: 'stone' })] },
      { entities: [crystal(vec(0, 0, -2.5))], opponent: vec(0, 0, -3.2) },
      { entities: [crystal(vec(0, 0, -1))], opponent: vec(0, 0, -2) },
      { health: 5, offhand: { ...EMPTY_ITEM } },
      { hotbar: swordHotbar(), offhand: { ...EMPTY_ITEM } },
      { opponent: null }
    ]
    for (const style of styles) {
      for (const situation of situations) {
        for (const tick of [0, 1, 13, 20, 21]) {
          for (const yaw of [0, 1.2, -2.4]) {
            for (const selected of [0, 2, 3, 7]) {
              const action = scriptedAction(observationOf({ ...situation, tick, yaw, selected }), style)
              expect(() => validateAction(action)).not.toThrow()
              expect(Math.abs(action.yaw_delta)).toBeLessThanOrEqual(Math.PI)
            }
          }
        }
      }
    }
  })
})
