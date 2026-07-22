import type { Bot } from 'mineflayer'
import { Vec3 } from 'vec3'
import { describe, expect, it, vi } from 'vitest'
import {
  attackActionLegal,
  combatAttackReady,
  crystalAttackReady,
  crystalPlaceReady,
  hasAssignedOpponentLineOfSight,
  ObservationBuilder,
  tacticalBlockPlaceReady,
  tacticalBlockBreakReady,
  type ControlTelemetry
} from '../src/observation.js'
import {
  assignedOpponentMeleeTarget,
  findAssignedOpponent,
  isLegalArenaAttackTarget,
  legalArenaAttackTargetAtCrosshair
} from '../src/targeting.js'
import { TacticalPlacementTracker } from '../src/tactical-blocks.js'

function player(id: number, username: string, x: number, z = 0): any {
  return {
    id,
    type: 'player',
    username,
    position: new Vec3(x, 64, z),
    width: 0.6,
    height: 1.8,
    velocity: new Vec3(0, 0, 0),
    yaw: 0,
    pitch: 0,
    onGround: true,
    equipment: []
  }
}

function fakeBot(cursor: any, cursorBlock: any = null): Bot {
  const spectator = player(2, 'HumanSpectator', 1)
  const unrelatedBot = player(3, 'MCAI_003', 2)
  const assignedBot = cursor?.username === 'MCAI_002' ? cursor : player(4, 'MCAI_002', 8)
  return {
    username: 'MCAI_001',
    entity: {
      id: 1,
      type: 'player',
      username: 'MCAI_001',
      position: new Vec3(0, 64, 0),
      velocity: new Vec3(0, 0, 0),
      yaw: 0,
      pitch: 0,
      eyeHeight: 1.62,
      onGround: true
    },
    entities: { 2: spectator, 3: unrelatedBot, 4: assignedBot },
    players: {
      HumanSpectator: { entity: spectator },
      MCAI_003: { entity: unrelatedBot },
      MCAI_002: { entity: assignedBot }
    },
    health: 20,
    food: 20,
    quickBarSlot: 0,
    heldItem: null,
    inventory: { slots: Array.from({ length: 46 }) },
    getControlState: () => false,
    entityAtCursor: () => cursor,
    blockAtCursor: () => cursorBlock,
    blockAt: () => null,
    canSeeEntity: () => true
  } as unknown as Bot
}

const READY_CONTROL: ControlTelemetry = {
  lastAttackTick: -1000,
  activeHand: 'none',
  useStartedTick: 0,
  miningProgress: 0
}

function observe(builder: ObservationBuilder, tick: number, episodeId = 'episode-geometry') {
  return builder.build({
    episode_id: episodeId,
    tick,
    policy_version: 0,
    arena_seed: 7,
    action_delay_ticks: 0,
    observation_delay_ticks: 0
  }, READY_CONTROL)
}

describe('arena-assigned targeting', () => {
  it('selects only the assigned opponent, even when a spectator and another bot are closer', () => {
    const bot = fakeBot(null)
    expect(findAssignedOpponent(bot, 'MCAI_002')?.username).toBe('MCAI_002')
    expect(findAssignedOpponent(bot, null)).toBeNull()
    expect(findAssignedOpponent(bot, 'missing')).toBeNull()
  })

  it('trusts only the assigned roster entry while its spawned entity username is pending', () => {
    const bot = fakeBot(null)
    const assigned = (bot.players as any).MCAI_002.entity
    delete assigned.username

    expect(findAssignedOpponent(bot, 'MCAI_002')).toBe(assigned)
    expect(legalArenaAttackTargetAtCrosshair(bot, 'MCAI_002', 3.4)).toBeNull()

    assigned.position = new Vec3(0, 64, -2.5)
    expect(legalArenaAttackTargetAtCrosshair(bot, 'MCAI_002', 3.4)).toBe(assigned)
    expect(assignedOpponentMeleeTarget(bot, 'MCAI_002', 3.4)).toBe(assigned)
    expect(legalArenaAttackTargetAtCrosshair(bot, 'HumanSpectator', 3.4)).toBeNull()
  })

  it('excludes the spectator from both opponent state and raycast input', () => {
    const spectator = player(2, 'HumanSpectator', 1)
    const bot = fakeBot(spectator)
    const builder = new ObservationBuilder(bot)
    builder.setOpponentUsername('MCAI_002')
    const observation = builder.build({
      episode_id: 'episode-1',
      tick: 1,
      policy_version: 0,
      arena_seed: 1,
      action_delay_ticks: 0,
      observation_delay_ticks: 0
    }, {
      lastAttackTick: -1000,
      activeHand: 'none',
      useStartedTick: 0,
      miningProgress: 0
    })

    expect(observation.opponent?.relative_position.x).toBe(8)
    expect(observation.self.raycast.kind).toBe('none')
    expect(observation.entities.every(entity => entity.kind !== 'player')).toBe(true)
  })

  it.each([
    { yaw: Math.PI / 2, position: new Vec3(-3, 64, 0) },
    { yaw: -Math.PI / 2, position: new Vec3(3, 64, 0) },
    { yaw: Math.PI, position: new Vec3(0, 64, 3) }
  ])('emits a corrected body frame while preserving legacy coordinates at yaw $yaw', ({ yaw, position }) => {
    const assigned = player(4, 'MCAI_002', position.x, position.z)
    const bot = fakeBot(assigned)
    bot.entity.yaw = yaw
    const builder = new ObservationBuilder(bot)
    builder.setOpponentUsername('MCAI_002')

    const opponent = observe(builder, 1).opponent!
    expect(opponent.body_relative_position.x).toBeCloseTo(0, 8)
    expect(opponent.body_relative_position.z).toBeCloseTo(-3, 8)
    expect(opponent.bearing_error).toBeCloseTo(0, 8)
    // The compatibility field deliberately retains the historical wrong sign.
    if (Math.abs(yaw) === Math.PI / 2) expect(opponent.relative_position.z).toBeCloseTo(3, 8)
  })

  it('emits explicit distance, bearing, reach, aim, facing and derived closing speed', () => {
    const assigned = player(4, 'MCAI_002', 3, -4)
    assigned.headYaw = Math.atan2(-3, 4) + Math.PI
    const bot = fakeBot(assigned)
    const builder = new ObservationBuilder(bot)
    builder.setOpponentUsername('MCAI_002')

    const initial = observe(builder, 10).opponent!
    expect(initial.distance).toBeCloseTo(5)
    expect(initial.horizontal_distance).toBeCloseTo(5)
    expect(initial.bearing_error).toBeCloseTo(Math.atan2(-3, 4))
    expect(initial.within_melee_reach).toBe(false)
    expect(initial.aim_alignment).toBeGreaterThan(0.79)
    expect(initial.facing_toward_self).toBeGreaterThan(0.99)
    expect(initial.head_yaw).toBeCloseTo(assigned.headYaw)

    // Mineflayer leaves ordinary player entity.velocity at zero. Position
    // deltas must still expose that the opponent is closing on us.
    assigned.position = new Vec3(2.4, 64, -3.2)
    const moved = observe(builder, 11).opponent!
    expect(moved.closing_speed).toBeCloseTo(0.45, 6)
    expect(moved.body_relative_velocity.x).toBeCloseTo(-0.27, 6)
    expect(moved.body_relative_velocity.z).toBeCloseTo(0.36, 6)
    expect(moved.relative_velocity).toEqual({ x: 0, y: 0, z: 0 })
    expect(moved.within_melee_reach).toBe(false)

    assigned.position = new Vec3(1.8, 64, -2.4)
    expect(observe(builder, 12).opponent?.within_melee_reach).toBe(true)
  })

  it('computes block-occluded LOS without a Mineflayer canSeeEntity helper', () => {
    const assigned = player(4, 'MCAI_002', 0, -4)
    const bot = fakeBot(assigned)
    const raycast = vi.fn((..._arguments: any[]) => null as any)
    ;(bot as any).world = { raycast }
    expect(hasAssignedOpponentLineOfSight(bot, assigned)).toBe(true)

    raycast.mockImplementation((...arguments_: any[]) => {
      const origin = arguments_[0] as Vec3
      const direction = arguments_[1] as Vec3
      return {
        position: new Vec3(0, 65, -2),
        intersect: origin.plus(direction.scaled(2))
      }
    })
    expect(hasAssignedOpponentLineOfSight(bot, assigned)).toBe(false)
  })

  it('falls back to eye-to-hitbox block sampling for LOS', () => {
    const assigned = player(4, 'MCAI_002', 0, -4)
    const bot = fakeBot(assigned)
    ;(bot as any).world = undefined
    ;(bot as any).blockAt = (position: Vec3) => position.z === -2 && position.y === 65
      ? { name: 'stone', type: 1, boundingBox: 'block' }
      : { name: 'air', type: 0, boundingBox: 'empty' }
    expect(hasAssignedOpponentLineOfSight(bot, assigned)).toBe(false)
    ;(bot as any).blockAt = () => ({ name: 'air', type: 0, boundingBox: 'empty' })
    expect(hasAssignedOpponentLineOfSight(bot, assigned)).toBe(true)
  })

  it('does not let spectator position, velocity or camera affect new opponent geometry', () => {
    const assigned = player(4, 'MCAI_002', 0, -3)
    assigned.headYaw = Math.PI
    const bot = fakeBot(assigned)
    const builder = new ObservationBuilder(bot)
    builder.setOpponentUsername('MCAI_002')
    const before = observe(builder, 20).opponent!

    const spectator: any = (bot.players as any).HumanSpectator.entity
    spectator.position = new Vec3(0, 100, 0)
    spectator.velocity = new Vec3(12, 12, 12)
    spectator.yaw = -Math.PI / 2
    spectator.headYaw = -Math.PI / 2
    const after = observe(builder, 21).opponent!

    expect(after.body_relative_position).toEqual(before.body_relative_position)
    expect(after.distance).toBe(before.distance)
    expect(after.bearing_error).toBe(before.bearing_error)
    expect(after.head_yaw).toBe(before.head_yaw)
    expect(after.facing_toward_self).toBe(before.facing_toward_self)
  })

  it('uses only a fresh matching arena fighter snapshot for authoritative combat state', () => {
    const assigned = player(4, 'MCAI_002', 0, -4)
    assigned.headYaw = 1.25
    assigned.yaw = 0.33
    assigned.pitch = 0.1
    const bot = fakeBot(assigned)
    const builder = new ObservationBuilder(bot)
    builder.setOpponentUsername('MCAI_002')

    const authoritative = {
      episode_id: 'episode-live',
      elapsed_ticks: 40,
      fighters: [
        {
          name: 'HumanSpectator', health: 1, absorption: 20, grounded: false,
          vx: 99, vy: 99, vz: 99, yaw: 0, pitch: 90
        },
        {
          name: 'MCAI_002', health: 7, absorption: 4, grounded: false,
          vx: 0, vy: 0, vz: 0.5, yaw: 180, pitch: 30
        }
      ]
    }
    expect(builder.acceptArenaSnapshot(authoritative, 'episode-wrong', 20)).toBe(false)
    expect(builder.acceptArenaSnapshot({
      ...authoritative,
      fighters: authoritative.fighters.slice(0, 1)
    }, 'episode-live', 20)).toBe(false)
    expect(builder.acceptArenaSnapshot(authoritative, 'episode-live', 20)).toBe(true)

    const fresh = observe(builder, 22, 'episode-live').opponent!
    expect(fresh.health).toBe(7)
    expect(fresh.absorption).toBe(4)
    expect(fresh.on_ground).toBe(false)
    expect(fresh.server_state_age_ticks).toBe(2)
    expect(fresh.body_relative_velocity).toEqual({ x: 0, y: 0, z: 0.5 })
    expect(fresh.closing_speed).toBeCloseTo(0.5)
    expect(fresh.yaw).toBeCloseTo(0)
    expect(fresh.pitch).toBeCloseTo(-Math.PI / 6)
    // Position and current head aim remain local packet state.
    expect(fresh.body_relative_position).toEqual({ x: 0, y: 0, z: -4 })
    expect(fresh.head_yaw).toBe(1.25)

    // A wrong-episode or unassigned fighter update cannot overwrite the valid one.
    expect(builder.acceptArenaSnapshot({
      ...authoritative,
      episode_id: 'episode-other',
      elapsed_ticks: 41,
      fighters: [{ ...authoritative.fighters[0], name: 'MCAI_002' }]
    }, 'episode-live', 23)).toBe(false)
    expect(observe(builder, 23, 'episode-live').opponent?.health).toBe(7)

    const stale = observe(builder, 27, 'episode-live').opponent!
    expect(stale.server_state_age_ticks).toBe(7)
    expect(stale.health).toBeNull()
    expect(stale.absorption).toBe(0)
    expect(stale.on_ground).toBe(true)
    expect(stale.yaw).toBeCloseTo(0.33)
  })

  it('emits the currently selected mainhand item explicitly', () => {
    const bot = fakeBot(null)
    ;(bot as any).heldItem = { name: 'end_crystal', count: 12, maxDurability: 0, durabilityUsed: 0 }
    const observation = observe(new ObservationBuilder(bot), 1)
    expect(observation.self.mainhand.name).toBe('end_crystal')
    expect(observation.self.mainhand.count).toBe(12)
  })

  it('emits corrected body coordinates for combat entities without changing legacy slots', () => {
    const bot = fakeBot(null)
    bot.entity.yaw = Math.PI / 2
    const crystal = {
      id: 9,
      type: 'object',
      name: 'end_crystal',
      position: new Vec3(-3, 64, 0),
      velocity: new Vec3(-0.2, 0, 0),
      width: 2,
      height: 2
    }
    ;(bot.entities as any)[9] = crystal
    const slot = observe(new ObservationBuilder(bot), 1).entities
      .find(entity => entity.kind === 'end_crystal')!
    expect(slot.body_relative_position.x).toBeCloseTo(0, 8)
    expect(slot.body_relative_position.z).toBeCloseTo(-3, 8)
    expect(slot.body_relative_velocity.z).toBeCloseTo(-0.2, 8)
    expect(slot.relative_position.z).toBeCloseTo(3, 8)
  })

  it('allows attacks only on the assigned fighter or an arena crystal', () => {
    expect(isLegalArenaAttackTarget(player(2, 'HumanSpectator', 1), 'MCAI_002')).toBe(false)
    expect(isLegalArenaAttackTarget(player(3, 'MCAI_003', 1), 'MCAI_002')).toBe(false)
    expect(isLegalArenaAttackTarget(player(4, 'MCAI_002', 1), 'MCAI_002')).toBe(true)
    expect(isLegalArenaAttackTarget({ type: 'object', name: 'end_crystal' }, 'MCAI_002')).toBe(true)
  })

  it('ray-tests the assigned hitbox at server reach without selecting by proximity', () => {
    const assigned = player(4, 'MCAI_002', 0, -3.3)
    const bot = fakeBot(assigned)
    expect(legalArenaAttackTargetAtCrosshair(bot, 'MCAI_002', 3.4)).toBe(assigned)
    bot.entity.yaw = Math.PI / 2
    expect(legalArenaAttackTargetAtCrosshair(bot, 'MCAI_002', 3.4)).toBeNull()
  })

  it('requires the assigned fighter inside the melee facing cone', () => {
    const assigned = player(4, 'MCAI_002', 0, -2)
    const bot = fakeBot(assigned)
    expect(assignedOpponentMeleeTarget(bot, 'MCAI_002', 3.4)).toBe(assigned)

    // A nearby assigned fighter still does not qualify while looking sideways.
    bot.entity.yaw = Math.PI / 2
    expect(legalArenaAttackTargetAtCrosshair(bot, 'MCAI_002', 3.0)).toBeNull()
    expect(assignedOpponentMeleeTarget(bot, 'MCAI_002', 3.4)).toBeNull()

    const spectator = player(2, 'HumanSpectator', 0, -1)
    expect(assignedOpponentMeleeTarget(fakeBot(spectator), 'MCAI_002', 3.4)).toBeNull()
  })

  it('ray-tests crystals even though Mineflayer classifies them as objects', () => {
    const crystal = {
      id: 9, type: 'object', name: 'end_crystal', position: new Vec3(0, 64, -3), width: 2, height: 2
    }
    const bot = fakeBot(null)
    ;(bot.entities as any)[9] = crystal
    expect(legalArenaAttackTargetAtCrosshair(bot, 'MCAI_002', 3.4)).toBe(crystal)
    expect(assignedOpponentMeleeTarget(bot, 'MCAI_002', 3.4)).toBeNull()
    expect(crystalAttackReady(bot, {
      lastAttackTick: 0, activeHand: 'none', useStartedTick: 0, miningProgress: 0
    }, 20, 'MCAI_002')).toBe(true)
  })

  it('advertises a clear top-face crystal base without requiring crystals to be selected yet', () => {
    const base = {
      name: 'obsidian', position: new Vec3(0, 63, -2), face: 1,
      intersect: new Vec3(0.5, 64, -1.5), diggable: true
    }
    const bot = fakeBot(null, base)
    const slots = bot.inventory.slots as any[]
    slots[36] = { name: 'diamond_sword', count: 1 }
    slots[39] = { name: 'end_crystal', count: 64 }
    ;(bot as any).heldItem = slots[36]
    ;(bot as any).blockAt = (position: Vec3) => [64, 65].includes(position.y)
      ? { name: 'air', type: 0, boundingBox: 'empty' }
      : null

    expect(crystalPlaceReady(bot)).toBe(true)
    base.face = 3
    expect(crystalPlaceReady(bot)).toBe(false)
    base.face = 1
    ;(bot as any).blockAt = (position: Vec3) => position.y === 64 ? { name: 'stone', boundingBox: 'block' } : null
    expect(crystalPlaceReady(bot)).toBe(false)
  })

  it('does not advertise a placement whose crystal would be outside reliable detonation reach', () => {
    const base = {
      name: 'obsidian', position: new Vec3(0, 63, -4), face: 1,
      intersect: new Vec3(0.5, 64, -3.5), diggable: true
    }
    const bot = fakeBot(null, base)
    const slots = bot.inventory.slots as any[]
    slots[39] = { name: 'end_crystal', count: 64 }
    ;(bot as any).blockAt = (position: Vec3) => [64, 65].includes(position.y)
      ? { name: 'air', type: 0, boundingBox: 'empty' }
      : null

    expect(crystalPlaceReady(bot)).toBe(false)
  })

  it('masks empty hotbar slots and exposes separate crystal readiness fields', () => {
    const base = {
      name: 'bedrock', position: new Vec3(0, 63, -2), face: 1,
      intersect: new Vec3(0.5, 64, -1.5), diggable: false
    }
    const bot = fakeBot(null, base)
    const slots = bot.inventory.slots as any[]
    slots[36] = { name: 'diamond_sword', count: 1 }
    slots[39] = { name: 'end_crystal', count: 64 }
    ;(bot as any).heldItem = slots[36]
    ;(bot as any).blockAt = (position: Vec3) => [64, 65].includes(position.y)
      ? { name: 'air', type: 0, boundingBox: 'empty' }
      : null
    const builder = new ObservationBuilder(bot)
    builder.setOpponentUsername('MCAI_002')
    const observation = builder.build({
      episode_id: 'crystal-mask', tick: 20, policy_version: 0, arena_seed: 2,
      action_delay_ticks: 0, observation_delay_ticks: 0
    }, {
      lastAttackTick: 0, activeHand: 'none', useStartedTick: 0, miningProgress: 0
    })

    expect(observation.action_mask.hotbar).toEqual([
      true, false, false, true, false, false, false, false, false
    ])
    expect(observation.action_mask.crystal_place_ready).toBe(true)
    expect(observation.action_mask.crystal_attack_ready).toBe(false)
    expect(observation.action_mask.use_main).toBe(true)
  })

  it('does not advertise meaningless sword/totem use or arbitrary offhand swaps', () => {
    const bot = fakeBot(null)
    const slots = bot.inventory.slots as any[]
    slots[36] = { name: 'diamond_sword', count: 1 }
    slots[45] = { name: 'totem_of_undying', count: 1 }
    ;(bot as any).heldItem = slots[36]
    const builder = new ObservationBuilder(bot)

    const ordinary = observe(builder, 1).action_mask
    expect(ordinary.use_main).toBe(false)
    expect(ordinary.use_offhand).toBe(false)
    expect(ordinary.swap_offhand).toBe(false)

    slots[40] = { name: 'golden_apple', count: 4 }
    ;(bot as any).heldItem = slots[40]
    ;(bot as any).quickBarSlot = 4
    expect(observe(builder, 2).action_mask.use_main).toBe(true)

    slots[45] = null
    ;(bot as any).heldItem = { name: 'totem_of_undying', count: 1 }
    expect(observe(builder, 3).action_mask.swap_offhand).toBe(true)
  })

  it('allows exact same-step obsidian selection/use for bounded distinct terrain targets', () => {
    const floor = {
      name: 'stone', type: 1, boundingBox: 'block', position: new Vec3(0, 63, -3), face: 1
    }
    const bot = fakeBot(null, floor)
    const slots = bot.inventory.slots as any[]
    slots[36] = { name: 'diamond_sword', count: 1 }
    slots[38] = { name: 'obsidian', count: 64 }
    ;(bot as any).heldItem = slots[36]
    ;(bot as any).quickBarSlot = 0
    ;(bot.entity as any).position = new Vec3(0.5, 64, 0.5)
    const assigned = (bot.players as any).MCAI_002.entity
    assigned.position = new Vec3(0.5, 64, -5.5)
    const placedTargets = new Set<string>()
    ;(bot as any).blockAt = (position: Vec3) => {
      if (placedTargets.has(`${position.x},${position.y},${position.z}`)) {
        return {
          name: 'obsidian', type: 49, boundingBox: 'block', hardness: 50,
          position: position.clone()
        }
      }
      if (position.y === 63) return { ...floor, position: position.clone() }
      return { name: 'air', type: 0, boundingBox: 'empty', position: position.clone() }
    }
    const tracker = new TacticalPlacementTracker(bot)
    tracker.beginEpisode('terrain-use')
    const selected = tracker.resolve('MCAI_002')!
    floor.position = selected.referenceBlock.position.clone()
    expect(tacticalBlockPlaceReady(bot, selected)).toBe(true)
    const builder = new ObservationBuilder(bot, tracker)
    builder.setOpponentUsername('MCAI_002')
    const context = {
      episode_id: 'terrain-use', tick: 1, policy_version: 0, arena_seed: 1,
      action_delay_ticks: 0, observation_delay_ticks: 0, mode: 'terrain' as const
    }
    const first = builder.build(context, READY_CONTROL)
    const marker = first.blocks.find(block => block.tactical_placement_target)
    expect(marker).toBeDefined()
    expect(tacticalBlockPlaceReady(bot, selected)).toBe(true)
    expect(first.action_mask.use_main).toBe(true)
    expect(builder.build({ ...context, mode: 'combined' }, READY_CONTROL).action_mask.use_main).toBe(false)

    const firstTarget = new Vec3(
      bot.entity.position.x + marker!.relative_position.x,
      bot.entity.position.y + marker!.relative_position.y + 1,
      bot.entity.position.z + marker!.relative_position.z
    )
    placedTargets.add(`${firstTarget.x},${firstTarget.y},${firstTarget.z}`)
    ;(bot.entity as any).position.z -= 2
    assigned.position.z -= 2
    const after = builder.build({ ...context, tick: 2 }, READY_CONTROL)
    const secondMarker = after.blocks.find(block => block.tactical_placement_target)
    expect(secondMarker).toBeDefined()
    const secondWorldTarget = new Vec3(
      bot.entity.position.x + secondMarker!.relative_position.x,
      bot.entity.position.y + secondMarker!.relative_position.y + 1,
      bot.entity.position.z + secondMarker!.relative_position.z
    )
    expect(secondWorldTarget).not.toEqual(firstTarget)
    // The crosshair still points at the first support, so only the stable new
    // target is visible; the click remains gated until the policy aims at it.
    expect(after.action_mask.use_main).toBe(false)

    floor.position = new Vec3(
      bot.entity.position.x + secondMarker!.relative_position.x,
      bot.entity.position.y + secondMarker!.relative_position.y,
      bot.entity.position.z + secondMarker!.relative_position.z
    )
    const secondReady = builder.build({ ...context, tick: 3 }, READY_CONTROL)
    expect(secondReady.action_mask.use_main).toBe(true)
    const secondTarget = floor.position.offset(0, 1, 0)
    placedTargets.add(`${secondTarget.x},${secondTarget.y},${secondTarget.z}`)

    const complete = builder.build({ ...context, tick: 4 }, READY_CONTROL)
    expect(complete.blocks.every(block => !block.tactical_placement_target)).toBe(true)
    expect(complete.action_mask.use_main).toBe(false)
  })

  it('advertises combat readiness only for a charged swing at a legal crosshair target', () => {
    const charged: ControlTelemetry = {
      lastAttackTick: 8,
      activeHand: 'none',
      useStartedTick: 0,
      miningProgress: 0
    }
    const assigned = player(4, 'MCAI_002', 0, -2.5)
    expect(combatAttackReady(fakeBot(assigned), charged, 20, 'MCAI_002')).toBe(true)
    expect(combatAttackReady(fakeBot(assigned), charged, 19, 'MCAI_002')).toBe(false)
    expect(combatAttackReady(fakeBot(player(2, 'HumanSpectator', 0, -2)), charged, 20, 'MCAI_002')).toBe(false)
    expect(combatAttackReady(
      (() => {
        const crystal = { id: 9, type: 'object', name: 'end_crystal', position: new Vec3(0, 64, -2), width: 2, height: 2 }
        const bot = fakeBot(null); (bot.entities as any)[9] = crystal; return bot
      })(), charged, 20, 'MCAI_002'
    )).toBe(false)
    const crystalBot = (() => {
      const crystal = { id: 9, type: 'object', name: 'end_crystal', position: new Vec3(0, 64, -2), width: 2, height: 2 }
      const bot = fakeBot(null); (bot.entities as any)[9] = crystal; return bot
    })()
    expect(crystalAttackReady(crystalBot, charged, 20, 'MCAI_002')).toBe(true)
    expect(crystalAttackReady(crystalBot, charged, 19, 'MCAI_002')).toBe(false)
  })

  it('keeps block mining legal without applying the combat attack prior', () => {
    const coolingDown: ControlTelemetry = {
      lastAttackTick: 18,
      activeHand: 'none',
      useStartedTick: 0,
      miningProgress: 0
    }
    const mineable = { name: 'obsidian', position: new Vec3(4, 64, 0), diggable: true }
    const miningBot = fakeBot(null, mineable)
    ;(miningBot as any).blockAt = (position: Vec3) => position.equals(mineable.position)
      ? mineable
      : { name: 'air', type: 0, position: position.clone(), boundingBox: 'empty' }
    expect(tacticalBlockBreakReady(miningBot, 'MCAI_002')).toBe(true)
    expect(attackActionLegal(miningBot, coolingDown, 20, 'MCAI_002')).toBe(true)
    expect(combatAttackReady(miningBot, coolingDown, 20, 'MCAI_002')).toBe(false)
    const builder = new ObservationBuilder(miningBot)
    builder.setOpponentUsername('MCAI_002')
    expect(builder.build({
      episode_id: 'tactical-break-mask', tick: 20, policy_version: 0, arena_seed: 3,
      action_delay_ticks: 0, observation_delay_ticks: 0
    }, coolingDown).action_mask.tactical_block_break_ready).toBe(true)

    const protectedBase = { name: 'obsidian', position: new Vec3(4, 63, 0), diggable: true }
    const protectedBot = fakeBot(null, protectedBase)
    ;(protectedBot as any).blockAt = (position: Vec3) => ({
      name: 'air', type: 0, position: position.clone(), boundingBox: 'empty'
    })
    expect(tacticalBlockBreakReady(protectedBot, 'MCAI_002')).toBe(false)
    expect(attackActionLegal(protectedBot, coolingDown, 20, 'MCAI_002')).toBe(false)

    const assigned = player(4, 'MCAI_002', 0, -2.5)
    expect(attackActionLegal(fakeBot(assigned, mineable), coolingDown, 20, 'MCAI_002')).toBe(false)
  })
})
