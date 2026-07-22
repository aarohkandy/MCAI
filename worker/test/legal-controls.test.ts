import type { Bot } from 'mineflayer'
import { Vec3 } from 'vec3'
import { describe, expect, it, vi } from 'vitest'
import {
  COMBINED_CRYSTAL_DEMO_PERIOD_TICKS,
  COMBINED_TACTICAL_BLOCK_DEMO_PERIOD_TICKS,
  EXTREME_PITCH_RECENTER_TICKS,
  LegalControlAdapter,
  MAX_TEACHER_CONTROL_TICKS,
  MAX_COMBAT_PITCH,
  MAX_CRYSTAL_PITCH,
  combinedCrystalDemoDelay,
  combinedRescueDelay,
  combinedTacticalBlockDemoDelay,
  SWORD_BOOTCAMP_ATTACK_COOLDOWN_TICKS
} from '../src/legal-controls.js'
import { ObservationBuilder } from '../src/observation.js'
import { TacticalPlacementTracker } from '../src/tactical-blocks.js'

function fighter(id: number, username: string, x: number): any {
  return {
    id, type: 'player', username, position: new Vec3(x, 64, 0),
    velocity: new Vec3(0, 0, 0), width: 0.6, height: 1.8
  }
}

function bootcampBot(): { bot: Bot; assigned: any; spectator: any; attack: ReturnType<typeof vi.fn> } {
  const assigned = fighter(2, 'MCAI_002', 3)
  const spectator = fighter(3, 'HumanSpectator', 0.5)
  const attack = vi.fn()
  const slots = Array.from({ length: 46 })
  slots[36] = { name: 'diamond_sword', count: 1 }
  slots[39] = { name: 'end_crystal', count: 64 }
  const entity: any = {
    id: 1, type: 'player', username: 'MCAI_001',
    position: new Vec3(0, 64, 0), yaw: Math.PI, pitch: 0, eyeHeight: 1.62
  }
  const look = vi.fn(async (yaw: number, pitch: number) => {
    entity.yaw = yaw
    entity.pitch = pitch
  })
  const bot = {
    entity,
    entities: { 2: assigned, 3: spectator },
    players: { MCAI_002: { entity: assigned }, HumanSpectator: { entity: spectator } },
    inventory: { slots }, quickBarSlot: 1, heldItem: null,
    setControlState: vi.fn(), getControlState: vi.fn(() => false),
    setQuickBarSlot: vi.fn(), clearControlStates: vi.fn(),
    deactivateItem: vi.fn(), stopDigging: vi.fn(), look, attack,
    blockAtCursor: vi.fn(() => null)
  } as unknown as Bot
  return { bot, assigned, spectator, attack }
}

function crystalDemoBot(): {
  bot: Bot
  basePosition: Vec3
  assigned: any
  spectator: any
  attack: ReturnType<typeof vi.fn>
  genericPlace: ReturnType<typeof vi.fn>
} {
  const basePosition = new Vec3(2, 63, 0)
  const assigned = fighter(2, 'MCAI_002', 1)
  const spectator = fighter(3, 'HumanSpectator', 0.5)
  const attack = vi.fn()
  const genericPlace = vi.fn(async () => basePosition)
  const slots = Array.from({ length: 46 })
  slots[36] = { name: 'diamond_sword', count: 1 }
  slots[39] = { name: 'end_crystal', count: 64 }
  const entity: any = {
    id: 1, type: 'player', username: 'MCAI_001',
    position: new Vec3(0, 64, 0), yaw: 0, pitch: 0, eyeHeight: 1.62
  }
  const bot: any = {
    entity,
    username: 'MCAI_001',
    entities: { 2: assigned, 3: spectator },
    players: { MCAI_002: { entity: assigned }, HumanSpectator: { entity: spectator } },
    inventory: { slots }, quickBarSlot: 0, heldItem: slots[36],
    setControlState: vi.fn(), getControlState: vi.fn(() => false),
    clearControlStates: vi.fn(), deactivateItem: vi.fn(), stopDigging: vi.fn(),
    attack, blockAtCursor: vi.fn(() => null), _genericPlace: genericPlace,
    setQuickBarSlot: vi.fn((slot: number) => {
      bot.quickBarSlot = slot
      bot.heldItem = slots[36 + slot]
    }),
    look: vi.fn(async (yaw: number, pitch: number) => {
      entity.yaw = yaw
      entity.pitch = pitch
    }),
    blockAt: vi.fn((position: Vec3) => {
      if (position.equals(basePosition)) return { name: 'obsidian', position: basePosition.clone() }
      return { name: 'air', position: position.clone(), boundingBox: 'empty' }
    }),
    world: {
      raycast: vi.fn(() => ({
        position: basePosition.clone(), face: 1,
        intersect: basePosition.offset(0.5, 1, 0.5)
      }))
    }
  }
  return { bot: bot as Bot, basePosition, assigned, spectator, attack, genericPlace }
}

function tacticalDemoBot(withStoneWall: boolean, wallMaterial: 'stone' | 'obsidian' = 'stone'): {
  bot: Bot
  blocks: Map<string, any>
  wallPosition: Vec3
  placementPosition: Vec3
  assigned: any
  spectator: any
  dig: ReturnType<typeof vi.fn>
  genericPlace: ReturnType<typeof vi.fn>
} {
  const wallPosition = new Vec3(0, 64, -3)
  const placementPosition = new Vec3(0, 64, -3)
  const self: any = {
    id: 1, type: 'player', username: 'MCAI_001',
    position: new Vec3(0.5, 64, 0.5), yaw: 0, pitch: 0,
    eyeHeight: 1.62, width: 0.6, height: 1.8
  }
  const assigned: any = {
    id: 2, type: 'player', username: 'MCAI_002',
    position: new Vec3(0.5, 64, -6.5), width: 0.6, height: 1.8
  }
  const spectator: any = {
    id: 3, type: 'player', username: 'HumanSpectator',
    position: new Vec3(8, 70, 8), width: 0.6, height: 1.8
  }
  const blocks = new Map<string, any>()
  if (withStoneWall) {
    blocks.set(positionKey(wallPosition), {
      name: wallMaterial, position: wallPosition.clone(), diggable: true, boundingBox: 'block'
    })
  }
  const slots = Array.from({ length: 46 })
  slots[36] = { name: 'diamond_sword', count: 1 }
  slots[37] = { name: 'diamond_pickaxe', count: 1 }
  slots[38] = { name: 'obsidian', count: 64 }
  slots[39] = { name: 'end_crystal', count: 64 }
  const dig = vi.fn(async () => undefined)
  const genericPlace = vi.fn(async () => placementPosition)
  const blockAt = (position: Vec3) => {
    const stored = blocks.get(positionKey(position))
    if (stored) return stored
    if (position.y === 63) {
      return { name: 'stone', position: position.clone(), diggable: true, boundingBox: 'block' }
    }
    return { name: 'air', type: 0, position: position.clone(), diggable: false, boundingBox: 'empty' }
  }
  const bot: any = {
    username: 'MCAI_001', entity: self,
    entities: { 2: assigned, 3: spectator },
    players: { MCAI_002: { entity: assigned }, HumanSpectator: { entity: spectator } },
    inventory: { slots }, quickBarSlot: 0, heldItem: slots[36],
    setControlState: vi.fn(), getControlState: vi.fn(() => false),
    clearControlStates: vi.fn(), deactivateItem: vi.fn(), stopDigging: vi.fn(),
    attack: vi.fn(), activateItem: vi.fn(), dig, digTime: vi.fn(() => 50),
    _genericPlace: genericPlace, blockAt, blockAtCursor: vi.fn(() => blockAt(
      withStoneWall ? wallPosition : placementPosition.offset(0, -1, 0)
    )),
    setQuickBarSlot: vi.fn((slot: number) => {
      bot.quickBarSlot = slot
      bot.heldItem = slots[36 + slot]
    }),
    look: vi.fn(async (yaw: number, pitch: number) => {
      self.yaw = yaw
      self.pitch = pitch
    }),
    world: {
      raycast: vi.fn(() => {
        const coverPresent = blocks.has(positionKey(wallPosition))
        const rayPosition = coverPresent ? wallPosition : placementPosition.offset(0, -1, 0)
        return {
          position: rayPosition.clone(),
          face: coverPresent ? 3 : 1,
          intersect: rayPosition.offset(0.5, 1, 0.5)
        }
      })
    }
  }
  return {
    bot: bot as Bot, blocks, wallPosition, placementPosition,
    assigned, spectator, dig, genericPlace
  }
}

function positionKey(position: Vec3): string {
  return `${position.x},${position.y},${position.z}`
}

describe('combat camera curriculum', () => {
  it('clamps ordinary exploratory pitch to 45 degrees', async () => {
    const look = vi.fn(async () => undefined)
    const bot = {
      entity: { position: new Vec3(0, 64, 0), yaw: 0, pitch: 0, eyeHeight: 1.62 },
      entities: {}, players: {}, inventory: { slots: Array.from({ length: 46 }) },
      quickBarSlot: 0, heldItem: null,
      setControlState: vi.fn(), getControlState: vi.fn(() => false),
      setQuickBarSlot: vi.fn(), clearControlStates: vi.fn(),
      deactivateItem: vi.fn(), stopDigging: vi.fn(), look,
      blockAtCursor: vi.fn(() => null)
    } as unknown as Bot
    const controls = new LegalControlAdapter(bot)

    await controls.apply({
      schema_version: 1, forward: 0, strafe: 0, jump: false, sprint: false,
      sneak: false, yaw_delta: 0, pitch_delta: Math.PI / 2, primary: 'none',
      release_use: false, hotbar: -1, swap_offhand: false
    }, 1)

    expect(MAX_COMBAT_PITCH).toBeCloseTo(Math.PI / 4)
    expect(look).toHaveBeenCalledWith(0, MAX_COMBAT_PITCH, true)
  })

  it('uses one bounded sword demonstration and attacks only the assigned fighter', async () => {
    const { bot, assigned, spectator, attack } = bootcampBot()
    const controls = new LegalControlAdapter(bot, 'sword')
    controls.setOpponentUsername('MCAI_002')

    await controls.applySwordBootcampAssist(1)
    await controls.applySwordBootcampAssist(2)
    await controls.applySwordBootcampAssist(1 + SWORD_BOOTCAMP_ATTACK_COOLDOWN_TICKS)

    expect(attack).toHaveBeenCalledTimes(1)
    expect(attack).toHaveBeenNthCalledWith(1, assigned)
    expect(attack).not.toHaveBeenCalledWith(spectator)
    expect(bot.look).toHaveBeenCalledTimes(2)
    expect(bot.setControlState).toHaveBeenCalledWith('forward', true)
    expect(bot.setControlState).toHaveBeenCalledWith('sprint', true)
    expect(bot.setQuickBarSlot).toHaveBeenCalledWith(0)
    expect(controls.telemetry(13).lastAttackTick).toBe(2)
  })

  it('uses the normal pitch range for every policy mode', async () => {
    const build = (mode: 'sword' | 'crystal' | 'combined' | 'terrain') => {
      const look = vi.fn(async () => undefined)
      const bot = {
        entity: { position: new Vec3(0, 64, 0), yaw: 0, pitch: 0, eyeHeight: 1.62 },
        entities: {}, players: {}, inventory: { slots: Array.from({ length: 46 }) },
        quickBarSlot: 0, heldItem: null,
        setControlState: vi.fn(), getControlState: vi.fn(() => false),
        setQuickBarSlot: vi.fn(), clearControlStates: vi.fn(),
        deactivateItem: vi.fn(), stopDigging: vi.fn(), look,
        blockAtCursor: vi.fn(() => null)
      } as unknown as Bot
      return { controls: new LegalControlAdapter(bot, mode), look }
    }
    const action = {
      schema_version: 1 as const, forward: 0 as const, strafe: 0 as const,
      jump: false, sprint: false, sneak: false, yaw_delta: 0,
      pitch_delta: Math.PI / 2, primary: 'none' as const,
      release_use: false, hotbar: -1, swap_offhand: false
    }

    const sword = build('sword')
    await sword.controls.apply(action, 1)
    expect(sword.look).toHaveBeenCalledWith(0, MAX_COMBAT_PITCH, true)

    for (const mode of ['crystal', 'combined', 'terrain'] as const) {
      const value = build(mode)
      await value.controls.apply(action, 1)
      expect(value.look).toHaveBeenCalledWith(0, MAX_COMBAT_PITCH, true)
    }
  })

  it('does not let bare hotbar selection suppress the combined sword rescue', async () => {
    const { bot, attack } = bootcampBot()
    const agentKey = 'local-rollout/agent-3'
    const rescueTick = combinedRescueDelay(agentKey)
    const controls = new LegalControlAdapter(bot, 'combined', agentKey)
    controls.setOpponentUsername('MCAI_002')

    expect(await controls.applySwordBootcampAssist(rescueTick - 1)).toBeNull()
    expect(attack).not.toHaveBeenCalled()

    await controls.apply({
      schema_version: 1, forward: 0, strafe: 0, jump: false, sprint: false,
      sneak: false, yaw_delta: 0, pitch_delta: 0, primary: 'none',
      release_use: false, hotbar: 3, swap_offhand: false
    }, rescueTick)
    expect((await controls.applySwordBootcampAssist(rescueTick))?.source).toBe('teacher_sword')
    expect((await controls.applySwordBootcampAssist(rescueTick + 1))?.source).toBe('teacher_sword')
    expect(attack).toHaveBeenCalledTimes(1)
    expect(controls.getAuditEvents().some(event => event.operation === 'combined_crystal_yield')).toBe(false)
    expect(controls.getAuditEvents().some(event => event.operation === 'combined_melee_rescue')).toBe(true)
  })

  it('gives combined rescues stable per-agent cadence jitter', () => {
    const keys = Array.from({ length: 8 }, (_, index) => `agent-${index}`)
    const delays = keys.map(combinedRescueDelay)
    expect(keys.map(combinedRescueDelay)).toEqual(delays)
    expect(new Set(delays).size).toBeGreaterThan(1)
    expect(Math.min(...delays)).toBeGreaterThanOrEqual(53)
    expect(Math.max(...delays)).toBeLessThanOrEqual(67)
  })

  it('does nothing in crystal mode or without an arena-assigned opponent', async () => {
    const disabled = bootcampBot()
    const disabledControls = new LegalControlAdapter(disabled.bot, 'crystal')
    disabledControls.setOpponentUsername('MCAI_002')
    expect(await disabledControls.applySwordBootcampAssist(1000)).toBeNull()
    expect(disabled.attack).not.toHaveBeenCalled()

    const unassigned = bootcampBot()
    const enabledControls = new LegalControlAdapter(unassigned.bot, 'sword')
    expect(await enabledControls.applySwordBootcampAssist(1)).toBeNull()
    expect(unassigned.attack).not.toHaveBeenCalled()
  })

  it('enables one crystal-retention sword rescue only after the autonomous crystal gate', async () => {
    const value = bootcampBot()
    const controls = new LegalControlAdapter(value.bot, 'crystal')
    controls.setOpponentUsername('MCAI_002')
    expect(await controls.applySwordBootcampAssist(1000)).toBeNull()

    controls.setCrystalRetentionSwordFallbackEnabled(true)
    expect((await controls.applySwordBootcampAssist(1001))?.source).toBe('teacher_sword')
  })

  it('disables every assist while preserving policy execution', async () => {
    const value = bootcampBot()
    const controls = new LegalControlAdapter(value.bot, 'combined', 'evaluation-agent')
    controls.setOpponentUsername('MCAI_002')
    controls.setMatchConfiguration({ teachersEnabled: false, terrainEnabled: true })

    expect(await controls.applySwordBootcampAssist(1000)).toBeNull()
    expect(await controls.applyCombinedCrystalDemonstration(1000)).toBeNull()
    expect(await controls.applyCombinedTacticalBlockDemonstration(1000)).toBeNull()
    const policy = await controls.apply({
      schema_version: 1, forward: 1, strafe: 0, jump: false, sprint: true,
      sneak: false, yaw_delta: 0, pitch_delta: 0, primary: 'none',
      release_use: false, hotbar: 3, swap_offhand: false
    }, 1000)
    expect(policy.source).toBe('policy')
    expect(value.bot.setControlState).toHaveBeenCalledWith('forward', true)
    expect(value.bot.setQuickBarSlot).toHaveBeenCalledWith(3)
  })

  it('recenters toward only the assigned opponent after ten extreme-pitch ticks', async () => {
    const value = bootcampBot()
    ;(value.bot.entity as any).pitch = 70 * Math.PI / 180
    const controls = new LegalControlAdapter(value.bot, 'combined')
    controls.setOpponentUsername('MCAI_002')
    for (let tick = 1; tick < EXTREME_PITCH_RECENTER_TICKS; tick++) {
      expect(await controls.applyPitchSafety(tick)).toBeNull()
    }
    const execution = await controls.applyPitchSafety(EXTREME_PITCH_RECENTER_TICKS)
    expect(execution?.source).toBe('safety')
    expect(Math.abs((value.bot.entity as any).pitch)).toBeLessThanOrEqual(MAX_COMBAT_PITCH)
    expect(value.bot.look).toHaveBeenCalled()
  })

  it('uses cooldown-limited sword pressure only against the assigned opponent during a stall', async () => {
    const value = bootcampBot()
    const controls = new LegalControlAdapter(value.bot, 'combined')
    controls.setOpponentUsername('MCAI_002')

    const execution = await controls.applyTrainerStallSafety(40)

    expect(execution).toMatchObject({
      source: 'safety',
      action: { primary: 'attack', forward: 1, sprint: true, hotbar: 0 }
    })
    expect(value.attack).toHaveBeenCalledTimes(1)
    expect(value.attack).toHaveBeenCalledWith(value.assigned)
    expect(value.attack).not.toHaveBeenCalledWith(value.spectator)
    expect(value.bot.setControlState).toHaveBeenCalledWith('forward', true)
    expect(value.bot.setControlState).toHaveBeenCalledWith('sprint', true)
    expect(value.bot.look).toHaveBeenCalledWith(
      -Math.PI / 2,
      expect.any(Number),
      true
    )

    await controls.applyTrainerStallSafety(41)
    expect(value.attack).toHaveBeenCalledTimes(1)
  })

  it('awaits the source marker before a teacher can send a combat packet', async () => {
    const value = bootcampBot()
    const order: string[] = []
    value.attack.mockImplementation(() => { order.push('attack') })
    const controls = new LegalControlAdapter(
      value.bot,
      'sword',
      'ordered-attribution',
      async source => {
        await Promise.resolve()
        order.push(`marker:${source}`)
        return true
      }
    )
    controls.setOpponentUsername('MCAI_002')

    await controls.applySwordBootcampAssist(1)
    await controls.applySwordBootcampAssist(2)
    expect(order).toContain('attack')
    expect(order.indexOf('marker:teacher_sword')).toBeLessThan(order.indexOf('attack'))
  })

  it('falls back to releasing every known control when clearControlStates is unavailable', () => {
    const value = bootcampBot()
    delete (value.bot as any).clearControlStates
    const controls = new LegalControlAdapter(value.bot, 'combined')

    expect(() => controls.emergencyStop()).not.toThrow()
    for (const control of ['forward', 'back', 'left', 'right', 'jump', 'sprint', 'sneak']) {
      expect(value.bot.setControlState).toHaveBeenCalledWith(control, false)
    }
    expect(controls.getAuditEvents().map(event => event.operation))
      .toContain('clear_control_states_fallback')
  })
})

describe('combined crystal demonstration rail', () => {
  const noop = {
    schema_version: 1 as const, forward: 0 as const, strafe: 0 as const,
    jump: false, sprint: false, sneak: false, yaw_delta: 0, pitch_delta: 0,
    primary: 'none' as const, release_use: false, hotbar: -1, swap_offhand: false
  }

  it('executes a legal crystal click before a simultaneous camera delta', async () => {
    const value = crystalDemoBot()
    const order: string[] = []
    value.genericPlace.mockImplementation(async () => {
      order.push('place')
      return value.basePosition
    })
    ;(value.bot.look as any).mockImplementation(async (yaw: number, pitch: number) => {
      order.push('look')
      ;(value.bot.entity as any).yaw = yaw
      ;(value.bot.entity as any).pitch = pitch
    })
    const controls = new LegalControlAdapter(value.bot, 'crystal')
    controls.setOpponentUsername('MCAI_002')

    const execution = await controls.apply({
      ...noop,
      yaw_delta: Math.PI / 2,
      primary: 'use_main',
      hotbar: 3
    }, 20)

    expect(execution.combatPriority).toBe(true)
    expect(value.genericPlace).toHaveBeenCalledTimes(1)
    expect(value.genericPlace).toHaveBeenCalledWith(
      expect.objectContaining({ name: 'obsidian', position: value.basePosition }),
      new Vec3(0, 1, 0),
      expect.objectContaining({ forceLook: 'ignore', swingArm: 'right' })
    )
    expect(order).toEqual(['place', 'look'])
  })

  it('stages a reachable edge-pad place and attacks only its exact newly spawned crystal', async () => {
    const value = crystalDemoBot()
    value.assigned.position = new Vec3(0, 64, 3)
    const agentKey = 'arena-2/MCAI_001'
    const due = combinedCrystalDemoDelay(agentKey)
    const controls = new LegalControlAdapter(value.bot, 'combined', agentKey)
    controls.setOpponentUsername('MCAI_002')

    expect(await controls.applyCombinedCrystalDemonstration(due - 1)).toBeNull()
    expect((await controls.applyCombinedCrystalDemonstration(due))?.source)
      .toBe('teacher_crystal')
    expect(value.genericPlace).not.toHaveBeenCalled()

    expect((await controls.applyCombinedCrystalDemonstration(due + 1))?.action.primary)
      .toBe('use_main')
    expect(value.attack).not.toHaveBeenCalled()
    expect(value.genericPlace).toHaveBeenCalledTimes(1)
    expect(value.genericPlace).toHaveBeenCalledWith(
      expect.objectContaining({ name: 'obsidian' }), new Vec3(0, 1, 0),
      expect.objectContaining({ forceLook: 'ignore', swingArm: 'right' })
    )

    const remoteCrystal: any = {
      id: 20, type: 'object', name: 'end_crystal',
      position: new Vec3(-2.5, 64, 0.5), width: 2, height: 2
    }
    ;(value.bot.entities as any)[20] = remoteCrystal
    const placedCrystal: any = {
      id: 21, type: 'object', name: 'end_crystal',
      position: value.basePosition.offset(0.5, 1, 0.5), width: 2, height: 2
    }
    ;(value.bot.entities as any)[21] = placedCrystal
    expect((await controls.applyCombinedCrystalDemonstration(due + 2))?.source)
      .toBe('teacher_crystal')
    expect(value.attack).not.toHaveBeenCalled()
    expect((await controls.applyCombinedCrystalDemonstration(due + 3))?.action.primary)
      .toBe('attack')

    expect(value.attack).toHaveBeenCalledTimes(1)
    expect(value.attack).toHaveBeenCalledWith(placedCrystal)
    expect(value.attack).not.toHaveBeenCalledWith(remoteCrystal)
    expect(value.attack).not.toHaveBeenCalledWith(value.spectator)
    expect(await controls.applySwordBootcampAssist(due + 4)).toBeNull()
    expect(await controls.applyCombinedCrystalDemonstration(
      due + 3 + COMBINED_CRYSTAL_DEMO_PERIOD_TICKS - 1
    )).toBeNull()
    expect(value.bot.quickBarSlot).toBe(0)
    const operations = controls.getAuditEvents().map(event => event.operation)
    expect(operations).toContain('combined_crystal_demo_aim_pad')
    expect(operations).toContain('combined_crystal_demo_place')
    expect(operations).toContain('combined_crystal_demo_reaim_crystal')
    expect(operations).toContain('combined_crystal_demo_attack')
  })

  it('lets legal readiness-prior actions place and attack before scripted fallbacks', async () => {
    const value = crystalDemoBot()
    ;(value.bot.entity as any).position = new Vec3(1, 64, 0)
    value.assigned.position = new Vec3(0, 64, 3)
    const agentKey = 'arena-1/MCAI_003'
    const due = combinedCrystalDemoDelay(agentKey)
    const controls = new LegalControlAdapter(value.bot, 'combined', agentKey)
    controls.setOpponentUsername('MCAI_002')

    expect((await controls.applyCombinedCrystalDemonstration(due))?.source)
      .toBe('teacher_crystal')
    const place = await controls.apply({ ...noop, primary: 'use_main', hotbar: 3 }, due + 1)
    expect(place.combatPriority).toBe(true)
    expect(value.genericPlace).toHaveBeenCalledTimes(1)

    const placedCrystal: any = {
      id: 31, type: 'object', name: 'end_crystal',
      position: value.basePosition.offset(0.5, 1, 0.5), width: 2, height: 2
    }
    ;(value.bot.entities as any)[31] = placedCrystal
    expect((await controls.applyCombinedCrystalDemonstration(due + 1))?.source)
      .toBe('teacher_crystal')
    const attack = await controls.apply({ ...noop, primary: 'attack' }, due + 2)
    expect(attack.combatPriority).toBe(true)

    expect(value.attack).toHaveBeenCalledTimes(1)
    expect(value.attack).toHaveBeenCalledWith(placedCrystal)
    const operations = controls.getAuditEvents().map(event => event.operation)
    expect(operations).toContain('activate_block_from_crosshair')
    expect(operations).toContain('attack_entity')
    expect(operations).not.toContain('combined_crystal_demo_place')
    expect(operations).not.toContain('combined_crystal_demo_attack')
  })

  it('uses deterministic, staggered phases and never runs outside combined mode', async () => {
    const keys = Array.from({ length: 8 }, (_, index) => `crystal-agent-${index}`)
    const delays = keys.map(combinedCrystalDemoDelay)
    expect(keys.map(combinedCrystalDemoDelay)).toEqual(delays)
    expect(new Set(delays).size).toBeGreaterThan(1)
    expect(Math.min(...delays)).toBeGreaterThanOrEqual(20)
    expect(Math.max(...delays)).toBeLessThan(100)

    const value = crystalDemoBot()
    const controls = new LegalControlAdapter(value.bot, 'sword', keys[0])
    controls.setOpponentUsername('MCAI_002')
    expect(await controls.applyCombinedCrystalDemonstration(1000)).toBeNull()
    expect(value.genericPlace).not.toHaveBeenCalled()
  })

  it('waits passively for entity spawn and spends four controls only on useful actions', async () => {
    const value = crystalDemoBot()
    value.assigned.position = new Vec3(0, 64, 3)
    const key = 'bounded-crystal-teacher'
    const due = combinedCrystalDemoDelay(key)
    const beforeExecution = vi.fn(async () => true)
    const controls = new LegalControlAdapter(value.bot, 'combined', key, beforeExecution)
    controls.setOpponentUsername('MCAI_002')

    const executions = [
      await controls.applyCombinedCrystalDemonstration(due),
      await controls.applyCombinedCrystalDemonstration(due + 1)
    ].filter(Boolean)
    expect(executions).toHaveLength(2)
    expect(executions[1]?.action.primary).toBe('use_main')

    for (let tick = due + 2; tick <= due + 5; tick++) {
      expect(await controls.applyCombinedCrystalDemonstration(tick)).toBeNull()
    }
    expect(beforeExecution).toHaveBeenCalledTimes(2)

    const placedCrystal: any = {
      id: 41, type: 'object', name: 'end_crystal',
      position: value.basePosition.offset(0.5, 1, 0.5), width: 2, height: 2
    }
    ;(value.bot.entities as any)[41] = placedCrystal
    executions.push((await controls.applyCombinedCrystalDemonstration(due + 6))!)
    executions.push((await controls.applyCombinedCrystalDemonstration(due + 7))!)

    expect(executions).toHaveLength(MAX_TEACHER_CONTROL_TICKS)
    expect(executions.every(execution => execution?.source === 'teacher_crystal')).toBe(true)
    expect(executions.map(execution => execution?.action.primary))
      .toEqual(['none', 'use_main', 'none', 'attack'])
    expect(beforeExecution).toHaveBeenCalledTimes(MAX_TEACHER_CONTROL_TICKS)
    expect(controls.getAuditEvents().filter(
      event => event.operation === 'combined_crystal_demo_wait_spawn'
    )).toHaveLength(4)
    expect(value.attack).toHaveBeenCalledWith(placedCrystal)
    expect(await controls.applyCombinedCrystalDemonstration(due + 1000)).toBeNull()
    expect(value.bot.quickBarSlot).toBe(0)
  })
})

describe('combined tactical block demonstration rail', () => {
  const noop = {
    schema_version: 1 as const, forward: 0 as const, strafe: 0 as const,
    jump: false, sprint: false, sneak: false, yaw_delta: 0, pitch_delta: 0,
    primary: 'none' as const, release_use: false, hotbar: -1, swap_offhand: false
  }

  it('executes only the exact marked policy foundation before camera and protects its priority', async () => {
    const value = tacticalDemoBot(false)
    const placeBlock = vi.fn(async () => undefined)
    ;(value.bot as any)._placeBlockWithOptions = placeBlock
    const controls = new LegalControlAdapter(value.bot, 'terrain', 'policy-foundation')
    controls.setOpponentUsername('MCAI_002')

    const exact = await controls.apply({
      ...noop, primary: 'use_main', hotbar: 2, yaw_delta: 0.25
    }, 1)
    expect(exact.combatPriority).toBe(true)
    expect(placeBlock).toHaveBeenCalledTimes(1)
    expect(placeBlock.mock.invocationCallOrder[0])
      .toBeLessThan((value.bot.look as any).mock.invocationCallOrder[0])

    await controls.apply(noop, 2)
    const unrelatedSupport = new Vec3(1, 63, -3)
    ;(value.bot.world as any).raycast.mockReturnValue({
      position: unrelatedSupport, face: 1,
      intersect: unrelatedSupport.offset(0.5, 1, 0.5)
    })
    const arbitrary = await controls.apply({
      ...noop, primary: 'use_main', hotbar: 2
    }, 3)
    expect(arbitrary.combatPriority).toBe(false)
    expect(placeBlock).toHaveBeenCalledTimes(1)
    expect(controls.getAuditEvents().map(event => event.operation))
      .toContain('reject_untargeted_obsidian_place')
  })

  it('adopts a policy-mined stone cell as the exact extra foundation after quota', async () => {
    const value = tacticalDemoBot(false)
    const tracker = new TacticalPlacementTracker(value.bot)
    const episodeId = 'policy-mine-replace'
    tracker.beginEpisode(episodeId, 1)

    // Complete the two ordinary stage-one foundations first. The mined cell
    // must still become one bounded extra target rather than being discarded.
    for (let index = 0; index < 2; index++) {
      const ordinary = tracker.resolve('MCAI_002', index)!
      value.blocks.set(positionKey(ordinary.targetPosition), {
        name: 'obsidian', position: ordinary.targetPosition.clone(),
        diggable: true, boundingBox: 'block'
      })
      ;(value.bot.entity as any).position.z -= 2
      value.assigned.position.z -= 2
    }
    expect(tracker.resolve('MCAI_002', 2)).toBeNull()
    expect(tracker.progress()).toEqual({ completed: 2, quota: 2 })

    const naturalStone = new Vec3(0, 64, -7)
    const support = naturalStone.offset(0, -1, 0)
    ;(value.bot.entity as any).position = new Vec3(0.5, 64, -3.5)
    value.assigned.position = new Vec3(0.5, 64, -10.5)
    value.blocks.set(positionKey(naturalStone), {
      name: 'stone', position: naturalStone.clone(), diggable: true, boundingBox: 'block'
    })
    ;(value.bot.world as any).raycast.mockImplementation(() => {
      const stonePresent = value.blocks.has(positionKey(naturalStone))
      const position = stonePresent ? naturalStone : support
      return {
        position: position.clone(), face: stonePresent ? 3 : 1,
        intersect: position.offset(0.5, 1, 0.5)
      }
    })
    ;(value.dig as any).mockImplementation(async () => {
      value.blocks.delete(positionKey(naturalStone))
    })
    const placeBlock = vi.fn(async () => {
      value.blocks.set(positionKey(naturalStone), {
        name: 'obsidian', position: naturalStone.clone(),
        diggable: true, boundingBox: 'block'
      })
    })
    ;(value.bot as any)._placeBlockWithOptions = placeBlock

    const controls = new LegalControlAdapter(
      value.bot, 'terrain', 'policy-mine-replace', async () => true, tracker
    )
    controls.setMatchConfiguration({ terrainEnabled: true, stage: 1 })
    controls.setOpponentUsername('MCAI_002')
    const mineTick = combinedTacticalBlockDemoDelay('policy-mine-replace')
    const mine = await controls.apply({ ...noop, primary: 'attack', hotbar: 1 }, mineTick)
    expect(mine.combatPriority).toBe(true)
    await vi.waitFor(() => expect(controls.getAuditEvents().map(event => event.operation))
      .toContain('policy_mine_replacement_complete'))
    expect(await controls.applyCombinedTacticalBlockDemonstration(mineTick + 1)).toBeNull()
    expect(value.genericPlace).not.toHaveBeenCalled()
    expect(controls.getAuditEvents().map(event => event.operation))
      .toContain('combined_tactical_block_demo_yield_policy_mine_replacement')

    const builder = new ObservationBuilder(value.bot, tracker)
    builder.setOpponentUsername('MCAI_002')
    const context = {
      episode_id: episodeId, tick: mineTick + 2, policy_version: 0, arena_seed: 1,
      action_delay_ticks: 0, observation_delay_ticks: 0,
      mode: 'terrain' as const, curriculum_stage: 1
    }
    const observation = builder.build(context, controls.telemetry(11))
    const marker = observation.blocks.find(block => block.tactical_placement_target)
    expect(marker).toBeDefined()
    expect(observation.action_mask.use_main).toBe(true)
    expect(tracker.hasAdoptedPolicyMineReplacement()).toBe(true)
    expect(tracker.hasPolicyMineReplacementSequence(mineTick + 2)).toBe(true)

    const place = await controls.apply({
      ...noop, primary: 'use_main', hotbar: 2
    }, mineTick + 3)
    expect(place.combatPriority).toBe(true)
    expect(placeBlock).toHaveBeenCalledTimes(1)
    const retired = builder.build(
      { ...context, tick: mineTick + 4 }, controls.telemetry(mineTick + 4)
    )
    expect(retired.blocks.every(block => !block.tactical_placement_target)).toBe(true)
    expect(tracker.progress()).toEqual({ completed: 3, quota: 2 })
    expect(tracker.isCompleted()).toBe(true)
  })

  it('never adopts or steals a teacher-owned mine as a policy replacement', async () => {
    const value = tacticalDemoBot(true)
    ;(value.bot.inventory.slots as any[])[38] = null
    ;(value.dig as any).mockImplementation(async (block: any) => {
      value.blocks.delete(positionKey(block.position))
    })
    const tracker = new TacticalPlacementTracker(value.bot)
    tracker.beginEpisode('teacher-mine-no-adopt', 1)
    const key = 'teacher-mine-no-adopt'
    const due = combinedTacticalBlockDemoDelay(key)
    const controls = new LegalControlAdapter(
      value.bot, 'terrain', key, async () => true, tracker
    )
    controls.setMatchConfiguration({ terrainEnabled: true })
    controls.setOpponentUsername('MCAI_002')

    expect((await controls.applyCombinedTacticalBlockDemonstration(due))?.source)
      .toBe('teacher_block')
    expect((await controls.applyCombinedTacticalBlockDemonstration(due + 1))?.action.primary)
      .toBe('attack')
    await vi.waitFor(() => expect(value.blocks.has(positionKey(value.wallPosition))).toBe(false))
    tracker.resolve('MCAI_002', due + 2)
    expect(tracker.hasAdoptedPolicyMineReplacement()).toBe(false)
    expect(tracker.hasPolicyMineReplacementSequence(due + 2)).toBe(false)
    expect(controls.getAuditEvents().map(event => event.operation))
      .not.toContain('policy_mine_replacement_complete')
  })

  it('spends the one block demonstration when policy already built the first target', async () => {
    const value = tacticalDemoBot(false)
    const key = 'policy-built-before-teacher'
    const due = combinedTacticalBlockDemoDelay(key)
    const tracker = new TacticalPlacementTracker(value.bot)
    tracker.beginEpisode('terrain-policy-first', 3)
    const policyTarget = tracker.resolve('MCAI_002')!
    value.blocks.set(positionKey(policyTarget.targetPosition), {
      name: 'obsidian', position: policyTarget.targetPosition.clone(),
      diggable: true, boundingBox: 'block'
    })
    ;(value.bot.entity as any).position.z -= 2
    value.assigned.position.z -= 2
    expect(tracker.resolve('MCAI_002')).not.toBeNull()
    expect(tracker.progress()).toEqual({ completed: 1, quota: 3 })

    const beforeExecution = vi.fn(async () => true)
    const controls = new LegalControlAdapter(
      value.bot, 'terrain', key, beforeExecution, tracker
    )
    controls.setMatchConfiguration({ terrainEnabled: true, stage: 3 })
    controls.setOpponentUsername('MCAI_002')

    expect(await controls.applyCombinedTacticalBlockDemonstration(due)).toBeNull()
    expect(beforeExecution).not.toHaveBeenCalled()
    expect(value.genericPlace).not.toHaveBeenCalled()
    expect(controls.getAuditEvents().map(event => event.operation))
      .toContain('combined_tactical_block_demo_policy_foundation_complete')
    expect(await controls.applyCombinedTacticalBlockDemonstration(
      due + COMBINED_TACTICAL_BLOCK_DEMO_PERIOD_TICKS + 1
    )).toBeNull()
    expect(tracker.progress()).toEqual({ completed: 1, quota: 3 })
  })

  it('falls back to mining only when no obsidian foundation is available', async () => {
    const value = tacticalDemoBot(true, 'obsidian')
    ;(value.bot.inventory.slots as any[])[38] = null
    const key = 'arena-3/MCAI_005'
    const due = combinedTacticalBlockDemoDelay(key)
    const controls = new LegalControlAdapter(value.bot, 'combined', key)
    controls.setMatchConfiguration({ terrainEnabled: true })
    controls.setOpponentUsername('MCAI_002')

    expect(await controls.applyCombinedTacticalBlockDemonstration(due - 1)).toBeNull()
    expect((await controls.applyCombinedTacticalBlockDemonstration(due))?.source)
      .toBe('teacher_block')
    expect(value.dig).not.toHaveBeenCalled()
    await controls.apply({ ...noop, primary: 'attack', hotbar: 1 }, due + 1)
    expect(value.dig).toHaveBeenCalledTimes(1)
    expect(value.dig).toHaveBeenCalledWith(
      expect.objectContaining({ name: 'obsidian', position: value.wallPosition }), 'ignore'
    )

    value.blocks.delete(positionKey(value.wallPosition))
    expect(await controls.applyCombinedTacticalBlockDemonstration(due + 1)).toBeNull()
    const operations = controls.getAuditEvents().map(event => event.operation)
    expect(operations).toContain('combined_tactical_block_demo_aim_mine')
    expect(operations).toContain('start_digging')
    expect(operations).toContain('combined_tactical_block_demo_mined')
    expect(operations).not.toContain('combined_tactical_block_demo_fallback_mine')
    for (const call of (value.bot.look as any).mock.calls) {
      expect(Math.abs(call[1])).toBeLessThanOrEqual(MAX_CRYSTAL_PITCH)
    }
    expect((value.bot as any).attack).not.toHaveBeenCalledWith(value.spectator)
  })

  it('falls back once to an offensive foundation and leaves it for crystals', async () => {
    const value = tacticalDemoBot(false)
    const key = 'arena-4/MCAI_007'
    const due = combinedTacticalBlockDemoDelay(key)
    const beforeExecution = vi.fn(async () => true)
    const controls = new LegalControlAdapter(value.bot, 'combined', key, beforeExecution)
    controls.setMatchConfiguration({ terrainEnabled: true })
    controls.setOpponentUsername('MCAI_002')

    expect((await controls.applyCombinedTacticalBlockDemonstration(due))?.source)
      .toBe('teacher_block')
    expect((await controls.applyCombinedTacticalBlockDemonstration(due + 1))?.action.primary)
      .toBe('use_main')
    expect(value.genericPlace).toHaveBeenCalledTimes(1)
    expect(value.genericPlace).toHaveBeenCalledWith(
      expect.objectContaining({ name: 'stone' }), new Vec3(0, 1, 0),
      expect.objectContaining({ forceLook: 'ignore', swingArm: 'right' })
    )

    value.blocks.set(positionKey(value.placementPosition), {
      name: 'obsidian', position: value.placementPosition.clone(), diggable: true, boundingBox: 'block'
    })
    expect(await controls.applyCombinedTacticalBlockDemonstration(due + 2)).toBeNull()
    expect(beforeExecution).toHaveBeenCalledTimes(2)
    expect(value.dig).not.toHaveBeenCalled()
    expect(value.blocks.get(positionKey(value.placementPosition))?.name).toBe('obsidian')
    expect(await controls.applyCombinedTacticalBlockDemonstration(
      due + 4 + COMBINED_TACTICAL_BLOCK_DEMO_PERIOD_TICKS - 1
    )).toBeNull()
    expect(value.bot.quickBarSlot).toBe(0)
    const operations = controls.getAuditEvents().map(event => event.operation)
    expect(operations).toContain('combined_tactical_block_demo_aim_place')
    expect(operations).toContain('combined_tactical_block_demo_fallback_place')
    expect(operations).toContain('combined_tactical_block_demo_placed')
    expect(operations).not.toContain('combined_tactical_block_demo_fallback_mine')
  })

  it('tracks a long fallback mine passively through its bounded estimate', async () => {
    const value = tacticalDemoBot(true)
    ;(value.bot.inventory.slots as any[])[38] = null
    ;(value.bot.digTime as any).mockReturnValue(5000)
    const key = 'arena-long-dig/MCAI_009'
    const due = combinedTacticalBlockDemoDelay(key)
    const controls = new LegalControlAdapter(value.bot, 'combined', key)
    controls.setMatchConfiguration({ terrainEnabled: true })
    controls.setOpponentUsername('MCAI_002')

    expect((await controls.applyCombinedTacticalBlockDemonstration(due))?.source)
      .toBe('teacher_block')
    expect((await controls.applyCombinedTacticalBlockDemonstration(due + 1))?.action.primary)
      .toBe('attack')
    expect(value.dig).toHaveBeenCalledTimes(1)

    // The original fixed deadline was due+20. A valid long dig remains active
    // beyond it, but still inside the clamped estimate-derived window.
    expect(await controls.applyCombinedTacticalBlockDemonstration(due + 25)).toBeNull()
    expect(value.bot.stopDigging).not.toHaveBeenCalled()
    value.blocks.delete(positionKey(value.wallPosition))
    expect(await controls.applyCombinedTacticalBlockDemonstration(due + 26)).toBeNull()
    const deadlineEvents = controls.getAuditEvents()
      .filter(event => event.operation === 'combined_tactical_block_demo_dig_deadline')
    expect(deadlineEvents.length).toBeGreaterThanOrEqual(1)
    expect(Number(deadlineEvents.at(-1)?.detail)).toBeGreaterThan(due + 100)
    expect(controls.getAuditEvents().some(
      event => event.operation === 'combined_tactical_block_demo_mined'
    )).toBe(true)
  })

  it('uses deterministic stagger and is disabled outside combined mode', async () => {
    const keys = Array.from({ length: 8 }, (_, index) => `terrain-agent-${index}`)
    const delays = keys.map(combinedTacticalBlockDemoDelay)
    expect(keys.map(combinedTacticalBlockDemoDelay)).toEqual(delays)
    expect(new Set(delays).size).toBeGreaterThan(1)
    expect(Math.min(...delays)).toBeGreaterThanOrEqual(5)
    expect(Math.max(...delays)).toBeLessThan(25)

    const value = tacticalDemoBot(true)
    const controls = new LegalControlAdapter(value.bot, 'sword', keys[0])
    controls.setOpponentUsername('MCAI_002')
    expect(await controls.applyCombinedTacticalBlockDemonstration(1000)).toBeNull()
    expect(value.dig).not.toHaveBeenCalled()
  })
})
