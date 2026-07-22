import type { Bot } from 'mineflayer'
import { Vec3 } from 'vec3'
import { describe, expect, it } from 'vitest'
import {
  findTacticalCrystalPadPlacement,
  findTacticalMiningTarget,
  findTacticalWallPlacement,
  isSafeTacticalMiningTarget,
  isUsefulPolicyMineReplacementCandidate,
  MINIMUM_TACTICAL_FOUNDATION_SPACING,
  POLICY_MINE_REPLACEMENT_TIMEOUT_TICKS,
  tacticalCrystalPadPlacementAt,
  tacticalPlacementQuotaForStage,
  TacticalPlacementTracker
} from '../src/tactical-blocks.js'

function arenaBot(): { bot: Bot; blocks: Map<string, any>; assigned: any; spectator: any } {
  const self: any = {
    id: 1, type: 'player', username: 'MCAI_001',
    position: new Vec3(0.5, 64, 0.5), eyeHeight: 1.62, width: 0.6, height: 1.8
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
  const bot: any = {
    entity: self,
    entities: { 2: assigned, 3: spectator },
    players: { MCAI_002: { entity: assigned }, HumanSpectator: { entity: spectator } },
    blockAt: (position: Vec3) => {
      const stored = blocks.get(key(position))
      if (stored) return stored
      if (position.y === 63) {
        return { name: 'stone', position: position.clone(), diggable: true, boundingBox: 'block' }
      }
      return { name: 'air', type: 0, position: position.clone(), diggable: false, boundingBox: 'empty' }
    }
  }
  return { bot: bot as Bot, blocks, assigned, spectator }
}

function put(blocks: Map<string, any>, position: Vec3, name: string, diggable = true): any {
  const block = { name, position: position.clone(), diggable, boundingBox: 'block' }
  blocks.set(key(position), block)
  return block
}

function key(position: Vec3): string {
  return `${position.x},${position.y},${position.z}`
}

describe('assigned-arena tactical terrain targets', () => {
  it('selects exposed above-floor stone or placed obsidian cover in the assigned corridor', () => {
    const { bot, blocks } = arenaBot()
    const useful = put(blocks, new Vec3(0, 64, -3), 'stone')
    put(blocks, new Vec3(3, 64, -3), 'stone')

    expect(isSafeTacticalMiningTarget(bot, useful, 'MCAI_002')).toBe(true)
    expect(findTacticalMiningTarget(bot, 'MCAI_002')?.position).toEqual(useful.position)

    const placedCover = put(blocks, new Vec3(0, 64, -3), 'obsidian')
    expect(isSafeTacticalMiningTarget(bot, placedCover, 'MCAI_002')).toBe(true)
    expect(findTacticalMiningTarget(bot, 'MCAI_002')?.position).toEqual(placedCover.position)
  })

  it('rejects floor/depth, protected materials, scenery, and cross-arena assignments', () => {
    const value = arenaBot()
    const floor = put(value.blocks, new Vec3(0, 63, -3), 'stone')
    const offAxis = put(value.blocks, new Vec3(3, 64, -3), 'stone')
    const crystalPad = put(value.blocks, new Vec3(0, 63, -3), 'obsidian')
    const bedrock = put(value.blocks, new Vec3(0, 64, -3), 'bedrock', false)
    const barrier = put(value.blocks, new Vec3(0, 65, -3), 'barrier', false)

    expect(isSafeTacticalMiningTarget(value.bot, floor, 'MCAI_002')).toBe(false)
    expect(isSafeTacticalMiningTarget(value.bot, offAxis, 'MCAI_002')).toBe(false)
    expect(isSafeTacticalMiningTarget(value.bot, crystalPad, 'MCAI_002')).toBe(false)
    expect(isSafeTacticalMiningTarget(value.bot, bedrock, 'MCAI_002')).toBe(false)
    expect(isSafeTacticalMiningTarget(value.bot, barrier, 'MCAI_002')).toBe(false)

    value.assigned.position = new Vec3(0.5, 64, -96)
    expect(findTacticalMiningTarget(value.bot, 'MCAI_002')).toBeNull()
    expect(findTacticalWallPlacement(value.bot, 'MCAI_002')).toBeNull()
  })

  it('chooses a reachable clear offensive foundation between only the assigned pair', () => {
    const value = arenaBot()
    const placement = findTacticalCrystalPadPlacement(value.bot, 'MCAI_002')

    expect(placement).not.toBeNull()
    expect(placement?.targetPosition.y).toBe(64)
    expect(placement?.targetPosition.x).toBe(0)
    expect(placement?.referenceBlock.name).toBe('stone')
    expect(placement?.face).toEqual(new Vec3(0, 1, 0))
    expect(placement?.targetPosition.distanceTo(value.spectator.position)).toBeGreaterThan(5)
    const eye = value.bot.entity.position.offset(0, 1.62, 0)
    expect(eye.distanceTo(placement!.targetPosition.offset(0.5, 2, 0.5)))
      .toBeLessThanOrEqual(3.4)
    expect(Math.hypot(
      placement!.targetPosition.x + 0.5 - value.assigned.position.x,
      placement!.targetPosition.z + 0.5 - value.assigned.position.z
    )).toBeGreaterThanOrEqual(1.5)

    value.spectator.position = placement?.targetPosition.offset(0.5, 0, 0.5)
    expect(findTacticalCrystalPadPlacement(value.bot, 'MCAI_002')?.targetPosition)
      .toEqual(placement?.targetPosition)
  })

  it('rejects blocked clearance and off-corridor foundations', () => {
    const value = arenaBot()
    const placement = findTacticalCrystalPadPlacement(value.bot, 'MCAI_002')!
    put(value.blocks, placement.targetPosition.offset(0, 1, 0), 'stone')
    expect(tacticalCrystalPadPlacementAt(
      value.bot, 'MCAI_002', placement.targetPosition
    )).toBeNull()

    value.blocks.delete(key(placement.targetPosition.offset(0, 1, 0)))
    put(value.blocks, placement.referenceBlock.position, 'obsidian')
    expect(tacticalCrystalPadPlacementAt(
      value.bot, 'MCAI_002', placement.targetPosition
    )).toBeNull()

    put(value.blocks, placement.referenceBlock.position, 'stone')
    expect(tacticalCrystalPadPlacementAt(
      value.bot, 'MCAI_002', new Vec3(2, 64, -3)
    )).toBeNull()
  })

  it('emits two stable, separated foundation opportunities in the baseline stage', () => {
    const value = arenaBot()
    const tracker = new TacticalPlacementTracker(value.bot)
    tracker.beginEpisode('episode-a')
    const first = tracker.resolve('MCAI_002')!
    expect(tracker.resolve(null)).toBeNull()
    expect(tracker.resolve('MCAI_002')?.targetPosition).toEqual(first.targetPosition)
    put(value.blocks, first.targetPosition, 'obsidian')
    value.bot.entity.position.z -= 2
    value.assigned.position.z -= 2

    const second = tracker.resolve('MCAI_002')!
    expect(second).not.toBeNull()
    expect(second.targetPosition).not.toEqual(first.targetPosition)
    expect(Math.hypot(
      second.targetPosition.x - first.targetPosition.x,
      second.targetPosition.z - first.targetPosition.z
    )).toBeGreaterThanOrEqual(MINIMUM_TACTICAL_FOUNDATION_SPACING)
    expect(tracker.progress()).toEqual({ completed: 1, quota: 2 })
    put(value.blocks, second.targetPosition, 'obsidian')

    expect(tracker.resolve('MCAI_002')).toBeNull()
    expect(tracker.isCompleted()).toBe(true)
    expect(tracker.progress()).toEqual({ completed: 2, quota: 2 })
    value.blocks.delete(key(first.targetPosition))
    value.blocks.delete(key(second.targetPosition))
    expect(tracker.resolve('MCAI_002')).toBeNull()

    tracker.beginEpisode('episode-b', 1)
    expect(tracker.resolve('MCAI_002')).not.toBeNull()
  })

  it('raises the bounded placement quota to three at stage three', () => {
    expect(tacticalPlacementQuotaForStage(Number.NaN)).toBe(2)
    expect(tacticalPlacementQuotaForStage(1)).toBe(2)
    expect(tacticalPlacementQuotaForStage(2)).toBe(2)
    expect(tacticalPlacementQuotaForStage(3)).toBe(3)
    expect(tacticalPlacementQuotaForStage(99)).toBe(3)

    const value = arenaBot()
    const tracker = new TacticalPlacementTracker(value.bot)
    tracker.beginEpisode('advanced', 3)
    const positions: Vec3[] = []
    for (let index = 0; index < 3; index++) {
      const target = tracker.resolve('MCAI_002')
      expect(target, `stage-three target ${index + 1}`).not.toBeNull()
      positions.push(target!.targetPosition.clone())
      put(value.blocks, target!.targetPosition, 'obsidian')
      value.bot.entity.position.z -= 2
      value.assigned.position.z -= 2
    }
    expect(tracker.resolve('MCAI_002')).toBeNull()
    expect(tracker.progress()).toEqual({ completed: 3, quota: 3 })
    expect(new Set(positions.map(key)).size).toBe(3)
    for (let first = 0; first < positions.length; first++) {
      for (let second = first + 1; second < positions.length; second++) {
        expect(Math.hypot(
          positions[first].x - positions[second].x,
          positions[first].z - positions[second].z
        )).toBeGreaterThanOrEqual(MINIMUM_TACTICAL_FOUNDATION_SPACING)
      }
    }
  })

  it('bounds policy mine replacement by confirmation, TTL, cap, and episode reset', () => {
    const value = arenaBot()
    const mined = put(value.blocks, new Vec3(0, 64, -3), 'stone')
    const tracker = new TacticalPlacementTracker(value.bot)
    tracker.beginEpisode('mine-bounds', 1)
    const reservation = tracker.reservePolicyMineReplacement(
      mined, 'MCAI_002', 10
    )!
    expect(reservation).not.toBeNull()

    // Air alone is not proof of a completed policy dig.
    value.blocks.delete(key(mined.position))
    expect(tracker.resolve('MCAI_002', 11)?.targetPosition).not.toEqual(mined.position)
    expect(tracker.hasAdoptedPolicyMineReplacement()).toBe(false)
    tracker.confirmPolicyMineReplacement(reservation)
    expect(tracker.resolve('MCAI_002', 12)?.targetPosition).toEqual(mined.position)
    expect(tracker.hasAdoptedPolicyMineReplacement()).toBe(true)

    // The exact marker expires before server reward attribution does, resumes
    // ordinary targets, and cannot be farmed a second time this episode.
    tracker.resolve('MCAI_002', 10 + POLICY_MINE_REPLACEMENT_TIMEOUT_TICKS + 1)
    expect(tracker.hasPolicyMineReplacementSequence()).toBe(false)
    put(value.blocks, mined.position, 'stone')
    expect(tracker.reservePolicyMineReplacement(mined, 'MCAI_002', 200)).toBeNull()

    tracker.beginEpisode('mine-bounds-reset', 1)
    expect(tracker.hasAdoptedPolicyMineReplacement()).toBe(false)
    expect(tracker.reservePolicyMineReplacement(mined, 'MCAI_002', 1)).not.toBeNull()
  })

  it('rejects cancelled, stale, fallen-plane, and opponent-distant mine replacements', () => {
    const value = arenaBot()
    const mined = put(value.blocks, new Vec3(0, 64, -3), 'stone')
    const tracker = new TacticalPlacementTracker(value.bot)
    tracker.beginEpisode('mine-reject', 1)
    const cancelled = tracker.reservePolicyMineReplacement(mined, 'MCAI_002', 1)!
    tracker.cancelPolicyMineReplacement(cancelled)
    value.blocks.delete(key(mined.position))
    tracker.confirmPolicyMineReplacement(cancelled)
    expect(tracker.resolve('MCAI_002', 2)?.targetPosition).not.toEqual(mined.position)
    expect(tracker.hasAdoptedPolicyMineReplacement()).toBe(false)

    put(value.blocks, mined.position, 'stone')
    tracker.beginEpisode('mine-fallen', 1)
    value.bot.entity.position.y = 63
    expect(tracker.reservePolicyMineReplacement(mined, 'MCAI_002', 1)).toBeNull()
    value.bot.entity.position.y = 64

    const tooFar = put(value.blocks, new Vec3(0, 64, -2), 'stone')
    expect(isSafeTacticalMiningTarget(value.bot, tooFar, 'MCAI_002')).toBe(true)
    expect(isUsefulPolicyMineReplacementCandidate(
      value.bot, tooFar, 'MCAI_002'
    )).toBe(false)
  })

  it('fails closed when a reserved crosshair block disappears or has malformed position', () => {
    const value = arenaBot()
    const mined = put(value.blocks, new Vec3(0, 64, -3), 'stone')
    const tracker = new TacticalPlacementTracker(value.bot)
    tracker.beginEpisode('mine-missing-crosshair', 1)
    expect(tracker.reservePolicyMineReplacement(
      mined, 'MCAI_002', 1
    )).not.toBeNull()

    expect(() => tracker.isPolicyMineReplacementPriorityCandidate(
      undefined, 'MCAI_002', 2
    )).not.toThrow()
    expect(tracker.isPolicyMineReplacementPriorityCandidate(
      undefined, 'MCAI_002', 2
    )).toBe(false)
    expect(tracker.isPolicyMineReplacementPriorityCandidate(
      { position: undefined }, 'MCAI_002', 2
    )).toBe(false)
    expect(tracker.isPolicyMineReplacementPriorityCandidate(
      { position: { x: Number.NaN, y: 64, z: -3 } }, 'MCAI_002', 2
    )).toBe(false)

    // The original exact target remains valid if it reappears before TTL.
    expect(tracker.isPolicyMineReplacementPriorityCandidate(
      mined, 'MCAI_002', 3
    )).toBe(true)
  })
})
