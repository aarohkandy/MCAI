import type { Bot } from 'mineflayer'
import { Vec3 } from 'vec3'
import { describe, expect, it, vi } from 'vitest'
import { BlockSampler } from '../src/block-sampler.js'

function blockWorld(initialPads: Vec3[]): {
  bot: Bot
  pads: Set<string>
  setCursor: (position: Vec3 | null) => void
  blockAt: ReturnType<typeof vi.fn>
} {
  const pads = new Set(initialPads.map(key))
  let cursor: Vec3 | null = null
  const blockAt = vi.fn((position: Vec3) => ({
    name: pads.has(key(position)) ? 'obsidian' : 'air',
    boundingBox: pads.has(key(position)) ? 'block' : 'empty',
    hardness: pads.has(key(position)) ? 50 : 0,
    position: position.clone()
  }))
  const bot = {
    entity: { position: new Vec3(0, 64, 0), yaw: 0 },
    blockAt,
    blockAtCursor: vi.fn(() => cursor ? blockAt(cursor) : null)
  } as unknown as Bot
  return { bot, pads, setCursor: position => { cursor = position }, blockAt }
}

describe('randomized crystal-pad observations', () => {
  it('staggeres recurring scans by agent cadence key', () => {
    const first = blockWorld([new Vec3(1, 63, 1)])
    const second = blockWorld([new Vec3(1, 63, 1)])
    const a = new BlockSampler(first.bot, 5, 'agent-0')
    const b = new BlockSampler(second.bot, 5, 'agent-1')
    a.sample(1, undefined, 'episode')
    b.sample(1, undefined, 'episode')
    let previousA = first.blockAt.mock.calls.length
    let previousB = second.blockAt.mock.calls.length
    const scanTicksA: number[] = []
    const scanTicksB: number[] = []
    for (let tick = 2; tick <= 15; tick += 1) {
      a.sample(tick, undefined, 'episode')
      b.sample(tick, undefined, 'episode')
      const nextA = first.blockAt.mock.calls.length
      const nextB = second.blockAt.mock.calls.length
      if (nextA - previousA > 10) scanTicksA.push(tick)
      if (nextB - previousB > 10) scanTicksB.push(tick)
      previousA = nextA
      previousB = nextB
    }
    expect(scanTicksA.length).toBeGreaterThan(0)
    expect(scanTicksB.length).toBeGreaterThan(0)
    expect(scanTicksA[0]).not.toBe(scanTicksB[0])
  })

  it('does not probe blocks in unloaded columns', () => {
    const world = blockWorld([new Vec3(2, 63, 0)])
    ;(world.bot as any).world = {
      getColumnAt: (position: Vec3) => Math.floor(position.x / 16) === 0 ? {} : null
    }
    const sampler = new BlockSampler(world.bot, 5)
    sampler.sample(1, { x: 18, y: 64, z: 0 }, 'episode')
    expect(world.blockAt.mock.calls.every(([position]) => Math.floor(position.x / 16) === 0)).toBe(true)
  })

  it('invalidates cached blocks when a new episode moves the pads', () => {
    const first = new Vec3(2, 63, -1)
    const second = new Vec3(-2, 63, 1)
    const world = blockWorld([first])
    const sampler = new BlockSampler(world.bot, 5)

    const before = sampler.sample(100, undefined, 'episode-a')
      .filter(block => block.name === 'obsidian')
    expect(before).toHaveLength(1)
    expect(before[0].relative_position).toEqual({ x: 2, y: -1, z: -1 })

    world.pads.delete(key(first))
    world.pads.add(key(second))
    const after = sampler.sample(0, undefined, 'episode-b')
      .filter(block => block.name === 'obsidian')
    expect(after).toHaveLength(1)
    expect(after[0].relative_position).toEqual({ x: -2, y: -1, z: 1 })
  })

  it('reprojects cached world positions after turning instead of repeating a stale aim', () => {
    const pad = new Vec3(2, 63, 0)
    const world = blockWorld([pad])
    const sampler = new BlockSampler(world.bot, 5)
    const first = sampler.sample(10, undefined, 'episode-a')
      .find(block => block.name === 'obsidian')
    const scanCalls = world.blockAt.mock.calls.length
    expect(first?.relative_position).toEqual({ x: 2, y: -1, z: 0 })
    expect(first?.body_relative_position).toEqual({ x: 2, y: -1, z: 0 })
    expect(first?.body_relative_velocity).toEqual({ x: 0, y: 0, z: 0 })
    expect(first?.raycastable).toBe(false)

    ;(world.bot.entity as any).yaw = Math.PI / 2
    world.setCursor(pad)
    const turned = sampler.sample(11, undefined, 'episode-a')
      .find(block => block.name === 'obsidian')

    // No full scan occurred, but the target is now expressed in the current
    // camera frame and current crosshair state.
    expect(world.blockAt.mock.calls.length).toBe(scanCalls + 1)
    expect(turned?.relative_position.x).toBeCloseTo(0, 6)
    expect(turned?.relative_position.y).toBe(-1)
    expect(turned?.relative_position.z).toBeCloseTo(-2, 6)
    expect(turned?.body_relative_position.x).toBeCloseTo(0, 6)
    expect(turned?.body_relative_position.z).toBeCloseTo(2, 6)
    expect(turned?.raycastable).toBe(true)
  })

  it('uses tactical quotas instead of filling 48 slots with air or self-nearest floor', () => {
    const validPad = new Vec3(8, 63, 1)
    const blockedPad = new Vec3(-2, 63, 2)
    const cursorPosition = new Vec3(5, 64, 2)
    const opponentObstacle = new Vec3(9, 64, 2)
    const solids = new Set([validPad, blockedPad, cursorPosition, opponentObstacle].map(key))
    const blockAt = vi.fn((position: Vec3) => {
      const name = key(position)
      if (name === key(blockedPad.offset(0, 1, 0))) {
        return { name: 'water', type: 9, boundingBox: 'empty', hardness: 100, position: position.clone() }
      }
      if (position.y === 63 || solids.has(name)) {
        const obsidian = name === key(validPad) || name === key(blockedPad)
        return {
          name: obsidian ? 'obsidian' : 'stone',
          type: obsidian ? 49 : 1,
          boundingBox: 'block',
          hardness: obsidian ? 50 : 1.5,
          position: position.clone()
        }
      }
      return { name: 'air', type: 0, boundingBox: 'empty', hardness: 0, position: position.clone() }
    })
    const bot = {
      entity: { position: new Vec3(0, 64, 0), velocity: new Vec3(0, 0, 0), yaw: 0 },
      blockAt,
      blockAtCursor: vi.fn(() => blockAt(cursorPosition))
    } as unknown as Bot
    const sampler = new BlockSampler(bot, 5)
    const first = sampler.sample(1, { x: 9, y: 64, z: 0 }, 'episode-tactical')

    expect(first.length).toBeLessThanOrEqual(48)
    expect(first.every(block => !block.replaceable && !['empty', 'liquid'].includes(block.collision))).toBe(true)
    expect(at(first, cursorPosition)?.raycastable).toBe(true)
    expect(at(first, validPad)?.crystal_clearance).toBe(true)
    expect(at(first, opponentObstacle)).toBeDefined()
    expect(first.filter(block => block.relative_position.y >= 0).length).toBeGreaterThanOrEqual(2)
    expect(first.filter(block => Math.hypot(
      block.relative_position.x - 9, block.relative_position.z
    ) <= 3).length).toBeGreaterThan(0)

    const second = sampler.sample(2, { x: 9, y: 64, z: 0 }, 'episode-tactical')
    expect(second.map(block => block.relative_position)).toEqual(first.map(block => block.relative_position))
    expect(second.every(block => block.sample_age_ticks === 1)).toBe(true)

    ;(bot.blockAtCursor as ReturnType<typeof vi.fn>).mockReturnValue(blockAt(blockedPad))
    const refreshed = sampler.sample(6, { x: 9, y: 64, z: 0 }, 'episode-tactical')
    expect(at(refreshed, blockedPad)?.raycastable).toBe(true)
    expect(at(refreshed, blockedPad)?.crystal_clearance).toBe(false)
  })

  it('guarantees exactly one marked support and clears it without stale cache state', () => {
    const support = new Vec3(2, 63, -2)
    const world = blockWorld([support])
    const sampler = new BlockSampler(world.bot, 5)

    const marked = sampler.sample(
      10, { x: 5, y: 64, z: -2 }, 'episode-foundation', support
    )
    expect(marked.filter(block => block.tactical_placement_target)).toHaveLength(1)
    expect(at(marked, support)?.tactical_placement_target).toBe(true)

    const cleared = sampler.sample(
      11, { x: 5, y: 64, z: -2 }, 'episode-foundation'
    )
    expect(cleared.every(block => !block.tactical_placement_target)).toBe(true)
  })
})

function at(blocks: ReturnType<BlockSampler['sample']>, position: Vec3) {
  return blocks.find(block => block.relative_position.x === position.x
    && block.relative_position.y === position.y - 64
    && block.relative_position.z === position.z)
}

function key(position: { x: number; y: number; z: number }): string {
  return `${position.x},${position.y},${position.z}`
}
