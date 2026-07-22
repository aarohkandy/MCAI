import { describe, expect, it } from 'vitest'
import { LoadController } from '../src/load-controller.js'

describe('LoadController', () => {
  it('backs off on every defined resource threshold', () => {
    const arena = { request: async () => ({}) } as any
    const overloaded = [
      [{ estimated_tps: 19.4, p95_tick_ms: 50, memory_fraction: 0.5 }, 2],
      [{ estimated_tps: 20, p95_tick_ms: 56, memory_fraction: 0.5 }, 2],
      [{ estimated_tps: 20, p95_tick_ms: 50, memory_fraction: 0.81 }, 2],
      [{ estimated_tps: 20, p95_tick_ms: 50, memory_fraction: 0.5 }, 10.1]
    ] as const
    for (const [status, delay] of overloaded) {
      const controller = new LoadController(arena, { initialPairs: 3, maximumPairs: 8, stableSamplesBeforeIncrease: 2 })
      expect(controller.evaluate(status, delay)).toBe('decrease')
    }
  })

  it('adds only one pair after a stable window', () => {
    const arena = { request: async () => ({}) } as any
    const controller = new LoadController(arena, { initialPairs: 2, maximumPairs: 4, stableSamplesBeforeIncrease: 2 })
    const healthy = { estimated_tps: 20, p95_tick_ms: 50, memory_fraction: 0.5 }
    expect(controller.evaluate(healthy, 2)).toBe('hold')
    expect(controller.evaluate(healthy, 2)).toBe('increase')
  })

  it('keeps ordinary load spikes above the configured parallel-match floor', () => {
    const arena = { request: async () => ({}) } as any
    const controller = new LoadController(arena, {
      initialPairs: 4,
      minimumPairs: 3,
      maximumPairs: 4
    })
    const moderatelyLoaded = { estimated_tps: 19, p95_tick_ms: 65, memory_fraction: 0.5 }
    expect(controller.evaluate(moderatelyLoaded, 2)).toBe('decrease')
    expect(controller.evaluate(moderatelyLoaded, 2)).toBe('hold')
  })

  it('treats the configured four-pair minimum as a hard floor during sustained emergencies', () => {
    const arena = { request: async () => ({}) } as any
    const controller = new LoadController(arena, {
      initialPairs: 6,
      minimumPairs: 4,
      maximumPairs: 6
    })
    const emergency = { estimated_tps: 14, p95_tick_ms: 170, memory_fraction: 0.5 }
    expect(controller.evaluate(emergency, 2)).toBe('decrease')
    expect(controller.evaluate(emergency, 2)).toBe('decrease')
    for (let sample = 0; sample < 20; sample += 1) {
      expect(controller.evaluate(emergency, 200)).toBe('hold')
    }
  })

  it('uses one pair only when the configured bot capacity cannot support two', () => {
    const arena = { request: async () => ({}) } as any
    const controller = new LoadController(arena, {
      initialPairs: 1,
      minimumPairs: 1,
      maximumPairs: 1
    })
    const emergency = { estimated_tps: 10, p95_tick_ms: 200, memory_fraction: 0.99 }
    expect(controller.evaluate(emergency, 200)).toBe('hold')
  })
})
