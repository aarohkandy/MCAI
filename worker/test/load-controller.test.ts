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
      const controller = new LoadController(arena, { initialPairs: 2, maximumPairs: 8, stableSamplesBeforeIncrease: 2 })
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
})

describe('LoadController.sample', () => {
  it('ramps back up on a healthy server instead of pinning at one pair', async () => {
    // Regression: sample() used to blend `1 - os.freemem()/os.totalmem()` into memory_fraction.
    // os.freemem() excludes reclaimable page cache, so on a warm host that term is ~0.97 and the
    // controller treated a perfectly healthy server as permanently overloaded, collapsing to one
    // pair forever and wasting most of the machine.
    const requests: Array<{ pairs: number }> = []
    const arena = {
      request: async (command: string, payload?: { pairs: number }) => {
        if (command === 'set_max_pairs' && payload) requests.push(payload)
        return { estimated_tps: 20, p95_tick_ms: 20, memory_fraction: 0.25 }
      }
    } as any
    const controller = new LoadController(arena, {
      initialPairs: 2, maximumPairs: 6, sampleIntervalMs: 5, stableSamplesBeforeIncrease: 1
    })
    await controller.start()
    await new Promise(resolve => setTimeout(resolve, 60))
    controller.stop()
    // start() sets the initial pair count; a healthy server must then push it UP, never down.
    const applied = requests.map(entry => entry.pairs)
    expect(applied[0]).toBe(2)
    expect(Math.max(...applied)).toBeGreaterThan(2)
    expect(Math.min(...applied)).toBe(2)
  })
})
