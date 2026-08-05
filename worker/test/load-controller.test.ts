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
  it('judges load on the server heap only, never on host free memory', async () => {
    // Regression: sample() used to fold `1 - os.freemem()/os.totalmem()` into memory_fraction.
    // os.freemem() excludes reclaimable page cache, so on a warm host that term is ~0.97 -- always
    // above the 0.8 overload threshold. The controller then treated a perfectly healthy server as
    // overloaded, collapsed to one pair on its first sample, and could never climb back, wasting
    // most of a large VM.
    //
    // Asserting on the resulting pair count would be flaky, because evaluate() also (correctly)
    // reacts to real event-loop delay on a busy CI machine. So assert the precise thing that broke:
    // the status handed to evaluate() must carry the server's OWN heap fraction, untouched.
    const seen: Array<Record<string, unknown>> = []
    const arena = {
      request: async () => ({ estimated_tps: 20, p95_tick_ms: 20, memory_fraction: 0.25 })
    } as any
    const controller = new LoadController(arena, {
      initialPairs: 2, maximumPairs: 6, sampleIntervalMs: 5, stableSamplesBeforeIncrease: 1
    })
    const realEvaluate = controller.evaluate.bind(controller)
    controller.evaluate = (status, delay) => { seen.push(status); return realEvaluate(status, delay) }

    await controller.start()
    await new Promise(resolve => setTimeout(resolve, 40))
    controller.stop()

    expect(seen.length).toBeGreaterThan(0)
    for (const status of seen) {
      // 0.25 is what the server reported. The old code raised this to ~0.97 from host memory.
      expect(status.memory_fraction).toBe(0.25)
      expect(Number(status.memory_fraction)).toBeLessThan(0.8)
    }
  })

  it('applies the initial pair count on start', async () => {
    const applied: number[] = []
    const arena = {
      request: async (command: string, payload?: { pairs: number }) => {
        if (command === 'set_max_pairs' && payload) applied.push(payload.pairs)
        return { estimated_tps: 20, p95_tick_ms: 20, memory_fraction: 0.25 }
      }
    } as any
    const controller = new LoadController(arena, { initialPairs: 3, maximumPairs: 6 })
    await controller.start()
    controller.stop()
    expect(applied).toEqual([3])
  })
})
