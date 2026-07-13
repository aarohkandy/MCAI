import os from 'node:os'
import { monitorEventLoopDelay, type IntervalHistogram } from 'node:perf_hooks'
import type { ArenaClient } from './arena-client.js'

export type LoadControllerOptions = {
  initialPairs: number
  maximumPairs: number
  sampleIntervalMs?: number
  stableSamplesBeforeIncrease?: number
}

export class LoadController {
  private readonly histogram: IntervalHistogram
  private readonly intervalMs: number
  private readonly stableSamplesBeforeIncrease: number
  private pairs: number
  private stableSamples = 0
  private timer: NodeJS.Timeout | null = null
  private stopped = false

  constructor(private readonly arena: ArenaClient, private readonly options: LoadControllerOptions) {
    this.pairs = Math.max(1, Math.min(options.initialPairs, options.maximumPairs))
    this.intervalMs = options.sampleIntervalMs ?? 30_000
    this.stableSamplesBeforeIncrease = options.stableSamplesBeforeIncrease ?? 10
    this.histogram = monitorEventLoopDelay({ resolution: 1 })
  }

  async start(): Promise<void> {
    this.stopped = false
    this.histogram.enable()
    await this.arena.request('set_max_pairs', { pairs: this.pairs })
    if (this.timer) clearInterval(this.timer)
    this.timer = setInterval(() => void this.sample(), this.intervalMs)
  }

  stop(): void {
    this.stopped = true
    if (this.timer) clearInterval(this.timer)
    this.timer = null
    this.histogram.disable()
  }

  evaluate(status: Record<string, unknown>, eventLoopP95Ms: number): 'hold' | 'increase' | 'decrease' {
    const overloaded = Number(status.estimated_tps ?? 0) < 19.5
      || Number(status.p95_tick_ms ?? Number.POSITIVE_INFINITY) > 55
      || Number(status.memory_fraction ?? 1) > 0.8
      || eventLoopP95Ms > 10
    if (overloaded) {
      this.stableSamples = 0
      if (this.pairs > 1) {
        this.pairs -= 1
        return 'decrease'
      }
      return 'hold'
    }
    this.stableSamples += 1
    if (this.stableSamples >= this.stableSamplesBeforeIncrease && this.pairs < this.options.maximumPairs) {
      this.stableSamples = 0
      this.pairs += 1
      return 'increase'
    }
    return 'hold'
  }

  private async sample(): Promise<void> {
    if (this.stopped) return
    const p95Ms = this.histogram.percentile(95) / 1e6
    this.histogram.reset()
    try {
      const status = await this.arena.request('status')
      status.memory_fraction = Math.max(
        Number(status.memory_fraction ?? 0),
        1 - os.freemem() / Math.max(1, os.totalmem())
      )
      const decision = this.evaluate(status, p95Ms)
      if (decision !== 'hold') {
        await this.arena.request('set_max_pairs', { pairs: this.pairs })
        console.info(`[load] ${decision} rollout concurrency to ${this.pairs} pairs`)
      }
    } catch (error) {
      console.warn('[load] unable to sample arena health:', String(error))
    }
  }
}
