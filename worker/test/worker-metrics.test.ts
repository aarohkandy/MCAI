import { describe, expect, it, vi } from 'vitest'
import { WorkerPerformanceMetrics } from '../src/worker-metrics.js'

describe('WorkerPerformanceMetrics', () => {
  it('reports latency percentiles, stalls, skipped ticks, and effective Hz', () => {
    const clock = vi.spyOn(performance, 'now').mockReturnValue(0)
    const metrics = new WorkerPerformanceMetrics()
    for (const value of [10, 20, 30, 250]) metrics.noteRoundTrip(value)
    metrics.noteControlApplication(5)
    metrics.noteControlApplication(15)
    metrics.noteSkippedTick(3)
    clock.mockReturnValue(100)
    metrics.noteDecisions(2)
    clock.mockReturnValue(1_100)
    metrics.noteDecisions(2)
    clock.mockReturnValue(2_100)
    expect(metrics.snapshot()).toEqual({
      decision_round_trip_ms_median: 20,
      decision_round_trip_ms_p95: 30,
      control_application_ms_median: 5,
      control_application_ms_p95: 5,
      skipped_50ms_ticks: 3,
      optimizer_stalls_over_200ms: 1,
      effective_decisions_hz_per_agent: 1,
      decision_rate_window_seconds: 2.1,
      decisions_in_window: 4,
      active_agent_seconds_in_window: 4
    })
    clock.mockRestore()
  })

  it('weights changing active-agent counts instead of dividing by the current count', () => {
    const clock = vi.spyOn(performance, 'now').mockReturnValue(0)
    const metrics = new WorkerPerformanceMetrics()
    clock.mockReturnValue(100)
    metrics.noteDecisions(8)
    clock.mockReturnValue(1_100)
    metrics.noteDecisions(4)
    clock.mockReturnValue(2_100)

    const snapshot = metrics.snapshot()
    expect(snapshot.decisions_in_window).toBe(12)
    expect(snapshot.active_agent_seconds_in_window).toBe(12)
    expect(snapshot.effective_decisions_hz_per_agent).toBe(1)
    clock.mockRestore()
  })
})
