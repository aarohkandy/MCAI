import { describe, expect, it } from 'vitest'

import {
  POLICY_DECISION_PERIOD_MS,
  responseTickDelay
} from '../src/worker.js'


describe('response-driven policy scheduling', () => {
  it('wakes immediately after a response that has already crossed the 20 Hz deadline', () => {
    expect(responseTickDelay(163, 150)).toBe(0)
  })

  it('waits only for the remainder of the 20 Hz deadline after a fast response', () => {
    expect(responseTickDelay(143.2, 150)).toBe(7)
  })

  it('removes fixed-timer quantization without exceeding 20 Hz', () => {
    const controlMs = 13
    const trainerRttMs = 50
    const responseDrivenInterval = Math.max(
      POLICY_DECISION_PERIOD_MS,
      controlMs + trainerRttMs
    )
    const fixedTimerInterval = Math.ceil(
      (controlMs + trainerRttMs) / POLICY_DECISION_PERIOD_MS
    ) * POLICY_DECISION_PERIOD_MS

    expect(responseDrivenInterval).toBe(63)
    expect(fixedTimerInterval).toBe(100)
    expect(1000 / responseDrivenInterval).toBeGreaterThan(15)
  })

  it('fails open for invalid clocks instead of stalling combat', () => {
    expect(responseTickDelay(Number.NaN, 150)).toBe(0)
    expect(responseTickDelay(100, Number.POSITIVE_INFINITY)).toBe(0)
  })
})
