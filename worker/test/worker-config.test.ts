import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  RolloutWorker,
  TRAINER_STALL_SAFETY_MS,
  matchConfigurationFromPayload,
  optionsFromEnvironment,
  trainerStallSafetyDue,
  trainerCapabilities
} from '../src/worker.js'

const originalTeachers = process.env.MCAI_TEACHERS_ENABLED

afterEach(() => {
  if (originalTeachers === undefined) delete process.env.MCAI_TEACHERS_ENABLED
  else process.env.MCAI_TEACHERS_ENABLED = originalTeachers
})

describe('per-match combat configuration', () => {
  it('forwards arena snapshots to every agent for episode/assignment filtering', () => {
    const first = { applyArenaSnapshot: vi.fn() }
    const second = { applyArenaSnapshot: vi.fn() }
    const worker = Object.create(RolloutWorker.prototype) as any
    worker.agents = [first, second]
    const payload = { episode_id: 'episode-live', fighters: [] }

    worker.onArenaEvent({ type: 'event', event: 'arena_snapshot', payload })
    expect(first.applyArenaSnapshot).toHaveBeenCalledWith(payload)
    expect(second.applyArenaSnapshot).toHaveBeenCalledWith(payload)
  })

  it('advertises delayed-action correlation in the trainer hello capability set', () => {
    expect(trainerCapabilities()).toContain('action-correlation-v1')
    expect(trainerCapabilities()).toContain('execution-source-v1')
    expect(trainerCapabilities()).toContain('legal-controls-v1')
    expect(trainerCapabilities()).toContain('finite-wire-v1')
  })

  it('lets the rotating lane override startup mode and consumes radius/stage', () => {
    expect(matchConfigurationFromPayload({
      lane: 'terrain', mode: 'combined', arena_radius: 8, curriculum_stage: 4
    }, 'sword', true)).toEqual({
      lane: 'terrain', mode: 'terrain', radius: 8, stage: 4,
      teachersEnabled: true, terrainEnabled: true
    })

    expect(matchConfigurationFromPayload({
      lane: 'sword_retention', arena_radius: 5, curriculum_stage: 1
    }, 'combined', true).mode).toBe('sword')
    expect(matchConfigurationFromPayload({ lane: 'unknown' }, 'combined', true).lane).toBe('')
  })

  it('defaults held-out evaluation to pure policy but permits an explicit override', () => {
    expect(matchConfigurationFromPayload({ evaluation: true }, 'combined', true).teachersEnabled)
      .toBe(false)
    expect(matchConfigurationFromPayload({
      evaluation: true, teachers_enabled: true
    }, 'combined', false).teachersEnabled).toBe(true)
  })

  it('supports a process-wide teacher kill switch', () => {
    process.env.MCAI_TEACHERS_ENABLED = 'off'
    expect(optionsFromEnvironment().teachersEnabled).toBe(false)
    process.env.MCAI_TEACHERS_ENABLED = 'yes'
    expect(optionsFromEnvironment().teachersEnabled).toBe(true)
  })

  it('starts local safety only after the bounded trainer-response grace period', () => {
    expect(trainerStallSafetyDue(1_000, 1_000 + TRAINER_STALL_SAFETY_MS - 1)).toBe(false)
    expect(trainerStallSafetyDue(1_000, 1_000 + TRAINER_STALL_SAFETY_MS)).toBe(true)
    expect(trainerStallSafetyDue(0, 10_000)).toBe(false)
  })
})
