import { describe, expect, it } from 'vitest'
import { NOOP_ACTION, SCHEMA_VERSION, validateAction } from '../src/contracts.js'

describe('ActionV1', () => {
  it('accepts the canonical no-op', () => {
    expect(validateAction({ ...NOOP_ACTION })).toEqual(NOOP_ACTION)
  })

  it('clamps camera deltas to game-facing bounds', () => {
    const value = validateAction({ ...NOOP_ACTION, yaw_delta: 99, pitch_delta: -99 })
    expect(value.yaw_delta).toBe(Math.PI)
    expect(value.pitch_delta).toBe(-Math.PI / 2)
  })

  it('rejects an incompatible schema', () => {
    expect(() => validateAction({ ...NOOP_ACTION, schema_version: 2 as typeof SCHEMA_VERSION })).toThrow()
  })
})
