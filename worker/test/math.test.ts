import { describe, expect, it } from 'vitest'
import { egocentric, normalizeAngle } from '../src/math.js'

describe('coordinate normalization', () => {
  it('keeps a vector stable at zero yaw', () => {
    expect(egocentric({ x: 1, y: 2, z: 3 }, 0)).toEqual({ x: 1, y: 2, z: 3 })
  })

  it('normalizes arbitrarily large angles', () => {
    expect(normalizeAngle(8 * Math.PI + 0.5)).toBeCloseTo(0.5)
  })
})
