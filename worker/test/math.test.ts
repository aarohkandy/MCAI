import { describe, expect, it } from 'vitest'
import { egocentric, normalizeAngle } from '../src/math.js'

describe('coordinate normalization', () => {
  it('keeps a vector stable at zero yaw', () => {
    expect(egocentric({ x: 1, y: 2, z: 3 }, 0)).toEqual({ x: 1, y: 2, z: 3 })
  })

  it('maps the forward direction to -z at every yaw (yaw-invariant frame)', () => {
    for (const yaw of [0, 0.3, 1.1, Math.PI / 2, 2.5, -1.7, Math.PI]) {
      // forward vector for canonical yaw is (-sin yaw, 0, -cos yaw)
      const forward = { x: -Math.sin(yaw), y: 0, z: -Math.cos(yaw) }
      const ego = egocentric(forward, yaw)
      expect(ego.x).toBeCloseTo(0)
      expect(ego.z).toBeCloseTo(-1)
    }
  })

  it('places an opponent directly ahead on -z regardless of yaw', () => {
    for (const yaw of [0, Math.PI / 2, 2.0, -1.2]) {
      const worldDelta = { x: -Math.sin(yaw) * 5, y: 1, z: -Math.cos(yaw) * 5 }
      const ego = egocentric(worldDelta, yaw)
      expect(ego.x).toBeCloseTo(0)
      expect(ego.y).toBeCloseTo(1)
      expect(ego.z).toBeCloseTo(-5)
    }
  })

  it('preserves vector length (pure rotation)', () => {
    const v = { x: 2, y: -1, z: 4 }
    const ego = egocentric(v, 1.234)
    expect(Math.hypot(ego.x, ego.y, ego.z)).toBeCloseTo(Math.hypot(v.x, v.y, v.z))
  })

  it('normalizes arbitrarily large angles', () => {
    expect(normalizeAngle(8 * Math.PI + 0.5)).toBeCloseTo(0.5)
  })
})
