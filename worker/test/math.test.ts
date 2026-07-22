import { describe, expect, it } from 'vitest'
import { egocentric, mineflayerBodyRelative, normalizeAngle } from '../src/math.js'

describe('coordinate normalization', () => {
  it('keeps a vector stable at zero yaw', () => {
    expect(egocentric({ x: 1, y: 2, z: 3 }, 0)).toEqual({ x: 1, y: 2, z: 3 })
    expect(mineflayerBodyRelative({ x: 1, y: 2, z: 3 }, 0)).toEqual({ x: 1, y: 2, z: 3 })
  })

  it.each([
    { yaw: Math.PI / 2, forward: { x: -2, y: 0, z: 0 }, right: { x: 0, y: 0, z: -2 } },
    { yaw: -Math.PI / 2, forward: { x: 2, y: 0, z: 0 }, right: { x: 0, y: 0, z: 2 } },
    { yaw: Math.PI, forward: { x: 0, y: 0, z: 2 }, right: { x: -2, y: 0, z: 0 } }
  ])('keeps -Z forward and +X right at yaw $yaw', ({ yaw, forward, right }) => {
    const localForward = mineflayerBodyRelative(forward, yaw)
    const localRight = mineflayerBodyRelative(right, yaw)
    expect(localForward.x).toBeCloseTo(0, 8)
    expect(localForward.z).toBeCloseTo(-2, 8)
    expect(localRight.x).toBeCloseTo(2, 8)
    expect(localRight.z).toBeCloseTo(0, 8)
  })

  it('leaves the legacy opposite-sign transform unchanged for old checkpoints', () => {
    expect(egocentric({ x: -2, y: 0, z: 0 }, Math.PI / 2).z).toBeCloseTo(2)
    expect(mineflayerBodyRelative({ x: -2, y: 0, z: 0 }, Math.PI / 2).z).toBeCloseTo(-2)
  })

  it('normalizes arbitrarily large angles', () => {
    expect(normalizeAngle(8 * Math.PI + 0.5)).toBeCloseTo(0.5)
  })
})
