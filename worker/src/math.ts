import type { Vec3Value } from './contracts.js'

export function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value))
}

export function normalizeAngle(value: number): number {
  let angle = value
  while (angle > Math.PI) angle -= Math.PI * 2
  while (angle < -Math.PI) angle += Math.PI * 2
  return angle
}

export function toVec3Value(value: { x?: number; y?: number; z?: number } | null | undefined): Vec3Value {
  return {
    x: finite(value?.x),
    y: finite(value?.y),
    z: finite(value?.z)
  }
}

/**
 * Rotate a world-space delta into the controlled player's egocentric frame.
 *
 * The player's forward vector is (-sin yaw, 0, -cos yaw) (mineflayer/eagler canonical
 * yaw). A correct egocentric frame is yaw-invariant: something directly ahead must map
 * to the same coordinates at every yaw. That requires R(+yaw) here — the previous
 * R(-yaw) sent "directly ahead" to (-sin 2·yaw, …), i.e. it rotated with the player
 * instead of cancelling the rotation, corrupting every relative position/velocity at
 * non-zero yaw. Verified: forward(yaw) -> (0, 0, -1) for all yaw with the signs below.
 */
export function egocentric(delta: Vec3Value, yaw: number): Vec3Value {
  const sin = Math.sin(yaw)
  const cos = Math.cos(yaw)
  return {
    x: delta.x * cos - delta.z * sin,
    y: delta.y,
    z: delta.x * sin + delta.z * cos
  }
}

export function subtract(a: Vec3Value, b: Vec3Value): Vec3Value {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z }
}

export function distance(a: Vec3Value, b: Vec3Value): number {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z)
}

function finite(value: number | undefined): number {
  return Number.isFinite(value) ? Number(value) : 0
}
