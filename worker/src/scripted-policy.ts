import { NOOP_ACTION, type ActionV1, type AnyObservation } from './contracts.js'

export type ScriptedStyle = 'rush' | 'strafe' | 'retreat' | 'jump_critical' | 'defensive' | 'erratic'

export function scriptedAction(observation: AnyObservation, style: ScriptedStyle = 'strafe'): ActionV1 {
  const opponent = observation.opponent
  if (!opponent) return { ...NOOP_ACTION, yaw_delta: 0.15 }
  const relative = opponent.relative_position
  const horizontal = Math.hypot(relative.x, relative.z)
  const yawDelta = clamp(Math.atan2(-relative.x, -relative.z), -0.8, 0.8)
  const pitchDelta = clamp(Math.atan2(relative.y, Math.max(0.1, horizontal)) - observation.self.pitch, -0.4, 0.4)
  const tick = observation.match.tick
  let forward: -1 | 0 | 1 = horizontal > 2.75 ? 1 : 0
  let strafe: -1 | 0 | 1 = tick % 80 < 40 ? -1 : 1
  if (style === 'rush') strafe = 0
  if (style === 'retreat' || style === 'defensive') forward = horizontal < 4.5 ? -1 : 0
  if (style === 'erratic') strafe = tick % 17 < 8 ? -1 : 1
  return {
    ...NOOP_ACTION,
    forward,
    strafe,
    sprint: forward === 1,
    jump: (style === 'erratic' && tick % 23 === 0) || (style === 'jump_critical' && tick % 13 === 0),
    yaw_delta: yawDelta,
    pitch_delta: pitchDelta,
    primary: horizontal <= 3.0 && observation.self.attack_cooldown >= 0.9 ? 'attack' : 'none'
  }
}

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value))
}
