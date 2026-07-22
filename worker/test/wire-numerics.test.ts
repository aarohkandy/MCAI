import { describe, expect, it } from 'vitest'
import { NOOP_ACTION, type StepBatch } from '../src/contracts.js'
import { sanitizeStepForWire } from '../src/wire-numerics.js'

function step(): StepBatch['steps'][number] {
  return {
    agent_id: 'worker/agent-1',
    observation: {
      schema_version: 1,
      match: {
        episode_id: 'episode-finite', tick: 4, policy_version: 2,
        arena_seed: 3, action_delay_ticks: 0, observation_delay_ticks: 0
      },
      self: {
        health: 20, absorption: 0, food: 20,
        position: { x: 0, y: 64, z: 0 }, velocity: { x: 0, y: 0, z: 0 },
        yaw: 0, pitch: 0, on_ground: true, sprinting: false, sneaking: false,
        hurt_time: 0, attack_cooldown: 1, active_hand: 'none', use_ticks: 0,
        mining_progress: 0, selected_hotbar: 0,
        mainhand: item(), hotbar: [], offhand: item(), armor: [],
        raycast: { kind: 'none', distance: 0, block_name: '', entity_kind: '' }
      },
      opponent: null, entities: [], blocks: [],
      action_mask: {
        attack: false, combat_attack_ready: false, crystal_place_ready: false,
        crystal_attack_ready: false, tactical_block_break_ready: false,
        use_main: false, use_offhand: false, release_use: false,
        swap_offhand: false, hotbar: Array(9).fill(false)
      }
    },
    reward: 0, terminated: false, truncated: false, info: {},
    execution: { source: 'policy', action: { ...NOOP_ACTION }, action_id: 91 }
  }
}

function item() {
  return { name: 'air', count: 0, durability: 0, max_durability: 0, enchant_hash: 0 }
}

describe('finite worker wire boundary', () => {
  it('leaves a finite step untouched', () => {
    const value = step()
    expect(sanitizeStepForWire(value)).toBe(value)
  })

  it('replaces only nonfinite leaves, reports paths, and excludes PPO correlation', () => {
    const value = step()
    value.observation.self.yaw = Number.NaN
    value.observation.self.velocity.x = Number.POSITIVE_INFINITY
    value.reward = Number.NEGATIVE_INFINITY
    value.info = { authoritative_snapshot: { health: 17, suspicious: Number.NaN } }
    value.execution.action.pitch_delta = Number.NaN

    const sanitized = sanitizeStepForWire(value)

    expect(sanitized.observation.self.yaw).toBe(0)
    expect(sanitized.observation.self.velocity.x).toBe(0)
    expect(sanitized.reward).toBe(0)
    expect((sanitized.info.authoritative_snapshot as any).health).toBe(17)
    expect((sanitized.info.authoritative_snapshot as any).suspicious).toBe(0)
    expect(sanitized.execution).toMatchObject({
      source: 'safety', action_id: 91, action: { pitch_delta: 0 }
    })
    expect(sanitized.info.worker_numeric_safety).toEqual({
      nonfinite_count: 5,
      paths: expect.arrayContaining([
        'step.observation.self.yaw:NaN',
        'step.observation.self.velocity.x:+Infinity',
        'step.reward:-Infinity',
        'step.info.authoritative_snapshot.suspicious:NaN',
        'step.execution.action.pitch_delta:NaN'
      ]),
      policy_transition_excluded: true
    })
    const wire = JSON.stringify(sanitized)
    expect(wire).not.toContain('"yaw":null')
    expect(wire).not.toContain('"pitch_delta":null')
    expect(wire).not.toContain('"reward":null')
  })
})
