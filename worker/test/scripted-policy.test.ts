import { describe, expect, it } from 'vitest'
import { scriptedAction } from '../src/scripted-policy.js'
import type { ObservationV1 } from '../src/contracts.js'

describe('scripted fallback', () => {
  it('attacks an opponent inside reach only after cooldown', () => {
    const observation = {
      match: { tick: 20 },
      self: { pitch: 0, attack_cooldown: 1 },
      opponent: { relative_position: { x: 0, y: 0, z: -2.5 } }
    } as ObservationV1
    expect(scriptedAction(observation).primary).toBe('attack')
  })
})
