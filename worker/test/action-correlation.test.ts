import { describe, expect, it } from 'vitest'
import {
  DelayedPolicyActionQueue,
  withConsumedPolicyActionId
} from '../src/action-correlation.js'
import { NOOP_ACTION, type ObservationV2, type StepExecution } from '../src/contracts.js'

describe('delayed policy action correlation', () => {
  it('keeps the exact proposal observation with its delayed action id', () => {
    const queue = new DelayedPolicyActionQueue()
    const observation = { schema_version: 2, match: { episode_id: 'e' } } as ObservationV2
    queue.enqueue(10, 2, { ...NOOP_ACTION }, 41, observation)
    expect(queue.shiftDue(12)?.proposalObservation).toBe(observation)
  })

  it.each([0, 1, 2, 5])('retains the assignment id across a %i-tick delay', delay => {
    const queue = new DelayedPolicyActionQueue()
    queue.enqueue(100, delay, { ...NOOP_ACTION }, 700 + delay)

    expect(queue.shiftDue(99 + delay)).toBeUndefined()
    expect(queue.shiftDue(100 + delay)).toMatchObject({
      due: 100 + delay,
      actionId: 700 + delay,
      action: NOOP_ACTION
    })
  })

  it('distinguishes repeated identical actions by id and preserves FIFO order', () => {
    const queue = new DelayedPolicyActionQueue()
    const identical = { ...NOOP_ACTION, forward: 1 as const }
    queue.enqueue(10, 2, identical, 41)
    queue.enqueue(10, 2, identical, 42)

    expect(queue.shiftDue(12)?.actionId).toBe(41)
    expect(queue.shiftDue(12)?.actionId).toBe(42)
    expect(queue.shiftDue(12)).toBeUndefined()
  })

  it.each(['teacher_sword', 'teacher_crystal', 'teacher_block', 'safety'] as const)(
    'preserves a consumed policy id when %s wins same-tick arbitration',
    source => {
      const override: StepExecution = {
        source,
        action: { ...NOOP_ACTION, primary: source === 'safety' ? 'none' : 'attack' }
      }
      expect(withConsumedPolicyActionId(override, 91)).toEqual({
        ...override,
        action_id: 91
      })
    }
  )

  it('does not invent an id for a teacher or bare execution without a due policy action', () => {
    const teacher: StepExecution = {
      source: 'teacher_sword',
      action: { ...NOOP_ACTION, primary: 'attack' }
    }
    const bare: StepExecution = { source: 'safety', action: { ...NOOP_ACTION } }

    expect(withConsumedPolicyActionId(teacher, undefined)).not.toHaveProperty('action_id')
    expect(withConsumedPolicyActionId(bare, undefined)).not.toHaveProperty('action_id')
  })

  it('rejects zero because correlated trainer assignments start at one', () => {
    const queue = new DelayedPolicyActionQueue()
    queue.enqueue(0, 0, { ...NOOP_ACTION }, 0)
    expect(queue.shiftDue(0)).not.toHaveProperty('actionId')

    const policy: StepExecution = { source: 'policy', action: { ...NOOP_ACTION } }
    expect(withConsumedPolicyActionId(policy, 0)).not.toHaveProperty('action_id')
  })
})
