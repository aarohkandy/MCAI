import type { AnyAction, ObservationV2, StepExecution } from './contracts.js'

export type QueuedPolicyAction = {
  due: number
  action: AnyAction
  actionId?: number
  /** Exact previously-sent state from which the trainer sampled this action. */
  proposalObservation?: ObservationV2
}

/** FIFO delay queue that keeps proposal identity separate from action equality. */
export class DelayedPolicyActionQueue {
  private readonly entries: QueuedPolicyAction[] = []

  enqueue(
    currentTick: number,
    delayTicks: number,
    action: AnyAction,
    actionId?: number,
    proposalObservation?: ObservationV2
  ): void {
    const entry: QueuedPolicyAction = {
      due: currentTick + Math.max(0, Math.trunc(delayTicks)),
      action
    }
    const valid = validActionId(actionId)
    if (valid !== undefined) entry.actionId = valid
    if (proposalObservation) entry.proposalObservation = proposalObservation
    this.entries.push(entry)
  }

  peekDue(tick: number): QueuedPolicyAction | undefined {
    const first = this.entries[0]
    return first && first.due <= tick ? first : undefined
  }

  shiftDue(tick: number): QueuedPolicyAction | undefined {
    if (!this.peekDue(tick)) return undefined
    return this.entries.shift()
  }

  clear(): void {
    this.entries.length = 0
  }
}

/**
 * A teacher/safety action can override a due policy action in the same tick.
 * The actual control remains the override, while the consumed proposal id tells
 * the trainer exactly which pending policy transition must be excluded.
 */
export function withConsumedPolicyActionId(
  execution: StepExecution,
  actionId: number | undefined
): StepExecution {
  const valid = validActionId(actionId)
  return valid === undefined ? execution : { ...execution, action_id: valid }
}

function validActionId(value: number | undefined): number | undefined {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 1
    ? value
    : undefined
}
