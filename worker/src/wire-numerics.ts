import type { StepBatch } from './contracts.js'

export const NUMERIC_SAFETY_PATH_LIMIT = 16

export type WireNumericSafetyMetric = {
  nonfinite_count: number
  paths: string[]
  policy_transition_excluded: true
}

type StepPayload = StepBatch['steps'][number]

/**
 * JSON.stringify silently serializes NaN and infinities as null. Make the
 * protocol boundary explicit instead: preserve the complete payload shape,
 * replace only invalid numeric leaves, report their paths, and exclude the
 * affected transition from PPO by marking its actual execution as safety.
 */
export function sanitizeStepForWire(step: StepPayload): StepPayload {
  const issues: string[] = []
  let issueCount = 0
  const visit = (value: unknown, path: string): unknown => {
    if (typeof value === 'number') {
      if (Number.isFinite(value)) return value
      issueCount += 1
      if (issues.length < NUMERIC_SAFETY_PATH_LIMIT) {
        issues.push(`${path}:${numericKind(value)}`)
      }
      return 0
    }
    if (Array.isArray(value)) {
      return value.map((entry, index) => visit(entry, `${path}[${index}]`))
    }
    if (value && typeof value === 'object') {
      return Object.fromEntries(Object.entries(value).map(([key, entry]) => [
        key, visit(entry, path ? `${path}.${key}` : key)
      ]))
    }
    return value
  }

  const sanitized = visit(step, 'step') as StepPayload
  if (issueCount === 0) return step
  const metric: WireNumericSafetyMetric = {
    nonfinite_count: issueCount,
    paths: issues,
    policy_transition_excluded: true
  }
  return {
    ...sanitized,
    info: { ...sanitized.info, worker_numeric_safety: metric },
    execution: { ...sanitized.execution, source: 'safety' }
  }
}

function numericKind(value: number): 'NaN' | '+Infinity' | '-Infinity' {
  if (Number.isNaN(value)) return 'NaN'
  return value > 0 ? '+Infinity' : '-Infinity'
}
