export type WorkerPerformanceSnapshot = {
  decision_round_trip_ms_median: number
  decision_round_trip_ms_p95: number
  control_application_ms_median: number
  control_application_ms_p95: number
  skipped_50ms_ticks: number
  optimizer_stalls_over_200ms: number
  effective_decisions_hz_per_agent: number
  decision_rate_window_seconds: number
  decisions_in_window: number
  active_agent_seconds_in_window: number
}

const DECISION_RATE_WINDOW_MS = 60_000

type DecisionEvent = { at: number, count: number }
type ActiveSegment = { startedAt: number, endedAt: number, agents: number }

export class WorkerPerformanceMetrics {
  private readonly roundTrips: number[] = []
  private readonly controlTimes: number[] = []
  private skippedTicks = 0
  private optimizerStalls = 0
  private readonly startedAt = performance.now()
  private readonly decisionEvents: DecisionEvent[] = []
  private readonly activeSegments: ActiveSegment[] = []
  private activeAgents = 0
  private activeSince = this.startedAt

  noteRoundTrip(milliseconds: number): void {
    pushBounded(this.roundTrips, milliseconds)
    if (milliseconds > 200) this.optimizerStalls += 1
  }

  noteControlApplication(milliseconds: number): void {
    pushBounded(this.controlTimes, milliseconds)
  }

  noteSkippedTick(count = 1): void {
    this.skippedTicks += Math.max(1, Math.trunc(count))
  }

  noteDecisions(count: number): void {
    const now = performance.now()
    if (now > this.activeSince && this.activeAgents > 0) {
      this.activeSegments.push({
        startedAt: this.activeSince,
        endedAt: now,
        agents: this.activeAgents
      })
    }
    const decisions = Math.max(0, Math.trunc(count))
    if (decisions > 0) this.decisionEvents.push({ at: now, count: decisions })
    // The returned action count is the number of agents which actually
    // received a policy decision. Using it as exposure avoids the old bug
    // where a transient four-agent snapshot doubled a lifetime eight-agent
    // rate from 8 Hz to 16 Hz.
    this.activeAgents = decisions
    this.activeSince = now
    this.pruneDecisionRateWindow(now)
  }

  snapshot(): WorkerPerformanceSnapshot {
    const now = performance.now()
    this.pruneDecisionRateWindow(now)
    const windowStart = Math.max(this.startedAt, now - DECISION_RATE_WINDOW_MS)
    const windowSeconds = Math.max(0, (now - windowStart) / 1000)
    const decisions = this.decisionEvents.reduce(
      (total, event) => total + (event.at >= windowStart ? event.count : 0), 0
    )
    let activeAgentMilliseconds = this.activeSegments.reduce((total, segment) => {
      const startedAt = Math.max(windowStart, segment.startedAt)
      return total + Math.max(0, segment.endedAt - startedAt) * segment.agents
    }, 0)
    if (this.activeAgents > 0) {
      activeAgentMilliseconds += Math.max(0, now - Math.max(windowStart, this.activeSince))
        * this.activeAgents
    }
    const activeAgentSeconds = activeAgentMilliseconds / 1000
    return {
      decision_round_trip_ms_median: percentile(this.roundTrips, 0.5),
      decision_round_trip_ms_p95: percentile(this.roundTrips, 0.95),
      control_application_ms_median: percentile(this.controlTimes, 0.5),
      control_application_ms_p95: percentile(this.controlTimes, 0.95),
      skipped_50ms_ticks: this.skippedTicks,
      optimizer_stalls_over_200ms: this.optimizerStalls,
      effective_decisions_hz_per_agent: activeAgentSeconds > 0
        ? decisions / activeAgentSeconds
        : 0,
      decision_rate_window_seconds: windowSeconds,
      decisions_in_window: decisions,
      active_agent_seconds_in_window: activeAgentSeconds
    }
  }

  private pruneDecisionRateWindow(now: number): void {
    const cutoff = now - DECISION_RATE_WINDOW_MS
    while (this.decisionEvents[0]?.at < cutoff) this.decisionEvents.shift()
    while (this.activeSegments[0]?.endedAt < cutoff) this.activeSegments.shift()
  }
}

function pushBounded(values: number[], value: number): void {
  if (!Number.isFinite(value) || value < 0) return
  values.push(value)
  if (values.length > 512) values.shift()
}

function percentile(values: readonly number[], fraction: number): number {
  if (values.length === 0) return 0
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * fraction))]
}
