import os from 'node:os'
import { randomUUID } from 'node:crypto'
import {
  SCHEMA_VERSION,
  type ActionBatch,
  type ExecutionSource,
  type StepBatch
} from './contracts.js'
import { ArenaClient, type ArenaEvent } from './arena-client.js'
import { BotAgent, workerCapabilities } from './bot-agent.js'
import { LoadController } from './load-controller.js'
import { scriptedAction, type ScriptedStyle } from './scripted-policy.js'
import { TrainerConnection } from './trainer-connection.js'
import type { CombatControlMode, CombatMatchConfiguration } from './legal-controls.js'
import { WorkerPerformanceMetrics } from './worker-metrics.js'
import { sanitizeStepForWire } from './wire-numerics.js'

/** Grace period before replacing held controls with PPO-excluded local safety. */
export const TRAINER_STALL_SAFETY_MS = 200
export const POLICY_DECISION_PERIOD_MS = 50

/** Delay a response-driven wakeup just enough to preserve the 20 Hz input cap. */
export function responseTickDelay(now: number, nextDecisionAt: number): number {
  if (!Number.isFinite(now) || !Number.isFinite(nextDecisionAt)) return 0
  return Math.max(0, Math.ceil(nextDecisionAt - now))
}

export type WorkerOptions = {
  workerId: string
  serverHost: string
  serverPort: number
  trainerUrl: string
  arenaHost: string
  arenaPort: number
  botCount: number
  initialPairs: number
  minimumPairs: number
  mode: CombatControlMode
  teachersEnabled: boolean
  usernamePrefix: string
}

export class RolloutWorker {
  private readonly agents: BotAgent[]
  private readonly byId: Map<string, BotAgent>
  private readonly byUsername: Map<string, BotAgent>
  private readonly trainer: TrainerConnection
  private readonly arena: ArenaClient
  private sequence = 0
  private policyVersion = 0
  private timer: NodeJS.Timeout | null = null
  private responseTickTimer: NodeJS.Timeout | null = null
  private stepping = false
  private trainerReady = false
  // Keep at most one rollout request in flight. PPO updates run synchronously in
  // the trainer process; without backpressure the worker can enqueue thousands
  // of observations while an update is running. Every returned action then
  // looks stale and gets discarded, leaving the bots frozen after an arena
  // reset.
  private trainerPendingSequence: number | null = null
  private trainerRequestStartedAt = 0
  private trainerPendingEpisodes = new Map<string, string>()
  private readonly performanceMetrics = new WorkerPerformanceMetrics()
  private expectedTickAt = performance.now() + 50
  private nextDecisionAt = 0
  private readonly loadController: LoadController
  private arenaReconnectTimer: NodeJS.Timeout | null = null
  private arenaConnecting = false
  private stopped = false
  private reconnectingAgents = new Map<string, NodeJS.Timeout>()
  private matchParticipants = new Map<string, Set<string>>()
  private executionMarkers = new Map<string, string>()

  constructor(private readonly options: WorkerOptions) {
    this.agents = Array.from({ length: options.botCount }, (_, index) => this.createAgent(index))
    this.byId = new Map(this.agents.map(agent => [agent.id, agent]))
    this.byUsername = new Map(this.agents.map(agent => [agent.options.username.toLowerCase(), agent]))
    this.trainer = new TrainerConnection(
      options.trainerUrl,
      options.workerId,
      this.agents.map(agent => agent.id),
      trainerCapabilities()
    )
    this.arena = new ArenaClient(options.arenaHost, options.arenaPort)
    const maximumPairs = Math.max(1, Math.floor(options.botCount / 2))
    this.loadController = new LoadController(this.arena, {
      initialPairs: options.initialPairs,
      minimumPairs: options.minimumPairs,
      maximumPairs
    })
  }

  async start(): Promise<void> {
    this.stopped = false
    this.trainer.on('ready', () => {
      this.resetTrainerSession()
      this.trainerReady = true
    })
    this.trainer.on('disconnected', () => {
      this.trainerReady = false
      this.resetTrainerSession()
    })
    this.trainer.on('actions', (batch: ActionBatch) => this.onActions(batch))
    this.trainer.on('error', error => console.warn('[trainer]', String(error)))
    this.trainer.connect()
    this.arena.on('event', event => this.onArenaEvent(event as ArenaEvent))
    this.arena.on('error', error => console.warn('[arena]', String(error)))
    this.arena.on('disconnected', () => {
      this.loadController.stop()
      this.scheduleArenaReconnect()
    })
    await this.connectArena()
    this.expectedTickAt = performance.now() + POLICY_DECISION_PERIOD_MS
    this.timer = setInterval(() => void this.tick(true), POLICY_DECISION_PERIOD_MS)
  }

  async stop(): Promise<void> {
    this.stopped = true
    if (this.timer) clearInterval(this.timer)
    if (this.responseTickTimer) clearTimeout(this.responseTickTimer)
    if (this.arenaReconnectTimer) clearTimeout(this.arenaReconnectTimer)
    for (const timer of this.reconnectingAgents.values()) clearTimeout(timer)
    this.reconnectingAgents.clear()
    this.timer = null
    this.responseTickTimer = null
    this.arenaReconnectTimer = null
    this.trainer.close()
    this.loadController.stop()
    this.arena.close()
    for (const agent of this.agents) agent.disconnect()
  }

  private createAgent(index: number): BotAgent {
    return new BotAgent({
      agentId: `${this.options.workerId}/agent-${index}`,
      username: `${this.options.usernamePrefix}${String(index + 1).padStart(3, '0')}`,
      host: this.options.serverHost,
      port: this.options.serverPort,
      mode: this.options.mode,
      teachersEnabled: this.options.teachersEnabled,
      beforeExecution: (agent, source, episodeId) =>
        this.markExecutionSource(agent, source, episodeId),
      onExecution: (agent, execution, episodeId) => {
        // Safety is an input-only no-op and cannot create a server combat,
        // crystal, block, or terminal event. Publishing it would flip the
        // authoritative marker away from policy and force the next ordinary
        // action to wait for another server round trip.
        if (execution.source !== 'safety') {
          void this.markExecutionSource(agent, execution.source, episodeId)
        }
      },
      onDisconnected: agent => this.scheduleAgentReconnect(agent)
    })
  }

  private scheduleAgentReconnect(agent: BotAgent): void {
    if (this.stopped || this.reconnectingAgents.has(agent.id)) return
    const timer = setTimeout(() => {
      this.reconnectingAgents.delete(agent.id)
      if (this.stopped) return
      const index = this.agents.indexOf(agent)
      if (index < 0) return
      const replacement = this.createAgent(index)
      this.agents[index] = replacement
      this.byId.set(replacement.id, replacement)
      this.byUsername.set(replacement.options.username.toLowerCase(), replacement)
      void this.arena.request('register_agent', {
        agent_id: replacement.id,
        username: replacement.options.username
      }).catch(() => undefined)
    }, 2_000)
    this.reconnectingAgents.set(agent.id, timer)
  }

  private async connectArena(): Promise<void> {
    if (this.stopped || this.arenaConnecting) return
    this.arenaConnecting = true
    try {
      await this.arena.connect()
      await this.arena.request('ping')
      for (const agent of this.agents) {
        await this.arena.request('register_agent', {
          agent_id: agent.id,
          username: agent.options.username
        })
      }
      await this.loadController.start()
      console.info('[arena] control connected')
    } catch (error) {
      console.warn('[arena] control unavailable; using client-local rewards:', String(error))
      this.arena.close()
      this.scheduleArenaReconnect()
    } finally {
      this.arenaConnecting = false
    }
  }

  private scheduleArenaReconnect(): void {
    if (this.stopped || this.arenaReconnectTimer) return
    this.arenaReconnectTimer = setTimeout(() => {
      this.arenaReconnectTimer = null
      void this.connectArena()
    }, 2_000)
  }

  private async tick(periodic = false): Promise<void> {
    const now = performance.now()
    if (periodic && now > this.expectedTickAt + 25) {
      this.performanceMetrics.noteSkippedTick(
        Math.floor((now - this.expectedTickAt) / POLICY_DECISION_PERIOD_MS) + 1
      )
    }
    if (periodic) {
      this.expectedTickAt = now + POLICY_DECISION_PERIOD_MS
    }
    if (this.stepping) {
      this.performanceMetrics.noteSkippedTick()
      return
    }
    this.stepping = true
    try {
      const ready = rolloutEligibleAgents(this.agents)
      if (ready.length === 0) return
      // Do not consume observations/rewards while the trainer still owes us
      // the previous action batch. This heartbeat advances only passive timing;
      // queued policy actions and teacher/safety arbitration run exclusively in
      // agent.step(), so one observation can never collapse multiple action IDs.
      if (this.trainerReady && this.trainerPendingSequence !== null) {
        ready.forEach(agent => agent.heartbeat())
        if (trainerStallSafetyDue(
          this.trainerRequestStartedAt,
          performance.now(),
          TRAINER_STALL_SAFETY_MS
        )) {
          await Promise.all(ready.map(agent => agent.continueDuringTrainerStall()))
        }
        return
      }
      // A terminal transition is irreplaceable. If the trainer is temporarily
      // offline, keep it pending instead of consuming it into an unsent batch.
      const retainedTerminals = this.trainerReady
        ? []
        : ready.filter(agent => agent.hasTerminalStepPending())
      if (retainedTerminals.length) {
        retainedTerminals.forEach(agent => agent.heartbeat())
      }
      const stepAgents = this.trainerReady
        ? ready
        : ready.filter(agent => !agent.hasTerminalStepPending())
      if (stepAgents.length === 0) return
      // Trainer responses are allowed to wake the worker between fixed timer
      // boundaries, but ordinary-player inputs remain capped at one decision
      // per Minecraft tick. This removes up to 50 ms of idle quantization when
      // a response narrowly misses the interval callback.
      if (performance.now() < this.nextDecisionAt) return
      const controlStartedAt = performance.now()
      this.nextDecisionAt = controlStartedAt + POLICY_DECISION_PERIOD_MS
      // Every step in this batch shares the same worker timing window. Compute
      // percentiles/exposure once instead of sorting the bounded samples once
      // per fighter on every control tick.
      const workerMetrics = this.performanceMetrics.snapshot()
      const steps = await Promise.all(stepAgents.map(async agent => {
        const { observation, feedback, execution } = await agent.step()
        if (!this.trainerReady) agent.queueAction(scriptedAction(observation, styleFor(agent.id)))
        return sanitizeStepForWire({
          agent_id: agent.id,
          observation,
          reward: feedback.reward,
          terminated: feedback.terminated,
          truncated: feedback.truncated,
          info: {
            ...feedback.info,
            worker_metrics: workerMetrics
          },
          execution
        })
      }))
      this.performanceMetrics.noteControlApplication(performance.now() - controlStartedAt)
      const batch: StepBatch = {
        schema_version: SCHEMA_VERSION,
        type: 'step_batch',
        sequence: ++this.sequence,
        policy_version: this.policyVersion,
        steps
      }
      if (this.trainer.sendSteps(batch)) {
        this.trainerPendingSequence = batch.sequence
        this.trainerRequestStartedAt = performance.now()
        this.trainerPendingEpisodes = new Map(steps.map(step => [
          step.agent_id, step.observation.match.episode_id
        ]))
      }
    } finally {
      this.stepping = false
    }
  }

  private onActions(batch: ActionBatch): void {
    // Only the response to the outstanding request may drive the clients.
    // Messages from an old socket can arrive during reconnect and are ignored.
    if (this.trainerPendingSequence !== batch.sequence) return
    this.performanceMetrics.noteRoundTrip(performance.now() - this.trainerRequestStartedAt)
    this.performanceMetrics.noteDecisions(batch.actions.length)
    this.trainerPendingSequence = null
    this.policyVersion = batch.policy_version
    for (const agent of this.agents) agent.setPolicyVersion(this.policyVersion)
    for (const entry of batch.actions) {
      this.byId.get(entry.agent_id)?.queueAction(
        entry.action,
        entry.action_id,
        this.trainerPendingEpisodes.get(entry.agent_id)
      )
    }
    this.trainerPendingEpisodes.clear()
    this.scheduleResponseTick()
  }

  private scheduleResponseTick(): void {
    if (this.stopped || !this.trainerReady || this.responseTickTimer) return
    const delay = responseTickDelay(performance.now(), this.nextDecisionAt)
    this.responseTickTimer = setTimeout(() => {
      this.responseTickTimer = null
      void this.tick(false)
    }, delay)
  }

  private onArenaEvent(event: ArenaEvent): void {
    if (event.event === 'arena_snapshot') {
      const payload = event.payload ?? {}
      for (const agent of this.agents) agent.applyArenaSnapshot(payload)
      return
    }
    if (event.event === 'match_started') {
      const payload = event.payload ?? {}
      const episodeId = String(payload.episode_id ?? randomUUID())
      const target = event.agent_id ? this.resolveAgent(event.agent_id) : undefined
      const targets = event.agent_id ? [target].filter(Boolean) as BotAgent[] : this.agents
      for (const agent of targets) {
        const match = matchConfigurationFromPayload(
          payload, this.options.mode, this.options.teachersEnabled
        )
        agent.setMatch({
          episode_id: episodeId,
          arena_seed: Number(payload.arena_seed ?? 0),
          action_delay_ticks: Number(payload.action_delay_ticks ?? 0),
          observation_delay_ticks: Number(payload.observation_delay_ticks ?? 0),
          ...match
        })
        const assigned = typeof payload.opponent_username === 'string' ? payload.opponent_username : null
        agent.setOpponentUsername(assigned)
        if (!assigned) this.pairParticipant(episodeId, agent)
      }
      return
    }
    if (event.event === 'emergency_stop') {
      this.matchParticipants.clear()
      this.executionMarkers.clear()
      for (const agent of this.agents) {
        agent.setOpponentUsername(null)
        agent.emergencyStop()
      }
      return
    }
    if ((event.event === 'step_feedback' || event.event === 'match_ended') && event.agent_id) {
      const payload = event.payload ?? {}
      const agent = this.resolveAgent(event.agent_id)
      agent?.applyArenaFeedback({
        reward: Number(payload.reward ?? 0),
        terminated: event.event === 'match_ended' && payload.truncated !== true,
        truncated: event.event === 'match_ended' && payload.truncated === true,
        info: payload
      })
      if (event.event === 'match_ended') {
        agent?.setOpponentUsername(null)
        if (typeof payload.episode_id === 'string') this.matchParticipants.delete(payload.episode_id)
      }
    }
  }

  private resetTrainerSession(): void {
    if (this.responseTickTimer) clearTimeout(this.responseTickTimer)
    this.responseTickTimer = null
    this.trainerPendingSequence = null
    this.trainerRequestStartedAt = 0
    this.trainerPendingEpisodes.clear()
    for (const agent of this.agents) agent.resetTrainerSession()
  }

  private resolveAgent(idOrUsername: string): BotAgent | undefined {
    return this.byId.get(idOrUsername) ?? this.byUsername.get(idOrUsername.toLowerCase())
  }

  private pairParticipant(episodeId: string, agent: BotAgent): void {
    const participants = this.matchParticipants.get(episodeId) ?? new Set<string>()
    participants.add(agent.id)
    this.matchParticipants.set(episodeId, participants)
    if (participants.size !== 2) return
    const [firstId, secondId] = [...participants]
    const first = this.byId.get(firstId)
    const second = this.byId.get(secondId)
    if (!first || !second) return
    first.setOpponentUsername(second.options.username)
    second.setOpponentUsername(first.options.username)
  }

  private async markExecutionSource(
    agent: BotAgent,
    source: ExecutionSource,
    episodeId: string
  ): Promise<boolean> {
    // Refresh safety ownership every five seconds during arbitrarily long
    // optimizer stalls. Its 10-second server TTL overlaps the refresh, while
    // ordinary policy/teacher transitions retain change-only marking.
    const refreshWindow = source === 'safety' ? `:${Math.floor(performance.now() / 5_000)}` : ''
    const marker = `${episodeId}:${source}${refreshWindow}`
    if (this.executionMarkers.get(agent.id) === marker) return true
    // A trainer update can take several seconds. Keep the whole local
    // continuation window server-attributed as safety; the next real policy
    // action immediately replaces this marker.
    const duration = source === 'safety' ? 200 : source.startsWith('teacher_') ? 4 : 1
    try {
      await this.arena.request('mark_execution_source', {
        username: agent.options.username,
        episode_id: episodeId,
        source,
        duration_ticks: duration
      })
      this.executionMarkers.set(agent.id, marker)
      return true
    } catch (error) {
      console.warn('[arena] execution marker unavailable:', String(error))
      return false
    }
  }
}

export function matchConfigurationFromPayload(
  payload: Record<string, unknown>,
  fallbackMode: CombatControlMode,
  teachersEnabledDefault: boolean
): CombatMatchConfiguration & { lane: string } {
  const requestedLane = String(payload.lane ?? payload.lane_id ?? '').trim().toLowerCase()
  const lane = [
    'sword_retention', 'crystal_retention', 'combined', 'terrain'
  ].includes(requestedLane) ? requestedLane : ''
  const requested = String(payload.mode ?? '').trim().toLowerCase()
  const laneMode: CombatControlMode | null = lane === 'sword_retention'
    ? 'sword'
    : lane === 'crystal_retention'
      ? 'crystal'
      : lane === 'terrain'
        ? 'terrain'
        : lane === 'combined'
          ? 'combined'
          : null
  const mode: CombatControlMode = laneMode ?? (
    requested === 'sword' || requested === 'crystal'
      || requested === 'combined' || requested === 'terrain'
      ? requested
      : fallbackMode
  )
  const evaluation = payload.evaluation === true || payload.training === false
  const teachersEnabled = typeof payload.teachers_enabled === 'boolean'
    ? payload.teachers_enabled
    : evaluation
      ? false
      : teachersEnabledDefault
  return {
    lane,
    mode,
    radius: positiveInteger(payload.arena_radius ?? payload.radius, 5),
    stage: positiveInteger(payload.curriculum_stage ?? payload.stage, 1),
    teachersEnabled,
    terrainEnabled: typeof payload.terrain_enabled === 'boolean'
      ? payload.terrain_enabled
      : mode === 'terrain'
  }
}

export function rolloutEligibleAgents(agents: readonly BotAgent[]): BotAgent[] {
  return agents.filter(agent => agent.isSpawned() && agent.isMatchActive())
}

export function trainerStallSafetyDue(
  startedAt: number,
  now: number,
  thresholdMs = TRAINER_STALL_SAFETY_MS
): boolean {
  return Number.isFinite(startedAt)
    && startedAt > 0
    && Number.isFinite(now)
    && now - startedAt >= Math.max(0, thresholdMs)
}

export function trainerCapabilities(): string[] {
  return [
    'minecraft-1.12.2',
    'structured-state',
    'legal-controls-v1',
    ...workerCapabilities()
  ]
}

export function optionsFromEnvironment(): WorkerOptions {
  const botCount = Math.max(2, integerEnv('MCAI_BOT_COUNT', 4))
  const maximumPairs = Math.max(1, Math.floor(botCount / 2))
  const requestedMode = (process.env.MCAI_MODE ?? 'sword').toLowerCase()
  const mode = requestedMode === 'crystal' || requestedMode === 'combined'
    || requestedMode === 'terrain'
    ? requestedMode
    : 'sword'
  return {
    workerId: process.env.MCAI_WORKER_ID ?? `${os.hostname()}-${process.pid}`,
    serverHost: process.env.MCAI_SERVER_HOST ?? '127.0.0.1',
    serverPort: integerEnv('MCAI_SERVER_PORT', 25565),
    trainerUrl: process.env.MCAI_TRAINER_URL ?? 'ws://127.0.0.1:8766',
    arenaHost: process.env.MCAI_ARENA_HOST ?? '127.0.0.1',
    arenaPort: integerEnv('MCAI_ARENA_PORT', 8765),
    botCount,
    initialPairs: boundedIntegerEnv('MCAI_INITIAL_PAIRS', maximumPairs, 1, maximumPairs),
    minimumPairs: boundedIntegerEnv('MCAI_MIN_PAIRS', Math.min(2, maximumPairs), 1, maximumPairs),
    mode,
    teachersEnabled: booleanEnv('MCAI_TEACHERS_ENABLED', true),
    usernamePrefix: process.env.MCAI_USERNAME_PREFIX ?? 'MCAI_'
  }
}

function integerEnv(name: string, fallback: number): number {
  const parsed = Number.parseInt(process.env[name] ?? '', 10)
  return Number.isFinite(parsed) ? parsed : fallback
}

function boundedIntegerEnv(name: string, fallback: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, integerEnv(name, fallback)))
}

function positiveInteger(value: unknown, fallback: number): number {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? Math.max(1, Math.floor(parsed)) : fallback
}

function booleanEnv(name: string, fallback: boolean): boolean {
  const value = process.env[name]?.trim().toLowerCase()
  if (!value) return fallback
  if (['1', 'true', 'yes', 'on'].includes(value)) return true
  if (['0', 'false', 'no', 'off'].includes(value)) return false
  return fallback
}

function styleFor(agentId: string): ScriptedStyle {
  const styles: ScriptedStyle[] = ['rush', 'strafe', 'retreat', 'jump_critical', 'defensive', 'erratic']
  let hash = 0
  for (const char of agentId) hash = (hash * 31 + char.charCodeAt(0)) | 0
  return styles[Math.abs(hash) % styles.length]
}
