import mineflayer, { type Bot } from 'mineflayer'
import {
  NOOP_ACTION,
  SCHEMA_VERSION,
  type ActionHistoryEntry,
  type AnyAction,
  type ActionV1,
  type ObservationV1,
  type ObservationV2,
  type StepExecution,
  type StepFeedback
} from './contracts.js'
import {
  DelayedPolicyActionQueue,
  withConsumedPolicyActionId
} from './action-correlation.js'
import {
  LegalControlAdapter,
  type CombatControlMode,
  type CombatMatchConfiguration
} from './legal-controls.js'
import { ObservationBuilder, type MatchContext } from './observation.js'
import { observationV2 } from './observation-v2.js'
import { resolveActionV2 } from './action-v2.js'
import { TacticalPlacementTracker } from './tactical-blocks.js'

export type BotAgentOptions = {
  agentId: string
  username: string
  host: string
  port: number
  version?: string
  mode?: CombatControlMode
  teachersEnabled?: boolean
  beforeExecution?: (
    agent: BotAgent,
    source: StepExecution['source'],
    episodeId: string
  ) => Promise<boolean>
  onExecution?: (agent: BotAgent, execution: StepExecution, episodeId: string) => void
  onDisconnected?: (agent: BotAgent) => void
}

export type BotMatchUpdate = Partial<MatchContext> & Partial<CombatMatchConfiguration> & {
  lane?: string
}

/** Crystal retention gets a focused two-second attempt before melee fallback. */
export const CRYSTAL_RETENTION_ACQUISITION_TICKS = 40

export class BotAgent {
  readonly bot: Bot
  private readonly controls: LegalControlAdapter
  private readonly observations: ObservationBuilder
  private readonly tacticalPlacements: TacticalPlacementTracker
  private tick = 0
  private episodeId = 'waiting'
  private policyVersion = 0
  private arenaSeed = 0
  private actionDelayTicks = 0
  private observationDelayTicks = 0
  private feedback: StepFeedback = { reward: 0, terminated: false, truncated: false, info: {} }
  private previousHealth = 20
  private readonly queuedActions = new DelayedPolicyActionQueue()
  private observationHistory: ObservationV2[] = []
  private lastTrainerObservation: ObservationV2 | null = null
  private actionHistory: ActionHistoryEntry[] = []
  private pendingPolicyIntent: ActionHistoryEntry['intent'] = 'legacy'
  private lastObservedHealth: number | null = null
  private lastObservedOpponentHealth: number | null = null
  private readonly v2Rejections = new Map<string, number>()
  private crystalRetentionAttempted = false
  private crystalRetentionGateDeadlineTick = 0
  private trainerStallSafetyActive = false
  private trainerStallExecution: StepExecution | null = null
  private stalledProposalPending = false
  private stalledProposalActionId: number | undefined
  private arenaManaged = false
  private pendingExecution: StepExecution = { source: 'safety', action: { ...NOOP_ACTION } }
  private hasPendingExecution = false
  private pendingTeacherObservation: ObservationV1 | null = null
  private liveExecutionSource: StepExecution['source'] = 'policy'
  private terminalStepPending = false
  private deferredMatch: BotMatchUpdate | null = null
  private deferredOpponentUsername: string | null | undefined
  private matchMode: CombatControlMode
  private matchLane = ''
  private matchRadius = 5
  private matchStage = 1

  constructor(readonly options: BotAgentOptions) {
    this.matchMode = options.mode ?? 'sword'
    this.bot = mineflayer.createBot({
      host: options.host,
      port: options.port,
      username: options.username,
      version: options.version ?? '1.12.2',
      auth: 'offline',
      // Mineflayer otherwise writes one warning for every block-entity packet
      // that races a not-yet-loaded arena chunk. Eight fighters can generate
      // thousands of synchronous console writes; actionable worker/control
      // failures are still surfaced by the explicit handlers below.
      hideErrors: true
    })
    this.tacticalPlacements = new TacticalPlacementTracker(this.bot)
    this.controls = new LegalControlAdapter(
      this.bot,
      options.mode ?? 'sword',
      options.agentId,
      async source => {
        const preExecutionObservation = source.startsWith('teacher_')
          ? this.observations.build(this.matchContext(), this.controls.telemetry(this.tick))
          : null
        const marked = await (this.options.beforeExecution?.(this, source, this.episodeId)
          ?? Promise.resolve(true))
        if (marked) {
          this.liveExecutionSource = source
          this.pendingTeacherObservation = preExecutionObservation
        }
        return marked
      },
      this.tacticalPlacements
    )
    this.controls.setMatchConfiguration({
      teachersEnabled: options.teachersEnabled ?? true,
      terrainEnabled: options.mode === 'terrain'
    })
    this.observations = new ObservationBuilder(
      this.bot, this.tacticalPlacements, options.agentId
    )
    this.bot.on('health', () => this.updateLocalReward())
    this.bot.on('death', () => {
      if (!this.arenaManaged) {
        this.feedback = { reward: -1, terminated: true, truncated: false, info: { reason: 'death' } }
        this.terminalStepPending = this.isMatchActive()
        this.queuedActions.clear()
      }
      this.controls.emergencyStop()
    })
    this.bot.on('kicked', reason => this.controls.emergencyStop())
    this.bot.on('end', () => {
      this.controls.emergencyStop()
      this.options.onDisconnected?.(this)
    })
  }

  get id(): string {
    return this.options.agentId
  }

  isSpawned(): boolean {
    return Boolean(this.bot.entity)
  }

  isMatchActive(): boolean {
    return this.episodeId !== 'waiting'
  }

  currentEpisodeId(): string {
    return this.episodeId
  }

  hasTerminalStepPending(): boolean {
    return this.terminalStepPending
  }

  setMatch(context: BotMatchUpdate): void {
    if (typeof context.episode_id === 'string'
      && context.episode_id !== this.episodeId
      && this.terminalStepPending) {
      // Preserve the old episode until its one terminal transition is sent.
      // Auto-matchmaking may announce the replacement while the trainer is
      // still updating, so activation is deferred instead of relabelling it.
      this.deferredMatch = { ...context }
      this.deferredOpponentUsername = undefined
      return
    }
    if (context.mode) this.matchMode = context.mode
    if (typeof context.lane === 'string') this.matchLane = context.lane
    if (Number.isFinite(context.radius)) this.matchRadius = Math.max(1, Math.floor(context.radius as number))
    if (Number.isFinite(context.stage)) this.matchStage = Math.max(1, Math.floor(context.stage as number))
    if (typeof context.episode_id === 'string' && context.episode_id !== this.episodeId) {
      this.episodeId = context.episode_id
      this.tacticalPlacements.beginEpisode(this.episodeId, this.matchStage)
      this.observationHistory = []
      this.lastTrainerObservation = null
      this.actionHistory = []
      this.lastObservedHealth = null
      this.lastObservedOpponentHealth = null
      this.crystalRetentionAttempted = false
      this.crystalRetentionGateDeadlineTick = this.tick + CRYSTAL_RETENTION_ACQUISITION_TICKS
      this.v2Rejections.clear()
      this.queuedActions.clear()
      this.controls.emergencyStop()
      this.controls.beginEpisode({
        mode: context.mode,
        radius: context.radius,
        stage: context.stage,
        teachersEnabled: context.teachersEnabled,
        terrainEnabled: context.terrainEnabled
      })
      this.controls.setCrystalRetentionSwordFallbackEnabled(false)
      this.clearTrainerStallSafety()
      this.pendingExecution = { source: 'safety', action: { ...NOOP_ACTION } }
      this.hasPendingExecution = false
      this.pendingTeacherObservation = null
      this.liveExecutionSource = 'policy'
      this.terminalStepPending = false
      this.feedback = { reward: 0, terminated: false, truncated: false, info: {} }
      this.arenaManaged = context.episode_id !== 'waiting'
    } else {
      this.controls.setMatchConfiguration({
        mode: context.mode,
        radius: context.radius,
        stage: context.stage,
        teachersEnabled: context.teachersEnabled,
        terrainEnabled: context.terrainEnabled
      })
    }
    if (typeof context.policy_version === 'number') this.policyVersion = context.policy_version
    if (typeof context.arena_seed === 'number') this.arenaSeed = context.arena_seed
    if (typeof context.action_delay_ticks === 'number') this.actionDelayTicks = context.action_delay_ticks
    if (typeof context.observation_delay_ticks === 'number') this.observationDelayTicks = context.observation_delay_ticks
  }

  setPolicyVersion(version: number): void {
    if (version !== this.policyVersion) {
      this.queuedActions.clear()
      this.controls.emergencyStop()
    }
    this.policyVersion = version
  }

  /** Drop controls owned by a websocket session without ending the arena match. */
  resetTrainerSession(): void {
    this.queuedActions.clear()
    this.clearTrainerStallSafety()
    this.pendingExecution = { source: 'safety', action: { ...NOOP_ACTION } }
    this.hasPendingExecution = false
    this.pendingTeacherObservation = null
    this.liveExecutionSource = 'policy'
    this.controls.emergencyStop()
  }

  setOpponentUsername(username: string | null): void {
    if (this.deferredMatch) {
      this.deferredOpponentUsername = username
      return
    }
    this.observations.setOpponentUsername(username)
    this.controls.setOpponentUsername(username)
  }

  applyArenaSnapshot(payload: Record<string, unknown>): boolean {
    if (String(payload.episode_id ?? '') !== this.episodeId) return false
    return this.observations.acceptArenaSnapshot(payload, this.episodeId, this.tick)
  }

  queueAction(action: AnyAction, actionId?: number, proposalEpisodeId?: string): void {
    if (!this.isMatchActive() || this.terminalStepPending) return
    if (proposalEpisodeId && proposalEpisodeId !== this.episodeId) {
      // The arena advanced while inference/update was outstanding. Consume the
      // old proposal as an exact correlated safety no-op; never apply it to the
      // replacement episode.
      this.trainerStallSafetyActive = true
      this.trainerStallExecution = { source: 'safety', action: { ...NOOP_ACTION } }
      this.stalledProposalPending = true
      this.stalledProposalActionId = validActionId(actionId)
      return
    }
    if (this.trainerStallSafetyActive) {
      // Safety changed the state from which this delayed proposal was sampled.
      // Consume its exact id on the next step without ever applying it.
      this.stalledProposalPending = true
      this.stalledProposalActionId = validActionId(actionId)
      return
    }
    const proposalObservation = this.lastTrainerObservation?.match.episode_id === this.episodeId
      ? this.lastTrainerObservation
      : undefined
    this.queuedActions.enqueue(
      this.tick, this.actionDelayTicks, action, actionId, proposalObservation
    )
  }

  heartbeat(): void {
    this.tick += 1
    this.controls.observeTeacherCompletion(this.tick)
  }

  async continueDuringTrainerStall(): Promise<void> {
    if (!this.isMatchActive() || this.terminalStepPending) return
    // Re-enter the marker hook so the worker can cheaply refresh long-running
    // safety ownership without sending one arena request per 50 ms control tick.
    const marked = await (this.options.beforeExecution?.(
      this, 'safety', this.episodeId
    ) ?? Promise.resolve(true))
    if (!marked) return
    if (!this.trainerStallSafetyActive) {
      this.trainerStallSafetyActive = true
      this.liveExecutionSource = 'safety'
    }
    this.trainerStallExecution = await this.controls.applyTrainerStallSafety(this.tick)
  }

  /** Execute at most one due trainer proposal and arbitrate its same-tick overrides. */
  private async executeControlStep(): Promise<void> {
    this.pendingTeacherObservation = null
    if (this.matchLane === 'crystal_retention' && this.crystalRetentionGateOpen()) {
      this.controls.setCrystalRetentionSwordFallbackEnabled(true)
    }
    let execution: StepExecution | null = null
    let consumedPolicyActionId: number | undefined
    let legalPolicyCombat = false
    let stallOverride = false
    if (this.trainerStallSafetyActive && this.stalledProposalPending) {
      execution = this.trainerStallExecution
        ?? { source: 'safety', action: { ...NOOP_ACTION } }
      consumedPolicyActionId = this.stalledProposalActionId
      stallOverride = true
      this.clearTrainerStallSafety()
    }
    if (!stallOverride && this.queuedActions.peekDue(this.tick)) {
      const mayExecute = this.liveExecutionSource === 'policy'
        || await (this.options.beforeExecution?.(this, 'policy', this.episodeId)
          ?? Promise.resolve(true))
      if (mayExecute) {
        this.liveExecutionSource = 'policy'
        const queued = this.queuedActions.shiftDue(this.tick)
        if (!queued) return
        consumedPolicyActionId = queued.actionId
        this.pendingPolicyIntent = queued.action.schema_version === 2 ? queued.action.intent : 'legacy'
        let executable: ActionV1 | null = null
        try {
          executable = queued.action.schema_version === 2
            ? resolveActionV2(
              queued.action,
              queued.proposalObservation ?? observationV2(
                this.observations.build(this.matchContext(), this.controls.telemetry(this.tick)),
                this.actionHistory
              )
            )
            : queued.action
          if (queued.action.schema_version === 2) {
            executable = this.applyCrystalRetentionGate(executable, queued.action.intent)
          }
        } catch (error) {
          this.noteV2Rejection(error)
          // Consume malformed/cross-family proposals as safety no-ops. They are
          // correlated but never allowed to crash or poison the control loop.
          execution = { source: 'safety', action: { ...NOOP_ACTION } }
        }
        if (executable) try {
          const policy = await this.controls.apply(executable, this.tick)
          execution = policy
          legalPolicyCombat = policy.combatPriority
          if (this.matchLane === 'crystal_retention' && policy.crystalPriority) {
            this.crystalRetentionAttempted = true
            this.controls.setCrystalRetentionSwordFallbackEnabled(true)
          }
        } catch (error) {
          this.noteV2Rejection(new Error(`control_apply:${errorMessage(error)}`))
          execution = { source: 'safety', action: { ...NOOP_ACTION } }
        }
      }
    }
    if (!stallOverride && !legalPolicyCombat) {
      const sword = await this.controls.applySwordBootcampAssist(this.tick)
      const crystal = sword ? null : await this.controls.applyCombinedCrystalDemonstration(this.tick)
      const tactical = sword || crystal
        ? null
        : await this.controls.applyCombinedTacticalBlockDemonstration(this.tick)
      execution = sword ?? crystal ?? tactical ?? execution
      if (!sword && !crystal && !tactical) {
        execution = await this.controls.applyPitchSafety(this.tick) ?? execution
      }
    }
    if (execution) {
      this.noteExecution(withConsumedPolicyActionId(execution, consumedPolicyActionId))
    } else if (this.liveExecutionSource !== 'policy') {
      const cleared = await (this.options.beforeExecution?.(this, 'policy', this.episodeId)
        ?? Promise.resolve(true))
      if (cleared) {
        const clear: StepExecution = { source: 'policy', action: { ...NOOP_ACTION } }
        this.liveExecutionSource = 'policy'
        this.options.onExecution?.(this, clear, this.episodeId)
      }
    }
  }

  async step(): Promise<{
    observation: ObservationV2
    feedback: StepFeedback
    execution: StepExecution
  }> {
    await this.heartbeat()
    if (!this.terminalStepPending) await this.executeControlStep()
    const context = this.matchContext()
    const baseObservation = this.observations.build(context, this.controls.telemetry(this.tick))
    const currentOpponentHealth = baseObservation.opponent?.health ?? null
    const currentHealth = baseObservation.self?.health ?? null
    const latestAction = this.actionHistory[this.actionHistory.length - 1]
    if (latestAction) {
      latestAction.health_delta = this.lastObservedHealth === null || currentHealth === null
        ? 0 : currentHealth - this.lastObservedHealth
      latestAction.opponent_health_delta = this.lastObservedOpponentHealth === null || currentOpponentHealth === null
        ? 0 : currentOpponentHealth - this.lastObservedOpponentHealth
    }
    this.lastObservedHealth = currentHealth
    this.lastObservedOpponentHealth = currentOpponentHealth
    const currentObservation = observationV2(baseObservation, this.actionHistory)
    this.observationHistory.push(currentObservation)
    const maximumHistory = Math.max(8, this.observationDelayTicks + 2)
    while (this.observationHistory.length > maximumHistory) this.observationHistory.shift()
    const delayedIndex = Math.max(0, this.observationHistory.length - 1 - this.observationDelayTicks)
    const delayed = this.observationHistory[delayedIndex]
    const observation: ObservationV2 = {
      ...delayed,
      match: context,
      blocks: delayed.blocks.map(block => ({
        ...block,
        sample_age_ticks: block.sample_age_ticks + this.observationDelayTicks
      })),
      tactical: delayed.tactical
    }
    this.lastTrainerObservation = observation
    const rejectionCounts = Object.fromEntries(this.v2Rejections)
    this.v2Rejections.clear()
    const workerInfo: Record<string, unknown> = {}
    if (Object.keys(rejectionCounts).length > 0) {
      workerInfo.worker_v2_rejections = rejectionCounts
    }
    if (this.matchLane === 'crystal_retention') {
      workerInfo.crystal_retention_crystal_gate_open = this.crystalRetentionGateOpen()
    }
    const feedback = Object.keys(workerInfo).length === 0
      ? this.feedback
      : {
        ...this.feedback,
        info: { ...this.feedback.info, ...workerInfo }
      }
    const execution = this.hasPendingExecution
      ? this.pendingExecution
      : { source: 'safety' as const, action: { ...NOOP_ACTION } }
    this.feedback = { reward: 0, terminated: false, truncated: false, info: {} }
    this.pendingExecution = { source: 'safety', action: { ...NOOP_ACTION } }
    this.hasPendingExecution = false
    if (this.terminalStepPending) this.finishTerminalStep()
    return { observation, feedback, execution }
  }

  applyArenaFeedback(feedback: StepFeedback): void {
    this.arenaManaged = true
    this.feedback = {
      reward: this.feedback.reward + feedback.reward,
      terminated: this.feedback.terminated || feedback.terminated,
      truncated: this.feedback.truncated || feedback.truncated,
      info: { ...this.feedback.info, ...feedback.info }
    }
    if (feedback.terminated || feedback.truncated) {
      this.terminalStepPending = this.isMatchActive()
      this.queuedActions.clear()
      this.controls.emergencyStop()
    }
  }

  emergencyStop(): void {
    this.retireImmediately()
  }

  disconnect(): void {
    this.controls.emergencyStop()
    this.bot.quit('worker shutdown')
  }

  private updateLocalReward(): void {
    const health = Number.isFinite(this.bot.health) ? this.bot.health : this.previousHealth
    const delta = health - this.previousHealth
    if (delta < 0 && !this.arenaManaged) this.feedback.reward += 0.02 * delta
    this.previousHealth = health
  }

  private noteExecution(execution: StepExecution): void {
    // Any override since the previous observation must exclude that PPO
    // transition. A later ordinary policy tick must not hide an earlier teacher.
    const inheritedActionId = this.hasPendingExecution
      ? this.pendingExecution.action_id
      : undefined
    const observedExecution = execution.source.startsWith('teacher_')
      && this.pendingTeacherObservation
      ? { ...execution, pre_execution_observation: this.pendingTeacherObservation }
      : execution
    const correlated = withConsumedPolicyActionId(
      observedExecution,
      execution.action_id ?? inheritedActionId
    )
    if (!this.hasPendingExecution
      || executionPriority(correlated.source) >= executionPriority(this.pendingExecution.source)) {
      this.pendingExecution = correlated
    } else if (this.pendingExecution.action_id === undefined && correlated.action_id !== undefined) {
      // Defensive path for future multi-operation control steps: a lower-priority
      // due policy action still identifies the proposal excluded by the winner.
      this.pendingExecution = { ...this.pendingExecution, action_id: correlated.action_id }
    }
    this.hasPendingExecution = true
    this.liveExecutionSource = correlated.source
    this.pendingTeacherObservation = null
    this.actionHistory ??= []
    this.actionHistory.push({
      tick: this.tick,
      intent: execution.source === 'policy' ? this.pendingPolicyIntent : execution.source,
      primary: execution.action.primary,
      hotbar: execution.action.hotbar,
      health_delta: 0,
      opponent_health_delta: 0
    })
    while (this.actionHistory.length > 8) this.actionHistory.shift()
    if (execution.source === 'policy') this.pendingPolicyIntent = 'legacy'
    this.options.onExecution?.(this, correlated, this.episodeId)
  }

  private noteV2Rejection(error: unknown): void {
    const reason = rejectionReason(error)
    this.v2Rejections.set(reason, (this.v2Rejections.get(reason) ?? 0) + 1)
  }

  private clearTrainerStallSafety(): void {
    this.trainerStallSafetyActive = false
    this.trainerStallExecution = null
    this.stalledProposalPending = false
    this.stalledProposalActionId = undefined
  }

  private applyCrystalRetentionGate(
    action: ActionV1,
    intent: ActionHistoryEntry['intent']
  ): ActionV1 {
    if (this.matchLane !== 'crystal_retention') return action
    if (this.crystalRetentionGateOpen()) {
      this.controls.setCrystalRetentionSwordFallbackEnabled(true)
      return action
    }
    if (intent === 'sword_engage') {
      return { ...action, primary: 'none', forward: 1, sprint: true }
    }
    if (intent === 'crystal_acquire' || intent === 'reposition') {
      return { ...action, forward: 1, sprint: true }
    }
    return action
  }

  private crystalRetentionGateOpen(): boolean {
    return this.crystalRetentionAttempted
      || this.tick >= this.crystalRetentionGateDeadlineTick
  }

  private matchContext(): MatchContext {
    const context: MatchContext = {
      episode_id: this.episodeId,
      tick: this.tick,
      policy_version: this.policyVersion,
      arena_seed: this.arenaSeed,
      action_delay_ticks: this.actionDelayTicks,
      observation_delay_ticks: this.observationDelayTicks,
      mode: this.matchMode,
      arena_radius: this.matchRadius,
      curriculum_stage: this.matchStage
    }
    if (this.matchLane) context.lane = this.matchLane
    return context
  }

  private finishTerminalStep(): void {
    const deferredMatch = this.deferredMatch
    const deferredOpponent = this.deferredOpponentUsername
    this.deferredMatch = null
    this.deferredOpponentUsername = undefined
    this.retireImmediately(false)
    if (deferredMatch) {
      this.setMatch(deferredMatch)
      if (deferredOpponent !== undefined) this.setOpponentUsername(deferredOpponent)
    }
  }

  private retireImmediately(clearDeferred = true): void {
    this.episodeId = 'waiting'
    this.tacticalPlacements.beginEpisode(this.episodeId, this.matchStage)
    this.arenaSeed = 0
    this.queuedActions.clear()
    this.observationHistory = []
    this.lastTrainerObservation = null
    this.actionHistory = []
    this.pendingPolicyIntent = 'legacy'
    this.lastObservedHealth = null
    this.lastObservedOpponentHealth = null
    this.v2Rejections.clear()
    this.crystalRetentionAttempted = false
    this.crystalRetentionGateDeadlineTick = 0
    this.clearTrainerStallSafety()
    this.feedback = { reward: 0, terminated: false, truncated: false, info: {} }
    this.pendingExecution = { source: 'safety', action: { ...NOOP_ACTION } }
    this.hasPendingExecution = false
    this.pendingTeacherObservation = null
    this.liveExecutionSource = 'policy'
    this.terminalStepPending = false
    this.arenaManaged = false
    if (clearDeferred) {
      this.deferredMatch = null
      this.deferredOpponentUsername = undefined
    }
    this.observations.setOpponentUsername(null)
    this.controls.setOpponentUsername(null)
    this.controls.emergencyStop()
  }
}

function executionPriority(source: StepExecution['source']): number {
  if (source.startsWith('teacher_')) return 3
  if (source === 'safety') return 2
  return 1
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function rejectionReason(error: unknown): string {
  const message = errorMessage(error).toLowerCase()
  if (message.includes('does not accept a candidate')) return 'implicit_intent_with_target'
  if (message.includes('does not name a base')) return 'crystal_place_wrong_target_kind'
  if (message.includes('does not name a crystal')) return 'crystal_detonate_wrong_target_kind'
  if (message.includes('does not name a block')) return 'block_target_missing'
  if (message.includes('target index')) return 'target_index_invalid'
  if (message.includes('combat intent')) return 'intent_invalid'
  if (message.startsWith('control_apply:')) return message.slice(0, 120)
  return `malformed_v2:${message.slice(0, 100)}`
}

function validActionId(value: number | undefined): number | undefined {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 1
    ? value
    : undefined
}

export function workerCapabilities(): string[] {
  return [
    `schema-${SCHEMA_VERSION}`,
    'observation-v2',
    'action-v2',
    'conditional-intent-targets',
    'legal-control-allowlist',
    'crosshair-placement',
    'execution-source-v1',
    'teacher-pre-execution-observation-v1',
    'action-correlation-v1',
    'finite-wire-v1',
    'per-match-combat-mode'
  ]
}
