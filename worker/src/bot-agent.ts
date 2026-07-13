import mineflayer, { type Bot } from 'mineflayer'
import {
  SCHEMA_VERSION,
  type ActionV1,
  type ObservationV1,
  type StepFeedback
} from './contracts.js'
import { LegalControlAdapter } from './legal-controls.js'
import { ObservationBuilder, type MatchContext } from './observation.js'

export type BotAgentOptions = {
  agentId: string
  username: string
  host: string
  port: number
  version?: string
  onDisconnected?: (agent: BotAgent) => void
}

export class BotAgent {
  readonly bot: Bot
  private readonly controls: LegalControlAdapter
  private readonly observations: ObservationBuilder
  private tick = 0
  private episodeId = 'waiting'
  private policyVersion = 0
  private arenaSeed = 0
  private actionDelayTicks = 0
  private observationDelayTicks = 0
  private feedback: StepFeedback = { reward: 0, terminated: false, truncated: false, info: {} }
  private previousHealth = 20
  private queuedActions: Array<{ due: number; action: ActionV1 }> = []
  private observationHistory: ObservationV1[] = []
  private arenaManaged = false

  constructor(readonly options: BotAgentOptions) {
    this.bot = mineflayer.createBot({
      host: options.host,
      port: options.port,
      username: options.username,
      version: options.version ?? '1.12.2',
      auth: 'offline',
      hideErrors: false
    })
    this.controls = new LegalControlAdapter(this.bot)
    this.observations = new ObservationBuilder(this.bot)
    this.bot.on('health', () => this.updateLocalReward())
    this.bot.on('death', () => {
      if (!this.arenaManaged) {
        this.feedback = { reward: -1, terminated: true, truncated: false, info: { reason: 'death' } }
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

  setMatch(context: Partial<MatchContext>): void {
    if (typeof context.episode_id === 'string' && context.episode_id !== this.episodeId) {
      this.episodeId = context.episode_id
      this.observationHistory = []
      this.queuedActions = []
      this.controls.emergencyStop()
      this.arenaManaged = context.episode_id !== 'waiting'
    }
    if (typeof context.policy_version === 'number') this.policyVersion = context.policy_version
    if (typeof context.arena_seed === 'number') this.arenaSeed = context.arena_seed
    if (typeof context.action_delay_ticks === 'number') this.actionDelayTicks = context.action_delay_ticks
    if (typeof context.observation_delay_ticks === 'number') this.observationDelayTicks = context.observation_delay_ticks
  }

  setPolicyVersion(version: number): void {
    if (version !== this.policyVersion) {
      this.queuedActions = []
      this.controls.emergencyStop()
    }
    this.policyVersion = version
  }

  queueAction(action: ActionV1): void {
    this.queuedActions.push({ due: this.tick + this.actionDelayTicks, action })
  }

  async step(): Promise<{ observation: ObservationV1; feedback: StepFeedback }> {
    this.tick += 1
    while (this.queuedActions.length && this.queuedActions[0].due <= this.tick) {
      const queued = this.queuedActions.shift()
      if (queued) await this.controls.apply(queued.action, this.tick)
    }
    const context: MatchContext = {
      episode_id: this.episodeId,
      tick: this.tick,
      policy_version: this.policyVersion,
      arena_seed: this.arenaSeed,
      action_delay_ticks: this.actionDelayTicks,
      observation_delay_ticks: this.observationDelayTicks
    }
    const currentObservation = this.observations.build(context, this.controls.telemetry(this.tick))
    this.observationHistory.push(currentObservation)
    const maximumHistory = Math.max(8, this.observationDelayTicks + 2)
    while (this.observationHistory.length > maximumHistory) this.observationHistory.shift()
    const delayedIndex = Math.max(0, this.observationHistory.length - 1 - this.observationDelayTicks)
    const delayed = this.observationHistory[delayedIndex]
    const observation: ObservationV1 = {
      ...delayed,
      match: context,
      blocks: delayed.blocks.map(block => ({
        ...block,
        sample_age_ticks: block.sample_age_ticks + this.observationDelayTicks
      }))
    }
    const feedback = this.feedback
    this.feedback = { reward: 0, terminated: false, truncated: false, info: {} }
    return { observation, feedback }
  }

  applyArenaFeedback(feedback: StepFeedback): void {
    this.arenaManaged = true
    this.feedback = {
      reward: this.feedback.reward + feedback.reward,
      terminated: this.feedback.terminated || feedback.terminated,
      truncated: this.feedback.truncated || feedback.truncated,
      info: { ...this.feedback.info, ...feedback.info }
    }
    if (feedback.terminated || feedback.truncated) this.controls.emergencyStop()
  }

  emergencyStop(): void {
    this.queuedActions = []
    this.controls.emergencyStop()
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
}

export function workerCapabilities(): string[] {
  return [`schema-${SCHEMA_VERSION}`, 'legal-control-allowlist', 'crosshair-placement']
}
