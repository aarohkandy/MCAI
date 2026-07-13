import os from 'node:os'
import { randomUUID } from 'node:crypto'
import {
  SCHEMA_VERSION,
  type ActionBatch,
  type StepBatch
} from './contracts.js'
import { ArenaClient, type ArenaEvent } from './arena-client.js'
import { BotAgent } from './bot-agent.js'
import { LoadController } from './load-controller.js'
import { scriptedAction, type ScriptedStyle } from './scripted-policy.js'
import { TrainerConnection } from './trainer-connection.js'

export type WorkerOptions = {
  workerId: string
  serverHost: string
  serverPort: number
  trainerUrl: string
  arenaHost: string
  arenaPort: number
  botCount: number
  usernamePrefix: string
}

export class RolloutWorker {
  private readonly agents: BotAgent[]
  private readonly byId: Map<string, BotAgent>
  private readonly trainer: TrainerConnection
  private readonly arena: ArenaClient
  private sequence = 0
  private policyVersion = 0
  private timer: NodeJS.Timeout | null = null
  private stepping = false
  private trainerReady = false
  private readonly loadController: LoadController
  private arenaReconnectTimer: NodeJS.Timeout | null = null
  private arenaConnecting = false
  private stopped = false
  private reconnectingAgents = new Map<string, NodeJS.Timeout>()

  constructor(private readonly options: WorkerOptions) {
    this.agents = Array.from({ length: options.botCount }, (_, index) => this.createAgent(index))
    this.byId = new Map(this.agents.map(agent => [agent.id, agent]))
    this.trainer = new TrainerConnection(options.trainerUrl, options.workerId, this.agents.map(agent => agent.id))
    this.arena = new ArenaClient(options.arenaHost, options.arenaPort)
    this.loadController = new LoadController(this.arena, {
      initialPairs: Math.min(2, Math.max(1, Math.floor(options.botCount / 2))),
      maximumPairs: Math.max(1, Math.floor(options.botCount / 2))
    })
  }

  async start(): Promise<void> {
    this.stopped = false
    this.trainer.on('ready', () => { this.trainerReady = true })
    this.trainer.on('disconnected', () => { this.trainerReady = false })
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
    this.timer = setInterval(() => void this.tick(), 50)
  }

  async stop(): Promise<void> {
    this.stopped = true
    if (this.timer) clearInterval(this.timer)
    if (this.arenaReconnectTimer) clearTimeout(this.arenaReconnectTimer)
    for (const timer of this.reconnectingAgents.values()) clearTimeout(timer)
    this.reconnectingAgents.clear()
    this.timer = null
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

  private async tick(): Promise<void> {
    if (this.stepping) return
    this.stepping = true
    try {
      const ready = this.agents.filter(agent => agent.isSpawned())
      if (ready.length === 0) return
      const steps = await Promise.all(ready.map(async agent => {
        const { observation, feedback } = await agent.step()
        if (!this.trainerReady) agent.queueAction(scriptedAction(observation, styleFor(agent.id)))
        return {
          agent_id: agent.id,
          observation,
          reward: feedback.reward,
          terminated: feedback.terminated,
          truncated: feedback.truncated,
          info: feedback.info
        }
      }))
      const batch: StepBatch = {
        schema_version: SCHEMA_VERSION,
        type: 'step_batch',
        sequence: ++this.sequence,
        policy_version: this.policyVersion,
        steps
      }
      this.trainer.sendSteps(batch)
    } finally {
      this.stepping = false
    }
  }

  private onActions(batch: ActionBatch): void {
    if (batch.sequence < this.sequence - 4) return
    this.policyVersion = batch.policy_version
    for (const agent of this.agents) agent.setPolicyVersion(this.policyVersion)
    for (const entry of batch.actions) this.byId.get(entry.agent_id)?.queueAction(entry.action)
  }

  private onArenaEvent(event: ArenaEvent): void {
    if (event.event === 'match_started') {
      const payload = event.payload ?? {}
      const targets = event.agent_id ? [this.byId.get(event.agent_id)].filter(Boolean) as BotAgent[] : this.agents
      for (const agent of targets) {
        agent.setMatch({
          episode_id: String(payload.episode_id ?? randomUUID()),
          arena_seed: Number(payload.arena_seed ?? 0),
          action_delay_ticks: Number(payload.action_delay_ticks ?? 0),
          observation_delay_ticks: Number(payload.observation_delay_ticks ?? 0)
        })
      }
      return
    }
    if (event.event === 'emergency_stop') {
      for (const agent of this.agents) agent.emergencyStop()
      return
    }
    if ((event.event === 'step_feedback' || event.event === 'match_ended') && event.agent_id) {
      const payload = event.payload ?? {}
      this.byId.get(event.agent_id)?.applyArenaFeedback({
        reward: Number(payload.reward ?? 0),
        terminated: event.event === 'match_ended' && payload.truncated !== true,
        truncated: event.event === 'match_ended' && payload.truncated === true,
        info: payload
      })
    }
  }
}

export function optionsFromEnvironment(): WorkerOptions {
  return {
    workerId: process.env.MCAI_WORKER_ID ?? `${os.hostname()}-${process.pid}`,
    serverHost: process.env.MCAI_SERVER_HOST ?? '127.0.0.1',
    serverPort: integerEnv('MCAI_SERVER_PORT', 25565),
    trainerUrl: process.env.MCAI_TRAINER_URL ?? 'ws://127.0.0.1:8766',
    arenaHost: process.env.MCAI_ARENA_HOST ?? '127.0.0.1',
    arenaPort: integerEnv('MCAI_ARENA_PORT', 8765),
    botCount: integerEnv('MCAI_BOT_COUNT', 4),
    usernamePrefix: process.env.MCAI_USERNAME_PREFIX ?? 'MCAI_'
  }
}

function integerEnv(name: string, fallback: number): number {
  const parsed = Number.parseInt(process.env[name] ?? '', 10)
  return Number.isFinite(parsed) ? parsed : fallback
}

function styleFor(agentId: string): ScriptedStyle {
  const styles: ScriptedStyle[] = ['rush', 'strafe', 'retreat', 'jump_critical', 'defensive', 'erratic']
  let hash = 0
  for (const char of agentId) hash = (hash * 31 + char.charCodeAt(0)) | 0
  return styles[Math.abs(hash) % styles.length]
}
