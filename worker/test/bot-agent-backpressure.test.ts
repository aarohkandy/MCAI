import { describe, expect, it, vi } from 'vitest'

vi.mock('mineflayer', () => ({
  default: {
    createBot: () => ({
      on: () => undefined,
      entity: { position: { x: 0, y: 0, z: 0 }, yaw: 0, pitch: 0 },
      health: 20,
      food: 20
    })
  }
}))

import { BotAgent } from '../src/bot-agent.js'
import { CRYSTAL_RETENTION_ACQUISITION_TICKS } from '../src/bot-agent.js'
import {
  ACTION_V2_SCHEMA_VERSION,
  NOOP_ACTION,
  type ActionV1
} from '../src/contracts.js'
import { rolloutEligibleAgents } from '../src/worker.js'

function createAgentHarness() {
  const apply = vi.fn(async (action: ActionV1) => ({
    source: 'policy' as const,
    action,
    combatPriority: true
  }))
  const controls = {
    observeTeacherCompletion: vi.fn(),
    apply,
    applySwordBootcampAssist: vi.fn(async () => null),
    applyCombinedCrystalDemonstration: vi.fn(async () => null),
    applyCombinedTacticalBlockDemonstration: vi.fn(async () => null),
    applyPitchSafety: vi.fn(async () => null),
    applyTrainerStallSafety: vi.fn(async () => ({
      source: 'safety' as const,
      action: { ...NOOP_ACTION, forward: 1 as const, sprint: true }
    })),
    telemetry: () => ({
      lastAttackTick: -1000,
      activeHand: 'none' as const,
      useStartedTick: 0,
      miningProgress: 0
    }),
    emergencyStop: vi.fn(),
    beginEpisode: vi.fn(),
    setMatchConfiguration: vi.fn(),
    setCrystalRetentionSwordFallbackEnabled: vi.fn(),
    setOpponentUsername: vi.fn()
  }
  const observations = {
    setOpponentUsername: vi.fn(),
    acceptArenaSnapshot: vi.fn(() => true),
    build: vi.fn((match: Record<string, unknown>) => ({
      schema_version: 1,
      match,
      blocks: []
    }))
  }
  const agent = new BotAgent({
    agentId: 'worker/agent-0',
    username: 'MCAI_001',
    host: '127.0.0.1',
    port: 25565
  })
  ;(agent as any).controls = controls
  ;(agent as any).observations = observations
  return { agent, apply, controls, observations }
}

describe('BotAgent trainer backpressure', () => {
  it('resolves V2 against its exact sent observation without a duplicate build', async () => {
    const { agent, observations } = createAgentHarness()
    agent.setMatch({ episode_id: 'episode-correlated-observation' })
    ;(agent as any).lastTrainerObservation = {
      schema_version: 2,
      match: { episode_id: 'episode-correlated-observation' },
      action_mask: {},
      tactical: {}
    }
    agent.queueAction({
      ...NOOP_ACTION,
      schema_version: ACTION_V2_SCHEMA_VERSION,
      intent: 'reposition',
      target_index: -1
    }, 73)

    await agent.step()
    expect(observations.build).toHaveBeenCalledTimes(1)
  })

  it('pressures forward and suppresses sword clicks until crystal retention attempts once', () => {
    const { agent, controls } = createAgentHarness()
    agent.setMatch({
      episode_id: 'episode-crystal-retention',
      lane: 'crystal_retention',
      mode: 'crystal'
    })
    const sword = { ...NOOP_ACTION, primary: 'attack' as const }
    expect((agent as any).applyCrystalRetentionGate(sword, 'sword_engage')).toMatchObject({
      primary: 'none', forward: 1, sprint: true
    })
    ;(agent as any).crystalRetentionAttempted = true
    expect((agent as any).applyCrystalRetentionGate(sword, 'sword_engage')).toEqual(sword)
    expect(controls.setCrystalRetentionSwordFallbackEnabled).toHaveBeenCalledWith(false)
  })

  it('opens crystal retention sword fallback after the bounded acquisition window', () => {
    const { agent, controls } = createAgentHarness()
    agent.setMatch({
      episode_id: 'episode-crystal-timeout',
      lane: 'crystal_retention',
      mode: 'crystal'
    })
    const sword = { ...NOOP_ACTION, primary: 'attack' as const }
    ;(agent as any).tick = CRYSTAL_RETENTION_ACQUISITION_TICKS - 1
    expect((agent as any).applyCrystalRetentionGate(sword, 'sword_engage').primary)
      .toBe('none')

    ;(agent as any).tick = CRYSTAL_RETENTION_ACQUISITION_TICKS
    expect((agent as any).applyCrystalRetentionGate(sword, 'sword_engage')).toEqual(sword)
    expect(controls.setCrystalRetentionSwordFallbackEnabled).toHaveBeenCalledWith(true)
    expect((agent as any).crystalRetentionAttempted).toBe(false)
  })

  it('reports malformed V2 target reasons instead of silently hiding the safety no-op', async () => {
    const { agent, apply } = createAgentHarness()
    agent.setMatch({ episode_id: 'episode-v2-rejection' })
    agent.queueAction({
      ...NOOP_ACTION,
      schema_version: ACTION_V2_SCHEMA_VERSION,
      intent: 'crystal_acquire',
      target_index: 0
    }, 99)

    const step = await agent.step()
    expect(apply).not.toHaveBeenCalled()
    expect(step.execution).toMatchObject({ source: 'safety', action_id: 99 })
    expect(step.feedback.info.worker_v2_rejections).toEqual({
      implicit_intent_with_target: 1
    })
  })

  it('accepts arena snapshots only for its currently active episode', () => {
    const { agent, observations } = createAgentHarness()
    agent.setMatch({ episode_id: 'episode-live' })
    expect(agent.applyArenaSnapshot({ episode_id: 'episode-old', fighters: [] })).toBe(false)
    expect(observations.acceptArenaSnapshot).not.toHaveBeenCalled()

    const payload = { episode_id: 'episode-live', fighters: [] }
    expect(agent.applyArenaSnapshot(payload)).toBe(true)
    expect(observations.acceptArenaSnapshot).toHaveBeenCalledWith(
      payload, 'episode-live', expect.any(Number)
    )
  })

  it('never consumes delayed IDs in passive heartbeats and emits one per step', async () => {
    const { agent, apply } = createAgentHarness()
    agent.setMatch({ episode_id: 'episode-a', action_delay_ticks: 2 })

    const repeated = { ...NOOP_ACTION, forward: 1 as const }
    agent.queueAction(repeated, 101)
    agent.queueAction(repeated, 102)

    await agent.heartbeat()
    await agent.heartbeat()
    await agent.heartbeat()
    expect(apply).not.toHaveBeenCalled()

    const first = await agent.step()
    expect(apply).toHaveBeenCalledTimes(1)
    expect(first.execution).toMatchObject({ source: 'policy', action_id: 101 })

    const second = await agent.step()
    expect(apply).toHaveBeenCalledTimes(2)
    expect(second.execution).toMatchObject({ source: 'policy', action_id: 102 })
  })

  it('emits one terminal transition, retires to waiting, and ignores waiting actions', async () => {
    const { agent, apply } = createAgentHarness()
    agent.setMatch({ episode_id: 'episode-old', action_delay_ticks: 0 })
    agent.queueAction({ ...NOOP_ACTION, primary: 'attack' }, 201)
    agent.applyArenaFeedback({
      reward: 1,
      terminated: true,
      truncated: false,
      info: { episode_id: 'episode-old', outcome: 'win' }
    })

    expect(agent.hasTerminalStepPending()).toBe(true)
    expect(rolloutEligibleAgents([agent])).toEqual([agent])
    const terminal = await agent.step()

    expect(terminal.observation.match.episode_id).toBe('episode-old')
    expect(terminal.feedback).toMatchObject({
      reward: 1,
      terminated: true,
      info: { episode_id: 'episode-old', outcome: 'win' }
    })
    expect(apply).not.toHaveBeenCalled()
    expect(agent.isMatchActive()).toBe(false)
    expect(agent.hasTerminalStepPending()).toBe(false)
    expect(rolloutEligibleAgents([agent])).toEqual([])

    agent.queueAction({ ...NOOP_ACTION, forward: 1 }, 202)
    await agent.heartbeat()
    for (const eligible of rolloutEligibleAgents([agent])) await eligible.step()
    expect(apply).not.toHaveBeenCalled()

    agent.setMatch({ episode_id: 'episode-new', action_delay_ticks: 0 })
    expect(rolloutEligibleAgents([agent])).toEqual([agent])
    agent.queueAction({ ...NOOP_ACTION, forward: 1 }, 203)
    const reactivated = await agent.step()
    expect(reactivated.observation.match.episode_id).toBe('episode-new')
    expect(reactivated.execution).toMatchObject({ source: 'policy', action_id: 203 })
  })

  it('flushes the old terminal before activating an early replacement match', async () => {
    const { agent, apply, controls } = createAgentHarness()
    agent.setMatch({ episode_id: 'episode-old', action_delay_ticks: 3 })
    agent.setOpponentUsername('MCAI_002')
    agent.queueAction({ ...NOOP_ACTION, forward: -1 }, 301)
    agent.applyArenaFeedback({
      reward: -1,
      terminated: true,
      truncated: false,
      info: { episode_id: 'episode-old', outcome: 'loss' }
    })

    agent.setMatch({ episode_id: 'episode-new', action_delay_ticks: 0 })
    agent.setOpponentUsername('MCAI_008')
    const terminal = await agent.step()

    expect(terminal.observation.match.episode_id).toBe('episode-old')
    expect(terminal.feedback.info).toMatchObject({
      episode_id: 'episode-old',
      outcome: 'loss'
    })
    expect(apply).not.toHaveBeenCalled()
    expect(agent.isMatchActive()).toBe(true)
    expect(controls.setOpponentUsername).toHaveBeenLastCalledWith('MCAI_008')

    const newAction = { ...NOOP_ACTION, strafe: 1 as const }
    agent.queueAction(newAction, 302)
    const replacement = await agent.step()
    expect(replacement.observation.match.episode_id).toBe('episode-new')
    expect(replacement.execution).toMatchObject({ action_id: 302, action: newAction })
    expect(apply).toHaveBeenCalledTimes(1)
    expect(apply).toHaveBeenLastCalledWith(newAction, expect.any(Number))
  })

  it('retains a consumed id when a higher-priority execution replaces it in-window', () => {
    const agent = Object.create(BotAgent.prototype) as any
    agent.hasPendingExecution = false
    agent.pendingExecution = { source: 'safety', action: { ...NOOP_ACTION } }
    agent.liveExecutionSource = 'policy'
    const teacherObservation = {
      schema_version: 1,
      match: { episode_id: 'episode-a', tick: 10 }
    }
    agent.options = { onExecution: vi.fn() }
    agent.episodeId = 'episode-a'

    agent.noteExecution({ source: 'policy', action: { ...NOOP_ACTION }, action_id: 77 })
    agent.pendingTeacherObservation = teacherObservation
    agent.noteExecution({
      source: 'teacher_sword',
      action: { ...NOOP_ACTION, primary: 'attack' }
    })

    expect(agent.pendingExecution).toMatchObject({
      source: 'teacher_sword',
      action_id: 77,
      action: { primary: 'attack' },
      pre_execution_observation: teacherObservation
    })
  })

  it('drops delayed trainer actions at a websocket-session boundary without retiring the match', async () => {
    const { agent, apply, controls } = createAgentHarness()
    agent.setMatch({ episode_id: 'episode-live', action_delay_ticks: 2 })
    agent.queueAction({ ...NOOP_ACTION, forward: 1 }, 401)

    agent.resetTrainerSession()
    await agent.heartbeat()
    await agent.heartbeat()
    await agent.step()

    expect(agent.isMatchActive()).toBe(true)
    expect(apply).not.toHaveBeenCalled()
    expect(controls.emergencyStop).toHaveBeenCalled()
  })

  it('drops a proposal sampled before local stall safety and reports its exact id as safety', async () => {
    const beforeExecution = vi.fn(async () => true)
    const { agent, apply, controls } = createAgentHarness()
    ;(agent as any).options.beforeExecution = beforeExecution
    agent.setMatch({ episode_id: 'episode-stalled', action_delay_ticks: 0 })

    await agent.heartbeat()
    await agent.continueDuringTrainerStall()
    agent.queueAction({ ...NOOP_ACTION, primary: 'attack' }, 501)
    const excluded = await agent.step()

    expect(controls.applyTrainerStallSafety).toHaveBeenCalled()
    expect(beforeExecution).toHaveBeenCalledWith(agent, 'safety', 'episode-stalled')
    expect(apply).not.toHaveBeenCalled()
    expect(excluded.execution).toEqual({
      source: 'safety', action_id: 501,
      action: { ...NOOP_ACTION, forward: 1, sprint: true }
    })

    agent.queueAction({ ...NOOP_ACTION, strafe: 1 }, 502)
    const resumed = await agent.step()
    expect(resumed.execution).toMatchObject({ source: 'policy', action_id: 502 })
    expect(apply).toHaveBeenCalledTimes(1)
  })

  it('clears local stall ownership when the trainer websocket session resets', async () => {
    const { agent, controls } = createAgentHarness()
    agent.setMatch({ episode_id: 'episode-reset-stall' })
    await agent.continueDuringTrainerStall()

    agent.resetTrainerSession()
    agent.queueAction({ ...NOOP_ACTION, forward: 1 }, 601)
    const step = await agent.step()

    expect(controls.emergencyStop).toHaveBeenCalled()
    expect(step.execution).toMatchObject({ source: 'policy', action_id: 601 })
  })

  it('never applies a delayed action to the episode replacing its proposal episode', async () => {
    const { agent, apply } = createAgentHarness()
    agent.setMatch({ episode_id: 'episode-new' })

    agent.queueAction(
      { ...NOOP_ACTION, primary: 'attack' }, 701, 'episode-old'
    )
    const discarded = await agent.step()

    expect(discarded.observation.match.episode_id).toBe('episode-new')
    expect(discarded.execution).toMatchObject({ source: 'safety', action_id: 701 })
    expect(apply).not.toHaveBeenCalled()
  })
})
