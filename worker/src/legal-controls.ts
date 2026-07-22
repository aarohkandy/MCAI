import type { Bot } from 'mineflayer'
import { Vec3 } from 'vec3'
import {
  NOOP_ACTION,
  type ActionV1,
  type StepExecution,
  validateAction
} from './contracts.js'
import { clamp, normalizeAngle } from './math.js'
import type { ControlTelemetry } from './observation.js'
import {
  assignedOpponentMeleeTarget,
  findAssignedOpponent,
  legalArenaAttackTargetAtCrosshair
} from './targeting.js'
import {
  findTacticalMiningTarget,
  isAir,
  isSafeTacticalMiningTarget,
  tacticalCrystalPadPlacementAt,
  TacticalPlacementTracker,
  type PolicyMineReplacementReservation,
  type TacticalWallPlacement
} from './tactical-blocks.js'

export type CombatControlMode = 'sword' | 'crystal' | 'combined' | 'terrain'

/** Ordinary policy exploration always keeps the assigned opponent in a usable view. */
export const MAX_COMBAT_PITCH = Math.PI / 4
/** Verified block/crystal teacher targets may briefly require steeper aim. */
export const MAX_CRYSTAL_PITCH = 75 * Math.PI / 180
export const EXTREME_PITCH = Math.PI / 3
export const EXTREME_PITCH_RECENTER_TICKS = 10
export const MAX_TEACHER_CONTROL_TICKS = 4
export const SWORD_BOOTCAMP_ATTACK_COOLDOWN_TICKS = 12
export const SWORD_BOOTCAMP_REACH = 3.4
export const COMBINED_RESCUE_BASE_TICKS = 60
export const COMBINED_CRYSTAL_DEMO_PERIOD_TICKS = 180
export const COMBINED_CRYSTAL_DEMO_ACQUIRE_TICKS = 8
export const COMBINED_CRYSTAL_DEMO_RETRY_TICKS = 20
export const COMBINED_TACTICAL_BLOCK_DEMO_PERIOD_TICKS = 300
export const COMBINED_TACTICAL_BLOCK_DEMO_RETRY_TICKS = 30
const COMBINED_RESCUE_JITTER_TICKS = 7
const COMBINED_CRYSTAL_DEMO_MIN_DELAY_TICKS = 20
const COMBINED_CRYSTAL_DEMO_STAGGER_TICKS = 80
// Keep every demonstrated placement close enough for the just-spawned
// crystal to remain a reliable next-tick attack target.  Vanilla placement
// reaches farther than combat, but advertising that extra distance teaches a
// dead-end half of the combo.
const COMBINED_CRYSTAL_DEMO_PLACE_REACH = 3.3
const COMBINED_CRYSTAL_DEMO_ATTACK_RAY_REACH = 3.4
const COMBINED_CRYSTAL_BASE_SCAN_RADIUS = 4
const COMBINED_TACTICAL_BLOCK_DEMO_MIN_DELAY_TICKS = 5
const COMBINED_TACTICAL_BLOCK_DEMO_STAGGER_TICKS = 20
const COMBINED_TACTICAL_BLOCK_DEMO_TIMEOUT_TICKS = 20
const COMBINED_TACTICAL_DIG_MARGIN_TICKS = 12
const COMBINED_TACTICAL_MAX_DIG_WINDOW_TICKS = 120
const SWORD_BOOTCAMP_APPROACH_DISTANCE = 2.2

export type CombatMatchConfiguration = {
  mode: CombatControlMode
  radius: number
  stage: number
  teachersEnabled: boolean
  terrainEnabled: boolean
}

export type PolicyExecution = StepExecution & {
  /** True only for an executable assigned-opponent melee or crystal action. */
  combatPriority: boolean
  /** True only for an executable autonomous crystal placement/detonation. */
  crystalPriority: boolean
}

type TeacherExecutionSource = Extract<StepExecution['source'], `teacher_${string}`>

type CombinedCrystalDemoState = {
  phase: 'place_pad' | 'acquire_crystal' | 'attack_crystal'
  basePosition: Vec3
  existingCrystalIds: Set<string>
  deadlineTick: number
  fallbackTick: number
  crystalId?: string
  controlTicks: number
  previousHotbar: number
  slotRestored: boolean
}

type CombinedTacticalBlockDemoState = {
  phase: 'mine_aim' | 'mine_wait' | 'place_aim' | 'place_wait'
  targetPosition: Vec3
  deadlineTick: number
  fallbackTick: number
  stateOwnedCover: boolean
  controlTicks: number
  previousHotbar: number
  slotRestored: boolean
}

/** A stable per-agent offset keeps all parallel arenas from entering rescue on the same tick. */
export function combinedRescueDelay(agentKey: string): number {
  const hash = stableAgentHash(agentKey)
  const width = COMBINED_RESCUE_JITTER_TICKS * 2 + 1
  return COMBINED_RESCUE_BASE_TICKS + (hash % width) - COMBINED_RESCUE_JITTER_TICKS
}

/** Stable first-attempt phase so parallel fighters do not all place on the same tick. */
export function combinedCrystalDemoDelay(agentKey: string): number {
  return COMBINED_CRYSTAL_DEMO_MIN_DELAY_TICKS
    + (stableAgentHash(`crystal-demo:${agentKey}`) % COMBINED_CRYSTAL_DEMO_STAGGER_TICKS)
}

/** Early per-agent phase lets one fighter build before its later-offset opponent mines. */
export function combinedTacticalBlockDemoDelay(agentKey: string): number {
  return COMBINED_TACTICAL_BLOCK_DEMO_MIN_DELAY_TICKS
    + (stableAgentHash(`tactical-block-demo:${agentKey}`)
      % COMBINED_TACTICAL_BLOCK_DEMO_STAGGER_TICKS)
}

export type ControlAuditEvent = {
  tick: number
  operation: string
  detail?: string
}

export class LegalControlAdapter {
  private tick = 0
  private activeHand: ControlTelemetry['activeHand'] = 'none'
  private useStartedTick = 0
  private lastAttackTick = -1000
  private miningProgress = 0
  private miningStartedTick = 0
  private miningExpectedTicks = 1
  private digging = false
  private activePolicyMineReplacement: PolicyMineReplacementReservation | null = null
  private combatAimHoldUntilTick = -1
  private lastAction: ActionV1 = { ...NOOP_ACTION }
  private audit: ControlAuditEvent[] = []
  private opponentUsername: string | null = null
  private lastCombinedAttackTick = 0
  private mode: CombatControlMode
  private matchRadius = 5
  private matchStage = 1
  private teachersEnabled = true
  private crystalRetentionSwordFallbackEnabled = false
  private terrainEnabled = false
  private extremePitchTicks = 0
  private recenterRequested = false
  private swordDemoUsed = false
  private swordDemoActive = false
  private swordDemoControlTicks = 0
  private swordDemoPreviousHotbar = -1
  private readonly combinedRescueDelayTicks: number
  private readonly combinedCrystalDemoDelayTicks: number
  private combinedCrystalDemoNextTick: number
  private combinedCrystalDemoState: CombinedCrystalDemoState | null = null
  private combinedCrystalDemoUsed = false
  private readonly combinedTacticalBlockDemoDelayTicks: number
  private combinedTacticalBlockDemoNextTick: number
  private combinedTacticalBlockDemoState: CombinedTacticalBlockDemoState | null = null
  private combinedTacticalBlockDemoUsed = false
  private readonly tacticalPlacements: TacticalPlacementTracker

  constructor(
    private readonly bot: Bot,
    mode: CombatControlMode = 'sword',
    cadenceKey = '',
    private readonly beforeTeacherExecution: (
      source: TeacherExecutionSource
    ) => Promise<boolean> = async () => true,
    tacticalPlacements?: TacticalPlacementTracker
  ) {
    this.tacticalPlacements = tacticalPlacements ?? new TacticalPlacementTracker(bot)
    this.mode = mode
    this.terrainEnabled = mode === 'terrain'
    this.combinedRescueDelayTicks = combinedRescueDelay(cadenceKey)
    this.combinedCrystalDemoDelayTicks = combinedCrystalDemoDelay(cadenceKey)
    this.combinedCrystalDemoNextTick = this.combinedCrystalDemoDelayTicks
    this.combinedTacticalBlockDemoDelayTicks = combinedTacticalBlockDemoDelay(cadenceKey)
    this.combinedTacticalBlockDemoNextTick = this.combinedTacticalBlockDemoDelayTicks
  }

  setOpponentUsername(username: string | null): void {
    this.opponentUsername = username?.trim() || null
  }

  setCrystalRetentionSwordFallbackEnabled(enabled: boolean): void {
    this.crystalRetentionSwordFallbackEnabled = enabled
  }

  setMatchConfiguration(configuration: Partial<CombatMatchConfiguration>): void {
    if (configuration.mode) this.mode = configuration.mode
    if (Number.isFinite(configuration.radius)) {
      this.matchRadius = Math.max(1, Math.floor(configuration.radius as number))
    }
    if (Number.isFinite(configuration.stage)) {
      this.matchStage = Math.max(1, Math.floor(configuration.stage as number))
    }
    if (typeof configuration.teachersEnabled === 'boolean') {
      this.teachersEnabled = configuration.teachersEnabled
    }
    if (typeof configuration.terrainEnabled === 'boolean') {
      this.terrainEnabled = configuration.terrainEnabled
    } else if (configuration.mode) {
      this.terrainEnabled = configuration.mode === 'terrain'
    }
  }

  matchConfiguration(): CombatMatchConfiguration {
    return {
      mode: this.mode,
      radius: this.matchRadius,
      stage: this.matchStage,
      teachersEnabled: this.teachersEnabled,
      terrainEnabled: this.terrainEnabled
    }
  }

  /** Reset one-demo-per-episode guards without reconstructing the bot client. */
  beginEpisode(configuration: Partial<CombatMatchConfiguration> = {}): void {
    this.restoreSwordDemoSlot()
    this.restoreCrystalDemoSlot(this.combinedCrystalDemoState)
    this.restoreTacticalDemoSlot(this.combinedTacticalBlockDemoState)
    this.setMatchConfiguration(configuration)
    this.swordDemoUsed = false
    this.swordDemoActive = false
    this.swordDemoControlTicks = 0
    this.swordDemoPreviousHotbar = -1
    this.combinedCrystalDemoState = null
    this.combinedCrystalDemoUsed = false
    this.combinedCrystalDemoNextTick = this.tick + this.combinedCrystalDemoDelayTicks
    this.combinedTacticalBlockDemoState = null
    this.combinedTacticalBlockDemoUsed = false
    this.combinedTacticalBlockDemoNextTick = this.tick + this.combinedTacticalBlockDemoDelayTicks
    this.extremePitchTicks = 0
    this.recenterRequested = false
    this.lastCombinedAttackTick = this.tick
  }

  telemetry(tick = this.tick): ControlTelemetry {
    if (this.digging) {
      this.miningProgress = clamp((tick - this.miningStartedTick) / this.miningExpectedTicks, 0.01, 0.99)
    }
    return {
      lastAttackTick: this.lastAttackTick,
      activeHand: this.activeHand,
      useStartedTick: this.useStartedTick,
      miningProgress: this.miningProgress
    }
  }

  getAuditEvents(): readonly ControlAuditEvent[] {
    return this.audit
  }

  /** Advance teacher completion state without taking control away from the policy. */
  observeTeacherCompletion(tick: number): void {
    this.tick = tick
    const crystal = this.combinedCrystalDemoState
    if (crystal) {
      if (tick > crystal.deadlineTick) {
        this.finishCombinedCrystalDemo(tick, 'combined_crystal_demo_timeout', true)
      } else if (crystal.phase === 'place_pad' || crystal.phase === 'acquire_crystal') {
        const spawned = this.newCrystalAtDemoBase(crystal)
        if (spawned) {
          crystal.phase = 'attack_crystal'
          crystal.crystalId = crystalIdentity(spawned)
          crystal.fallbackTick = tick + 1
          this.record('combined_crystal_demo_place_observed', crystal.crystalId)
        }
      } else if (crystal.crystalId && !this.loadedCrystalById(crystal.crystalId)) {
        this.finishCombinedCrystalDemo(
          tick, 'combined_crystal_demo_crystal_gone', false, crystal.crystalId
        )
      }
      if (crystal.controlTicks >= MAX_TEACHER_CONTROL_TICKS) {
        this.restoreCrystalDemoSlot(crystal)
      }
    }

    const tactical = this.combinedTacticalBlockDemoState
    if (tactical) {
      // Resolving retires this exact shared marker. The bounded tracker may
      // then advertise a distinct later pad, but this one-demo teacher stays
      // spent for the rest of the episode.
      this.tacticalPlacements.resolve(this.opponentUsername, tick)
      if (tick > tactical.deadlineTick) {
        this.finishCombinedTacticalBlockDemo(tick, 'combined_tactical_block_demo_timeout', true)
      } else {
        const block: any = this.bot.blockAt(tactical.targetPosition, false)
        if (tactical.phase === 'mine_wait') {
          const name = String(block?.name ?? '').toLowerCase()
          if (isAir(block) || (name !== 'stone' && name !== 'obsidian')) {
            this.digging = false
            this.miningProgress = 1
            this.finishCombinedTacticalBlockDemo(tick, 'combined_tactical_block_demo_mined')
          }
        } else if (tactical.phase === 'place_wait'
          && String(block?.name ?? '').toLowerCase() === 'obsidian') {
          // The new foundation is the lesson. Leave it in the arena so the
          // next autonomous decision can place and detonate a crystal on it.
          this.finishCombinedTacticalBlockDemo(
            tick, 'combined_tactical_block_demo_placed'
          )
        }
      }
      if (tactical.controlTicks >= MAX_TEACHER_CONTROL_TICKS) {
        this.restoreTacticalDemoSlot(tactical)
      }
    }
  }

  /** Recenter only after teacher control, or after ten consecutive extreme-pitch ticks. */
  async applyPitchSafety(tick: number): Promise<StepExecution | null> {
    this.tick = tick
    const self: any = this.bot.entity
    const pitch = Number(self?.pitch ?? 0)
    this.extremePitchTicks = Math.abs(pitch) > EXTREME_PITCH
      ? this.extremePitchTicks + 1
      : 0
    if (!this.recenterRequested && this.extremePitchTicks < EXTREME_PITCH_RECENTER_TICKS) {
      return null
    }
    const target: any = findAssignedOpponent(this.bot, this.opponentUsername)
    if (!self?.position || !target?.position) return null
    const before = this.lookSnapshot()
    const delta = target.position.minus(self.position)
    const horizontal = Math.hypot(delta.x, delta.z)
    const eyeHeight = Number(self.eyeHeight ?? 1.62)
    const targetHeight = Math.max(1, Number(target.height ?? 1.8) * 0.75)
    const yaw = horizontal > 1e-9
      ? normalizeAngle(Math.atan2(-delta.x, -delta.z))
      : Number(self.yaw ?? 0)
    const safePitch = clamp(
      Math.atan2(delta.y + targetHeight - eyeHeight, Math.max(horizontal, 1e-9)),
      -MAX_COMBAT_PITCH,
      MAX_COMBAT_PITCH
    )
    await this.bot.look(yaw, safePitch, true)
    this.extremePitchTicks = 0
    this.recenterRequested = false
    this.record('recenter_assigned_opponent', this.opponentUsername ?? undefined)
    return {
      source: 'safety',
      action: { ...NOOP_ACTION, ...this.lookDelta(before) }
    }
  }

  /**
   * Bounded local continuation while the trainer cannot answer. It uses only
   * the arena-assigned opponent and permits only cooldown/reach/LOS-validated
   * sword clicks. Everything is reported as safety so the delayed PPO proposal
   * is excluded rather than trained on stale state.
   */
  async applyTrainerStallSafety(tick: number): Promise<StepExecution> {
    this.tick = tick
    const self: any = this.bot.entity
    const target: any = findAssignedOpponent(this.bot, this.opponentUsername)
    const before = this.lookSnapshot()
    if (!self?.position || !target?.position) {
      this.clearControlStatesSafely()
      return { source: 'safety', action: { ...NOOP_ACTION } }
    }
    const delta = target.position.minus(self.position)
    const horizontal = Math.hypot(delta.x, delta.z)
    const approach = horizontal > SWORD_BOOTCAMP_APPROACH_DISTANCE
    this.bot.setControlState('forward', approach)
    this.bot.setControlState('back', false)
    this.bot.setControlState('left', false)
    this.bot.setControlState('right', false)
    this.bot.setControlState('jump', false)
    this.bot.setControlState('sprint', approach)
    this.bot.setControlState('sneak', false)
    if (this.activeHand !== 'none') this.releaseUse()
    if (horizontal > 1e-6) {
      const eyeHeight = Number(self.eyeHeight ?? 1.62)
      const targetHeight = Math.max(1, Number(target.height ?? 1.8) * 0.75)
      await this.bot.look(
        normalizeAngle(Math.atan2(-delta.x, -delta.z)),
        clamp(
          Math.atan2(delta.y + targetHeight - eyeHeight, horizontal),
          -MAX_COMBAT_PITCH,
          MAX_COMBAT_PITCH
        ),
        true
      )
    }
    const swordSlot = this.swordHotbarSlot()
    if (swordSlot >= 0 && swordSlot !== this.bot.quickBarSlot) {
      this.bot.setQuickBarSlot(swordSlot)
    }
    let attacked = false
    if (swordSlot >= 0
      && tick - this.lastAttackTick >= SWORD_BOOTCAMP_ATTACK_COOLDOWN_TICKS) {
      // Require the assigned fighter to be the exact unobstructed crosshair
      // hit after aiming. This preserves ordinary reach/LOS/cooldown legality
      // and can never substitute a spectator or neighboring-lane player.
      const legalTarget = legalArenaAttackTargetAtCrosshair(
        this.bot, this.opponentUsername, SWORD_BOOTCAMP_REACH
      )
      if (legalTarget === target) {
        this.bot.attack(legalTarget)
        this.lastAttackTick = tick
        this.lastCombinedAttackTick = tick
        attacked = true
        this.record('trainer_stall_safety_attack', String(legalTarget.id))
      }
    }
    this.record('trainer_stall_safety_pressure', this.opponentUsername ?? undefined)
    return {
      source: 'safety',
      action: {
        ...NOOP_ACTION,
        forward: approach ? 1 : 0,
        sprint: approach,
        primary: attacked ? 'attack' : 'none',
        hotbar: swordSlot,
        ...this.lookDelta(before)
      }
    }
  }

  /**
   * Give the initial sword curriculum reliable, repeated examples of real
   * melee contact. The arena assignment is the authority for target identity:
   * this never falls back to the nearest player, so a spectator or a fighter
   * from another arena cannot be selected.
   *
   * The assist deliberately runs after the sampled policy action. The policy
   * still receives observations and all resulting arena rewards, while this
   * small curriculum rail keeps the bots facing, closing distance, holding a
   * sword and swinging only at the normal 1.12 cooldown.
   */
  async applySwordBootcampAssist(tick: number): Promise<StepExecution | null> {
    this.tick = tick
    if (!this.teachersEnabled
      || (this.mode === 'crystal' && !this.crystalRetentionSwordFallbackEnabled)
      || !this.opponentUsername) {
      this.finishSwordDemo()
      return null
    }
    if (!this.swordDemoActive) {
      if (this.swordDemoUsed) return null
      if ((this.mode === 'combined' || this.mode === 'terrain')
        && tick - this.lastCombinedAttackTick < this.combinedRescueDelayTicks) return null
      this.swordDemoUsed = true
      this.swordDemoActive = true
      this.swordDemoControlTicks = 0
      this.swordDemoPreviousHotbar = Number(this.bot.quickBarSlot ?? -1)
      if (this.mode === 'combined' || this.mode === 'terrain') this.record('combined_melee_rescue')
    }
    if (this.swordDemoControlTicks >= MAX_TEACHER_CONTROL_TICKS) {
      this.finishSwordDemo()
      return null
    }
    const self: any = this.bot.entity
    const target: any = findAssignedOpponent(this.bot, this.opponentUsername)
    if (!self?.position || !target?.position) {
      this.finishSwordDemo()
      return null
    }
    if (!await this.beforeTeacherExecution('teacher_sword')) return null

    const delta = target.position.minus(self.position)
    const horizontalDistance = Math.hypot(delta.x, delta.z)
    const distance = self.position.distanceTo(target.position)
    const shouldApproach = horizontalDistance > SWORD_BOOTCAMP_APPROACH_DISTANCE
    const previousYaw = Number(self.yaw ?? 0)
    const previousPitch = Number(self.pitch ?? 0)

    // Override only the locomotion controls that can pull a bootcamp fighter
    // away from its assigned opponent. There are no obstacles in this first
    // flat sword stage, so jumping/side-stepping only delays first contact.
    this.bot.setControlState('forward', shouldApproach)
    this.bot.setControlState('back', false)
    this.bot.setControlState('left', false)
    this.bot.setControlState('right', false)
    this.bot.setControlState('jump', false)
    this.bot.setControlState('sprint', shouldApproach)
    this.bot.setControlState('sneak', false)
    if (shouldApproach) this.record('bootcamp_approach', this.opponentUsername)

    const swordSlot = this.swordHotbarSlot()
    if (swordSlot >= 0 && swordSlot !== this.bot.quickBarSlot) {
      this.bot.setQuickBarSlot(swordSlot)
      this.record('bootcamp_sword', String(swordSlot))
    }
    if (this.activeHand !== 'none') this.releaseUse()

    // Aim at the upper torso rather than the feet. This makes the server's aim
    // cone and the worker's assigned-opponent melee predicate agree.
    if (horizontalDistance > 1e-6) {
      const eyeHeight = Number(self.eyeHeight ?? 1.62)
      const targetHeight = Math.max(1.0, Number(target.height ?? 1.8) * 0.75)
      const yaw = normalizeAngle(Math.atan2(-delta.x, -delta.z))
      const pitch = clamp(
        Math.atan2(delta.y + targetHeight - eyeHeight, horizontalDistance),
        -MAX_COMBAT_PITCH,
        MAX_COMBAT_PITCH
      )
      await this.bot.look(yaw, pitch, true)
      this.combatAimHoldUntilTick = tick + SWORD_BOOTCAMP_ATTACK_COOLDOWN_TICKS
      this.record('bootcamp_face', this.opponentUsername)
    }

    let attacked = false
    if (this.swordDemoControlTicks > 0
      && distance <= SWORD_BOOTCAMP_REACH
      && tick - this.lastAttackTick >= SWORD_BOOTCAMP_ATTACK_COOLDOWN_TICKS) {
      // Re-resolve after the awaited look so an arena reset cannot turn a stale
      // entity reference into a cross-match click.
      const currentTarget = findAssignedOpponent(this.bot, this.opponentUsername)
      if (currentTarget === target
        && self.position.distanceTo(currentTarget.position) <= SWORD_BOOTCAMP_REACH) {
        this.bot.attack(currentTarget)
        this.lastAttackTick = tick
        this.lastCombinedAttackTick = tick
        attacked = true
        this.record('bootcamp_attack_entity', String(currentTarget.id))
      }
    }
    this.swordDemoControlTicks += 1
    const execution: StepExecution = {
      source: 'teacher_sword',
      action: {
        ...NOOP_ACTION,
        forward: shouldApproach ? 1 : 0,
        sprint: shouldApproach,
        yaw_delta: normalizeAngle(Number(self.yaw ?? previousYaw) - previousYaw),
        pitch_delta: Number(self.pitch ?? previousPitch) - previousPitch,
        primary: attacked ? 'attack' : 'none',
        hotbar: swordSlot
      }
    }
    if (this.swordDemoControlTicks >= MAX_TEACHER_CONTROL_TICKS || attacked) {
      this.finishSwordDemo()
    }
    return execution
  }

  /**
   * Periodically demonstrate the shortest legal crystal sequence in combined
   * mode. This is intentionally a low-duty-cycle rail: each agent has a
   * stable phase offset, and between demonstrations the policy (plus the
   * sparse melee rescue) owns every tick.
   *
   * Target authority is entirely local. A placement base must be a loaded,
   * reachable obsidian/bedrock block with the vanilla two-air-block
   * clearance. The attack phase accepts only a newly spawned End Crystal at
   * that exact base. Human players, the spectator camera, remote arenas and
   * pre-existing crystals can never become demonstration targets.
   */
  async applyCombinedCrystalDemonstration(tick: number): Promise<StepExecution | null> {
    this.tick = tick
    this.observeTeacherCompletion(tick)
    if (!this.teachersEnabled
      || (this.mode !== 'crystal' && this.mode !== 'combined' && this.mode !== 'terrain')
      || !this.opponentUsername) return null

    let state = this.combinedCrystalDemoState
    if (!state) {
      if (this.combinedCrystalDemoUsed || tick < this.combinedCrystalDemoNextTick) return null
      const crystalSlot = this.crystalHotbarSlot()
      const base = crystalSlot >= 0 ? this.findReachableCrystalBase() : null
      if (crystalSlot < 0 || !base) {
        this.combinedCrystalDemoNextTick = tick + COMBINED_CRYSTAL_DEMO_RETRY_TICKS
        this.record('combined_crystal_demo_unavailable', crystalSlot < 0 ? 'no_crystal' : 'no_base')
        return null
      }

      this.combinedCrystalDemoUsed = true
      state = {
        phase: 'place_pad',
        basePosition: base.position.clone(),
        existingCrystalIds: this.loadedCrystalIds(),
        deadlineTick: tick + Math.max(20, COMBINED_CRYSTAL_DEMO_ACQUIRE_TICKS + 3),
        fallbackTick: tick + 1,
        controlTicks: 0,
        previousHotbar: Number(this.bot.quickBarSlot ?? -1),
        slotRestored: false
      }
      this.combinedCrystalDemoState = state
      if (!await this.beginCrystalTeacherControl()) return null
      this.selectCrystalSlot(crystalSlot)
      if (this.activeHand !== 'none') this.releaseUse()
      const before = this.lookSnapshot()
      await this.aimAtWorldPoint(base.position.offset(0.5, 1, 0.5), MAX_CRYSTAL_PITCH)
      this.record('combined_crystal_demo_aim_pad', positionDetail(base.position))
      return this.takeCrystalTeacherTick(state, {
        ...NOOP_ACTION,
        ...this.lookDelta(before),
        hotbar: crystalSlot
      })
    }

    if (state.controlTicks >= MAX_TEACHER_CONTROL_TICKS) return null
    if (state.phase === 'place_pad') {
      const crystal = this.newCrystalAtDemoBase(state)
      if (crystal) {
        if (!await this.beginCrystalTeacherControl()) return null
        state.phase = 'attack_crystal'
        state.crystalId = crystalIdentity(crystal)
        state.fallbackTick = tick + 1
        const before = this.lookSnapshot()
        await this.aimAtWorldPoint(crystalAimPoint(crystal), MAX_CRYSTAL_PITCH)
        this.record('combined_crystal_demo_policy_place_observed', state.crystalId)
        this.record('combined_crystal_demo_aim_crystal', state.crystalId)
        return this.takeCrystalTeacherTick(state, {
          ...NOOP_ACTION,
          ...this.lookDelta(before)
        })
      }
      const base = this.validCrystalBaseAt(state.basePosition)
      const crystalSlot = this.crystalHotbarSlot()
      if (!base || crystalSlot < 0) {
        this.finishCombinedCrystalDemo(tick, 'combined_crystal_demo_place_invalid', true)
        return null
      }
      if (!await this.beginCrystalTeacherControl()) return null
      this.selectCrystalSlot(crystalSlot)
      if (!this.crosshairHitsTopOf(state.basePosition) || tick < state.fallbackTick) {
        const before = this.lookSnapshot()
        await this.aimAtWorldPoint(state.basePosition.offset(0.5, 1, 0.5), MAX_CRYSTAL_PITCH)
        this.record('combined_crystal_demo_reaim_pad', positionDetail(state.basePosition))
        return this.takeCrystalTeacherTick(state, {
          ...NOOP_ACTION,
          ...this.lookDelta(before),
          hotbar: crystalSlot
        })
      }
      try {
        const placement = (this.bot as any)._genericPlace(base, new Vec3(0, 1, 0), {
          forceLook: 'ignore', swingArm: 'right', delta: new Vec3(0.5, 1, 0.5)
        })
        void Promise.resolve(placement).catch(error => {
          this.record('combined_crystal_demo_place_rejected', String(error).slice(0, 160))
        })
      } catch (error) {
        this.record('combined_crystal_demo_place_rejected', String(error).slice(0, 160))
        this.finishCombinedCrystalDemo(tick, 'combined_crystal_demo_place_failed', true)
        return null
      }
      state.phase = 'acquire_crystal'
      this.record('combined_crystal_demo_place', positionDetail(state.basePosition))
      return this.takeCrystalTeacherTick(state, {
        ...NOOP_ACTION,
        primary: 'use_main',
        hotbar: crystalSlot
      })
    }

    if (state.phase === 'acquire_crystal') {
      const crystal = this.newCrystalAtDemoBase(state)
      // Entity creation can arrive several protocol ticks after the placement
      // packet.  Merely waiting is not a control override: leave the policy in
      // charge, do not consume the four useful teacher controls, and do not
      // emit a no-op imitation sample.
      if (!crystal) {
        this.record('combined_crystal_demo_wait_spawn')
        return null
      }
      if (!await this.beginCrystalTeacherControl()) return null
      state.phase = 'attack_crystal'
      state.crystalId = crystalIdentity(crystal)
      state.fallbackTick = tick + 1
      const before = this.lookSnapshot()
      await this.aimAtWorldPoint(crystalAimPoint(crystal), MAX_CRYSTAL_PITCH)
      this.record('combined_crystal_demo_aim_crystal', state.crystalId)
      return this.takeCrystalTeacherTick(state, {
        ...NOOP_ACTION,
        ...this.lookDelta(before)
      })
    }

    const crystal = state.crystalId ? this.loadedCrystalById(state.crystalId) : null
    if (!crystal || !this.crystalBelongsToDemo(crystal, state)) {
      this.finishCombinedCrystalDemo(tick, 'combined_crystal_demo_crystal_gone', false, state.crystalId)
      return null
    }
    if (legalArenaAttackTargetAtCrosshair(
      this.bot, this.opponentUsername, COMBINED_CRYSTAL_DEMO_ATTACK_RAY_REACH
    ) !== crystal || tick < state.fallbackTick) {
      if (!await this.beginCrystalTeacherControl()) return null
      const before = this.lookSnapshot()
      await this.aimAtWorldPoint(crystalAimPoint(crystal), MAX_CRYSTAL_PITCH)
      this.record('combined_crystal_demo_reaim_crystal', state.crystalId)
      return this.takeCrystalTeacherTick(state, {
        ...NOOP_ACTION,
        ...this.lookDelta(before)
      })
    }
    if (!await this.beginCrystalTeacherControl()) return null
    this.bot.attack(crystal)
    this.lastAttackTick = tick
    this.lastCombinedAttackTick = tick
    this.combatAimHoldUntilTick = -1
    const execution = this.takeCrystalTeacherTick(state, {
      ...NOOP_ACTION,
      primary: 'attack'
    })
    this.finishCombinedCrystalDemo(tick, 'combined_crystal_demo_attack', false, state.crystalId)
    return execution
  }

  /**
   * Sparse tactical terrain teacher. It first demonstrates one reachable
   * offensive obsidian foundation and leaves it available for crystal play.
   * Mining is only a fallback when no valid foundation exists. Completion is
   * observed passively and the per-episode rail never exceeds four controls.
   */
  async applyCombinedTacticalBlockDemonstration(tick: number): Promise<StepExecution | null> {
    this.tick = tick
    this.observeTeacherCompletion(tick)
    if (!this.teachersEnabled || !this.terrainEnabled
      || (this.mode !== 'combined' && this.mode !== 'terrain')
      || !this.opponentUsername) return null

    let state = this.combinedTacticalBlockDemoState
    if (!state) {
      if (this.combinedTacticalBlockDemoUsed
        || tick < this.combinedTacticalBlockDemoNextTick) return null
      if (this.tacticalPlacements.hasPolicyMineReplacementSequence(tick)) {
        this.combinedTacticalBlockDemoUsed = true
        this.record('combined_tactical_block_demo_yield_policy_mine_replacement')
        return null
      }
      const placement = this.obsidianHotbarSlot() >= 0
        ? this.tacticalPlacements.resolve(this.opponentUsername, tick)
        : null
      // A policy-owned foundation before the demonstration is already the
      // desired lesson. Spend the one-demo guard now: every later foundation
      // in the bounded sequence must remain policy-owned.
      if (this.tacticalPlacements.progress().completed > 0) {
        this.combinedTacticalBlockDemoUsed = true
        this.record('combined_tactical_block_demo_policy_foundation_complete')
        return null
      }
      const mineTarget = !placement && this.pickaxeHotbarSlot() >= 0
        ? findTacticalMiningTarget(this.bot, this.opponentUsername)
        : null
      if (!mineTarget && !placement) {
        this.combinedTacticalBlockDemoNextTick = tick + COMBINED_TACTICAL_BLOCK_DEMO_RETRY_TICKS
        this.record('combined_tactical_block_demo_unavailable')
        return null
      }

      if (!await this.beforeTeacherExecution('teacher_block')) return null
      this.combinedTacticalBlockDemoUsed = true
      this.holdStillForCrystalDemo()
      const deadlineTick = tick + COMBINED_TACTICAL_BLOCK_DEMO_TIMEOUT_TICKS
      const previousHotbar = Number(this.bot.quickBarSlot ?? -1)
      const before = this.lookSnapshot()
      let teacherHotbar = -1
      if (placement) {
        const target = placement as TacticalWallPlacement
        teacherHotbar = this.obsidianHotbarSlot()
        this.selectTacticalSlot(teacherHotbar, 'obsidian')
        state = {
          phase: 'place_aim', targetPosition: target.targetPosition.clone(),
          deadlineTick, fallbackTick: tick + 1,
          stateOwnedCover: false, controlTicks: 0, previousHotbar, slotRestored: false
        }
        this.combinedTacticalBlockDemoState = state
        await this.aimAtWorldPoint(
          target.referenceBlock.position.offset(0.5, 1, 0.5), MAX_CRYSTAL_PITCH
        )
        this.record('combined_tactical_block_demo_aim_place', positionDetail(state.targetPosition))
      } else {
        teacherHotbar = this.pickaxeHotbarSlot()
        this.selectTacticalSlot(teacherHotbar, 'pickaxe')
        state = {
          phase: 'mine_aim', targetPosition: mineTarget.position.clone(),
          deadlineTick, fallbackTick: tick + 1,
          stateOwnedCover: false, controlTicks: 0, previousHotbar, slotRestored: false
        }
        this.combinedTacticalBlockDemoState = state
        await this.aimAtWorldPoint(mineTarget.position.offset(0.5, 0.5, 0.5), MAX_CRYSTAL_PITCH)
        this.record('combined_tactical_block_demo_aim_mine', positionDetail(state.targetPosition))
      }
      return this.takeTacticalTeacherTick(state, {
        ...NOOP_ACTION,
        ...this.lookDelta(before),
        hotbar: teacherHotbar
      })
    }

    const current: any = this.bot.blockAt(state.targetPosition, false)
    // A wait is passive: heartbeat observation owns completion/timeout. Do
    // not mark teacher ownership or silently cancel policy locomotion here.
    if (state.phase === 'mine_wait' || state.phase === 'place_wait') {
      return null
    }
    if (state.phase === 'place_aim'
      && String(current?.name ?? '').toLowerCase() === 'obsidian') {
      this.tacticalPlacements.resolve(this.opponentUsername, tick)
      this.finishCombinedTacticalBlockDemo(
        tick, 'combined_tactical_block_demo_policy_place_observed'
      )
      return null
    }
    if (state.controlTicks >= MAX_TEACHER_CONTROL_TICKS) return null
    if (!await this.beforeTeacherExecution('teacher_block')) return null
    this.holdStillForCrystalDemo()

    if (state.phase === 'mine_aim') {
      const safeMine = state.stateOwnedCover
        ? this.isExactStateOwnedCover(current, state)
        : isSafeTacticalMiningTarget(this.bot, current, this.opponentUsername)
      if (!safeMine) {
        this.finishCombinedTacticalBlockDemo(tick, 'combined_tactical_block_demo_mine_invalid', true)
        return null
      }
      const pickaxeSlot = this.pickaxeHotbarSlot()
      this.selectTacticalSlot(pickaxeSlot, 'pickaxe')
      if (!this.crosshairHitsBlock(state.targetPosition)) {
        const before = this.lookSnapshot()
        await this.aimAtWorldPoint(state.targetPosition.offset(0.5, 0.5, 0.5), MAX_CRYSTAL_PITCH)
        this.record('combined_tactical_block_demo_reaim_mine', positionDetail(state.targetPosition))
        return this.takeTacticalTeacherTick(state, {
          ...NOOP_ACTION,
          ...this.lookDelta(before),
          hotbar: pickaxeSlot
        })
      }
      if (tick < state.fallbackTick) {
        return this.takeTacticalTeacherTick(state, { ...NOOP_ACTION, hotbar: pickaxeSlot })
      }
      this.beginTacticalDig(current, 'combined_tactical_block_demo_fallback_mine')
      state.phase = 'mine_wait'
      return this.takeTacticalTeacherTick(state, {
        ...NOOP_ACTION,
        primary: 'attack',
        hotbar: pickaxeSlot
      })
    }

    const placement = tacticalCrystalPadPlacementAt(
      this.bot, this.opponentUsername, state.targetPosition
    )
    if (!placement || this.obsidianHotbarSlot() < 0) {
      this.finishCombinedTacticalBlockDemo(tick, 'combined_tactical_block_demo_place_invalid', true)
      return null
    }
    const obsidianSlot = this.obsidianHotbarSlot()
    this.selectTacticalSlot(obsidianSlot, 'obsidian')
    if (!this.crosshairHitsFace(placement.referenceBlock.position, placement.face)) {
      const before = this.lookSnapshot()
      await this.aimAtWorldPoint(
        placement.referenceBlock.position.offset(0.5, 1, 0.5), MAX_CRYSTAL_PITCH
      )
      this.record('combined_tactical_block_demo_reaim_place', positionDetail(state.targetPosition))
      return this.takeTacticalTeacherTick(state, {
        ...NOOP_ACTION,
        ...this.lookDelta(before),
        hotbar: obsidianSlot
      })
    }
    if (tick < state.fallbackTick) {
      return this.takeTacticalTeacherTick(state, { ...NOOP_ACTION, hotbar: obsidianSlot })
    }
    this.sendTacticalWallPlacement(placement, 'combined_tactical_block_demo_fallback_place')
    state.phase = 'place_wait'
    const execution = this.takeTacticalTeacherTick(state, {
      ...NOOP_ACTION,
      primary: 'use_main',
      hotbar: obsidianSlot
    })
    this.restoreTacticalDemoSlot(state)
    return execution
  }

  async apply(input: ActionV1, tick: number): Promise<PolicyExecution> {
    const action = validateAction(input)
    this.tick = tick
    const preCameraCrystalPlacement = this.isLegalPolicyCrystalPlacement(action)
    const preCameraCrystalAttack = this.isLegalPolicyCrystalAttack(action)
    const preCameraTacticalPlacement = this.isLegalPolicyTacticalPlacement(action)
    const preCameraPolicyMineReplacement = this.isLegalPolicyMineReplacement(action)
    const combatPriority = this.isLegalPolicyCombatAction(action)
      || preCameraTacticalPlacement
      || preCameraPolicyMineReplacement
    this.applyMovement(action)
    if (action.hotbar >= 0 && action.hotbar !== this.bot.quickBarSlot) {
      this.bot.setQuickBarSlot(action.hotbar)
      this.record('hotbar', String(action.hotbar))
    }
    if (action.release_use) this.releaseUse()
    if (action.swap_offhand && !this.lastAction.swap_offhand) this.swapOffhand()
    // The action mask describes the crosshair in the observation that selected
    // this action. Consume combat, crystal-placement and the current verified
    // tactical-foundation opportunity before
    // applying this tick's camera delta; otherwise a simultaneous exploratory
    // mouse move can invalidate a correctly selected click before it reaches
    // Minecraft.
    if (action.primary === 'attack'
      || preCameraCrystalPlacement
      || preCameraTacticalPlacement) {
      await this.applyPrimary(action)
    }
    await this.applyCamera(action)
    if (action.primary !== 'attack'
      && !preCameraCrystalPlacement
      && !preCameraTacticalPlacement) {
      await this.applyPrimary(action)
    }
    this.lastAction = action
    return {
      source: 'policy',
      action,
      combatPriority,
      crystalPriority: preCameraCrystalPlacement || preCameraCrystalAttack
    }
  }

  emergencyStop(): void {
    this.clearControlStatesSafely()
    this.releaseUse()
    if (this.digging && typeof (this.bot as any).stopDigging === 'function') {
      this.bot.stopDigging()
    }
    this.cancelActivePolicyMineReplacement()
    this.digging = false
    this.combatAimHoldUntilTick = -1
    this.lastCombinedAttackTick = this.tick
    this.finishSwordDemo()
    this.restoreCrystalDemoSlot(this.combinedCrystalDemoState)
    this.combinedCrystalDemoState = null
    this.combinedCrystalDemoNextTick = this.tick + this.combinedCrystalDemoDelayTicks
    this.restoreTacticalDemoSlot(this.combinedTacticalBlockDemoState)
    this.combinedTacticalBlockDemoState = null
    this.combinedTacticalBlockDemoNextTick = this.tick + this.combinedTacticalBlockDemoDelayTicks
    this.miningProgress = 0
    this.lastAction = { ...NOOP_ACTION }
    this.record('emergency_stop')
  }

  private applyMovement(action: ActionV1): void {
    this.bot.setControlState('forward', action.forward === 1)
    this.bot.setControlState('back', action.forward === -1)
    this.bot.setControlState('left', action.strafe === -1)
    this.bot.setControlState('right', action.strafe === 1)
    this.bot.setControlState('jump', action.jump)
    this.bot.setControlState('sprint', action.sprint)
    this.bot.setControlState('sneak', action.sneak)
  }

  private async applyCamera(action: ActionV1): Promise<void> {
    if (action.yaw_delta === 0 && action.pitch_delta === 0) return
    if (this.tick < this.combatAimHoldUntilTick) {
      const target = assignedOpponentMeleeTarget(this.bot, this.opponentUsername, 3.4)
      if (target) {
        this.record('hold_combat_aim')
        return
      }
    }
    const yaw = normalizeAngle((this.bot.entity?.yaw ?? 0) + action.yaw_delta)
    const pitch = clamp(
      (this.bot.entity?.pitch ?? 0) + action.pitch_delta,
      -MAX_COMBAT_PITCH,
      MAX_COMBAT_PITCH
    )
    await this.bot.look(yaw, pitch, true)
    this.record('look')
  }

  private async applyPrimary(action: ActionV1): Promise<void> {
    if (action.primary !== 'attack') {
      if (this.digging) {
        this.bot.stopDigging()
        this.cancelActivePolicyMineReplacement()
        this.digging = false
        this.record('stop_digging')
      }
      this.miningProgress = 0
    }
    if (action.primary === 'attack') await this.attackOrMine()
    else if (action.primary === 'use_main' && this.lastAction.primary !== 'use_main') await this.useHand(false)
    else if (action.primary === 'use_offhand' && this.lastAction.primary !== 'use_offhand') await this.useHand(true)
  }

  private async attackOrMine(): Promise<void> {
    // Only send a player attack when it matches the same centered, assigned-
    // opponent predicate advertised by combat_attack_ready. This preserves
    // exact identity isolation and prevents stale/edge rays from becoming
    // server-scored misses. Crystals use the broader arena ray without ever
    // permitting a spectator or unrelated player.
    const assigned = assignedOpponentMeleeTarget(
      this.bot, this.opponentUsername, 3.4
    )
    const arenaTarget = legalArenaAttackTargetAtCrosshair(this.bot, this.opponentUsername, 3.4)
    const entity = assigned ?? (arenaTarget?.type !== 'player' ? arenaTarget : null)
    if (entity) {
      this.bot.attack(entity)
      this.lastAttackTick = this.tick
      this.lastCombinedAttackTick = this.tick
      this.combatAimHoldUntilTick = assigned ? this.tick + 12 : -1
      this.record('attack_entity', String(entity.id))
      return
    }
    const block = this.crosshairBlock()?.block
    if (!isSafeTacticalMiningTarget(this.bot, block, this.opponentUsername)) return
    const mineReplacement = this.terrainEnabled && this.mode === 'terrain'
      ? this.tacticalPlacements.reservePolicyMineReplacement(
        block, this.opponentUsername, this.tick
      )
      : null
    this.beginTacticalDig(block, 'start_digging', mineReplacement)
  }

  private async useHand(offhand: boolean): Promise<void> {
    const item = offhand ? this.bot.inventory.slots[45] : this.bot.heldItem
    const name = String(item?.name ?? '')
    const target = this.crosshairBlock()
    if (!offhand && target && name === 'obsidian') {
      const placement = this.terrainEnabled && this.mode === 'terrain'
        ? this.tacticalPlacements.resolve(this.opponentUsername, this.tick)
        : null
      if (!placement?.referenceBlock?.position
        || target.face.y !== 1
        || !sameBlockPosition(target.block?.position, placement.referenceBlock.position)) {
        // The observation/action mask exposes exactly one bounded foundation
        // support. Refuse raw unmarked clicks so policy noise cannot make a
        // vertical stack, carpet the floor, or send avoidable place packets.
        this.record('reject_untargeted_obsidian_place')
        return
      }
      try {
        await (this.bot as any)._placeBlockWithOptions(target.block, target.face, {
          forceLook: 'ignore', swingArm: 'right', delta: target.cursor
        })
        this.record('place_from_crosshair', name)
      } catch (error) {
        this.record('place_rejected', String(error).slice(0, 160))
      }
      return
    }
    if (!offhand && target && name.includes('crystal')) {
      try {
        await (this.bot as any)._genericPlace(target.block, target.face, {
          forceLook: 'ignore', swingArm: 'right', delta: target.cursor
        })
        this.record('activate_block_from_crosshair', name)
      } catch (error) {
        this.record('activate_rejected', String(error).slice(0, 160))
      }
      return
    }
    this.bot.activateItem(offhand)
    this.activeHand = offhand ? 'off' : 'main'
    this.useStartedTick = this.tick
    this.record(offhand ? 'use_offhand' : 'use_main', name)
  }

  private crosshairBlock(): { block: any; face: Vec3; cursor: Vec3 } | null {
    const entity = this.bot.entity
    if (!entity) return null
    const yaw = entity.yaw ?? 0
    const pitch = entity.pitch ?? 0
    const cosPitch = Math.cos(pitch)
    const direction = new Vec3(-Math.sin(yaw) * cosPitch, Math.sin(pitch), -Math.cos(yaw) * cosPitch)
    const eye = entity.position.offset(0, (entity as any).eyeHeight ?? 1.62, 0)
    const hit = (this.bot.world as any).raycast(eye, direction, 5.0)
    if (!hit?.position) return null
    const block = this.bot.blockAt(hit.position, false)
    if (!block) return null
    const faces = [
      new Vec3(0, -1, 0), new Vec3(0, 1, 0), new Vec3(0, 0, -1),
      new Vec3(0, 0, 1), new Vec3(-1, 0, 0), new Vec3(1, 0, 0)
    ]
    const face = faces[Number(hit.face)] ?? new Vec3(0, 1, 0)
    const intersect = hit.intersect ?? hit.position.offset(0.5, 0.5, 0.5)
    const cursor = new Vec3(
      clamp(intersect.x - hit.position.x, 0, 1),
      clamp(intersect.y - hit.position.y, 0, 1),
      clamp(intersect.z - hit.position.z, 0, 1)
    )
    return { block, face, cursor }
  }

  private async applyCrystalDemoPolicyOpportunity(action: ActionV1): Promise<void> {
    const state = this.combinedCrystalDemoState
    if (!state) return
    this.holdStillForCrystalDemo()

    if (state.phase === 'place_pad') {
      if (action.hotbar >= 0) {
        const requestedName = String(
          this.bot.inventory?.slots?.[36 + action.hotbar]?.name ?? ''
        ).toLowerCase()
        if (requestedName.includes('crystal')) this.selectCrystalSlot(action.hotbar)
      }
      const heldName = String(this.bot.heldItem?.name ?? '').toLowerCase()
      if (action.primary === 'use_main'
        && heldName.includes('crystal')
        && this.crosshairHitsTopOf(state.basePosition)) {
        await this.useHand(false)
        state.phase = 'acquire_crystal'
        this.record('combined_crystal_demo_policy_place')
        return
      }
      this.record('hold_crystal_demo_policy', 'await_place')
      return
    }

    const crystal = state.phase === 'attack_crystal' && state.crystalId
      ? this.loadedCrystalById(state.crystalId)
      : this.newCrystalAtDemoBase(state)
    if (crystal && this.crystalBelongsToDemo(crystal, state)
      && action.primary === 'attack'
      && legalArenaAttackTargetAtCrosshair(
        this.bot, this.opponentUsername, COMBINED_CRYSTAL_DEMO_ATTACK_RAY_REACH
      ) === crystal) {
      this.bot.attack(crystal)
      this.lastAttackTick = this.tick
      this.lastCombinedAttackTick = this.tick
      this.combatAimHoldUntilTick = -1
      this.finishCombinedCrystalDemo(
        this.tick, 'combined_crystal_demo_policy_attack', false, crystalIdentity(crystal)
      )
      return
    }
    if (state.phase === 'attack_crystal' && !crystal) {
      this.finishCombinedCrystalDemo(
        this.tick, 'combined_crystal_demo_crystal_gone', false, state.crystalId
      )
      return
    }
    this.record('hold_crystal_demo_policy', state.phase)
  }

  private async applyTacticalBlockDemoPolicyOpportunity(action: ActionV1): Promise<void> {
    const state = this.combinedTacticalBlockDemoState
    if (!state) return
    this.holdStillForCrystalDemo()
    if (state.phase === 'mine_aim') {
      if (action.hotbar >= 0 && this.hotbarItemName(action.hotbar).includes('pickaxe')) {
        this.selectTacticalSlot(action.hotbar, 'pickaxe')
      }
      const block: any = this.bot.blockAt(state.targetPosition, false)
      const safeMine = state.stateOwnedCover
        ? this.isExactStateOwnedCover(block, state)
        : isSafeTacticalMiningTarget(this.bot, block, this.opponentUsername)
      if (action.primary === 'attack'
        && safeMine
        && this.crosshairHitsBlock(state.targetPosition)) {
        this.beginTacticalDig(block, 'combined_tactical_block_demo_policy_mine')
        state.phase = 'mine_wait'
        return
      }
      this.record('hold_tactical_block_demo_policy', 'await_mine')
      return
    }
    if (state.phase === 'place_aim') {
      if (action.hotbar >= 0 && this.hotbarItemName(action.hotbar) === 'obsidian') {
        this.selectTacticalSlot(action.hotbar, 'obsidian')
      }
      const placement = tacticalCrystalPadPlacementAt(
        this.bot, this.opponentUsername, state.targetPosition
      )
      if (placement && action.primary === 'use_main'
        && String(this.bot.heldItem?.name ?? '').toLowerCase() === 'obsidian'
        && this.crosshairHitsFace(placement.referenceBlock.position, placement.face)) {
        this.sendTacticalWallPlacement(placement, 'combined_tactical_block_demo_policy_place')
        state.phase = 'place_wait'
        return
      }
      this.record('hold_tactical_block_demo_policy', 'await_place')
      return
    }
    this.record('hold_tactical_block_demo_policy', state.phase)
  }

  private isExactStateOwnedCover(block: any, state: CombinedTacticalBlockDemoState): boolean {
    return state.stateOwnedCover
      && Boolean(block?.diggable)
      && String(block?.name ?? '').toLowerCase() === 'obsidian'
      && sameBlockPosition(block?.position, state.targetPosition)
  }

  private beginTacticalDig(
    block: any,
    operation: string,
    mineReplacement: PolicyMineReplacementReservation | null = null
  ): void {
    if (this.digging || !block?.position) {
      if (mineReplacement) this.tacticalPlacements.cancelPolicyMineReplacement(mineReplacement)
      return
    }
    this.digging = true
    this.activePolicyMineReplacement = mineReplacement
    this.miningProgress = 0.01
    this.miningStartedTick = this.tick
    const expectedMilliseconds = Number(this.bot.digTime?.(block) ?? 50)
    this.miningExpectedTicks = Math.max(1, Math.ceil(expectedMilliseconds / 50))
    const state = this.combinedTacticalBlockDemoState
    if (state && sameBlockPosition(block.position, state.targetPosition)) {
      this.extendTacticalDigDeadline(state, block, this.tick)
      state.phase = 'mine_wait'
    }
    this.record(operation, positionDetail(block.position))
    void this.bot.dig(block, 'ignore').then(() => {
      this.digging = false
      this.miningProgress = 1
      if (mineReplacement && this.activePolicyMineReplacement === mineReplacement) {
        this.tacticalPlacements.confirmPolicyMineReplacement(mineReplacement)
        this.activePolicyMineReplacement = null
        this.record('policy_mine_replacement_complete', positionDetail(mineReplacement.position))
      }
    }).catch(error => {
      this.digging = false
      this.miningProgress = 0
      if (mineReplacement) {
        this.tacticalPlacements.cancelPolicyMineReplacement(mineReplacement)
        if (this.activePolicyMineReplacement === mineReplacement) {
          this.activePolicyMineReplacement = null
        }
      }
      this.record('tactical_dig_rejected', String(error).slice(0, 160))
    })
  }

  private cancelActivePolicyMineReplacement(): void {
    const reservation = this.activePolicyMineReplacement
    if (!reservation) return
    this.tacticalPlacements.cancelPolicyMineReplacement(reservation)
    this.activePolicyMineReplacement = null
  }

  private extendTacticalDigDeadline(
    state: CombinedTacticalBlockDemoState,
    block: any,
    tick: number
  ): void {
    const milliseconds = Number(this.bot.digTime?.(block) ?? 50)
    const expectedTicks = Number.isFinite(milliseconds)
      ? Math.max(1, Math.ceil(Math.max(0, milliseconds) / 50))
      : COMBINED_TACTICAL_MAX_DIG_WINDOW_TICKS - COMBINED_TACTICAL_DIG_MARGIN_TICKS
    const window = Math.min(
      COMBINED_TACTICAL_MAX_DIG_WINDOW_TICKS,
      expectedTicks + COMBINED_TACTICAL_DIG_MARGIN_TICKS
    )
    state.deadlineTick = Math.max(state.deadlineTick, tick + window)
    this.record('combined_tactical_block_demo_dig_deadline', String(state.deadlineTick))
  }

  private sendTacticalWallPlacement(placement: TacticalWallPlacement, operation: string): void {
    try {
      const result = (this.bot as any)._genericPlace(
        placement.referenceBlock, placement.face,
        { forceLook: 'ignore', swingArm: 'right', delta: placement.cursor }
      )
      void Promise.resolve(result).catch(error => {
        this.record('tactical_place_rejected', String(error).slice(0, 160))
      })
      this.record(operation, positionDetail(placement.targetPosition))
    } catch (error) {
      this.record('tactical_place_rejected', String(error).slice(0, 160))
    }
  }

  private finishCombinedTacticalBlockDemo(
    tick: number,
    operation: string,
    retry = false
  ): void {
    const state = this.combinedTacticalBlockDemoState
    if (this.digging && retry) this.bot.stopDigging()
    if (retry) {
      this.digging = false
      this.miningProgress = 0
    }
    this.restoreTacticalDemoSlot(state)
    this.combinedTacticalBlockDemoState = null
    this.combinedTacticalBlockDemoNextTick = tick + (retry
      ? COMBINED_TACTICAL_BLOCK_DEMO_RETRY_TICKS
      : COMBINED_TACTICAL_BLOCK_DEMO_PERIOD_TICKS)
    this.record(operation)
  }

  private pickaxeHotbarSlot(): number {
    return this.hotbarSlotNamed(name => name.includes('pickaxe'))
  }

  private obsidianHotbarSlot(): number {
    return this.hotbarSlotNamed(name => name === 'obsidian')
  }

  private hotbarSlotNamed(predicate: (name: string) => boolean): number {
    for (let slot = 0; slot < 9; slot++) {
      if (predicate(this.hotbarItemName(slot))
        && Number(this.bot.inventory?.slots?.[36 + slot]?.count ?? 0) > 0) return slot
    }
    return -1
  }

  private hotbarItemName(slot: number): string {
    return String(this.bot.inventory?.slots?.[36 + slot]?.name ?? '').toLowerCase()
  }

  private selectTacticalSlot(slot: number, kind: string): void {
    if (slot < 0 || slot === this.bot.quickBarSlot) return
    this.bot.setQuickBarSlot(slot)
    this.record('combined_tactical_block_demo_hotbar', `${kind}:${slot}`)
  }

  private findReachableCrystalBase(): any | null {
    const self: any = this.bot.entity
    const opponent: any = findAssignedOpponent(this.bot, this.opponentUsername)
    if (!self?.position || !opponent?.position) return null
    const originX = Math.floor(self.position.x)
    const originY = Math.floor(self.position.y)
    const originZ = Math.floor(self.position.z)
    const eye = self.position.offset(0, Number(self.eyeHeight ?? 1.62), 0)
    const scanRadius = Math.max(1, Math.min(COMBINED_CRYSTAL_BASE_SCAN_RADIUS, this.matchRadius))
    let best: any | null = null
    let bestScore = Number.POSITIVE_INFINITY
    for (let y = originY - 3; y <= originY; y++) {
      for (let x = originX - scanRadius; x <= originX + scanRadius; x++) {
        for (let z = originZ - scanRadius; z <= originZ + scanRadius; z++) {
          const block = this.validCrystalBaseAt(new Vec3(x, y, z))
          if (!block) continue
          const spawn = block.position.offset(0.5, 1, 0.5)
          const placementDistance = eye.distanceTo(spawn)
          const selfDistance = self.position.distanceTo(spawn)
          const detonationAim = spawn.offset(0, 1, 0)
          if (placementDistance > COMBINED_CRYSTAL_DEMO_PLACE_REACH
            || eye.distanceTo(detonationAim) > COMBINED_CRYSTAL_DEMO_ATTACK_RAY_REACH) continue
          const opponentDistance = opponent.position.distanceTo(spawn)
          // Prefer an offensive crystal near the assigned fighter, while a
          // sharp close-range penalty avoids teaching point-blank self-damage.
          const closePenalty = selfDistance < 2
            ? 3 + (2 - selfDistance) * 4
            : 0
          const score = opponentDistance * 2 + placementDistance * 0.1 + closePenalty
          if (score < bestScore) {
            best = block
            bestScore = score
          }
        }
      }
    }
    return best
  }

  private validCrystalBaseAt(position: Vec3): any | null {
    const base: any = this.bot.blockAt(position, false)
    const baseName = String(base?.name ?? '').toLowerCase()
    if (baseName !== 'obsidian' && baseName !== 'bedrock') return null
    const first = this.bot.blockAt(position.offset(0, 1, 0), false)
    const second = this.bot.blockAt(position.offset(0, 2, 0), false)
    if (!isAirBlock(first) || !isAirBlock(second)) return null
    const centerX = position.x + 0.5
    const centerZ = position.z + 0.5
    const occupied = Object.values(this.bot.entities ?? {}).some((entity: any) => {
      const value = entity?.position
      if (!value) return false
      return Math.abs(value.x - centerX) < 0.9
        && Math.abs(value.z - centerZ) < 0.9
        && value.y >= position.y + 0.5
        && value.y < position.y + 3
    })
    return occupied ? null : base
  }

  private crystalHotbarSlot(): number {
    const slots = this.bot.inventory?.slots ?? []
    for (let hotbar = 0; hotbar < 9; hotbar++) {
      const name = String(slots[36 + hotbar]?.name ?? '').toLowerCase()
      if (name.includes('crystal') && Number(slots[36 + hotbar]?.count ?? 0) > 0) return hotbar
    }
    return -1
  }

  private selectCrystalSlot(slot: number): void {
    if (slot === this.bot.quickBarSlot) return
    this.bot.setQuickBarSlot(slot)
    this.record('combined_crystal_demo_hotbar', String(slot))
  }

  private holdStillForCrystalDemo(): void {
    this.bot.setControlState('forward', false)
    this.bot.setControlState('back', false)
    this.bot.setControlState('left', false)
    this.bot.setControlState('right', false)
    this.bot.setControlState('jump', false)
    this.bot.setControlState('sprint', false)
    this.bot.setControlState('sneak', false)
  }

  private async beginCrystalTeacherControl(): Promise<boolean> {
    if (!await this.beforeTeacherExecution('teacher_crystal')) return false
    this.holdStillForCrystalDemo()
    return true
  }

  private async aimAtWorldPoint(point: Vec3, maximumPitch = MAX_CRYSTAL_PITCH): Promise<void> {
    const self: any = this.bot.entity
    if (!self?.position) return
    const eye = self.position.offset(0, Number(self.eyeHeight ?? 1.62), 0)
    const delta = point.minus(eye)
    const horizontal = Math.hypot(delta.x, delta.z)
    const yaw = horizontal > 1e-9
      ? normalizeAngle(Math.atan2(-delta.x, -delta.z))
      : Number(self.yaw ?? 0)
    const pitch = clamp(Math.atan2(delta.y, horizontal), -maximumPitch, maximumPitch)
    await this.bot.look(yaw, pitch, true)
  }

  private crosshairHitsTopOf(position: Vec3): boolean {
    return this.crosshairHitsFace(position, new Vec3(0, 1, 0))
  }

  private crosshairHitsBlock(position: Vec3): boolean {
    const hit = this.crosshairBlock()
    return Boolean(hit?.block?.position && sameBlockPosition(hit.block.position, position))
  }

  private crosshairHitsFace(position: Vec3, face: Vec3): boolean {
    const hit = this.crosshairBlock()
    return Boolean(hit
      && sameBlockPosition(hit.block?.position, position)
      && hit.face.x === face.x && hit.face.y === face.y && hit.face.z === face.z)
  }

  private loadedCrystalIds(): Set<string> {
    return new Set(Object.values(this.bot.entities ?? {})
      .filter(isCrystalEntity)
      .map(crystalIdentity))
  }

  private loadedCrystalById(id: string): any | null {
    return Object.values(this.bot.entities ?? {}).find((entity: any) =>
      isCrystalEntity(entity) && crystalIdentity(entity) === id
    ) ?? null
  }

  private newCrystalAtDemoBase(state: CombinedCrystalDemoState): any | null {
    const self: any = this.bot.entity
    if (!self?.position) return null
    const candidates = Object.values(this.bot.entities ?? {}).filter((entity: any) =>
      !state.existingCrystalIds.has(crystalIdentity(entity))
      && this.crystalBelongsToDemo(entity, state)
    ) as any[]
    candidates.sort((first, second) =>
      self.position.distanceTo(first.position) - self.position.distanceTo(second.position)
    )
    return candidates[0] ?? null
  }

  private crystalBelongsToDemo(crystal: any, state: CombinedCrystalDemoState): boolean {
    const self: any = this.bot.entity
    if (!isCrystalEntity(crystal) || !self?.position || !crystal.position) return false
    if (state.existingCrystalIds.has(crystalIdentity(crystal))) return false
    const expectedX = state.basePosition.x + 0.5
    const expectedZ = state.basePosition.z + 0.5
    const atSelectedBase = Math.abs(crystal.position.x - expectedX) <= 0.9
      && Math.abs(crystal.position.z - expectedZ) <= 0.9
      && crystal.position.y >= state.basePosition.y + 0.5
      && crystal.position.y <= state.basePosition.y + 3
    const eye = self.position.offset(0, Number(self.eyeHeight ?? 1.62), 0)
    return atSelectedBase
      && eye.distanceTo(crystalAimPoint(crystal)) <= COMBINED_CRYSTAL_DEMO_ATTACK_RAY_REACH
  }

  private finishCombinedCrystalDemo(
    tick: number,
    operation: string,
    retry = false,
    detail?: string
  ): void {
    const state = this.combinedCrystalDemoState
    this.restoreCrystalDemoSlot(state)
    this.combinedCrystalDemoState = null
    this.combinedCrystalDemoNextTick = tick + (retry
      ? COMBINED_CRYSTAL_DEMO_RETRY_TICKS
      : COMBINED_CRYSTAL_DEMO_PERIOD_TICKS)
    this.record(operation, detail)
  }

  private swordHotbarSlot(): number {
    const slots = this.bot.inventory?.slots ?? []
    for (let hotbar = 0; hotbar < 9; hotbar++) {
      const name = String(slots[36 + hotbar]?.name ?? '').toLowerCase()
      if (name.endsWith('_sword') || name === 'sword') return hotbar
    }
    return -1
  }

  private isLegalPolicyCombatAction(action: ActionV1): boolean {
    if (action.primary === 'attack'
      && this.tick - this.lastAttackTick >= SWORD_BOOTCAMP_ATTACK_COOLDOWN_TICKS) {
      if (assignedOpponentMeleeTarget(
        this.bot, this.opponentUsername, SWORD_BOOTCAMP_REACH
      )) return true
      const target = legalArenaAttackTargetAtCrosshair(
        this.bot, this.opponentUsername, SWORD_BOOTCAMP_REACH
      )
      if (target && target.type !== 'player' && isCrystalEntity(target)) return true
    }
    if (this.isLegalPolicyCrystalPlacement(action)) return true
    if (action.primary !== 'use_offhand') return false
    const itemName = action.primary === 'use_offhand'
      ? String(this.bot.inventory?.slots?.[45]?.name ?? '').toLowerCase()
      : ''
    if (!itemName.includes('crystal')) return false
    const hit = this.crosshairBlock()
    return Boolean(hit?.block?.position
      && hit.face.y === 1
      && this.validCrystalBaseAt(hit.block.position))
  }

  private isLegalPolicyCrystalAttack(action: ActionV1): boolean {
    if (action.primary !== 'attack'
      || this.tick - this.lastAttackTick < SWORD_BOOTCAMP_ATTACK_COOLDOWN_TICKS) return false
    const target = legalArenaAttackTargetAtCrosshair(
      this.bot, this.opponentUsername, SWORD_BOOTCAMP_REACH
    )
    return Boolean(target && target.type !== 'player' && isCrystalEntity(target))
  }

  private isLegalPolicyCrystalPlacement(action: ActionV1): boolean {
    if (action.primary !== 'use_main' || this.lastAction.primary === 'use_main') return false
    const itemName = action.hotbar >= 0
      ? this.hotbarItemName(action.hotbar)
      : String(this.bot.heldItem?.name ?? '').toLowerCase()
    if (!itemName.includes('crystal')) return false
    const hit = this.crosshairBlock()
    if (!hit?.block?.position || hit.face.y !== 1 || !this.validCrystalBaseAt(hit.block.position)) {
      return false
    }
    const self: any = this.bot.entity
    if (!self?.position) return false
    const eye = self.position.offset(0, Number(self.eyeHeight ?? 1.62), 0)
    const topCenter = hit.block.position.offset(0.5, 1, 0.5)
    const crystalAim = topCenter.offset(0, 1, 0)
    return eye.distanceTo(topCenter) <= COMBINED_CRYSTAL_DEMO_PLACE_REACH
      && eye.distanceTo(crystalAim) <= COMBINED_CRYSTAL_DEMO_ATTACK_RAY_REACH
  }

  private isLegalPolicyTacticalPlacement(action: ActionV1): boolean {
    if (!this.terrainEnabled || this.mode !== 'terrain'
      || action.primary !== 'use_main'
      || this.lastAction.primary === 'use_main') return false
    const itemName = action.hotbar >= 0
      ? this.hotbarItemName(action.hotbar)
      : String(this.bot.heldItem?.name ?? '').toLowerCase()
    if (itemName !== 'obsidian') return false
    const placement = this.tacticalPlacements.resolve(this.opponentUsername, this.tick)
    if (!placement?.referenceBlock?.position) return false
    const hit = this.crosshairBlock()
    return Boolean(hit
      && hit.face.y === 1
      && sameBlockPosition(hit.block?.position, placement.referenceBlock.position))
  }

  private isLegalPolicyMineReplacement(action: ActionV1): boolean {
    if (!this.terrainEnabled || this.mode !== 'terrain'
      || action.primary !== 'attack') return false
    const assigned = assignedOpponentMeleeTarget(
      this.bot, this.opponentUsername, SWORD_BOOTCAMP_REACH
    )
    const arenaTarget = legalArenaAttackTargetAtCrosshair(
      this.bot, this.opponentUsername, SWORD_BOOTCAMP_REACH
    )
    if (assigned || arenaTarget) return false
    return this.tacticalPlacements.isPolicyMineReplacementPriorityCandidate(
      this.crosshairBlock()?.block,
      this.opponentUsername,
      this.tick
    )
  }

  private clearControlStatesSafely(): void {
    const bot: any = this.bot
    if (typeof bot.clearControlStates === 'function') {
      bot.clearControlStates()
      return
    }
    if (typeof bot.setControlState !== 'function') return
    for (const control of ['forward', 'back', 'left', 'right', 'jump', 'sprint', 'sneak']) {
      bot.setControlState(control, false)
    }
    this.record('clear_control_states_fallback')
  }

  private takeCrystalTeacherTick(
    state: CombinedCrystalDemoState,
    action: ActionV1
  ): StepExecution {
    state.controlTicks += 1
    if (state.controlTicks >= MAX_TEACHER_CONTROL_TICKS) {
      this.restoreCrystalDemoSlot(state)
    }
    return { source: 'teacher_crystal', action }
  }

  private takeTacticalTeacherTick(
    state: CombinedTacticalBlockDemoState,
    action: ActionV1
  ): StepExecution {
    state.controlTicks += 1
    if (state.controlTicks >= MAX_TEACHER_CONTROL_TICKS) {
      this.restoreTacticalDemoSlot(state)
    }
    return { source: 'teacher_block', action }
  }

  private finishSwordDemo(): void {
    if (!this.swordDemoActive) return
    this.restoreSwordDemoSlot()
    this.swordDemoActive = false
    this.recenterRequested = this.swordDemoControlTicks > 0
  }

  private restoreSwordDemoSlot(): void {
    const slot = this.swordDemoPreviousHotbar
    if (slot >= 0 && slot <= 8 && slot !== this.bot.quickBarSlot) {
      this.bot.setQuickBarSlot(slot)
      this.record('bootcamp_restore_hotbar', String(slot))
    }
    this.swordDemoPreviousHotbar = -1
  }

  private restoreCrystalDemoSlot(state: CombinedCrystalDemoState | null): void {
    if (!state || state.slotRestored) return
    if (state.previousHotbar >= 0 && state.previousHotbar <= 8
      && state.previousHotbar !== this.bot.quickBarSlot) {
      this.bot.setQuickBarSlot(state.previousHotbar)
      this.record('combined_crystal_demo_restore_hotbar', String(state.previousHotbar))
    }
    state.slotRestored = true
    this.recenterRequested = state.controlTicks > 0
  }

  private restoreTacticalDemoSlot(state: CombinedTacticalBlockDemoState | null): void {
    if (!state || state.slotRestored) return
    if (state.previousHotbar >= 0 && state.previousHotbar <= 8
      && state.previousHotbar !== this.bot.quickBarSlot) {
      this.bot.setQuickBarSlot(state.previousHotbar)
      this.record('combined_tactical_block_demo_restore_hotbar', String(state.previousHotbar))
    }
    state.slotRestored = true
    this.recenterRequested = state.controlTicks > 0
  }

  private lookSnapshot(): { yaw: number; pitch: number } {
    return {
      yaw: Number(this.bot.entity?.yaw ?? 0),
      pitch: Number(this.bot.entity?.pitch ?? 0)
    }
  }

  private lookDelta(before: { yaw: number; pitch: number }): Pick<ActionV1, 'yaw_delta' | 'pitch_delta'> {
    return {
      yaw_delta: normalizeAngle(Number(this.bot.entity?.yaw ?? before.yaw) - before.yaw),
      pitch_delta: Number(this.bot.entity?.pitch ?? before.pitch) - before.pitch
    }
  }

  private releaseUse(): void {
    if (typeof (this.bot as any).deactivateItem === 'function') this.bot.deactivateItem()
    this.activeHand = 'none'
    this.record('release_use')
  }

  /**
   * Minecraft 1.12 has no Mineflayer high-level swap-key method. This is the
   * one allowlisted raw write: status 6 is the vanilla SWAP_ITEM_WITH_OFFHAND
   * digging action generated by the F key. No coordinates or motion are sent.
   */
  private swapOffhand(): void {
    ;(this.bot as any)._client.write('block_dig', {
      status: 6,
      location: new Vec3(0, 0, 0),
      face: 0
    })
    this.record('swap_offhand_key')
  }

  private record(operation: string, detail?: string): void {
    this.audit.push({ tick: this.tick, operation, detail })
    if (this.audit.length > 2048) this.audit.splice(0, 512)
  }
}

function stableAgentHash(value: string): number {
  let hash = 0x811c9dc5
  for (const character of value) {
    hash ^= character.charCodeAt(0)
    hash = Math.imul(hash, 0x01000193)
  }
  return hash >>> 0
}

function isAirBlock(block: any): boolean {
  const name = String(block?.name ?? '').toLowerCase()
  return name === 'air' || name === 'cave_air' || name === 'void_air'
}

function isCrystalEntity(entity: any): boolean {
  if (!entity || entity.type === 'player' || !entity.position) return false
  const kind = String(entity.name ?? entity.displayName ?? entity.type ?? '').toLowerCase()
  return kind.includes('crystal')
}

function crystalIdentity(entity: any): string {
  return String(entity?.id ?? entity?.uuid ?? '')
}

function crystalAimPoint(entity: any): Vec3 {
  const height = Math.max(1, Number(entity?.height ?? 2))
  return entity.position.offset(0, height * 0.5, 0)
}

function sameBlockPosition(first: any, second: Vec3): boolean {
  return Boolean(first)
    && first.x === second.x
    && first.y === second.y
    && first.z === second.z
}

function positionDetail(position: Vec3): string {
  return `${position.x},${position.y},${position.z}`
}
