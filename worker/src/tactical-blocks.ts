import type { Bot } from 'mineflayer'
import { Vec3 } from 'vec3'
import { findAssignedOpponent } from './targeting.js'

export const TACTICAL_BLOCK_REACH = 5.0
export const TACTICAL_LOCAL_PAIR_RANGE = 14.0
const TACTICAL_SCAN_RADIUS = 5
const MINIMUM_HORIZONTAL_AIM_DISTANCE = 1.25
const MINIMUM_BUILDER_PLACEMENT_DISTANCE = 1.35
const MINIMUM_OPPONENT_PLACEMENT_DISTANCE = 1.5
const MAXIMUM_OPPONENT_PLACEMENT_DISTANCE = 4.5
export const ARENA_FOUNDATION_TARGET_Y = 64
const TACTICAL_CRYSTAL_CHAIN_REACH = 3.4
const IDEAL_SELF_PAD_DISTANCE = 2.65
const IDEAL_OPPONENT_PAD_DISTANCE = 2.35
/** Face-adjacent pads form a carpet; diagonal or wider separation stays tactical. */
export const MINIMUM_TACTICAL_FOUNDATION_SPACING = 1.35
export const MINIMUM_TACTICAL_PLACEMENT_QUOTA = 2
export const MAXIMUM_TACTICAL_PLACEMENT_QUOTA = 3
/** Worker/control ticks run at 20 Hz; expire before the server's 120-tick credit window. */
export const POLICY_MINE_REPLACEMENT_TIMEOUT_TICKS = 100

export type TacticalWallPlacement = {
  targetPosition: Vec3
  referenceBlock: any
  face: Vec3
  cursor: Vec3
}

/** Opaque episode-bound proof that a legal policy action started this mine. */
export type PolicyMineReplacementReservation = Readonly<{
  episodeId: string
  reservationId: number
  position: Vec3
}>

type ActiveTargetKind = 'ordinary' | 'mine_replacement'

type PendingPolicyMineReplacement = {
  episodeId: string
  reservationId: number
  position: Vec3
  digCompleted: boolean
  expiresTick: number
}

type LocalPair = {
  self: any
  opponent: any
  feetY: number
  actualFeetY: number
  horizontalDistance: number
}

/**
 * A bounded sequence of stable, offensive obsidian-foundation opportunities.
 *
 * Both observations and legal controls share this tracker in BotAgent. The
 * selected cell therefore cannot jump between the observation that proposed
 * an action and the tick that executes it. Once a cell becomes obsidian it is
 * permanently retired for the episode and the tracker may expose the next
 * well-spaced cell. Stage 1/2 permit two foundations and stage 3+ permits
 * three. Only one target is visible at a time, so this cannot become a broad
 * place-anywhere prior or an unbounded carpet/stack loop.
 */
export class TacticalPlacementTracker {
  private episodeId = 'waiting'
  private targetPosition: Vec3 | null = null
  private activeTargetKind: ActiveTargetKind | null = null
  private completedPositions: Vec3[] = []
  private policyMineAttemptPositions: Vec3[] = []
  private ordinaryCompleted = 0
  private placementQuota = MINIMUM_TACTICAL_PLACEMENT_QUOTA
  private nextReservationId = 1
  private pendingPolicyMineReplacement: PendingPolicyMineReplacement | null = null
  private mineReplacementAdopted = false
  private mineReplacementDeadlineTick: number | null = null
  private episodeFoundationY: number | null = null
  private latestTick = 0

  constructor(private readonly bot: Bot) {}

  beginEpisode(episodeId: string, curriculumStage = 1): void {
    const normalized = episodeId.trim() || 'waiting'
    const nextQuota = tacticalPlacementQuotaForStage(curriculumStage)
    if (normalized === this.episodeId) {
      // Stage is normally fixed for an episode, but accepting a later control
      // update avoids permanently using a startup-wide curriculum value.
      this.placementQuota = Math.max(
        this.placementQuota,
        this.ordinaryCompleted,
        nextQuota
      )
      return
    }
    this.episodeId = normalized
    this.targetPosition = null
    this.activeTargetKind = null
    this.completedPositions = []
    this.policyMineAttemptPositions = []
    this.ordinaryCompleted = 0
    this.placementQuota = nextQuota
    this.pendingPolicyMineReplacement = null
    this.mineReplacementAdopted = false
    this.mineReplacementDeadlineTick = null
    this.episodeFoundationY = null
    this.latestTick = 0
  }

  resolve(opponentUsername: string | null, tick = this.latestTick): TacticalWallPlacement | null {
    this.advanceTick(tick)
    // Assignment packets can briefly lag an episode transition. Do not erase
    // a stable target during that gap; simply withhold it until the exact
    // assigned opponent is available again.
    if (!opponentUsername?.trim()) return null
    this.captureEpisodeFoundationY(opponentUsername)
    this.promoteCompletedPolicyMine(opponentUsername)
    if (this.targetPosition) {
      const current: any = this.bot.blockAt(this.targetPosition, false)
      if (String(current?.name ?? '').toLowerCase() === 'obsidian') {
        if (!containsPosition(this.completedPositions, this.targetPosition)) {
          this.completedPositions.push(this.targetPosition.clone())
        }
        if (this.activeTargetKind === 'ordinary') this.ordinaryCompleted += 1
        if (this.activeTargetKind === 'mine_replacement') {
          this.mineReplacementDeadlineTick = null
        }
        this.targetPosition = null
        this.activeTargetKind = null
        if (this.isCompleted()) return null
      }
      if (this.targetPosition) {
        const retained = tacticalCrystalPadPlacementAt(
          this.bot, opponentUsername, this.targetPosition, this.completedPositions
        )
        if (retained) return retained
        // A successfully policy-mined cell is an exact sequence target, not a
        // suggestion that may jump elsewhere as the fighters move. Keep it
        // reserved and simply withhold the marker until it is legal/reachable.
        if (this.activeTargetKind === 'mine_replacement') return null
        this.targetPosition = null
        this.activeTargetKind = null
      }
    }
    if (this.ordinaryCompleted >= this.placementQuota) return null
    const selected = findTacticalCrystalPadPlacement(
      this.bot,
      opponentUsername,
      [...this.completedPositions, ...this.policyMineAttemptPositions]
    )
    this.targetPosition = selected?.targetPosition.clone() ?? null
    this.activeTargetKind = selected ? 'ordinary' : null
    return selected
  }

  isCompleted(): boolean {
    return this.ordinaryCompleted >= this.placementQuota
      && this.activeTargetKind !== 'mine_replacement'
  }

  progress(): Readonly<{ completed: number; quota: number }> {
    return { completed: this.completedPositions.length, quota: this.placementQuota }
  }

  hasAdoptedPolicyMineReplacement(): boolean {
    return this.mineReplacementAdopted
  }

  hasPolicyMineReplacementSequence(tick = this.latestTick): boolean {
    this.advanceTick(tick)
    return Boolean(this.pendingPolicyMineReplacement
      || this.activeTargetKind === 'mine_replacement')
  }

  /**
   * Reserve one exact natural-stone cell for a policy-owned mine->replace
   * sequence. Merely starting or teaching a dig cannot adopt it: the opaque
   * token must later be confirmed by the policy dig promise.
   */
  reservePolicyMineReplacement(
    block: any,
    opponentUsername: string | null,
    tick = this.latestTick
  ): PolicyMineReplacementReservation | null {
    this.advanceTick(tick)
    if (this.mineReplacementAdopted || this.pendingPolicyMineReplacement) return null
    const pair = localPair(this.bot, opponentUsername)
    if (!pair) return null
    if (this.episodeFoundationY === null) {
      if (pair.actualFeetY !== ARENA_FOUNDATION_TARGET_Y) return null
      this.episodeFoundationY = pair.actualFeetY
    }
    if (this.episodeFoundationY !== ARENA_FOUNDATION_TARGET_Y
      || pair.actualFeetY !== this.episodeFoundationY) return null
    if (!isUsefulPolicyMineReplacementCandidate(
      this.bot,
      block,
      opponentUsername,
      [...this.completedPositions, ...this.policyMineAttemptPositions],
      this.episodeFoundationY
    )) return null
    const reservation: PendingPolicyMineReplacement = {
      episodeId: this.episodeId,
      reservationId: this.nextReservationId++,
      position: block.position.clone(),
      digCompleted: false,
      expiresTick: this.latestTick + POLICY_MINE_REPLACEMENT_TIMEOUT_TICKS
    }
    this.pendingPolicyMineReplacement = reservation
    this.policyMineAttemptPositions.push(reservation.position.clone())
    this.mineReplacementDeadlineTick = reservation.expiresTick
    return {
      episodeId: reservation.episodeId,
      reservationId: reservation.reservationId,
      position: reservation.position.clone()
    }
  }

  confirmPolicyMineReplacement(reservation: PolicyMineReplacementReservation): void {
    const pending = this.matchingReservation(reservation)
    if (pending) pending.digCompleted = true
  }

  cancelPolicyMineReplacement(reservation: PolicyMineReplacementReservation): void {
    if (this.matchingReservation(reservation)) {
      this.pendingPolicyMineReplacement = null
      this.mineReplacementDeadlineTick = null
    }
  }

  isPolicyMineReplacementPriorityCandidate(
    block: any,
    opponentUsername: string | null,
    tick = this.latestTick
  ): boolean {
    this.advanceTick(tick)
    const pending = this.pendingPolicyMineReplacement
    if (pending && samePosition(block?.position, pending.position)) return true
    if (this.mineReplacementAdopted || pending) return false
    const pair = localPair(this.bot, opponentUsername)
    if (!pair) return false
    const expectedY = this.episodeFoundationY ?? pair.actualFeetY
    return expectedY === ARENA_FOUNDATION_TARGET_Y
      && pair.actualFeetY === expectedY
      && isUsefulPolicyMineReplacementCandidate(
        this.bot,
        block,
        opponentUsername,
        [...this.completedPositions, ...this.policyMineAttemptPositions],
        expectedY
      )
  }

  private matchingReservation(
    reservation: PolicyMineReplacementReservation
  ): PendingPolicyMineReplacement | null {
    const pending = this.pendingPolicyMineReplacement
    if (!pending || pending.episodeId !== this.episodeId
      || reservation.episodeId !== pending.episodeId
      || reservation.reservationId !== pending.reservationId
      || !samePosition(reservation.position, pending.position)) return null
    return pending
  }

  private promoteCompletedPolicyMine(opponentUsername: string): void {
    const pending = this.pendingPolicyMineReplacement
    if (!pending?.digCompleted || pending.episodeId !== this.episodeId) return
    if (!isAir(this.bot.blockAt(pending.position, false))) return
    // Adoption, rather than reservation, spends the one-extra-sequence cap.
    // Preempt an ordinary marker so the just-mined cell is always next.
    this.mineReplacementAdopted = true
    this.pendingPolicyMineReplacement = null
    this.targetPosition = pending.position.clone()
    this.activeTargetKind = 'mine_replacement'
    // Do not clear an invalid exact cell or fall back to an ordinary pad. The
    // resolve path above will expose it only while support/clearance/reach and
    // current assigned-opponent geometry still make the placement legal.
    void opponentUsername
  }

  private captureEpisodeFoundationY(opponentUsername: string): void {
    if (this.episodeFoundationY !== null) return
    const pair = localPair(this.bot, opponentUsername)
    if (pair?.actualFeetY === ARENA_FOUNDATION_TARGET_Y) {
      this.episodeFoundationY = pair.actualFeetY
    }
  }

  private advanceTick(tick: number): void {
    if (Number.isFinite(tick)) this.latestTick = Math.max(this.latestTick, Math.floor(tick))
    if (this.mineReplacementDeadlineTick === null
      || this.latestTick <= this.mineReplacementDeadlineTick) return
    this.pendingPolicyMineReplacement = null
    this.mineReplacementDeadlineTick = null
    if (this.activeTargetKind === 'mine_replacement') {
      this.targetPosition = null
      this.activeTargetKind = null
    }
  }
}

/** Keep early terrain learnable while making later stages build one extra base. */
export function tacticalPlacementQuotaForStage(curriculumStage: number): number {
  const stage = Number.isFinite(curriculumStage)
    ? Math.max(1, Math.floor(curriculumStage))
    : 1
  return stage >= 3
    ? MAXIMUM_TACTICAL_PLACEMENT_QUOTA
    : MINIMUM_TACTICAL_PLACEMENT_QUOTA
}

/**
 * A safe mining target is generated arena cover, never arena structure.
 * Generated stone or player-placed obsidian cover must be above the fighters'
 * floor, reachable, exposed, and directly obstruct the local corridor to the
 * exact server-assigned opponent. The Y bound is what keeps floor-level
 * obsidian crystal pads and the arena's stone floor/depth permanently safe.
 */
export function isSafeTacticalMiningTarget(
  bot: Bot,
  block: any,
  opponentUsername: string | null
): boolean {
  const pair = localPair(bot, opponentUsername)
  if (!pair || !block?.position || !block.diggable) return false
  const name = String(block.name ?? '').toLowerCase()
  if (name !== 'stone' && name !== 'obsidian') return false
  const position: Vec3 = block.position
  if (position.y < pair.feetY || position.y > pair.feetY + 1) return false

  const center = position.offset(0.5, 0.5, 0.5)
  const eye = pair.self.position.offset(0, Number(pair.self.eyeHeight ?? 1.62), 0)
  if (eye.distanceTo(center) > TACTICAL_BLOCK_REACH) return false
  if (Math.hypot(center.x - pair.self.position.x, center.z - pair.self.position.z)
    < MINIMUM_HORIZONTAL_AIM_DISTANCE) return false

  const corridor = horizontalCorridorPosition(pair.self.position, pair.opponent.position, center)
  if (!corridor || corridor.fraction < 0.12 || corridor.fraction > 0.88
    || corridor.distance > 0.85) return false
  return exposedFaceCount(bot, position) > 0
}

/**
 * Stronger pre-mine check for the single autonomous mine->replace sequence.
 * The occupied cell itself must be natural stone at fighter-foot height and
 * must become a legal crystal foundation after removal.
 */
export function isUsefulPolicyMineReplacementCandidate(
  bot: Bot,
  block: any,
  opponentUsername: string | null,
  completedPositions: readonly Vec3[] = [],
  expectedTargetY = ARENA_FOUNDATION_TARGET_Y
): boolean {
  const pair = localPair(bot, opponentUsername)
  if (!pair || !block?.position
    || String(block.name ?? '').toLowerCase() !== 'stone'
    || block.position.y !== pair.feetY
    || pair.actualFeetY !== expectedTargetY
    || block.position.y !== expectedTargetY
    || !isSafeTacticalMiningTarget(bot, block, opponentUsername)) return false
  const position: Vec3 = block.position
  if (completedPositions.some(completed =>
    horizontalDistance(completed, position) < MINIMUM_TACTICAL_FOUNDATION_SPACING
  )) return false
  const support: any = bot.blockAt(position.offset(0, -1, 0), false)
  if (String(support?.name ?? '').toLowerCase() !== 'stone') return false
  if (!isAir(bot.blockAt(position.offset(0, 1, 0), false))
    || !isAir(bot.blockAt(position.offset(0, 2, 0), false))) return false
  const center = position.offset(0.5, 0.5, 0.5)
  if (horizontalDistance(center, pair.self.position) < MINIMUM_BUILDER_PLACEMENT_DISTANCE
    || horizontalDistance(center, pair.opponent.position) < MINIMUM_OPPONENT_PLACEMENT_DISTANCE
    || horizontalDistance(center, pair.opponent.position) > MAXIMUM_OPPONENT_PLACEMENT_DISTANCE
    || placementOccupied(bot, position)) return false
  const eye = pair.self.position.offset(0, Number(pair.self.eyeHeight ?? 1.62), 0)
  return eye.distanceTo(position.offset(0.5, 0, 0.5)) <= TACTICAL_BLOCK_REACH
    && eye.distanceTo(position.offset(0.5, 2, 0.5)) <= TACTICAL_CRYSTAL_CHAIN_REACH
}

export function findTacticalMiningTarget(
  bot: Bot,
  opponentUsername: string | null
): any | null {
  const pair = localPair(bot, opponentUsername)
  if (!pair) return null
  const originX = Math.floor(pair.self.position.x)
  const originZ = Math.floor(pair.self.position.z)
  let best: any | null = null
  let bestScore = Number.POSITIVE_INFINITY
  for (let y = pair.feetY; y <= pair.feetY + 1; y++) {
    for (let x = originX - TACTICAL_SCAN_RADIUS; x <= originX + TACTICAL_SCAN_RADIUS; x++) {
      for (let z = originZ - TACTICAL_SCAN_RADIUS; z <= originZ + TACTICAL_SCAN_RADIUS; z++) {
        const block: any = bot.blockAt(new Vec3(x, y, z), false)
        if (!isSafeTacticalMiningTarget(bot, block, opponentUsername)) continue
        const center = block.position.offset(0.5, 0.5, 0.5)
        const corridor = horizontalCorridorPosition(pair.self.position, pair.opponent.position, center)
        if (!corridor) continue
        // Open the first meaningful obstruction. It is normally visible and
        // creates a real route toward the opponent instead of mining scenery.
        const score = corridor.fraction * 4 + corridor.distance * 3
          + Math.abs(center.y - (pair.feetY + 0.9)) * 0.25
        if (score < bestScore) {
          best = block
          bestScore = score
        }
      }
    }
  }
  return best
}

/**
 * Choose an executable floor-supported obsidian foundation for an offensive
 * crystal chain. Candidate generation uses only the bot and its exact
 * server-assigned opponent. A spectator or unrelated arena player cannot
 * influence the selected coordinate.
 */
export function findTacticalCrystalPadPlacement(
  bot: Bot,
  opponentUsername: string | null,
  completedPositions: readonly Vec3[] = []
): TacticalWallPlacement | null {
  const pair = localPair(bot, opponentUsername)
  if (!pair) return null
  const originX = Math.floor(pair.self.position.x)
  const originZ = Math.floor(pair.self.position.z)
  let best: TacticalWallPlacement | null = null
  let bestScore = Number.POSITIVE_INFINITY
  for (let x = originX - TACTICAL_SCAN_RADIUS; x <= originX + TACTICAL_SCAN_RADIUS; x++) {
    for (let z = originZ - TACTICAL_SCAN_RADIUS; z <= originZ + TACTICAL_SCAN_RADIUS; z++) {
      const targetPosition = new Vec3(x, pair.feetY, z)
      const placement = tacticalCrystalPadPlacementAt(
        bot, opponentUsername, targetPosition, completedPositions
      )
      if (!placement) continue
      const center = targetPosition.offset(0.5, 0, 0.5)
      const selfDistance = horizontalDistance(center, pair.self.position)
      const opponentDistance = horizontalDistance(center, pair.opponent.position)
      // Near-opponent weight makes the foundation offensive. A gentler self
      // term retains enough spacing to avoid point-blank self-crystals.
      const score = Math.abs(opponentDistance - IDEAL_OPPONENT_PAD_DISTANCE) * 1.35
        + Math.abs(selfDistance - IDEAL_SELF_PAD_DISTANCE)
        + coordinateTieBreak(targetPosition)
      if (score < bestScore) {
        best = placement
        bestScore = score
      }
    }
  }
  return best
}

export function tacticalCrystalPadPlacementAt(
  bot: Bot,
  opponentUsername: string | null,
  targetPosition: Vec3,
  completedPositions: readonly Vec3[] = []
): TacticalWallPlacement | null {
  const pair = localPair(bot, opponentUsername)
  if (!pair || targetPosition.y !== pair.feetY) return null
  const target: any = bot.blockAt(targetPosition, false)
  if (!isAir(target)) return null
  const referenceBlock: any = bot.blockAt(targetPosition.offset(0, -1, 0), false)
  const referenceName = String(referenceBlock?.name ?? '').toLowerCase()
  // Existing obsidian is already a legal crystal base. Marking its top as a
  // build target would compete with base acquisition and teach vertical
  // obsidian stacking, so new foundations start only from the stone floor.
  if (!referenceBlock?.position || referenceName !== 'stone') return null
  // Never reuse a destroyed foundation or choose a face-adjacent cell beside
  // one already built this episode. This keeps the curriculum about multiple
  // offensive anchors instead of repeatedly farming one square or carpeting.
  if (completedPositions.some(position =>
    horizontalDistance(position, targetPosition) < MINIMUM_TACTICAL_FOUNDATION_SPACING
  )) return null
  const center = targetPosition.offset(0.5, 0.5, 0.5)
  const corridor = horizontalCorridorPosition(
    pair.self.position, pair.opponent.position, center
  )
  // Keep the foundation in the active fight lane so server-side useful-pad
  // attribution agrees with the worker marker. This is still wide enough for
  // side-step placement without selecting off-axis scenery.
  if (!corridor || corridor.fraction < 0.1 || corridor.fraction > 0.9
    || corridor.distance > 1.05) return null
  if (horizontalDistance(center, pair.self.position) < MINIMUM_BUILDER_PLACEMENT_DISTANCE
    || horizontalDistance(center, pair.opponent.position) < MINIMUM_OPPONENT_PLACEMENT_DISTANCE
    || horizontalDistance(center, pair.opponent.position) > MAXIMUM_OPPONENT_PLACEMENT_DISTANCE) {
    return null
  }
  // The placed block itself becomes the crystal base. Require the two vanilla
  // clearance cells now so the demonstrated action always leads to a usable
  // place->detonate chain rather than inert cover.
  if (!isAir(bot.blockAt(targetPosition.offset(0, 1, 0), false))
    || !isAir(bot.blockAt(targetPosition.offset(0, 2, 0), false))) return null
  if (placementOccupied(bot, targetPosition)) return null
  const eye = pair.self.position.offset(0, Number(pair.self.eyeHeight ?? 1.62), 0)
  const clickPoint = referenceBlock.position.offset(0.5, 1, 0.5)
  const futureCrystalAim = targetPosition.offset(0.5, 2, 0.5)
  if (eye.distanceTo(clickPoint) > TACTICAL_BLOCK_REACH
    || eye.distanceTo(futureCrystalAim) > TACTICAL_CRYSTAL_CHAIN_REACH) return null
  return {
    targetPosition: targetPosition.clone(),
    referenceBlock,
    face: new Vec3(0, 1, 0),
    cursor: new Vec3(0.5, 1, 0.5)
  }
}

/** Compatibility aliases for existing callers while the terminology migrates. */
export const findTacticalWallPlacement = findTacticalCrystalPadPlacement
export const tacticalWallPlacementAt = tacticalCrystalPadPlacementAt

export function isAir(block: any): boolean {
  if (!block) return false
  const name = String(block.name ?? '').toLowerCase()
  return name === 'air' || Number(block.type) === 0
}

function localPair(bot: Bot, opponentUsername: string | null): LocalPair | null {
  const self: any = bot.entity
  const opponent: any = findAssignedOpponent(bot, opponentUsername)
  if (!self?.position || !opponent?.position) return null
  const horizontal = horizontalDistance(self.position, opponent.position)
  if (!Number.isFinite(horizontal) || horizontal > TACTICAL_LOCAL_PAIR_RANGE) return null
  return {
    self,
    opponent,
    // MCAI arenas have a fixed y=63 floor and y=64 foundation/fighter plane.
    // Reach checks below naturally withhold targets while a fighter is above
    // or below it; target generation itself must never drift with jumping.
    feetY: ARENA_FOUNDATION_TARGET_Y,
    actualFeetY: Math.floor(Math.min(self.position.y, opponent.position.y)),
    horizontalDistance: horizontal
  }
}

function horizontalCorridorPosition(
  start: Vec3,
  end: Vec3,
  point: Vec3
): { fraction: number; distance: number } | null {
  const dx = end.x - start.x
  const dz = end.z - start.z
  const lengthSquared = dx * dx + dz * dz
  if (lengthSquared < 1e-6) return null
  const fraction = ((point.x - start.x) * dx + (point.z - start.z) * dz) / lengthSquared
  const projectedX = start.x + dx * fraction
  const projectedZ = start.z + dz * fraction
  return { fraction, distance: Math.hypot(point.x - projectedX, point.z - projectedZ) }
}

function exposedFaceCount(bot: Bot, position: Vec3): number {
  const faces = [
    new Vec3(1, 0, 0), new Vec3(-1, 0, 0), new Vec3(0, 1, 0),
    new Vec3(0, -1, 0), new Vec3(0, 0, 1), new Vec3(0, 0, -1)
  ]
  return faces.reduce((count, face) => count + (isAir(bot.blockAt(position.plus(face), false)) ? 1 : 0), 0)
}

function placementOccupied(bot: Bot, target: Vec3): boolean {
  // The spectator and unrelated fighters are never curriculum inputs. The
  // exact assigned opponent is already excluded geometrically above; retain
  // only self plus non-player entities that Minecraft placement can collide
  // with (crystals, items, projectiles, etc.).
  const entities = new Set<any>([
    bot.entity,
    ...Object.values(bot.entities ?? {}).filter((entity: any) => entity?.type !== 'player')
  ])
  for (const entity of entities) {
    if (!entity?.position) continue
    const width = Math.max(0.1, Number(entity.width ?? 0.6))
    const height = Math.max(0.1, Number(entity.height ?? 1.8))
    const half = width / 2
    const overlapsX = entity.position.x + half > target.x && entity.position.x - half < target.x + 1
    const overlapsZ = entity.position.z + half > target.z && entity.position.z - half < target.z + 1
    const overlapsY = entity.position.y + height > target.y && entity.position.y < target.y + 1
    if (overlapsX && overlapsY && overlapsZ) return true
  }
  return false
}

function horizontalDistance(first: Vec3, second: Vec3): number {
  return Math.hypot(first.x - second.x, first.z - second.z)
}

function coordinateTieBreak(position: Vec3): number {
  // A tiny deterministic term makes equal geometry stable across JS sort or
  // object iteration differences without materially changing tactical score.
  const hash = ((position.x * 73856093) ^ (position.z * 19349663)) >>> 0
  return (hash % 1000) * 1e-9
}

function samePosition(first: unknown, second: unknown): boolean {
  // Mineflayer can invalidate a crosshair block between observation and
  // execution (chunk refresh, explosion, or another fighter mining it). A
  // stale reservation must fail closed instead of dereferencing undefined and
  // killing every parallel rollout worker.
  if (!finitePosition(first) || !finitePosition(second)) return false
  return first.x === second.x && first.y === second.y && first.z === second.z
}

function finitePosition(value: unknown): value is Vec3 {
  if (!value || typeof value !== 'object') return false
  const candidate = value as { x?: unknown; y?: unknown; z?: unknown }
  return typeof candidate.x === 'number' && Number.isFinite(candidate.x)
    && typeof candidate.y === 'number' && Number.isFinite(candidate.y)
    && typeof candidate.z === 'number' && Number.isFinite(candidate.z)
}

function containsPosition(positions: readonly Vec3[], target: Vec3): boolean {
  return positions.some(position => samePosition(position, target))
}
