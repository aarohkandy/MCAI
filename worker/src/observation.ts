import type { Bot } from 'mineflayer'
import { Vec3 } from 'vec3'
import {
  MAX_ENTITY_SLOTS,
  SCHEMA_VERSION,
  type ActionMask,
  type EntitySlot,
  type ObservationV1,
  type OpponentState,
  type RaycastState,
  type Vec3Value
} from './contracts.js'
import { BlockSampler } from './block-sampler.js'
import { EMPTY_ITEM, itemState } from './items.js'
import {
  distance,
  egocentric,
  mineflayerBodyRelative,
  normalizeAngle,
  subtract,
  toVec3Value
} from './math.js'
import {
  assignedOpponentMeleeTarget,
  findAssignedOpponent,
  legalArenaAttackTargetAtCrosshair
} from './targeting.js'
import {
  isSafeTacticalMiningTarget,
  TACTICAL_BLOCK_REACH,
  TacticalPlacementTracker,
  type TacticalWallPlacement
} from './tactical-blocks.js'

// Keep the melee curriculum inside vanilla survival reach.  The generic
// arena ray remains wider for crystals and mining, but it must not advertise
// an edge-of-range sword swing as the high-priority combat action.
const RELIABLE_MELEE_REACH = 3.4
const CRYSTAL_PLACE_REACH = 3.3
const CRYSTAL_DETONATION_REACH = 3.4

export type MatchContext = ObservationV1['match']

export type ControlTelemetry = {
  lastAttackTick: number
  activeHand: 'none' | 'main' | 'off'
  useStartedTick: number
  miningProgress: number
}

type OpponentMotionEstimate = {
  episodeId: string
  entityId: number
  tick: number
  position: Vec3Value
  velocity: Vec3Value
}

type AuthoritativeOpponentState = {
  episodeId: string
  opponentUsername: string
  receivedTick: number
  serverElapsedTicks: number
  velocity: Vec3Value
  yaw: number
  pitch: number
  health: number
  absorption: number
  grounded: boolean
}

const SERVER_STATE_FRESH_TICKS = 6

export class ObservationBuilder {
  private readonly blocks: BlockSampler
  private entityBornTick = new Map<number, number>()
  private opponentUsername: string | null = null
  private opponentMotion: OpponentMotionEstimate | null = null
  private authoritativeOpponent: AuthoritativeOpponentState | null = null

  constructor(
    private readonly bot: Bot,
    private readonly tacticalPlacements = new TacticalPlacementTracker(bot),
    cadenceKey = ''
  ) {
    this.blocks = new BlockSampler(bot, 10, cadenceKey)
  }

  setOpponentUsername(username: string | null): void {
    const normalized = username?.trim() || null
    if (normalized?.toLowerCase() !== this.opponentUsername?.toLowerCase()) {
      this.opponentMotion = null
      this.authoritativeOpponent = null
    }
    this.opponentUsername = normalized
  }

  /** Accept only the active episode's exact server-assigned opponent fighter. */
  acceptArenaSnapshot(
    payload: Record<string, unknown>,
    activeEpisodeId: string,
    receivedTick: number
  ): boolean {
    if (!this.opponentUsername || String(payload.episode_id ?? '') !== activeEpisodeId) return false
    const fighters = Array.isArray(payload.fighters) ? payload.fighters : []
    const expected = this.opponentUsername.toLowerCase()
    const matches = fighters.filter((fighter): fighter is Record<string, unknown> =>
      Boolean(fighter) && typeof fighter === 'object'
        && String((fighter as Record<string, unknown>).name ?? '').toLowerCase() === expected
    )
    if (matches.length !== 1) return false
    const fighter = matches[0]
    const velocity = vectorFromFields(fighter, 'vx', 'vy', 'vz')
    const yawDegrees = finiteNumber(fighter.yaw)
    const pitchDegrees = finiteNumber(fighter.pitch)
    const health = finiteNumber(fighter.health)
    const absorption = finiteNumber(fighter.absorption)
    const serverElapsedTicks = finiteNumber(payload.elapsed_ticks)
    if (!velocity || yawDegrees === null || pitchDegrees === null || health === null
      || absorption === null || serverElapsedTicks === null || typeof fighter.grounded !== 'boolean') {
      return false
    }
    if (this.authoritativeOpponent
      && this.authoritativeOpponent.episodeId === activeEpisodeId
      && serverElapsedTicks < this.authoritativeOpponent.serverElapsedTicks) {
      return false
    }
    this.authoritativeOpponent = {
      episodeId: activeEpisodeId,
      opponentUsername: this.opponentUsername,
      receivedTick,
      serverElapsedTicks,
      velocity,
      yaw: notchDegreesToMineflayerYaw(yawDegrees),
      pitch: notchDegreesToMineflayerPitch(pitchDegrees),
      health: Math.max(0, health),
      absorption: Math.max(0, absorption),
      grounded: fighter.grounded
    }
    return true
  }

  build(match: MatchContext, control: ControlTelemetry): ObservationV1 {
    this.tacticalPlacements.beginEpisode(
      match.episode_id,
      match.curriculum_stage ?? 1
    )
    const selfPosition = toVec3Value(this.bot.entity?.position)
    const opponentEntity = findAssignedOpponent(this.bot, this.opponentUsername)
    if (!opponentEntity) this.opponentMotion = null
    const opponent = opponentEntity
      ? this.opponentState(opponentEntity, selfPosition, match.tick, match.episode_id)
      : null
    const opponentWorld = opponentEntity ? toVec3Value(opponentEntity.position) : undefined
    const tacticalPlacement = (match.mode ?? 'sword') === 'terrain'
      ? this.tacticalPlacements.resolve(this.opponentUsername, match.tick)
      : null
    return {
      schema_version: SCHEMA_VERSION,
      match,
      self: {
        health: finite(this.bot.health, 0),
        absorption: finite((this.bot as any).entity?.metadata?.absorption ?? (this.bot as any).absorption, 0),
        food: finite(this.bot.food, 0),
        position: selfPosition,
        velocity: toVec3Value(this.bot.entity?.velocity),
        yaw: finite(this.bot.entity?.yaw, 0),
        pitch: finite(this.bot.entity?.pitch, 0),
        on_ground: Boolean(this.bot.entity?.onGround),
        sprinting: Boolean(this.bot.getControlState('sprint')),
        sneaking: Boolean(this.bot.getControlState('sneak')),
        hurt_time: entityHurtTime(this.bot.entity),
        attack_cooldown: Math.min(1, Math.max(0, (match.tick - control.lastAttackTick) / 13)),
        active_hand: control.activeHand,
        use_ticks: control.activeHand === 'none' ? 0 : Math.max(0, match.tick - control.useStartedTick),
        mining_progress: Math.min(1, Math.max(0, control.miningProgress)),
        selected_hotbar: Math.min(8, Math.max(0, this.bot.quickBarSlot ?? 0)),
        mainhand: itemState(this.bot.heldItem),
        hotbar: Array.from({ length: 9 }, (_, index) => itemState(this.bot.inventory.slots[36 + index])),
        offhand: itemState(this.bot.inventory.slots[45]),
        armor: [5, 6, 7, 8].map(slot => itemState(this.bot.inventory.slots[slot])),
        raycast: raycastState(this.bot, this.opponentUsername)
      },
      opponent,
      entities: this.entitySlots(match.tick, selfPosition),
      blocks: this.blocks.sample(
        match.tick,
        opponentWorld,
        match.episode_id,
        tacticalPlacement?.referenceBlock?.position
      ).map(block => ({
        ...block,
        break_progress: block.raycastable ? control.miningProgress : block.break_progress
      })),
      action_mask: actionMask(
        this.bot,
        control,
        match.tick,
        this.opponentUsername,
        match.mode ?? 'sword',
        tacticalPlacement
      )
    }
  }

  private opponentState(
    entity: any,
    selfPosition: Vec3Value,
    tick: number,
    episodeId: string
  ): OpponentState {
    const position = toVec3Value(entity.position)
    const selfVelocity = toVec3Value(this.bot.entity?.velocity)
    const legacyRelativeVelocity = subtract(toVec3Value(entity.velocity), selfVelocity)
    const localEstimatedVelocity = this.estimateOpponentVelocity(entity, tick, episodeId)
    const serverState = this.matchingServerState(episodeId, tick)
    const opponentVelocity = serverState?.fresh
      ? serverState.state.velocity
      : localEstimatedVelocity
    const derivedRelativeVelocity = subtract(
      opponentVelocity,
      selfVelocity
    )
    const delta = subtract(position, selfPosition)
    const yaw = finite(this.bot.entity?.yaw, 0)
    const pitch = finite(this.bot.entity?.pitch, 0)
    const bodyPosition = mineflayerBodyRelative(delta, yaw)
    const bodyVelocity = mineflayerBodyRelative(derivedRelativeVelocity, yaw)
    const horizontalDistance = Math.hypot(delta.x, delta.z)
    const opponentDistance = Math.hypot(delta.x, delta.y, delta.z)
    const selfEyeHeight = finite(
      (this.bot.entity as any)?.eyeHeight ?? (this.bot.entity as any)?.height,
      1.62
    )
    const opponentTorsoHeight = Math.max(1, finite(entity.height, 1.8) * 0.75)
    const torsoDelta = {
      x: delta.x,
      y: delta.y + opponentTorsoHeight - selfEyeHeight,
      z: delta.z
    }
    const desiredPitch = Math.atan2(torsoDelta.y, Math.max(horizontalDistance, 1e-9))
    const headYaw = finite(entity.headYaw, finite(entity.yaw, 0))
    const equipment: any[] = Array.isArray(entity.equipment) ? entity.equipment : []
    return {
      // Preserve the original transform and packet velocity for existing policies.
      relative_position: egocentric(delta, this.bot.entity.yaw ?? 0),
      relative_velocity: egocentric(legacyRelativeVelocity, this.bot.entity.yaw ?? 0),
      body_relative_position: bodyPosition,
      body_relative_velocity: bodyVelocity,
      distance: opponentDistance,
      horizontal_distance: horizontalDistance,
      bearing_error: normalizeAngle(Math.atan2(-bodyPosition.x, -bodyPosition.z)),
      pitch_error: normalizeAngle(desiredPitch - pitch),
      closing_speed: radialClosingSpeed(delta, derivedRelativeVelocity),
      within_melee_reach: opponentDistance <= RELIABLE_MELEE_REACH,
      aim_alignment: directionalAlignment(lookDirection(yaw, pitch), torsoDelta),
      facing_toward_self: opponentFacingAlignment(this.bot, entity, selfPosition, headYaw),
      yaw: serverState?.fresh ? serverState.state.yaw : finite(entity.yaw, 0),
      head_yaw: headYaw,
      pitch: serverState?.fresh ? serverState.state.pitch : finite(entity.pitch, 0),
      health: serverState?.fresh
        ? serverState.state.health
        : (typeof entity.health === 'number' ? entity.health : metadataHealth(entity)),
      absorption: serverState?.fresh ? serverState.state.absorption : 0,
      server_state_age_ticks: serverState?.age ?? null,
      hurt_time: entityHurtTime(entity),
      on_ground: serverState?.fresh ? serverState.state.grounded : Boolean(entity.onGround),
      line_of_sight: hasAssignedOpponentLineOfSight(this.bot, entity),
      mainhand: itemState(equipment[0] ?? entity.heldItem),
      offhand: itemState(equipment[1]),
      armor: [equipment[5], equipment[4], equipment[3], equipment[2]].map(itemState)
    }
  }

  private matchingServerState(
    episodeId: string,
    tick: number
  ): { state: AuthoritativeOpponentState; age: number; fresh: boolean } | null {
    const state = this.authoritativeOpponent
    if (!state || state.episodeId !== episodeId || !this.opponentUsername
      || state.opponentUsername.toLowerCase() !== this.opponentUsername.toLowerCase()) return null
    const age = Math.max(0, tick - state.receivedTick)
    return { state, age, fresh: age <= SERVER_STATE_FRESH_TICKS }
  }

  private estimateOpponentVelocity(entity: any, tick: number, episodeId: string): Vec3Value {
    const position = toVec3Value(entity.position)
    const packetVelocity = toVec3Value(entity.velocity)
    const entityId = Number(entity.id ?? -1)
    const previous = this.opponentMotion
    if (!previous || previous.episodeId !== episodeId || previous.entityId !== entityId) {
      this.opponentMotion = { episodeId, entityId, tick, position, velocity: packetVelocity }
      return packetVelocity
    }
    if (tick <= previous.tick) return previous.velocity

    const elapsed = tick - previous.tick
    const moved = subtract(position, previous.position)
    const displacement = Math.hypot(moved.x, moved.y, moved.z)
    if (displacement > Math.max(4, elapsed * 1.5)) {
      this.opponentMotion = { episodeId, entityId, tick, position, velocity: packetVelocity }
      return packetVelocity
    }

    const measured = {
      x: moved.x / elapsed,
      y: moved.y / elapsed,
      z: moved.z / elapsed
    }
    const alpha = 0.45
    const velocity = {
      x: previous.velocity.x * (1 - alpha) + measured.x * alpha,
      y: previous.velocity.y * (1 - alpha) + measured.y * alpha,
      z: previous.velocity.z * (1 - alpha) + measured.z * alpha
    }
    this.opponentMotion = { episodeId, entityId, tick, position, velocity }
    return velocity
  }

  private entitySlots(tick: number, selfPosition: Vec3Value): EntitySlot[] {
    const cursor = legalArenaAttackTargetAtCrosshair(this.bot, this.opponentUsername, 6)
    const entities = Object.values(this.bot.entities)
      .filter((entity: any) => entity && entity !== this.bot.entity && isCombatEntity(entity))
      .map((entity: any) => {
        const id = Number(entity.id ?? -1)
        if (!this.entityBornTick.has(id)) this.entityBornTick.set(id, tick)
        const position = toVec3Value(entity.position)
        const relativePosition = subtract(position, selfPosition)
        const relativeVelocity = subtract(
          toVec3Value(entity.velocity),
          toVec3Value(this.bot.entity.velocity)
        )
        const yaw = this.bot.entity.yaw ?? 0
        return {
          id,
          slot: {
            kind: entityKind(entity),
            relative_position: egocentric(relativePosition, yaw),
            relative_velocity: egocentric(relativeVelocity, yaw),
            body_relative_position: mineflayerBodyRelative(relativePosition, yaw),
            body_relative_velocity: mineflayerBodyRelative(relativeVelocity, yaw),
            age_ticks: Math.max(0, tick - (this.entityBornTick.get(id) ?? tick)),
            distance: distance(selfPosition, position),
            raycastable: Boolean(cursor && cursor.id === entity.id)
          } satisfies EntitySlot
        }
      })
      .sort((a, b) => a.slot.distance - b.slot.distance || a.id - b.id)
      .slice(0, MAX_ENTITY_SLOTS)
      .map(value => value.slot)

    const liveIds = new Set(Object.keys(this.bot.entities).map(Number))
    for (const id of this.entityBornTick.keys()) if (!liveIds.has(id)) this.entityBornTick.delete(id)
    return entities
  }
}

/**
 * Visibility for the one already-resolved arena opponent. This function never
 * discovers a target and therefore cannot be influenced by a spectator.
 */
export function hasAssignedOpponentLineOfSight(bot: Bot, opponent: any): boolean {
  const self: any = bot.entity
  if (!self?.position || !opponent?.position) return false
  const eye = self.position.offset(0, finite(self.eyeHeight ?? self.height, 1.62), 0)
  const height = Math.max(0.2, finite(opponent.height, 1.8))
  const targetPoints = [
    opponent.position.offset(0, Math.max(0.2, height * 0.75), 0),
    opponent.position.offset(0, Math.max(0.2, height * 0.92), 0)
  ]
  return targetPoints.some(target => unobstructedRay(bot, eye, target))
}

function unobstructedRay(bot: Bot, origin: Vec3, target: Vec3): boolean {
  const delta = target.minus(origin)
  const length = delta.norm()
  if (!Number.isFinite(length) || length <= 1e-6) return true
  const direction = delta.scaled(1 / length)
  const worldRaycast = (bot.world as any)?.raycast
  if (typeof worldRaycast === 'function') {
    const hit: any = worldRaycast.call(bot.world, origin, direction, length)
    if (!hit) return true
    const intersection = hit.intersect
      ?? hit.position?.offset?.(0.5, 0.5, 0.5)
    return Boolean(intersection) && origin.distanceTo(intersection) >= length - 0.08
  }

  // Older/fake Mineflayer worlds need no extra dependency: sample the segment
  // tightly enough that a full block cannot be skipped.
  for (let travelled = 0.15; travelled < length - 0.15; travelled += 0.15) {
    const point = origin.plus(direction.scaled(travelled))
    const block: any = bot.blockAt?.(point.floored(), false)
    if (visionBlockingBlock(block)) return false
  }
  return true
}

function visionBlockingBlock(block: any): boolean {
  if (!block) return false
  const name = String(block.name ?? '').toLowerCase()
  if (['air', 'water', 'flowing_water', 'lava', 'flowing_lava', 'fire'].includes(name)) return false
  return block.boundingBox !== 'empty' && Number(block.type ?? 1) !== 0
}

function lookDirection(yaw: number, pitch: number): Vec3Value {
  const cosPitch = Math.cos(pitch)
  return {
    x: -Math.sin(yaw) * cosPitch,
    y: Math.sin(pitch),
    z: -Math.cos(yaw) * cosPitch
  }
}

function directionalAlignment(direction: Vec3Value, targetDelta: Vec3Value): number {
  const magnitude = Math.hypot(targetDelta.x, targetDelta.y, targetDelta.z)
  if (magnitude <= 1e-9) return 1
  return clampUnit(
    (direction.x * targetDelta.x + direction.y * targetDelta.y + direction.z * targetDelta.z)
      / magnitude
  )
}

function opponentFacingAlignment(
  bot: Bot,
  opponent: any,
  selfPosition: Vec3Value,
  headYaw: number
): number {
  const selfHeight = Math.max(1, finite((bot.entity as any)?.height, 1.8) * 0.75)
  const opponentEyeHeight = finite(
    opponent.eyeHeight ?? Math.max(1, finite(opponent.height, 1.8) * 0.9),
    1.62
  )
  const towardSelf = {
    x: selfPosition.x - finite(opponent.position?.x, 0),
    y: selfPosition.y + selfHeight - (finite(opponent.position?.y, 0) + opponentEyeHeight),
    z: selfPosition.z - finite(opponent.position?.z, 0)
  }
  const headPitch = finite(opponent.headPitch, finite(opponent.pitch, 0))
  return directionalAlignment(lookDirection(headYaw, headPitch), towardSelf)
}

function radialClosingSpeed(delta: Vec3Value, relativeVelocity: Vec3Value): number {
  const magnitude = Math.hypot(delta.x, delta.y, delta.z)
  if (magnitude <= 1e-9) return 0
  return -(
    delta.x * relativeVelocity.x
    + delta.y * relativeVelocity.y
    + delta.z * relativeVelocity.z
  ) / magnitude
}

function clampUnit(value: number): number {
  return Math.max(-1, Math.min(1, Number.isFinite(value) ? value : 0))
}

function isCombatEntity(entity: any): boolean {
  const kind = entityKind(entity)
  return kind.includes('crystal') || kind.includes('arrow') || kind.includes('projectile') || kind.includes('pearl')
}

function entityKind(entity: any): string {
  const raw = String(entity.name ?? entity.displayName ?? entity.type ?? 'unknown').toLowerCase()
  if (raw.includes('crystal')) return 'end_crystal'
  if (raw.includes('arrow')) return 'arrow'
  if (raw.includes('snowball')) return 'snowball'
  if (raw.includes('fireball')) return 'fireball'
  if (raw.includes('pearl')) return 'ender_pearl'
  if (raw === 'player' || raw.includes('player')) return 'player'
  if (raw.includes('egg')) return 'egg'
  if (raw.includes('projectile')) return 'projectile'
  return raw.replace(/^entity/, '').replace(/[^a-z0-9]+/g, '_') || 'unknown'
}

function entityHurtTime(entity: any): number {
  if (typeof entity?.hurtTime === 'number') return entity.hurtTime
  const metadata = entity?.metadata
  if (Array.isArray(metadata)) {
    for (const entry of metadata) {
      if (entry && typeof entry === 'object' && entry.key === 'hurtTime') return finite(entry.value, 0)
    }
  }
  return 0
}

function metadataHealth(entity: any): number | null {
  const metadata = entity?.metadata
  if (!Array.isArray(metadata)) return null
  for (const entry of metadata) {
    if (entry && typeof entry === 'object' && entry.key === 'health' && typeof entry.value === 'number') return entry.value
  }
  return null
}

function raycastState(bot: Bot, opponentUsername: string | null): RaycastState {
  const eye = bot.entity.position.offset(0, (bot.entity as any).eyeHeight ?? 1.62, 0)
  const entity = legalArenaAttackTargetAtCrosshair(bot, opponentUsername, 6)
  const block = bot.blockAtCursor?.(6)
  const entityDistance = entity?.position ? eye.distanceTo(entity.position) : Number.POSITIVE_INFINITY
  const blockDistance = block?.position ? eye.distanceTo(block.position.offset(0.5, 0.5, 0.5)) : Number.POSITIVE_INFINITY
  if (entity && entityDistance <= blockDistance) {
    return { kind: 'entity', distance: entityDistance, block_name: '', entity_kind: entityKind(entity) }
  }
  if (block) return { kind: 'block', distance: blockDistance, block_name: String(block.name ?? ''), entity_kind: '' }
  return { kind: 'none', distance: 0, block_name: '', entity_kind: '' }
}

export function combatAttackReady(
  bot: Bot,
  control: ControlTelemetry,
  tick: number,
  opponentUsername: string | null
): boolean {
  // A diamond sword is effectively charged after about 12 ticks in 1.12.2.
  // Masking the intervening ticks prevents the policy from learning to spam
  // weak swings that continually reset its own cooldown. Requiring a legal
  // crosshair target also keeps the advertised action space identical to what
  // LegalControlAdapter can actually execute.
  if (tick - control.lastAttackTick < 12) return false
  return Boolean(assignedOpponentMeleeTarget(
    bot, opponentUsername, RELIABLE_MELEE_REACH
  ))
}

/**
 * A crystal placement is executable when the ordinary client crosshair is on
 * the top of a legal base, both 1.12 clearance blocks are empty, the placement
 * volume contains no entity, and the hotbar still contains a crystal. The
 * crystal does not need to be selected yet: the policy can select slot three
 * and use it in the same sampled action, which LegalControlAdapter applies in
 * that order.
 */
export function crystalPlaceReady(bot: Bot): boolean {
  const slots = bot.inventory?.slots ?? []
  const hasCrystal = Array.from({ length: 9 }, (_, index) => slots[36 + index])
    .some(item => Number(item?.count ?? 0) > 0 && crystalName(item?.name))
  if (!hasCrystal) return false

  const base: any = crosshairBlock(bot, CRYSTAL_PLACE_REACH)
  if (!base?.position || !['obsidian', 'bedrock'].includes(String(base.name ?? '').toLowerCase())) return false
  if (Number(base.face) !== 1) return false
  if (!crystalPlacementWithinChainReach(bot, base)) return false
  if (!isReplaceableCrystalSpace(bot.blockAt?.(base.position.offset(0, 1, 0), false))) return false
  if (!isReplaceableCrystalSpace(bot.blockAt?.(base.position.offset(0, 2, 0), false))) return false
  return !crystalPlacementOccupied(bot, base.position)
}

/** A charged attack on a centered arena crystal, separate from melee readiness. */
export function crystalAttackReady(
  bot: Bot,
  control: ControlTelemetry,
  tick: number,
  opponentUsername: string | null
): boolean {
  if (tick - control.lastAttackTick < 12) return false
  const target = legalArenaAttackTargetAtCrosshair(bot, opponentUsername, RELIABLE_MELEE_REACH)
  return Boolean(target && target.type !== 'player' && crystalName(target.name ?? target.displayName ?? target.type))
}

export function attackActionLegal(
  bot: Bot,
  control: ControlTelemetry,
  tick: number,
  opponentUsername: string | null
): boolean {
  if (combatAttackReady(bot, control, tick, opponentUsername)) return true
  if (crystalAttackReady(bot, control, tick, opponentUsername)) return true
  // Attack is also Minecraft's ordinary destroy-block input. Preserve mining,
  // but do not advertise it as the combat opportunity used by the trainer's
  // attack prior. If a legal fighter/crystal is under the crosshair during its
  // cooldown, hiding attack avoids a weak swing even when a block is behind it.
  const cursorEntity = legalArenaAttackTargetAtCrosshair(bot, opponentUsername, 3.4)
  if (cursorEntity) {
    // Crystals remain a legal explicit attack, but they do not receive the
    // melee-ready prior. A poorly centered opponent ray is deliberately
    // withheld so it cannot create another server-scored miss.
    return cursorEntity.type !== 'player' && tick - control.lastAttackTick >= 12
  }
  return tacticalBlockBreakReady(bot, opponentUsername)
}

/** A reachable above-floor stone/obsidian obstruction in the assigned fight corridor. */
export function tacticalBlockBreakReady(
  bot: Bot,
  opponentUsername: string | null
): boolean {
  return isSafeTacticalMiningTarget(
    bot,
    crosshairBlock(bot, TACTICAL_BLOCK_REACH),
    opponentUsername
  )
}

function actionMask(
  bot: Bot,
  control: ControlTelemetry,
  tick: number,
  opponentUsername: string | null,
  mode: NonNullable<ObservationV1['match']['mode']>,
  tacticalPlacement: TacticalWallPlacement | null
): ActionMask {
  const main = itemState(bot.heldItem)
  const off = itemState(bot.inventory.slots[45])
  const combatReady = combatAttackReady(bot, control, tick, opponentUsername)
  const crystalPlace = crystalPlaceReady(bot)
  const crystalAttack = crystalAttackReady(bot, control, tick, opponentUsername)
  const tacticalBreak = tacticalBlockBreakReady(bot, opponentUsername)
  const slots = bot.inventory?.slots ?? []
  const mainName = main.name.toLowerCase()
  const offName = off.name.toLowerCase()
  const terrainPlacement = mode === 'terrain'
    && tacticalBlockPlaceReady(bot, tacticalPlacement)
  const mainUse = crystalPlace
    || goldenAppleName(mainName)
    // Like crystals, an obsidian slot can be selected and used in the same
    // sampled ActionV1. The exact marked support face still gates legality.
    || terrainPlacement
  const offhandUse = (crystalPlace && crystalName(offName))
    || goldenAppleName(offName)
    || (terrainPlacement && offName === 'obsidian')
  const restoreTotem = totemName(mainName) && !totemName(offName)
  return {
    attack: combatReady || attackActionLegal(bot, control, tick, opponentUsername),
    combat_attack_ready: combatReady,
    crystal_place_ready: crystalPlace,
    crystal_attack_ready: crystalAttack,
    tactical_block_break_ready: tacticalBreak,
    tactical_block_place_ready: terrainPlacement,
    use_main: mainUse,
    use_offhand: off.count > 0 && offhandUse,
    release_use: control.activeHand !== 'none',
    swap_offhand: main.count > 0 && restoreTotem,
    hotbar: Array.from({ length: 9 }, (_, index) => itemState(slots[36 + index]).count > 0)
  }
}

export function tacticalBlockPlaceReady(
  bot: Bot,
  placement: TacticalWallPlacement | null
): boolean {
  if (!placement?.referenceBlock?.position) return false
  const slots = bot.inventory?.slots ?? []
  const hasObsidian = Array.from({ length: 9 }, (_, index) => slots[36 + index])
    .some(item => Number(item?.count ?? 0) > 0
      && String(item?.name ?? '').toLowerCase() === 'obsidian')
  if (!hasObsidian) return false
  const hit: any = crosshairBlock(bot, TACTICAL_BLOCK_REACH)
  return Boolean(hit?.position
    && Number(hit.face) === 1
    && sameBlockPosition(hit.position, placement.referenceBlock.position))
}

function goldenAppleName(value: unknown): boolean {
  return String(value ?? '').toLowerCase().includes('golden_apple')
}

function totemName(value: unknown): boolean {
  return String(value ?? '').toLowerCase().includes('totem')
}

function crystalName(value: unknown): boolean {
  return String(value ?? '').toLowerCase().includes('crystal')
}

function crosshairBlock(bot: Bot, reach: number): any | null {
  const entity: any = bot.entity
  if (!entity?.position) return null
  const yaw = Number(entity.yaw ?? 0)
  const pitch = Number(entity.pitch ?? 0)
  const cosPitch = Math.cos(pitch)
  const direction = new Vec3(
    -Math.sin(yaw) * cosPitch,
    Math.sin(pitch),
    -Math.cos(yaw) * cosPitch
  )
  const eye = entity.position.offset(0, Number(entity.eyeHeight ?? entity.height ?? 1.62), 0)
  // Mineflayer 4.32's blockAtEntityCursor treats exact zero yaw/pitch as
  // missing values. Use the same ordinary world ray directly so cardinal aim
  // at an arena pad remains observable, retaining the helper as a test/client
  // fallback when the synchronized world ray is unavailable.
  return (bot.world as any)?.raycast?.(eye, direction, reach) ?? bot.blockAtCursor?.(reach) ?? null
}

function isReplaceableCrystalSpace(block: any): boolean {
  if (!block) return false
  const name = String(block.name ?? '').toLowerCase()
  // ItemEndCrystal in 1.12 checks isAirBlock for both cells. Plants, fire,
  // liquids and unloaded/null blocks are not executable placement clearance.
  return name === 'air' || Number(block.type) === 0
}

function crystalPlacementWithinChainReach(bot: Bot, base: any): boolean {
  const entity: any = bot.entity
  if (!entity?.position || !base?.position) return false
  const eye = entity.position.offset(0, Number(entity.eyeHeight ?? entity.height ?? 1.62), 0)
  const topCenter = base.position.offset(0.5, 1, 0.5)
  const clickPoint = base.intersect ?? topCenter
  const crystalAim = topCenter.offset(0, 1, 0)
  return eye.distanceTo(clickPoint) <= CRYSTAL_PLACE_REACH
    && eye.distanceTo(crystalAim) <= CRYSTAL_DETONATION_REACH
}

function crystalPlacementOccupied(bot: Bot, base: any): boolean {
  const minimumY = Number(base.y) + 1
  const maximumY = minimumY + 2
  const entities = new Set<any>([bot.entity, ...Object.values(bot.entities ?? {})])
  for (const entity of entities) {
    if (!entity?.position) continue
    const width = Math.max(0.1, Number(entity.width ?? 0.6))
    const height = Math.max(0.1, Number(entity.height ?? 1.8))
    const half = width / 2
    const overlapsX = entity.position.x + half > base.x && entity.position.x - half < base.x + 1
    const overlapsZ = entity.position.z + half > base.z && entity.position.z - half < base.z + 1
    const overlapsY = entity.position.y + height > minimumY && entity.position.y < maximumY
    if (overlapsX && overlapsZ && overlapsY) return true
  }
  return false
}

function finite(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function sameBlockPosition(first: any, second: any): boolean {
  return Boolean(first && second
    && Number(first.x) === Number(second.x)
    && Number(first.y) === Number(second.y)
    && Number(first.z) === Number(second.z))
}

function finiteNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function vectorFromFields(
  value: Record<string, unknown>,
  xField: string,
  yField: string,
  zField: string
): Vec3Value | null {
  const x = finiteNumber(value[xField])
  const y = finiteNumber(value[yField])
  const z = finiteNumber(value[zField])
  return x === null || y === null || z === null ? null : { x, y, z }
}

/** Bukkit/Notch yaw zero is +Z; Mineflayer yaw zero is -Z. */
function notchDegreesToMineflayerYaw(degrees: number): number {
  return normalizeAngle(Math.PI - degrees * Math.PI / 180)
}

/** Bukkit/Notch positive pitch points down; Mineflayer positive points up. */
function notchDegreesToMineflayerPitch(degrees: number): number {
  return normalizeAngle(-degrees * Math.PI / 180)
}

export function emptyEquipment(): ReturnType<typeof itemState> {
  return { ...EMPTY_ITEM }
}
