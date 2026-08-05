import {
  NOOP_ACTION,
  type ActionV1,
  type BlockSlot,
  type EntitySlot,
  type ItemState,
  type ObservationV1,
  type OpponentState,
  type Vec3Value
} from './contracts.js'
import { clamp, egocentric } from './math.js'

export type ScriptedStyle =
  | 'rush'
  | 'strafe'
  | 'retreat'
  | 'jump_critical'
  | 'defensive'
  | 'erratic'
  | 'crystal'
  | 'crystal_melee'

/** Standing eye height. Every relative_position in the observation is measured from the feet. */
const EYE_HEIGHT = 1.62
/** Torso height, used as the blast reference point for both players. */
const BODY_HEIGHT = 0.9
/**
 * Largest camera correction the teacher makes in a single tick. It doubles as the "can I be on
 * target when this action lands?" test: LegalControlAdapter looks before it clicks, so an aim
 * whose raw delta fits inside this clamp is already on the crosshair by the time the click runs.
 */
const MAX_AIM_TURN = 0.6
/** Conservative build reach; the control adapter raycasts 5.0 blocks from the eye. */
const PLACE_REACH = 4.4
/** LegalControlAdapter attacks entityAtCursor(3.0), so a crystal further out cannot be popped. */
const ATTACK_REACH = 3.2
/** Never pop a blast whose centre is closer to us than this, whatever the opponent is doing. */
const MIN_BLAST_GAP = 1.8
/** A crystal further than this from the opponent is not worth the placement ticks. */
const CRYSTAL_RANGE = 3.0
/** Back off from a blast we refuse to trigger while it is inside this radius. */
const DANGER_RADIUS = 3.4
const SELF_BODY: Vec3Value = { x: 0, y: BODY_HEIGHT, z: 0 }

type Kit = { sword: number; obsidian: number; crystal: number; totem: number }
type Aim = { yaw_delta: number; pitch_delta: number; onTarget: boolean }
type Spot = { aimPoint: Vec3Value; blast: Vec3Value; score: number }

export function scriptedAction(observation: ObservationV1, style: ScriptedStyle = 'strafe'): ActionV1 {
  const opponent = observation.opponent
  if (!opponent) return { ...NOOP_ACTION, yaw_delta: 0.15 }
  const kit = readKit(observation)
  // The SWORD arena mode hands out a sword and nothing else. Without both obsidian and crystals
  // there is no crystal technique to demonstrate, so the teacher stays byte-for-byte melee.
  if (kit.crystal < 0 || kit.obsidian < 0) return meleeAction(observation, style, opponent)
  return crystalAction(observation, style, opponent, kit)
}

function meleeAction(observation: ObservationV1, style: ScriptedStyle, opponent: OpponentState): ActionV1 {
  const base = meleeBaseStyle(style)
  const relative = opponent.relative_position
  const horizontal = Math.hypot(relative.x, relative.z)
  const yawDelta = clamp(Math.atan2(-relative.x, -relative.z), -0.8, 0.8)
  const pitchDelta = clamp(Math.atan2(relative.y, Math.max(0.1, horizontal)) - observation.self.pitch, -0.4, 0.4)
  const tick = observation.match.tick
  let forward: -1 | 0 | 1 = horizontal > 2.75 ? 1 : 0
  let strafe: -1 | 0 | 1 = tick % 80 < 40 ? -1 : 1
  if (base === 'rush') strafe = 0
  if (base === 'retreat' || base === 'defensive') forward = horizontal < 4.5 ? -1 : 0
  if (base === 'erratic') strafe = tick % 17 < 8 ? -1 : 1
  return {
    ...NOOP_ACTION,
    forward,
    strafe,
    sprint: forward === 1,
    jump: (base === 'erratic' && tick % 23 === 0) || (base === 'jump_critical' && tick % 13 === 0),
    yaw_delta: yawDelta,
    pitch_delta: pitchDelta,
    primary: horizontal <= 3.0 && observation.self.attack_cooldown >= 0.9 ? 'attack' : 'none'
  }
}

/** The crystal styles borrow their footwork from an existing melee style. */
function meleeBaseStyle(style: ScriptedStyle): ScriptedStyle {
  if (style === 'crystal') return 'strafe'
  if (style === 'crystal_melee') return 'rush'
  return style
}

/**
 * Crystal PvP as a fixed ladder: keep a totem up, pop anything already burning, then rebuild the
 * obsidian -> crystal -> detonate cycle, and fall through to the sword when none of that is on.
 * Every rung emits ordinary controls only: a hotbar index, a camera delta and one click.
 */
function crystalAction(
  observation: ObservationV1,
  style: ScriptedStyle,
  opponent: OpponentState,
  kit: Kit
): ActionV1 {
  const yaw = observation.self.yaw
  const opponentBody = add(opponent.relative_position, SELF_BODY)
  const crystals = observation.entities.filter(isCrystalEntity)
  const danger = crystals.find(entity => {
    const gaps = blastGaps(entity.relative_position, opponentBody)
    return gaps.self <= DANGER_RADIUS && !safeToPop(gaps)
  })
  const base = retreatFromBlast(meleeAction(observation, style, opponent), danger)

  const totem = totemAction(observation, kit, base, opponentBody)
  if (totem) return totem

  const target = bestDetonation(crystals, opponentBody)
  if (target) {
    // The crystal model is two blocks tall; aiming a block above its origin puts the crosshair in
    // the middle of that hitbox. No cooldown gate: any hit detonates, full charge or not.
    const shot = aim(add(target.relative_position, { x: 0, y: 1, z: 0 }), observation.self.pitch)
    return {
      ...base,
      yaw_delta: shot.yaw_delta,
      pitch_delta: shot.pitch_delta,
      jump: false,
      hotbar: -1,
      primary: shot.onTarget || target.raycastable ? 'attack' : 'none'
    }
  }

  const meleeReady = Math.hypot(opponent.relative_position.x, opponent.relative_position.z) <= 3.0 &&
    observation.self.attack_cooldown >= 0.9
  if (meleeFirst(style) && meleeReady) return swordAction(observation, kit, base)

  const spot = bestCrystalSpot(observation, crystals, opponentBody, yaw)
  if (spot) return placeAction(observation, base, spot, kit.crystal)

  const support = bestObsidianSupport(observation, crystals, opponentBody, yaw)
  if (support) return placeAction(observation, base, support, kit.obsidian)

  return swordAction(observation, kit, base)
}

/** Rushers keep swinging when the sword is up; spacing styles always prefer the crystal cycle. */
function meleeFirst(style: ScriptedStyle): boolean {
  return style === 'rush' || style === 'jump_critical' || style === 'erratic' || style === 'crystal_melee'
}

function swordAction(observation: ObservationV1, kit: Kit, base: ActionV1): ActionV1 {
  const holdingSword = kit.sword < 0 || observation.self.selected_hotbar === kit.sword
  return {
    ...base,
    hotbar: hotbarRequest(observation, kit.sword),
    // Swinging obsidian at someone wastes the swing and the cooldown; wait for the slot to land.
    primary: holdingSword ? base.primary : 'none'
  }
}

/**
 * A popped totem leaves the offhand empty, and the spare totems sit in the hotbar. Re-arming is
 * a hotbar select plus the vanilla swap key, both already in the action space.
 */
function totemAction(observation: ObservationV1, kit: Kit, base: ActionV1, opponentBody: Vec3Value): ActionV1 | null {
  if (kit.totem < 0) return null
  if (itemMatches(observation.self.offhand, name => name.includes('totem'))) return null
  if (observation.self.health > 12) return null
  const look = aim(opponentBody, observation.self.pitch)
  const ready = observation.self.selected_hotbar === kit.totem
  return {
    ...base,
    yaw_delta: look.yaw_delta,
    pitch_delta: look.pitch_delta,
    jump: false,
    primary: 'none',
    hotbar: hotbarRequest(observation, kit.totem),
    // swap_offhand is edge-triggered by the control adapter, so it has to be pulsed to repeat.
    swap_offhand: ready && pulse(observation)
  }
}

function placeAction(observation: ObservationV1, base: ActionV1, spot: Spot, slot: number): ActionV1 {
  const look = aim(spot.aimPoint, observation.self.pitch)
  const ready = observation.self.selected_hotbar === slot
  return {
    ...base,
    yaw_delta: look.yaw_delta,
    pitch_delta: look.pitch_delta,
    jump: false,
    hotbar: hotbarRequest(observation, slot),
    // use_main is edge-triggered too: alternating ticks re-arm it so a failed place retries.
    primary: ready && look.onTarget && pulse(observation) ? 'use_main' : 'none'
  }
}

/** Pick the crystal that hurts the opponent most while staying inside our own attack reach. */
function bestDetonation(crystals: EntitySlot[], opponentBody: Vec3Value): EntitySlot | null {
  let best: EntitySlot | null = null
  let bestScore = Number.POSITIVE_INFINITY
  for (const entity of crystals) {
    if (entity.distance > ATTACK_REACH) continue
    const gaps = blastGaps(entity.relative_position, opponentBody)
    if (!safeToPop(gaps)) continue
    const score = gaps.opponent - gaps.self * 0.25
    if (score < bestScore) {
      bestScore = score
      best = entity
    }
  }
  return best
}

/**
 * A crystal detonation is symmetric: only trigger one that is well clear of us and closer to the
 * opponent than to us, so the damage split is in our favour even before totems come into it.
 */
function safeToPop(gaps: { self: number; opponent: number }): boolean {
  return gaps.self >= MIN_BLAST_GAP && gaps.opponent < gaps.self
}

function blastGaps(origin: Vec3Value, opponentBody: Vec3Value): { self: number; opponent: number } {
  return { self: gap(origin, SELF_BODY), opponent: gap(origin, opponentBody) }
}

/** An obsidian/bedrock top with clear air above it: a crystal can be placed here right now. */
function bestCrystalSpot(
  observation: ObservationV1,
  crystals: EntitySlot[],
  opponentBody: Vec3Value,
  yaw: number
): Spot | null {
  return bestSpot(observation.blocks.filter(block => block.crystal_clearance), crystals, opponentBody, yaw, 0)
}

/**
 * No legal spot yet, so build one: a solid block near the opponent gets obsidian dropped on its
 * top face, which becomes next tick's crystal base.
 */
function bestObsidianSupport(
  observation: ObservationV1,
  crystals: EntitySlot[],
  opponentBody: Vec3Value,
  yaw: number
): Spot | null {
  const supports = observation.blocks.filter(block => block.collision === 'solid' && !block.replaceable)
  const scored: Array<{ block: BlockSlot; penalty: number }> = []
  for (const block of supports) {
    const above = blockAbove(observation.blocks, block)
    // Air is sampled opportunistically, so an unknown neighbour is possible rather than illegal.
    if (above && above.collision !== 'empty') continue
    scored.push({ block, penalty: above ? 0 : 0.5 })
  }
  // The obsidian lands one block higher than the support, so the blast geometry shifts up with it.
  return bestSpot(scored.map(entry => entry.block), crystals, opponentBody, yaw, 1,
    scored.map(entry => entry.penalty))
}

/**
 * Shared placement search. `lift` is how many blocks above the sampled block the crystal will
 * end up (0 when the block is already a crystal base, 1 when obsidian goes on top of it).
 */
function bestSpot(
  blocks: BlockSlot[],
  crystals: EntitySlot[],
  opponentBody: Vec3Value,
  yaw: number,
  lift: number,
  penalties: number[] = []
): Spot | null {
  let best: Spot | null = null
  for (let index = 0; index < blocks.length; index += 1) {
    const block = blocks[index]
    // relative_position addresses the block's minimum corner. The 0.5 offsets are world-axis, so
    // they have to be rotated by the same yaw the observation used to build the egocentric frame.
    const aimPoint = add(block.relative_position, egocentric({ x: 0.5, y: lift + 0.95, z: 0.5 }, yaw))
    if (length(aimPoint) > PLACE_REACH) continue
    const blast = add(block.relative_position, egocentric({ x: 0.5, y: lift + 1, z: 0.5 }, yaw))
    if (crystals.some(entity => gap(entity.relative_position, blast) < 1.2)) continue
    const gaps = blastGaps(blast, opponentBody)
    // Same rule as detonation, applied before the crystal exists: never build a bomb we cannot
    // safely trigger, and never build one under our own feet.
    if (gaps.opponent > CRYSTAL_RANGE) continue
    if (!safeToPop(gaps)) continue
    const score = gaps.opponent + length(aimPoint) * 0.15 + (penalties[index] ?? 0)
    if (!best || score < best.score) best = { aimPoint, blast, score }
  }
  return best
}

function blockAbove(blocks: BlockSlot[], block: BlockSlot): BlockSlot | null {
  const target = block.relative_position
  for (const candidate of blocks) {
    const position = candidate.relative_position
    if (Math.abs(position.y - target.y - 1) > 1e-3) continue
    if (Math.abs(position.x - target.x) > 1e-3 || Math.abs(position.z - target.z) > 1e-3) continue
    return candidate
  }
  return null
}

/**
 * Camera delta onto an egocentric point. Forward is -z and the frame is yaw-invariant, so the
 * yaw that faces (x, y, z) is atan2(-x, -z); pitch is measured from the eye, not the feet.
 */
function aim(point: Vec3Value, pitch: number): Aim {
  const horizontal = Math.max(0.05, Math.hypot(point.x, point.z))
  const rawYaw = Math.atan2(-point.x, -point.z)
  const rawPitch = Math.atan2(point.y - EYE_HEIGHT, horizontal) - pitch
  return {
    yaw_delta: clamp(rawYaw, -MAX_AIM_TURN, MAX_AIM_TURN),
    pitch_delta: clamp(rawPitch, -MAX_AIM_TURN, MAX_AIM_TURN),
    onTarget: Math.abs(rawYaw) <= MAX_AIM_TURN && Math.abs(rawPitch) <= MAX_AIM_TURN
  }
}

/** Standing still or jumping next to a blast we will not trigger is how the teacher dies. */
function retreatFromBlast(action: ActionV1, danger: EntitySlot | undefined): ActionV1 {
  if (!danger) return action
  return { ...action, forward: -1, sprint: false, jump: false }
}

function readKit(observation: ObservationV1): Kit {
  const hotbar = observation.self.hotbar ?? []
  return {
    sword: findSlot(hotbar, name => name.includes('sword')),
    obsidian: findSlot(hotbar, name => name === 'obsidian'),
    crystal: findSlot(hotbar, name => name.includes('crystal')),
    totem: findSlot(hotbar, name => name.includes('totem'))
  }
}

function findSlot(hotbar: ItemState[], match: (name: string) => boolean): number {
  const limit = Math.min(hotbar.length, 9)
  for (let index = 0; index < limit; index += 1) {
    if (itemMatches(hotbar[index], match)) return index
  }
  return -1
}

function itemMatches(item: ItemState | undefined, match: (name: string) => boolean): boolean {
  if (!item || item.count <= 0) return false
  return match(String(item.name ?? '').toLowerCase())
}

/** -1 means "leave the selection alone"; the adapter also ignores a redundant switch. */
function hotbarRequest(observation: ObservationV1, slot: number): number {
  if (slot < 0 || observation.self.selected_hotbar === slot) return -1
  return slot
}

function pulse(observation: ObservationV1): boolean {
  return observation.match.tick % 2 === 0
}

function isCrystalEntity(entity: EntitySlot): boolean {
  return String(entity.kind ?? '').includes('crystal')
}

function add(a: Vec3Value, b: Vec3Value): Vec3Value {
  return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z }
}

function gap(a: Vec3Value, b: Vec3Value): number {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z)
}

function length(value: Vec3Value): number {
  return Math.hypot(value.x, value.y, value.z)
}
