import type { Bot } from 'mineflayer'
import { Vec3 } from 'vec3'
import { MAX_BLOCK_SLOTS, type BlockSlot, type Vec3Value } from './contracts.js'
import { distance, egocentric, mineflayerBodyRelative, subtract, toVec3Value } from './math.js'

type CachedSlot = Omit<
  BlockSlot,
  'relative_position' | 'body_relative_position' | 'body_relative_velocity'
    | 'distance' | 'within_reach' | 'raycastable' | 'sample_age_ticks'
> & {
  world_position: Vec3Value
  sampled_tick: number
}

type ScoredSlot = CachedSlot & {
  key: string
  cursor: boolean
  obstacle: boolean
  self_distance: number
  opponent_distance: number
  near_fighter_distance: number
  corridor_distance: number
}

const FACE_VECTORS = [
  new Vec3(1, 0, 0), new Vec3(-1, 0, 0), new Vec3(0, 1, 0),
  new Vec3(0, -1, 0), new Vec3(0, 0, 1), new Vec3(0, 0, -1)
]

export class BlockSampler {
  private cache: CachedSlot[] = []
  private episodeId: string | null = null
  private tacticalSupportKey: string | null = null

  private readonly refreshPhase: number | null

  constructor(
    private readonly bot: Bot,
    private readonly refreshTicks = 10,
    cadenceKey = ''
  ) {
    this.refreshPhase = cadenceKey
      ? stableHash(cadenceKey) % Math.max(1, refreshTicks)
      : null
  }

  sample(
    tick: number,
    opponentPosition?: Vec3Value,
    episodeId?: string,
    tacticalSupportPosition?: Vec3Value
  ): BlockSlot[] {
    const normalizedEpisodeId = episodeId?.trim() || null
    const episodeChanged = normalizedEpisodeId !== null && normalizedEpisodeId !== this.episodeId
    const nextTacticalSupportKey = tacticalSupportPosition
      ? positionKey(tacticalSupportPosition)
      : null
    const tacticalTargetChanged = nextTacticalSupportKey !== this.tacticalSupportKey
    const sampledTick = this.cache[0]?.sampled_tick
    const clockRewound = sampledTick !== undefined && tick < sampledTick
    if (episodeChanged) this.episodeId = normalizedEpisodeId
    if (episodeChanged || tacticalTargetChanged) {
      this.tacticalSupportKey = nextTacticalSupportKey
    }
    const refreshDue = tick - Number(sampledTick) >= this.refreshTicks
      && (this.refreshPhase === null || tick % this.refreshTicks === this.refreshPhase)
    if (this.cache.length === 0 || episodeChanged || tacticalTargetChanged || clockRewound
      || refreshDue) {
      this.cache = this.scan(tick, opponentPosition, tacticalSupportPosition)
    }
    const self = toVec3Value(this.bot.entity?.position)
    const selfVelocity = toVec3Value(this.bot.entity?.velocity)
    const yaw = Number(this.bot.entity?.yaw ?? 0)
    const cursorBlock = this.bot.blockAtCursor?.(6)
    return this.cache.map(({ sampled_tick, world_position, ...slot }) => {
      const delta = subtract(world_position, self)
      const dist = distance(self, world_position)
      return {
        ...slot,
        // Cached block identity is stable for a few ticks, but its observation
        // coordinates are not: movement and camera turns change the agent's
        // frame every tick. Reprojecting here prevents a five-tick stale target
        // from repeatedly steering toward where a randomized pad used to be.
        relative_position: egocentric(delta, yaw),
        body_relative_position: mineflayerBodyRelative(delta, yaw),
        body_relative_velocity: mineflayerBodyRelative({
          x: -selfVelocity.x,
          y: -selfVelocity.y,
          z: -selfVelocity.z
        }, yaw),
        distance: dist,
        within_reach: dist <= 5,
        raycastable: samePosition(cursorBlock?.position, world_position),
        sample_age_ticks: Math.max(0, tick - sampled_tick)
      }
    })
  }

  private scan(
    tick: number,
    opponentPosition?: Vec3Value,
    tacticalSupportPosition?: Vec3Value
  ): CachedSlot[] {
    const self = toVec3Value(this.bot.entity?.position)
    const centers = [self]
    if (opponentPosition && distance(self, opponentPosition) > 2) {
      centers.push({
        x: (self.x + opponentPosition.x) / 2,
        y: (self.y + opponentPosition.y) / 2,
        z: (self.z + opponentPosition.z) / 2
      })
      centers.push(opponentPosition)
    }
    if (tacticalSupportPosition) centers.push(tacticalSupportPosition)
    const seen = new Set<string>()
    const candidates: ScoredSlot[] = []
    // One arena scan previously asked Mineflayer for every exposed neighbor
    // again for every solid. Memoizing by world position reduces the hot path
    // from several thousand duplicate lookups to one lookup per cell.
    const blockCache = new Map<string, any | null>()
    const readBlock = (position: Vec3): any | null => {
      const key = positionKey(position)
      if (blockCache.has(key)) return blockCache.get(key) ?? null
      if (!loadedAt(this.bot, position)) {
        blockCache.set(key, null)
        return null
      }
      const block = this.bot.blockAt(position, false)
      blockCache.set(key, block ?? null)
      return block ?? null
    }
    const cursorBlock = this.bot.blockAtCursor?.(6)
    if (cursorBlock?.position && distance(self, toVec3Value(cursorBlock.position)) > 5) {
      centers.push(toVec3Value(cursorBlock.position))
    }
    const fighterFloorY = Math.floor(Math.min(self.y, opponentPosition?.y ?? self.y))
    for (const center of centers) {
      const baseX = Math.floor(center.x)
      const baseY = Math.floor(center.y)
      const baseZ = Math.floor(center.z)
      for (let dy = -2; dy <= 3; dy += 1) {
        for (let dx = -5; dx <= 5; dx += 1) {
          for (let dz = -5; dz <= 5; dz += 1) {
            if (dx * dx + dz * dz > 26) continue
            const position = new Vec3(baseX + dx, baseY + dy, baseZ + dz)
            const key = `${position.x},${position.y},${position.z}`
            if (seen.has(key)) continue
            seen.add(key)
            const block = readBlock(position)
            if (!block) continue
            const replaceable = isReplaceable(block)
            // Empty/liquid cells previously consumed most of the 48 slots. Air
            // remains represented as clearance/support attributes of solids.
            if (replaceable) continue
            const world = toVec3Value(position)
            const selfDistance = distance(self, world)
            const opponentDistance = opponentPosition
              ? distance(opponentPosition, world)
              : Number.POSITIVE_INFINITY
            const faces = exposedFaces(position, readBlock)
            const clearance = isCrystalBase(block)
              && isCrystalAir(readBlock(position.offset(0, 1, 0)))
              && isCrystalAir(readBlock(position.offset(0, 2, 0)))
            candidates.push({
              key,
              name: String(block.name ?? ''),
              collision: collisionKind(block),
              hardness: finite(block.hardness, 0),
              replaceable,
              break_progress: 0,
              crystal_clearance: clearance,
              tactical_placement_target: Boolean(
                tacticalSupportPosition && samePosition(position, tacticalSupportPosition)
              ),
              exposed_faces: faces,
              world_position: world,
              sampled_tick: tick,
              cursor: samePosition(cursorBlock?.position, world),
              obstacle: world.y >= fighterFloorY && faces > 0,
              self_distance: selfDistance,
              opponent_distance: opponentDistance,
              near_fighter_distance: Math.min(selfDistance, opponentDistance),
              corridor_distance: opponentPosition
                ? horizontalSegmentDistance(world, self, opponentPosition)
                : selfDistance
            })
          }
        }
      }
    }

    return selectRelevantSlots(candidates).map(({
      key: _key,
      cursor: _cursor,
      obstacle: _obstacle,
      self_distance: _selfDistance,
      opponent_distance: _opponentDistance,
      near_fighter_distance: _nearFighterDistance,
      corridor_distance: _corridorDistance,
      ...slot
    }) => slot)
  }
}

function selectRelevantSlots(candidates: ScoredSlot[]): ScoredSlot[] {
  const selected: ScoredSlot[] = []
  const selectedKeys = new Set<string>()
  const take = (
    limit: number,
    predicate: (candidate: ScoredSlot) => boolean,
    compare: (a: ScoredSlot, b: ScoredSlot) => number
  ) => {
    for (const candidate of candidates.filter(predicate).sort(compare)) {
      if (selected.length >= MAX_BLOCK_SLOTS || limit <= 0) break
      if (selectedKeys.has(candidate.key)) continue
      selected.push(candidate)
      selectedKeys.add(candidate.key)
      limit -= 1
    }
  }
  const stable = (a: ScoredSlot, b: ScoredSlot) =>
    comparePosition(a.world_position, b.world_position)
  const byNearFighter = (a: ScoredSlot, b: ScoredSlot) =>
    a.near_fighter_distance - b.near_fighter_distance || stable(a, b)
  const byOpponent = (a: ScoredSlot, b: ScoredSlot) =>
    a.opponent_distance - b.opponent_distance || a.self_distance - b.self_distance || stable(a, b)
  const byCorridor = (a: ScoredSlot, b: ScoredSlot) =>
    a.corridor_distance - b.corridor_distance || byNearFighter(a, b)

  // Quotas keep rare tactical affordances from being buried by a flat floor.
  take(1, candidate => candidate.tactical_placement_target, byNearFighter)
  take(1, candidate => candidate.cursor, byNearFighter)
  take(12, candidate => candidate.crystal_clearance, byOpponent)
  take(16, candidate => candidate.obstacle, byCorridor)
  take(8, candidate => Number.isFinite(candidate.opponent_distance), byOpponent)
  take(8, candidate => candidate.corridor_distance <= 2.25, byCorridor)
  take(MAX_BLOCK_SLOTS, candidate => candidate.exposed_faces > 0, byNearFighter)
  take(MAX_BLOCK_SLOTS, () => true, byNearFighter)
  return selected.slice(0, MAX_BLOCK_SLOTS)
}

function isReplaceable(block: any): boolean {
  if (!block) return true
  const name = String(block.name ?? '')
  return block.boundingBox === 'empty' || [
    'air', 'tall_grass', 'double_plant', 'fire', 'snow_layer', 'water', 'flowing_water',
    'lava', 'flowing_lava'
  ].includes(name)
}

function isCrystalBase(block: any): boolean {
  return block?.name === 'obsidian' || block?.name === 'bedrock'
}

/** ItemEndCrystal in 1.12 requires two actual air cells, not generic replaceables. */
function isCrystalAir(block: any): boolean {
  if (!block) return false
  return String(block.name ?? '').toLowerCase() === 'air' || Number(block.type) === 0
}

function collisionKind(block: any): BlockSlot['collision'] {
  const name = String(block?.name ?? '')
  if (isReplaceable(block)) return name.includes('water') || name.includes('lava') ? 'liquid' : 'empty'
  return block?.boundingBox === 'block' ? 'solid' : 'partial'
}

function exposedFaces(position: Vec3, blockAt: (position: Vec3) => any | null): number {
  let count = 0
  for (const face of FACE_VECTORS) {
    if (isReplaceable(blockAt(position.plus(face)))) count += 1
  }
  return count
}

/** Never ask Mineflayer for a block in an absent column; it logs per probe. */
function loadedAt(bot: Bot, position: Vec3): boolean {
  const getColumnAt = (bot.world as any)?.getColumnAt
  if (typeof getColumnAt !== 'function') return true
  try {
    return Boolean(getColumnAt.call(bot.world, position))
  } catch {
    return false
  }
}

function stableHash(value: string): number {
  let hash = 0
  for (const char of value) hash = ((hash * 31) + char.charCodeAt(0)) >>> 0
  return hash
}

function comparePosition(a: Vec3Value, b: Vec3Value): number {
  return a.y - b.y || a.x - b.x || a.z - b.z
}

function horizontalSegmentDistance(point: Vec3Value, start: Vec3Value, end: Vec3Value): number {
  const dx = end.x - start.x
  const dz = end.z - start.z
  const lengthSquared = dx * dx + dz * dz
  if (lengthSquared <= 1e-9) return Math.hypot(point.x - start.x, point.z - start.z)
  const projection = Math.max(0, Math.min(1,
    ((point.x - start.x) * dx + (point.z - start.z) * dz) / lengthSquared
  ))
  return Math.hypot(
    point.x - (start.x + dx * projection),
    point.z - (start.z + dz * projection)
  )
}

function samePosition(a: Vec3Value | null | undefined, b: Vec3Value): boolean {
  return Boolean(a) && a!.x === b.x && a!.y === b.y && a!.z === b.z
}

function positionKey(position: Vec3Value): string {
  return `${position.x},${position.y},${position.z}`
}

function finite(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}
