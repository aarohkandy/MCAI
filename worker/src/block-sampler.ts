import type { Bot } from 'mineflayer'
import { Vec3 } from 'vec3'
import { MAX_BLOCK_SLOTS, type BlockSlot, type Vec3Value } from './contracts.js'
import { distance, egocentric, subtract, toVec3Value } from './math.js'

type CachedSlot = Omit<BlockSlot, 'sample_age_ticks'> & { sampled_tick: number }

const FACE_VECTORS = [
  new Vec3(1, 0, 0), new Vec3(-1, 0, 0), new Vec3(0, 1, 0),
  new Vec3(0, -1, 0), new Vec3(0, 0, 1), new Vec3(0, 0, -1)
]

export class BlockSampler {
  private cache: CachedSlot[] = []

  constructor(private readonly bot: Bot, private readonly refreshTicks = 5) {}

  sample(tick: number, opponentPosition?: Vec3Value): BlockSlot[] {
    if (this.cache.length === 0 || tick - this.cache[0].sampled_tick >= this.refreshTicks) {
      this.cache = this.scan(tick, opponentPosition)
    }
    return this.cache.map(({ sampled_tick, ...slot }) => ({
      ...slot,
      sample_age_ticks: Math.max(0, tick - sampled_tick)
    }))
  }

  private scan(tick: number, opponentPosition?: Vec3Value): CachedSlot[] {
    const self = toVec3Value(this.bot.entity?.position)
    const centers = [self]
    if (opponentPosition && distance(self, opponentPosition) > 2) centers.push(opponentPosition)
    const seen = new Set<string>()
    const candidates: Array<CachedSlot & { score: number }> = []
    const cursorBlock = this.bot.blockAtCursor?.(6)

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
            const block = this.bot.blockAt(position, false)
            if (!block) continue
            const replaceable = isReplaceable(block)
            const relevant = !replaceable || isCrystalBase(block) || exposedFaces(this.bot, position) > 0
            if (!relevant) continue
            const world = toVec3Value(position)
            const dist = distance(self, world)
            const clearance = isCrystalBase(block) && isReplaceable(this.bot.blockAt(position.offset(0, 1, 0), false)) &&
              isReplaceable(this.bot.blockAt(position.offset(0, 2, 0), false))
            const cursor = Boolean(cursorBlock && cursorBlock.position.equals(position))
            const placementBonus = clearance ? -4 : isCrystalBase(block) ? -2 : 0
            const cursorBonus = cursor ? -8 : 0
            candidates.push({
              name: String(block.name ?? ''),
              relative_position: egocentric(subtract(world, self), this.bot.entity.yaw ?? 0),
              collision: collisionKind(block),
              hardness: finite(block.hardness, 0),
              replaceable,
              break_progress: 0,
              crystal_clearance: clearance,
              exposed_faces: exposedFaces(this.bot, position),
              distance: dist,
              within_reach: dist <= 5,
              raycastable: cursor,
              sampled_tick: tick,
              score: dist + placementBonus + cursorBonus
            })
          }
        }
      }
    }

    candidates.sort((a, b) => a.score - b.score || comparePosition(a.relative_position, b.relative_position))
    return candidates.slice(0, MAX_BLOCK_SLOTS).map(({ score: _score, ...slot }) => slot)
  }
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

function collisionKind(block: any): BlockSlot['collision'] {
  const name = String(block?.name ?? '')
  if (isReplaceable(block)) return name.includes('water') || name.includes('lava') ? 'liquid' : 'empty'
  return block?.boundingBox === 'block' ? 'solid' : 'partial'
}

function exposedFaces(bot: Bot, position: Vec3): number {
  let count = 0
  for (const face of FACE_VECTORS) {
    if (isReplaceable(bot.blockAt(position.plus(face), false))) count += 1
  }
  return count
}

function comparePosition(a: Vec3Value, b: Vec3Value): number {
  return a.y - b.y || a.x - b.x || a.z - b.z
}

function finite(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}
