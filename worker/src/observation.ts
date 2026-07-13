import type { Bot } from 'mineflayer'
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
import { distance, egocentric, subtract, toVec3Value } from './math.js'

export type MatchContext = ObservationV1['match']

export type ControlTelemetry = {
  lastAttackTick: number
  activeHand: 'none' | 'main' | 'off'
  useStartedTick: number
  miningProgress: number
}

export class ObservationBuilder {
  private readonly blocks: BlockSampler
  private entityBornTick = new Map<number, number>()

  constructor(private readonly bot: Bot) {
    this.blocks = new BlockSampler(bot)
  }

  build(match: MatchContext, control: ControlTelemetry): ObservationV1 {
    const selfPosition = toVec3Value(this.bot.entity?.position)
    const opponentEntity = nearestOpponent(this.bot)
    const opponent = opponentEntity ? this.opponentState(opponentEntity, selfPosition) : null
    const opponentWorld = opponentEntity ? toVec3Value(opponentEntity.position) : undefined
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
        hotbar: Array.from({ length: 9 }, (_, index) => itemState(this.bot.inventory.slots[36 + index])),
        offhand: itemState(this.bot.inventory.slots[45]),
        armor: [5, 6, 7, 8].map(slot => itemState(this.bot.inventory.slots[slot])),
        raycast: raycastState(this.bot)
      },
      opponent,
      entities: this.entitySlots(match.tick, selfPosition),
      blocks: this.blocks.sample(match.tick, opponentWorld).map(block => ({
        ...block,
        break_progress: block.raycastable ? control.miningProgress : block.break_progress
      })),
      action_mask: actionMask(this.bot, control)
    }
  }

  private opponentState(entity: any, selfPosition: Vec3Value): OpponentState {
    const position = toVec3Value(entity.position)
    const equipment: any[] = Array.isArray(entity.equipment) ? entity.equipment : []
    return {
      relative_position: egocentric(subtract(position, selfPosition), this.bot.entity.yaw ?? 0),
      relative_velocity: egocentric(
        subtract(toVec3Value(entity.velocity), toVec3Value(this.bot.entity.velocity)),
        this.bot.entity.yaw ?? 0
      ),
      yaw: finite(entity.yaw, 0),
      pitch: finite(entity.pitch, 0),
      health: typeof entity.health === 'number' ? entity.health : metadataHealth(entity),
      hurt_time: entityHurtTime(entity),
      on_ground: Boolean(entity.onGround),
      line_of_sight: typeof (this.bot as any).canSeeEntity === 'function' ? Boolean((this.bot as any).canSeeEntity(entity)) : false,
      mainhand: itemState(equipment[0] ?? entity.heldItem),
      offhand: itemState(equipment[1]),
      armor: [equipment[5], equipment[4], equipment[3], equipment[2]].map(itemState)
    }
  }

  private entitySlots(tick: number, selfPosition: Vec3Value): EntitySlot[] {
    const cursor = this.bot.entityAtCursor?.(6)
    const entities = Object.values(this.bot.entities)
      .filter((entity: any) => entity && entity !== this.bot.entity && isCombatEntity(entity))
      .map((entity: any) => {
        const id = Number(entity.id ?? -1)
        if (!this.entityBornTick.has(id)) this.entityBornTick.set(id, tick)
        const position = toVec3Value(entity.position)
        return {
          id,
          slot: {
            kind: entityKind(entity),
            relative_position: egocentric(subtract(position, selfPosition), this.bot.entity.yaw ?? 0),
            relative_velocity: egocentric(
              subtract(toVec3Value(entity.velocity), toVec3Value(this.bot.entity.velocity)),
              this.bot.entity.yaw ?? 0
            ),
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

function nearestOpponent(bot: Bot): any | null {
  const self = bot.entity?.position
  if (!self) return null
  return Object.values(bot.entities)
    .filter((entity: any) => entity && entity.type === 'player' && entity.username !== bot.username)
    .sort((a: any, b: any) => self.distanceTo(a.position) - self.distanceTo(b.position) || a.id - b.id)[0] ?? null
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

function raycastState(bot: Bot): RaycastState {
  const eye = bot.entity.position.offset(0, (bot.entity as any).eyeHeight ?? 1.62, 0)
  const entity = bot.entityAtCursor?.(6)
  const block = bot.blockAtCursor?.(6)
  const entityDistance = entity?.position ? eye.distanceTo(entity.position) : Number.POSITIVE_INFINITY
  const blockDistance = block?.position ? eye.distanceTo(block.position.offset(0.5, 0.5, 0.5)) : Number.POSITIVE_INFINITY
  if (entity && entityDistance <= blockDistance) {
    return { kind: 'entity', distance: entityDistance, block_name: '', entity_kind: entityKind(entity) }
  }
  if (block) return { kind: 'block', distance: blockDistance, block_name: String(block.name ?? ''), entity_kind: '' }
  return { kind: 'none', distance: 0, block_name: '', entity_kind: '' }
}

function actionMask(bot: Bot, control: ControlTelemetry): ActionMask {
  const main = itemState(bot.heldItem)
  const off = itemState(bot.inventory.slots[45])
  return {
    attack: Boolean(bot.entity),
    use_main: main.count > 0,
    use_offhand: off.count > 0,
    release_use: control.activeHand !== 'none',
    swap_offhand: main.count > 0 || off.count > 0,
    hotbar: Array.from({ length: 9 }, () => true)
  }
}

function finite(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

export function emptyEquipment(): ReturnType<typeof itemState> {
  return { ...EMPTY_ITEM }
}
