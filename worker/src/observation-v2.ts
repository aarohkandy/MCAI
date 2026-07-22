import {
  OBSERVATION_V2_SCHEMA_VERSION,
  type ActionHistoryEntry,
  type ObservationV1,
  type ObservationV2,
  type Vec3Value
} from './contracts.js'

const MAX_CRYSTAL_CANDIDATES = 12
const MAX_BLOCK_CANDIDATES = 16

/** Builds only from state the client already knows; no spectator/camera input is consulted. */
export function observationV2(
  base: ObservationV1,
  recentHistory: readonly ActionHistoryEntry[] = []
): ObservationV2 {
  const opponent = base.opponent
  const opponentPosition = opponent?.body_relative_position
  const crystalCandidates = [
    ...(base.entities ?? []).map((entity, source_index) => ({ entity, source_index }))
      .filter(({ entity }) => entity.kind.toLowerCase().includes('crystal'))
      .map(({ entity, source_index }) => {
        const opponentDistance = opponentPosition
          ? vectorDistance(entity.body_relative_position, opponentPosition)
          : 12
        const opponentDamage = explosionDamageEstimate(opponentDistance)
        const selfDamage = explosionDamageEstimate(entity.distance)
        return {
          kind: 'crystal' as const,
          source_index,
          body_relative_position: entity.body_relative_position,
          distance: entity.distance,
          reachable: entity.distance <= 3.4,
          visible: entity.raycastable,
          placement_legal: false,
          estimated_opponent_damage: opponentDamage,
          estimated_self_damage: selfDamage,
          pop_potential: opponent && opponent.health !== null && opponent.health + opponent.absorption <= opponentDamage ? 1 : 0,
          escape_direction: escapeDirection(entity.body_relative_position)
        }
      }),
    ...(base.blocks ?? []).map((block, source_index) => ({ block, source_index }))
      .filter(({ block }) => block.crystal_clearance && /obsidian|bedrock/.test(block.name.toLowerCase()))
      .map(({ block, source_index }) => {
        const crystalPosition = { ...block.body_relative_position, y: block.body_relative_position.y + 1 }
        const opponentDistance = opponentPosition ? vectorDistance(crystalPosition, opponentPosition) : 12
        const opponentDamage = explosionDamageEstimate(opponentDistance)
        const selfDamage = explosionDamageEstimate(Math.max(0, block.distance - 1))
        return {
          kind: 'base' as const,
          source_index,
          body_relative_position: block.body_relative_position,
          distance: block.distance,
          reachable: block.within_reach,
          visible: block.raycastable,
          placement_legal: block.within_reach && block.raycastable && block.crystal_clearance,
          estimated_opponent_damage: opponentDamage,
          estimated_self_damage: selfDamage,
          pop_potential: opponent && opponent.health !== null && opponent.health + opponent.absorption <= opponentDamage ? 1 : 0,
          escape_direction: escapeDirection(block.body_relative_position)
        }
      })
  ].sort((a, b) => candidateScore(b) - candidateScore(a)).slice(0, MAX_CRYSTAL_CANDIDATES)

  const blockCandidates = (base.blocks ?? [])
    .map((block, source_index) => ({
      source_index,
      body_relative_position: block.body_relative_position,
      distance: block.distance,
      purpose: block.tactical_placement_target
        ? 'cover' as const
        : block.crystal_clearance
          ? 'crystal_base' as const
          : block.relative_position.y > 1
            ? 'mine_path' as const
            : 'high_ground' as const,
      reachable: block.within_reach,
      visible: block.raycastable,
      cover_value: clamp01((block.exposed_faces <= 3 ? 0.7 : 0.2) + (block.tactical_placement_target ? 0.3 : 0)),
      followup_crystal_viability: block.crystal_clearance ? 1 : 0
    }))
    .sort((a, b) => blockScore(b) - blockScore(a))
    .slice(0, MAX_BLOCK_CANDIDATES)

  const opponentItem = equipmentClass(opponent?.mainhand.name)
  const priorItem = recentHistory.length > 1 ? recentHistory[recentHistory.length - 2] : undefined
  const closingDirection: -1 | 0 | 1 = !opponent || Math.abs(opponent.closing_speed) < 0.02
    ? 0 : opponent.closing_speed > 0 ? 1 : -1
  const nearbyExplosionRisk = Math.max(0, ...crystalCandidates
    .filter(candidate => candidate.kind === 'crystal')
    .map(candidate => candidate.estimated_self_damage / 20))
  const armorPieces = (base.self?.armor ?? []).filter(item => item.count > 0).length
  const cover = clamp01(blockCandidates.filter(candidate => candidate.cover_value >= 0.7 && candidate.distance <= 3).length / 2)
  const radius = base.match?.arena_radius ?? 5
  const centerDistance = Math.hypot(base.self?.position?.x ?? 0, base.self?.position?.z ?? 0)

  return {
    ...base,
    schema_version: OBSERVATION_V2_SCHEMA_VERSION,
    tactical: {
      threat: {
        score: clamp01((opponent?.within_melee_reach ? 0.35 : 0)
          + (opponent?.facing_toward_self ?? 0) * 0.2
          + nearbyExplosionRisk * 0.35
          + (armorPieces < 4 ? 0.1 : 0)),
        opponent_attack_cooldown: opponent?.hurt_time ? 0 : 1,
        opponent_equipment_class: opponentItem,
        closing_direction: closingDirection,
        recent_item_changed: Boolean(priorItem && priorItem.intent !== 'safety' && opponentItem !== 'none')
      },
      crystal_candidates: crystalCandidates,
      block_candidates: blockCandidates,
      recent_history: recentHistory.slice(-8),
      survival: {
        has_totem: String(base.self?.offhand?.name ?? '').toLowerCase().includes('totem'),
        spare_totems: countItems(base, 'totem'),
        heal_available: countItems(base, 'golden_apple') > 0,
        nearby_explosion_risk: clamp01(nearbyExplosionRisk),
        cover,
        arena_boundary_distance: Math.max(0, radius - centerDistance),
        vertical_escape: (base.blocks ?? []).some(block => block.relative_position.y < -1 && block.distance < 2) ? -1 : 1
      }
    }
  }
}

function countItems(base: ObservationV1, needle: string): number {
  return [...(base.self?.hotbar ?? []), base.self?.offhand].filter(Boolean).reduce((sum, item) =>
    sum + (item.name.toLowerCase().includes(needle) ? item.count : 0), 0)
}

function equipmentClass(name: string | undefined): 'none' | 'sword' | 'crystal' | 'tool' | 'other' {
  const value = String(name ?? '').toLowerCase()
  if (!value || value === 'air') return 'none'
  if (value.includes('sword')) return 'sword'
  if (value.includes('crystal')) return 'crystal'
  if (/pickaxe|axe|shovel/.test(value)) return 'tool'
  return 'other'
}

function explosionDamageEstimate(distance: number): number {
  return Math.max(0, 20 * (1 - Math.min(1, distance / 12)) ** 2)
}

function candidateScore(candidate: { estimated_opponent_damage: number; estimated_self_damage: number; reachable: boolean }): number {
  return candidate.estimated_opponent_damage - candidate.estimated_self_damage * 1.25 + (candidate.reachable ? 3 : 0)
}

function blockScore(candidate: { reachable: boolean; visible: boolean; cover_value: number; followup_crystal_viability: number }): number {
  return (candidate.reachable ? 2 : 0) + (candidate.visible ? 1 : 0)
    + candidate.cover_value + candidate.followup_crystal_viability * 2
}

function escapeDirection(position: Vec3Value): -1 | 0 | 1 {
  if (Math.abs(position.x) < 0.1) return 0
  return position.x > 0 ? -1 : 1
}

function vectorDistance(first: Vec3Value, second: Vec3Value): number {
  return Math.hypot(first.x - second.x, first.y - second.y, first.z - second.z)
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value))
}
