import type { Bot } from 'mineflayer'
import { Vec3 } from 'vec3'

/**
 * Resolve the one player the arena server assigned to this bot. Deliberately
 * return null when there is no assignment: proximity is not proof of an
 * opponent, especially while a human spectator is orbiting the arena.
 */
export function findAssignedOpponent(bot: Bot, opponentUsername: string | null): any | null {
  if (!opponentUsername) return null
  const expected = opponentUsername.toLowerCase()
  const fromEntities = Object.values(bot.entities).find((entity: any) =>
    isPlayer(entity) && usernameOf(entity) === expected
  )
  if (fromEntities) return fromEntities

  const player = Object.entries(bot.players ?? {}).find(([username]) => username.toLowerCase() === expected)?.[1]
  const entity = (player as any)?.entity
  // During a cross-arena teleport Mineflayer can populate bot.players[username]
  // one packet before copying that username onto the spawned entity. The map
  // key is already the server-assigned identity, so it is safe to trust that
  // exact entry while still excluding every spectator/unrelated player.
  return entity?.type === 'player' ? entity : null
}

export function isAssignedOpponent(entity: any, opponentUsername: string | null): boolean {
  return Boolean(opponentUsername) && isPlayer(entity) && usernameOf(entity) === opponentUsername?.toLowerCase()
}

/** Player attacks are restricted to the assigned fighter; crystals remain legal arena targets. */
export function isLegalArenaAttackTarget(entity: any, opponentUsername: string | null): boolean {
  if (isAssignedOpponent(entity, opponentUsername)) return true
  if (!entity || entity.type === 'player') return false
  const kind = String(entity.name ?? entity.displayName ?? entity.type ?? '').toLowerCase()
  return kind.includes('crystal')
}

/**
 * Exact crosshair hit test for only the server-assigned opponent or an end
 * crystal. Mineflayer's stock helper rejects `object` entities (all crystals)
 * and filters player centers before testing their hitbox, which loses valid
 * edge-of-reach sword targets. This never selects a spectator or turns aim.
 */
export function legalArenaAttackTargetAtCrosshair(
  bot: Bot,
  opponentUsername: string | null,
  maxDistance: number
): any | null {
  const self: any = bot.entity
  if (!self?.position || !Number.isFinite(self.yaw) || !Number.isFinite(self.pitch)) return null
  const opponent = findAssignedOpponent(bot, opponentUsername)
  const candidates = Object.values(bot.entities ?? {}).filter((entity: any) =>
    entity && (entity === opponent || (entity.type !== 'player' && crystalKind(entity)))
  ) as any[]
  if (opponent && !candidates.includes(opponent)) candidates.push(opponent)

  const eye = self.position.offset(0, Number(self.eyeHeight ?? 1.62), 0)
  const cosPitch = Math.cos(self.pitch)
  const direction = new Vec3(
    -Math.sin(self.yaw) * cosPitch,
    Math.sin(self.pitch),
    -Math.cos(self.yaw) * cosPitch
  )
  const block: any = bot.blockAtCursor?.(maxDistance)
  const blockDistance = block?.intersect
    ? eye.distanceTo(block.intersect)
    : maxDistance

  let nearest: any | null = null
  let nearestDistance = Math.min(maxDistance, blockDistance)
  for (const entity of candidates) {
    if (entity !== opponent && !isLegalArenaAttackTarget(entity, opponentUsername)) continue
    const hitDistance = rayHitboxDistance(eye, direction, entity, maxDistance)
    if (hitDistance !== null && hitDistance <= nearestDistance) {
      nearest = entity
      nearestDistance = hitDistance
    }
  }
  return nearest
}

/**
 * Return the exact assigned fighter only while it is inside survival reach and
 * the configured melee-facing cone. The broader exact ray above remains for
 * crystals and block interaction; neither spectators nor unrelated players can
 * satisfy this roster-identity check.
 */
export function assignedOpponentMeleeTarget(
  bot: Bot,
  opponentUsername: string | null,
  maxDistance: number,
  minimumTorsoDot = 0.7
): any | null {
  const assignedOpponent = findAssignedOpponent(bot, opponentUsername)
  if (!assignedOpponent) return null
  const self: any = bot.entity
  const target = assignedOpponent
  if (!self?.position || !target?.position) return null
  // A vanilla attack packet names an entity; the server validates reach, not a
  // one-pixel client ray. Use the exact assigned roster identity plus the same
  // center-distance bound Paper scores. This preserves real melee constraints
  // while avoiding brittle hitbox-edge/block-ray disagreements.
  if (self.position.distanceTo(target.position) > maxDistance) return null
  const eye = self.position.offset(0, Number(self.eyeHeight ?? 1.62), 0)
  const torso = target.position.offset(0, 1.0, 0)
  const toTorso = torso.minus(eye)
  const length = toTorso.norm()
  if (!Number.isFinite(length) || length < 1e-9) return target
  const cosPitch = Math.cos(self.pitch)
  const direction = new Vec3(
    -Math.sin(self.yaw) * cosPitch,
    Math.sin(self.pitch),
    -Math.cos(self.yaw) * cosPitch
  )
  const dot = direction.dot(toTorso.scaled(1 / length))
  return dot >= minimumTorsoDot ? target : null
}

function rayHitboxDistance(origin: Vec3, direction: Vec3, entity: any, maxDistance: number): number | null {
  if (!entity?.position) return null
  const width = Math.max(0.1, Number(entity.width ?? (crystalKind(entity) ? 2 : 0.6)))
  const height = Math.max(0.1, Number(entity.height ?? (crystalKind(entity) ? 2 : 1.8)))
  const halfWidth = width / 2
  const minimum = [entity.position.x - halfWidth, entity.position.y, entity.position.z - halfWidth]
  const maximum = [entity.position.x + halfWidth, entity.position.y + height, entity.position.z + halfWidth]
  const origins = [origin.x, origin.y, origin.z]
  const directions = [direction.x, direction.y, direction.z]
  let near = 0
  let far = maxDistance
  for (let axis = 0; axis < 3; axis++) {
    if (Math.abs(directions[axis]) < 1e-9) {
      if (origins[axis] < minimum[axis] || origins[axis] > maximum[axis]) return null
      continue
    }
    let first = (minimum[axis] - origins[axis]) / directions[axis]
    let second = (maximum[axis] - origins[axis]) / directions[axis]
    if (first > second) [first, second] = [second, first]
    near = Math.max(near, first)
    far = Math.min(far, second)
    if (near > far) return null
  }
  return far >= 0 && near <= maxDistance ? Math.max(0, near) : null
}

function crystalKind(entity: any): boolean {
  const kind = String(entity?.name ?? entity?.displayName ?? entity?.type ?? '').toLowerCase()
  return kind.includes('crystal')
}

function isPlayer(entity: any): boolean {
  return Boolean(entity) && entity.type === 'player' && typeof entity.username === 'string'
}

function usernameOf(entity: any): string {
  return String(entity?.username ?? '').toLowerCase()
}
