import type { ItemState } from './contracts.js'

export const EMPTY_ITEM: ItemState = Object.freeze({
  name: '',
  count: 0,
  durability: 0,
  max_durability: 0,
  enchant_hash: 0
})

export function itemState(item: any): ItemState {
  if (!item) return { ...EMPTY_ITEM }
  const name = String(item.name ?? item.displayName ?? '')
  const max = numberOr(item.maxDurability, 0)
  const used = numberOr(item.durabilityUsed, 0)
  return {
    name,
    count: Math.max(0, Math.trunc(numberOr(item.count, 0))),
    durability: max > 0 ? Math.max(0, max - used) : 0,
    max_durability: max,
    enchant_hash: enchantHash(item, name)
  }
}

/**
 * Compatibility fallback for legacy fixtures/clients that expose no enchant
 * list at all. Live Prismarine items use their actual NBT-backed `enchants`
 * getter. The armor fallback describes the standard non-crystal kit; crystal
 * Prot II is preserved whenever the live enchant list is available.
 */
export function knownKitEnchantHash(name: string): number {
  let signature = ''
  if (name === 'diamond_sword') signature = 'sharpness:5'
  else if (name === 'diamond_pickaxe') signature = 'efficiency:5'
  else if (name.startsWith('diamond_') && /helmet|chestplate|leggings|boots/.test(name)) {
    signature = 'protection:4|unbreaking:3'
  }
  return signature ? hashString(signature) : 0
}

function enchantHash(item: any, name: string): number {
  const actualSignature = actualEnchantSignature(item)
  if (actualSignature !== null) return actualSignature ? hashString(actualSignature) : 0
  return knownKitEnchantHash(name)
}

/** Null means unavailable; an empty string means the item is authoritatively unenchanted. */
function actualEnchantSignature(item: any): string | null {
  if (!item || !('enchants' in Object(item))) return null
  let enchants: unknown
  try {
    enchants = item.enchants
  } catch {
    return null
  }
  if (!Array.isArray(enchants)) return null
  return enchants
    .map(entry => normalizeEnchant(entry))
    .filter((entry): entry is string => entry !== null)
    .sort((left, right) => left < right ? -1 : left > right ? 1 : 0)
    .join('|')
}

function normalizeEnchant(entry: unknown): string | null {
  if (!entry || typeof entry !== 'object') return null
  const value = entry as Record<string, unknown>
  const normalizedName = String(value.name ?? '')
    .trim()
    .toLowerCase()
    .replace(/^minecraft:/, '')
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
  const rawLevel = value.lvl ?? value.level
  const level = typeof rawLevel === 'number'
    ? rawLevel
    : (typeof rawLevel === 'string' && rawLevel.trim() ? Number(rawLevel) : Number.NaN)
  if (!normalizedName || !Number.isFinite(level) || level <= 0) return null
  return `${normalizedName}:${Math.trunc(level)}`
}

function hashString(value: string): number {
  let hash = 0x811c9dc5
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i)
    hash = Math.imul(hash, 0x01000193)
  }
  return hash | 0
}

function numberOr(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}
