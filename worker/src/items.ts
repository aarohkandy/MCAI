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
    enchant_hash: knownKitEnchantHash(name)
  }
}

export function knownKitEnchantHash(name: string): number {
  let signature = ''
  if (name === 'diamond_sword') signature = 'knockback:1|sharpness:5'
  else if (name === 'diamond_pickaxe') signature = 'efficiency:5'
  else if (name.startsWith('diamond_') && /helmet|chestplate|leggings|boots/.test(name)) {
    signature = 'protection:4|unbreaking:3'
  }
  return signature ? hashString(signature) : 0
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
