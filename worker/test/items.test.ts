import { describe, expect, it } from 'vitest'
import { itemState, knownKitEnchantHash } from '../src/items.js'

describe('portable item features', () => {
  it('uses a client-independent signature for the fixed evaluation kit', () => {
    expect(itemState({ name: 'diamond_sword', count: 1 }).enchant_hash)
      .toBe(knownKitEnchantHash('diamond_sword'))
    expect(knownKitEnchantHash('diamond_sword')).not.toBe(knownKitEnchantHash('diamond_pickaxe'))
  })

  it('hashes actual enchantments independently of packet/NBT order', () => {
    const first = itemState({
      name: 'diamond_sword', count: 1,
      enchants: [
        { name: 'minecraft:Sharpness', lvl: 5 },
        { name: 'unbreaking', lvl: 3 }
      ]
    })
    const reversed = itemState({
      name: 'diamond_sword', count: 1,
      enchants: [
        { name: 'UNBREAKING', level: '3' },
        { name: 'sharpness', lvl: 5 }
      ]
    })
    expect(first.enchant_hash).toBe(reversed.enchant_hash)
  })

  it('distinguishes the actual crystal Prot II kit from the Prot IV kit', () => {
    const armor = (level: number) => itemState({
      name: 'diamond_chestplate', count: 1,
      enchants: [
        { name: 'unbreaking', lvl: 3 },
        { name: 'protection', lvl: level }
      ]
    }).enchant_hash
    expect(armor(2)).not.toBe(armor(4))
  })

  it('does not fabricate fallback enchants when an authoritative empty list exists', () => {
    expect(itemState({ name: 'diamond_sword', count: 1, enchants: [] }).enchant_hash).toBe(0)
    expect(itemState({
      name: 'diamond_sword', count: 1,
      enchants: [{ name: 'sharpness', lvl: 5 }]
    }).enchant_hash).toBe(knownKitEnchantHash('diamond_sword'))
    expect(itemState({
      name: 'diamond_sword', count: 1,
      enchants: [{ name: 'sharpness', lvl: 5 }, { name: 'knockback', lvl: 1 }]
    }).enchant_hash).not.toBe(knownKitEnchantHash('diamond_sword'))
  })
})
