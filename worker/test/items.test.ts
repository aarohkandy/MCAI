import { describe, expect, it } from 'vitest'
import { itemState, knownKitEnchantHash } from '../src/items.js'

describe('portable item features', () => {
  it('uses a client-independent signature for the fixed evaluation kit', () => {
    expect(itemState({ name: 'diamond_sword', count: 1 }).enchant_hash)
      .toBe(knownKitEnchantHash('diamond_sword'))
    expect(knownKitEnchantHash('diamond_sword')).not.toBe(knownKitEnchantHash('diamond_pickaxe'))
  })
})
