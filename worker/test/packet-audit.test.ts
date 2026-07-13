import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

describe('gameplay packet allowlist', () => {
  it('contains exactly the documented vanilla swap-hands raw packet', () => {
    const source = readFileSync(new URL('../src/legal-controls.ts', import.meta.url), 'utf8')
    const writes = source.match(/_client\.write\(/g) ?? []
    expect(writes).toHaveLength(1)
    expect(source).toContain("_client.write('block_dig'")
    expect(source).toMatch(/status:\s*6/)
  })

  it('does not expose a position or velocity mutation path', () => {
    const source = readFileSync(new URL('../src/legal-controls.ts', import.meta.url), 'utf8')
    expect(source).not.toMatch(/entity\.position\s*=/)
    expect(source).not.toMatch(/entity\.velocity\s*=/)
    expect(source).not.toContain("_client.write('position'")
    expect(source).not.toContain("_client.write('position_look'")
  })
})
