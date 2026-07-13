import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import vm from 'node:vm'

const source = await readFile(new URL('../coordinate-contract.js', import.meta.url), 'utf8')
const context = vm.createContext({})
vm.runInContext(source, context)
const coordinates = context.MCAICoordinates

assert.ok(Math.abs(coordinates.canonicalYaw(0) - Math.PI) < 1e-12)
assert.ok(Math.abs(coordinates.canonicalYaw(90) - Math.PI / 2) < 1e-12)
assert.ok(Math.abs(coordinates.canonicalPitch(-30) - Math.PI / 6) < 1e-12)
assert.ok(Math.abs(coordinates.minecraftYawDelta(0.25) + 0.25 * 180 / Math.PI) < 1e-12)
assert.ok(Math.abs(coordinates.minecraftPitchDelta(-0.5) - 0.5 * 180 / Math.PI) < 1e-12)
