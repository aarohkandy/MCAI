import { readFile } from 'node:fs/promises'
import { pathToFileURL } from 'node:url'

const [runtimePath, manifestPath, weightsPath, observationPath, expectedPath] = process.argv.slice(2)
if (!expectedPath) throw new Error('usage: flat-parity runtime manifest weights observation expected')
await import(pathToFileURL(runtimePath))
const manifest = JSON.parse(await readFile(manifestPath, 'utf8'))
const weightsBytes = await readFile(weightsPath)
const buffer = weightsBytes.buffer.slice(weightsBytes.byteOffset, weightsBytes.byteOffset + weightsBytes.byteLength)
const observation = JSON.parse(await readFile(observationPath, 'utf8'))
const expected = JSON.parse(await readFile(expectedPath, 'utf8'))
const policy = new globalThis.MCAIFlatRuntime.FlatPolicy(manifest, buffer)
const actual = policy.step(observation)
let maximumDifference = 0
compare(actual.value, expected.value, 'value')
compare(Array.from(actual.camera_mean), expected.camera_mean, 'camera_mean')
compare(Array.from(actual.hidden), expected.hidden, 'hidden')
for (const [name, values] of Object.entries(expected.logits)) compare(Array.from(actual.logits[name]), values, `logits.${name}`)
if (maximumDifference > 1e-5) throw new Error(`flat policy parity exceeded 1e-5: ${maximumDifference}`)
console.log(JSON.stringify({ flat_policy_maximum_difference: maximumDifference }))

function compare(actualValue, expectedValue, label) {
  if (Array.isArray(expectedValue)) {
    if (actualValue.length !== expectedValue.length) throw new Error(`${label} length differs`)
    for (let index = 0; index < expectedValue.length; index++) compare(actualValue[index], expectedValue[index], `${label}[${index}]`)
    return
  }
  const difference = Math.abs(Number(actualValue) - Number(expectedValue))
  if (!Number.isFinite(difference)) throw new Error(`${label} is not finite`)
  maximumDifference = Math.max(maximumDifference, difference)
}
