import { readFile, writeFile, mkdir } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const directory = dirname(fileURLToPath(import.meta.url))
const output = resolve(directory, 'build/mcai.bundle.js')
const banner = `/* MCAI EaglerForge 1.12 adapter. Contains no Minecraft/Eaglercraft code or assets. */\n`
const runtime = await readFile(resolve(directory, 'flat-policy.js'), 'utf8')
const coordinates = await readFile(resolve(directory, 'coordinate-contract.js'), 'utf8')
const adapter = await readFile(resolve(directory, 'mcai.mod.js'), 'utf8')
await mkdir(dirname(output), { recursive: true })
await writeFile(output, `${banner}${runtime}\n${coordinates}\n${adapter}\n`, 'utf8')
console.log(output)
