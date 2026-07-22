import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import fs from 'node:fs'
import net from 'node:net'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
const dashboard = path.resolve(here, '..', 'server.mjs')
const run = fs.mkdtempSync(path.join(os.tmpdir(), 'mcai-dashboard-'))
const arenaPort = 18765 + Math.floor(Math.random() * 500)
const dashboardPort = arenaPort + 600
let emergencyStopped = false

fs.writeFileSync(path.join(run, 'trainer.log'), [
  JSON.stringify({ event: 'trainer_ready', device: 'cpu', policy_version: 2, parameters: 296000 }),
  JSON.stringify({ event: 'rollout_progress', policy_version: 2, collected_agent_ticks: 2048,
    target_agent_ticks: 8192, total_agent_ticks: 16384 }),
  JSON.stringify({ event: 'ppo_update', policy_version: 2, total_agent_ticks: 16384,
    policy_loss: -0.1, value_loss: 0.2, entropy: 1.3, approximate_kl: 0.004,
    reward_mean_raw: 0.02, reward_mean_training: 0.015, reward_clipped_transitions: 3 })
].join('\n') + '\n')

const mock = net.createServer(socket => {
  socket.setEncoding('utf8')
  socket.on('error', () => undefined)
  let buffer = ''
  const snapshot = setInterval(() => socket.write(`${JSON.stringify({
    type: 'event', event: 'arena_snapshot', arena_id: 'arena-1', payload: {
      episode_id: 'smoke', arena_seed: 7, mode: 'sword', arena_size: 21,
      remaining_ticks: 1200, fighters: [
        { name: 'MCAI_001', x: -2, y: 1, z: 0, yaw: 90, health: 18, stats: { damage_dealt: 2 } },
        { name: 'MCAI_002', x: 2, y: 1, z: 0, yaw: -90, health: 16, stats: { damage_dealt: 4 } }
      ], entities: [], blocks: []
    }
  })}\n`), 100)
  socket.on('data', data => {
    buffer += data
    for (;;) {
      const newline = buffer.indexOf('\n')
      if (newline < 0) break
      const line = buffer.slice(0, newline).trim(); buffer = buffer.slice(newline + 1)
      if (!line) continue
      const request = JSON.parse(line)
      if (request.command === 'stop_all') emergencyStopped = true
      const payload = request.command === 'status'
        ? { estimated_tps: 20, p95_tick_ms: 49, active_pairs: 1, max_concurrent_pairs: 2, mode: 'sword' }
        : { accepted: request.command }
      socket.write(`${JSON.stringify({ type: 'response', id: request.id, ok: true, payload })}\n`)
    }
  })
  socket.on('close', () => clearInterval(snapshot))
})

await new Promise((resolve, reject) => { mock.once('error', reject); mock.listen(arenaPort, '127.0.0.1', resolve) })
const child = spawn(process.execPath, [dashboard], {
  env: { ...process.env, MCAI_RUN_DIR: run, MCAI_ARENA_PORT: String(arenaPort),
    MCAI_DASHBOARD_PORT: String(dashboardPort), MCAI_ROLLOUT_STEPS: '8192' },
  stdio: ['ignore', 'pipe', 'pipe']
})

async function waitForState() {
  for (let attempt = 0; attempt < 60; attempt++) {
    try {
      const response = await fetch(`http://127.0.0.1:${dashboardPort}/api/state`)
      const state = await response.json()
      if (state.arena_connected && state.arena && state.snapshots.length) return state
    } catch { }
    await new Promise(resolve => setTimeout(resolve, 100))
  }
  throw new Error('dashboard did not become ready')
}

try {
  const state = await waitForState()
  assert.equal(state.trainer.policy_version, 2)
  assert.equal(state.trainer.total_agent_ticks, 16384)
  assert.equal(state.trainer.last_update.reward_mean_training, 0.015)
  assert.equal(state.arena.active_pairs, 1)
  assert.equal(state.snapshots[0].fighters.length, 2)
  const page = await fetch(`http://127.0.0.1:${dashboardPort}/`).then(response => response.text())
  assert.match(page, /MCAI Live Training/)
  assert.match(page, /Mean reward/)
  assert.match(page, /Adaptive rewards/)
  const command = await fetch(`http://127.0.0.1:${dashboardPort}/api/command`, {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ command: 'stop_all' })
  }).then(response => response.json())
  assert.equal(command.ok, true)
  assert.equal(emergencyStopped, true)
  console.log('dashboard smoke test passed')
} finally {
  child.kill('SIGTERM')
  mock.close()
  fs.rmSync(run, { recursive: true, force: true })
}
