import http from 'node:http'
import net from 'node:net'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const directory = path.dirname(fileURLToPath(import.meta.url))
const publicDirectory = path.join(directory, 'public')
const runDirectory = process.env.MCAI_RUN_DIR || path.join(path.dirname(directory), 'runs')
const port = Number.parseInt(process.env.MCAI_DASHBOARD_PORT || '8788', 10)
const arenaPort = Number.parseInt(process.env.MCAI_ARENA_PORT || '8765', 10)
const startedAt = new Date().toISOString()
const clients = new Set()
const snapshots = new Map()
let arenaStatus = null
let arenaConnected = false
let trainer = { phase: 'starting', policy_version: 0, total_agent_ticks: 0, collected_agent_ticks: 0,
  target_agent_ticks: Number.parseInt(process.env.MCAI_ROLLOUT_STEPS || '8192', 10), updates: [] }
let sequence = 0
let socket = null
let socketBuffer = ''
const pending = new Map()
let broadcastTimer = null

function connectArena() {
  if (socket && !socket.destroyed) return
  socket = net.createConnection({ host: '127.0.0.1', port: arenaPort })
  socket.setEncoding('utf8')
  socket.on('connect', () => { arenaConnected = true; scheduleBroadcast() })
  socket.on('data', onArenaData)
  socket.on('error', () => undefined)
  socket.on('close', () => {
    arenaConnected = false
    socket = null
    for (const request of pending.values()) request.reject(new Error('arena disconnected'))
    pending.clear()
    scheduleBroadcast()
    setTimeout(connectArena, 1500).unref()
  })
}

function onArenaData(data) {
  socketBuffer += data
  for (;;) {
    const newline = socketBuffer.indexOf('\n')
    if (newline < 0) break
    const line = socketBuffer.slice(0, newline).trim()
    socketBuffer = socketBuffer.slice(newline + 1)
    if (!line) continue
    try {
      const message = JSON.parse(line)
      if (message.type === 'response' && pending.has(message.id)) {
        const request = pending.get(message.id)
        clearTimeout(request.timeout)
        pending.delete(message.id)
        if (message.ok) request.resolve(message.payload || {})
        else request.reject(new Error(String(message.error || 'command failed')))
      } else if (message.type === 'event' && message.event === 'arena_snapshot') {
        snapshots.set(message.arena_id, { ...message.payload, arena_id: message.arena_id, received_at: Date.now() })
        scheduleBroadcast()
      } else if (message.type === 'event' && message.event === 'match_ended') {
        const old = snapshots.get(message.arena_id)
        if (old) snapshots.set(message.arena_id, { ...old, ended: true, end_reason: message.payload?.reason })
      }
    } catch { }
  }
}

function arenaCommand(command, payload = {}) {
  return new Promise((resolve, reject) => {
    if (!socket || socket.destroyed || !arenaConnected) return reject(new Error('arena is not ready'))
    const id = ++sequence
    const timeout = setTimeout(() => { pending.delete(id); reject(new Error('command timed out')) }, 5000)
    pending.set(id, { resolve, reject, timeout })
    socket.write(`${JSON.stringify({ type: 'command', id, command, payload })}\n`)
  })
}

async function refreshArenaStatus() {
  try { arenaStatus = await arenaCommand('status') } catch { arenaStatus = null }
  const cutoff = Date.now() - 5000
  for (const [id, value] of snapshots) if (value.received_at < cutoff) snapshots.delete(id)
  scheduleBroadcast()
}

function readTail(file, maximum = 768 * 1024) {
  try {
    const stat = fs.statSync(file)
    const length = Math.min(stat.size, maximum)
    const descriptor = fs.openSync(file, 'r')
    const buffer = Buffer.alloc(length)
    fs.readSync(descriptor, buffer, 0, length, stat.size - length)
    fs.closeSync(descriptor)
    return buffer.toString('utf8')
  } catch { return '' }
}

function refreshTrainer() {
  const lines = readTail(path.join(runDirectory, 'trainer.log')).split(/\r?\n/)
  const events = []
  for (let index = 0; index < lines.length; index++) {
    try {
      const value = JSON.parse(lines[index])
      if (value && typeof value.event === 'string') events.push({ ...value, _order: index })
    } catch { }
  }
  const latest = name => events.filter(event => event.event === name).at(-1)
  const ready = latest('trainer_ready')
  const progress = latest('rollout_progress')
  const training = latest('ppo_training_started')
  const update = latest('ppo_update')
  const updates = events.filter(event => event.event === 'ppo_update').slice(-40)
  const newest = [progress, training, update].filter(Boolean).sort((a, b) => a._order - b._order).at(-1)
  let phase = ready ? 'collecting' : 'starting'
  let collected = 0
  let target = trainer.target_agent_ticks
  if (newest?.event === 'rollout_progress') {
    phase = 'collecting'
    collected = newest.collected_agent_ticks || 0
    target = newest.target_agent_ticks || target
  } else if (newest?.event === 'ppo_training_started') {
    phase = 'updating'
    collected = newest.batch_agent_ticks || target
  }
  trainer = {
    phase,
    device: ready?.device || trainer.device || 'cpu',
    parameters: ready?.parameters || trainer.parameters || 0,
    policy_version: update?.policy_version ?? progress?.policy_version ?? ready?.policy_version ?? 0,
    total_agent_ticks: update?.total_agent_ticks ?? progress?.total_agent_ticks ?? 0,
    collected_agent_ticks: collected,
    target_agent_ticks: target,
    last_update: update || null,
    updates: updates.map(({ _order, ...value }) => value)
  }
  scheduleBroadcast()
}

function currentState() {
  return {
    now: new Date().toISOString(),
    started_at: startedAt,
    arena_connected: arenaConnected,
    arena: arenaStatus,
    snapshots: [...snapshots.values()].sort((a, b) => a.arena_id.localeCompare(b.arena_id)),
    trainer,
    system: {
      hostname: os.hostname(),
      cpu_threads: os.cpus().length,
      total_memory_bytes: os.totalmem(),
      free_memory_bytes: os.freemem(),
      uptime_seconds: os.uptime()
    }
  }
}

function broadcast() {
  broadcastTimer = null
  const frame = `data: ${JSON.stringify(currentState())}\n\n`
  for (const response of clients) response.write(frame)
}

function scheduleBroadcast() {
  if (!broadcastTimer) broadcastTimer = setTimeout(broadcast, 90)
}

function sendJson(response, status, value) {
  const body = JSON.stringify(value)
  response.writeHead(status, { 'content-type': 'application/json; charset=utf-8', 'content-length': Buffer.byteLength(body),
    'cache-control': 'no-store' })
  response.end(body)
}

function serveFile(response, file, contentType) {
  try {
    const body = fs.readFileSync(path.join(publicDirectory, file))
    response.writeHead(200, { 'content-type': contentType, 'content-length': body.length,
      'cache-control': 'no-cache',
      'content-security-policy': "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'" })
    response.end(body)
  } catch { response.writeHead(404).end('Not found') }
}

const server = http.createServer(async (request, response) => {
  const url = new URL(request.url || '/', `http://127.0.0.1:${port}`)
  if (request.method === 'GET' && url.pathname === '/') return serveFile(response, 'index.html', 'text/html; charset=utf-8')
  if (request.method === 'GET' && url.pathname === '/app.css') return serveFile(response, 'app.css', 'text/css; charset=utf-8')
  if (request.method === 'GET' && url.pathname === '/app.js') return serveFile(response, 'app.js', 'text/javascript; charset=utf-8')
  if (request.method === 'GET' && url.pathname === '/api/state') return sendJson(response, 200, currentState())
  if (request.method === 'GET' && url.pathname === '/api/events') {
    response.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' })
    clients.add(response)
    response.write(`data: ${JSON.stringify(currentState())}\n\n`)
    request.on('close', () => clients.delete(response))
    return
  }
  if (request.method === 'POST' && url.pathname === '/api/command') {
    let body = ''
    request.on('data', chunk => { body += chunk; if (body.length > 4096) request.destroy() })
    request.on('end', async () => {
      try {
        const input = JSON.parse(body || '{}')
        if (!['stop_all', 'resume'].includes(input.command)) throw new Error('command is not allowed')
        sendJson(response, 200, { ok: true, payload: await arenaCommand(input.command) })
      } catch (error) { sendJson(response, 400, { ok: false, error: String(error.message || error) }) }
    })
    return
  }
  response.writeHead(404).end('Not found')
})

server.listen(port, '127.0.0.1', () => {
  console.log(JSON.stringify({ event: 'dashboard_ready', url: `http://127.0.0.1:${port}`, run_directory: runDirectory }))
})
connectArena()
setInterval(refreshArenaStatus, 1000).unref()
setInterval(refreshTrainer, 1000).unref()
refreshTrainer()

for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => server.close(() => process.exit(0)))
