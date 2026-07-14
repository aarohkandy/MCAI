const elements = Object.fromEntries([
  'connection-dot', 'connection-text', 'policy', 'phase', 'ticks', 'device', 'tps', 'tick-time', 'memory',
  'pairs', 'progress-value', 'progress-bar', 'arena-title', 'arena-subtitle', 'arena-select', 'fighters',
  'policy-loss', 'value-loss', 'entropy', 'kl', 'pause', 'resume', 'command-result', 'arena-canvas'
].map(id => [id, document.getElementById(id)]))
const canvas = elements['arena-canvas']
const context = canvas.getContext('2d')
let latestState = null
let selectedArena = ''

function number(value, digits = 3) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '—'
}
function integer(value) { return Number(value || 0).toLocaleString() }
function bytes(value) {
  if (!Number.isFinite(Number(value))) return '—'
  return `${(Number(value) / 1073741824).toFixed(1)} GB`
}

function render(state) {
  latestState = state
  const connected = state.arena_connected
  elements['connection-dot'].className = `dot ${connected ? 'live' : 'offline'}`
  elements['connection-text'].textContent = connected ? 'Live' : 'Connecting to arena…'
  const trainer = state.trainer || {}
  elements.policy.textContent = `v${trainer.policy_version || 0}`
  elements.phase.textContent = trainer.phase === 'updating' ? 'learning from batch' : trainer.phase || 'starting'
  elements.ticks.textContent = integer(trainer.total_agent_ticks)
  elements.device.textContent = `${String(trainer.device || 'cpu').toUpperCase()} · ${integer(trainer.parameters)} params`
  const arena = state.arena || {}
  elements.tps.textContent = arena.estimated_tps ? `${Number(arena.estimated_tps).toFixed(1)} TPS` : '— TPS'
  elements['tick-time'].textContent = arena.p95_tick_ms ? `p95 ${Number(arena.p95_tick_ms).toFixed(1)} ms` : 'waiting for Paper'
  elements.memory.textContent = bytes((state.system?.total_memory_bytes || 0) - (state.system?.free_memory_bytes || 0))
  elements.pairs.textContent = `${arena.active_pairs || 0} active pair${arena.active_pairs === 1 ? '' : 's'} · ${state.system?.cpu_threads || 0} threads`
  const target = trainer.target_agent_ticks || 8192
  const collected = Math.min(target, trainer.collected_agent_ticks || 0)
  elements['progress-value'].textContent = trainer.phase === 'updating' ? 'Optimizing policy…' : `${integer(collected)} / ${integer(target)}`
  elements['progress-bar'].style.width = `${Math.max(0, Math.min(100, collected / target * 100))}%`
  const update = trainer.last_update || {}
  elements['policy-loss'].textContent = number(update.policy_loss)
  elements['value-loss'].textContent = number(update.value_loss)
  elements.entropy.textContent = number(update.entropy)
  elements.kl.textContent = number(update.approximate_kl, 5)
  updateArenaChoices(state.snapshots || [])
  const snapshot = (state.snapshots || []).find(value => value.arena_id === selectedArena) || state.snapshots?.[0]
  drawArena(snapshot)
}

function updateArenaChoices(snapshots) {
  const ids = snapshots.map(value => value.arena_id)
  if (!ids.includes(selectedArena)) selectedArena = ids[0] || ''
  const current = [...elements['arena-select'].options].map(option => option.value)
  if (current.join('|') !== ids.join('|')) {
    elements['arena-select'].replaceChildren(...ids.map(id => {
      const option = document.createElement('option'); option.value = id; option.textContent = id; return option
    }))
  }
  elements['arena-select'].value = selectedArena
  elements['arena-select'].disabled = ids.length < 2
}

function drawArena(snapshot) {
  resizeCanvas()
  const width = canvas.clientWidth
  const height = canvas.clientHeight
  context.clearRect(0, 0, width, height)
  const styles = getComputedStyle(document.documentElement)
  const color = name => styles.getPropertyValue(name).trim()
  context.fillStyle = '#0c1410'
  context.fillRect(0, 0, width, height)
  if (!snapshot) {
    context.fillStyle = color('--muted')
    context.textAlign = 'center'; context.textBaseline = 'middle'; context.font = '15px ui-monospace, monospace'
    context.fillText('Waiting for the first match…', width / 2, height / 2)
    elements['arena-title'].textContent = 'Waiting for fighters'
    elements['arena-subtitle'].textContent = 'The first arena appears after the bots join.'
    elements.fighters.replaceChildren()
    return
  }
  const size = Number(snapshot.arena_size || 21)
  const padding = 28
  const scale = Math.min((width - padding * 2) / size, (height - padding * 2) / size)
  const left = (width - size * scale) / 2
  const top = (height - size * scale) / 2
  const toX = x => left + (Number(x) + size / 2) * scale
  const toY = z => top + (Number(z) + size / 2) * scale
  context.strokeStyle = color('--grid'); context.lineWidth = 1
  for (let i = 0; i <= size; i++) {
    context.beginPath(); context.moveTo(left + i * scale, top); context.lineTo(left + i * scale, top + size * scale); context.stroke()
    context.beginPath(); context.moveTo(left, top + i * scale); context.lineTo(left + size * scale, top + i * scale); context.stroke()
  }
  context.strokeStyle = color('--green'); context.lineWidth = 2
  context.strokeRect(left, top, size * scale, size * scale)
  for (const block of snapshot.blocks || []) {
    context.fillStyle = block.type === 'obsidian' ? '#4f3f66' : '#667067'
    context.fillRect(toX(block.x) - scale / 2, toY(block.z) - scale / 2, Math.max(2, scale), Math.max(2, scale))
  }
  for (const entity of snapshot.entities || []) {
    const x = toX(entity.x), y = toY(entity.z), radius = Math.max(4, scale * .3)
    context.save(); context.translate(x, y); context.rotate(Math.PI / 4)
    context.fillStyle = color('--gold'); context.fillRect(-radius, -radius, radius * 2, radius * 2); context.restore()
  }
  const fighterColors = [color('--green'), color('--blue')]
  ;(snapshot.fighters || []).forEach((fighter, index) => {
    const x = toX(fighter.x), y = toY(fighter.z), radius = Math.max(7, scale * .38)
    const direction = (Number(fighter.yaw) + 90) * Math.PI / 180
    context.strokeStyle = fighterColors[index % fighterColors.length]; context.lineWidth = 3
    context.beginPath(); context.moveTo(x, y); context.lineTo(x + Math.cos(direction) * radius * 2.1, y + Math.sin(direction) * radius * 2.1); context.stroke()
    context.fillStyle = fighterColors[index % fighterColors.length]
    context.beginPath(); context.arc(x, y, radius, 0, Math.PI * 2); context.fill()
    context.fillStyle = color('--text'); context.font = '12px ui-monospace, monospace'; context.textAlign = 'center'; context.textBaseline = 'bottom'
    context.fillText(fighter.name, x, y - radius - 5)
  })
  const seconds = Math.ceil(Number(snapshot.remaining_ticks || 0) / 20)
  elements['arena-title'].textContent = `${snapshot.arena_id} · ${String(snapshot.mode || 'sword').toUpperCase()}`
  elements['arena-subtitle'].textContent = `${seconds}s remaining · seed ${snapshot.arena_seed}`
  renderFighters(snapshot.fighters || [], fighterColors)
}

function renderFighters(fighters, colors) {
  elements.fighters.replaceChildren(...fighters.map((fighter, index) => {
    const node = document.createElement('article'); node.className = 'fighter'; node.style.borderColor = colors[index % colors.length]
    const line = document.createElement('div'); line.className = 'fighter-line'
    const name = document.createElement('strong'); name.textContent = fighter.name
    const health = document.createElement('span'); health.textContent = `${Number(fighter.health || 0).toFixed(1)} HP`
    line.append(name, health)
    const details = document.createElement('small')
    const stats = fighter.stats || {}
    details.textContent = `${Number(stats.damage_dealt || 0).toFixed(1)} damage · ${stats.invalid_interactions || 0} invalid`
    const track = document.createElement('div'); track.className = 'health-track'
    const fill = document.createElement('div'); fill.className = 'health-fill'; fill.style.width = `${Math.max(0, Math.min(100, Number(fighter.health || 0) / 20 * 100))}%`
    track.append(fill); node.append(line, details, track); return node
  }))
}

function resizeCanvas() {
  const ratio = window.devicePixelRatio || 1
  const width = Math.max(1, Math.floor(canvas.clientWidth * ratio))
  const height = Math.max(1, Math.floor(canvas.clientHeight * ratio))
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width; canvas.height = height; context.setTransform(ratio, 0, 0, ratio, 0, 0)
  }
}

async function command(name) {
  elements['command-result'].textContent = 'Sending…'
  try {
    const response = await fetch('/api/command', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ command: name }) })
    const result = await response.json()
    if (!result.ok) throw new Error(result.error)
    elements['command-result'].textContent = name === 'stop_all' ? 'Fights stopped; all controls released.' : 'Fights resumed.'
  } catch (error) { elements['command-result'].textContent = `Could not send command: ${error.message || error}` }
}

elements['arena-select'].addEventListener('change', event => { selectedArena = event.target.value; render(latestState) })
elements.pause.addEventListener('click', () => command('stop_all'))
elements.resume.addEventListener('click', () => command('resume'))
new ResizeObserver(() => { if (latestState) render(latestState) }).observe(canvas)
const stream = new EventSource('/api/events')
stream.onmessage = event => { try { render(JSON.parse(event.data)) } catch { } }
stream.onerror = () => { elements['connection-dot'].className = 'dot offline'; elements['connection-text'].textContent = 'Dashboard reconnecting…' }
