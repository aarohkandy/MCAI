(function MCAIEaglerAdapter(global) {
  'use strict'

  if (!global.ModAPI) throw new Error('MCAI requires an EaglerForgeInjector client')
  if (!global.MCAICoordinates) throw new Error('MCAI coordinate contract must load before the adapter')
  const API = global.ModAPI
  const Coordinates = global.MCAICoordinates
  if (!API.is_1_12) throw new Error('MCAI currently supports only legitimate Eaglercraft 1.12 builds')
  API.meta.title('MCAI Combat Adapter')
  API.meta.version('0.1.0')
  API.meta.description('Structured-state combat policy adapter using only ordinary game controls. F8 is emergency stop.')
  API.meta.credits('MCAI project')
  for (const component of ['player', 'world', 'network', 'resolution']) API.require(component)

  const config = Object.assign({
    bridgeUrl: 'ws://127.0.0.1:8767', enabled: false, overlay: true,
    episodeId: 'browser-session', arenaSeed: 0, actionDelayTicks: 0, observationDelayTicks: 0
  }, global.MCAI_CONFIG || {})
  const EMPTY_ITEM = Object.freeze({ name: '', count: 0, durability: 0, max_durability: 0, enchant_hash: 0 })
  const state = {
    tick: 0, sequence: 0, policyVersion: 0, socket: null, connected: false, pending: false,
    flatPolicy: null, lastAction: noopAction(), currentAction: noopAction(),
    lastYaw: 0, lastPitch: 0, lastHotbar: -1, lastAttackTick: -1000,
    activeHand: 'none', useStartedTick: 0, blockCache: [], blockSampleTick: -100,
    entityBorn: new Map(), recording: false, demonstration: [],
    matchId: makeId(), target: null, value: 0, latencyMs: 0, lastRequestMs: 0, error: '',
    overlayElement: null, lastPrimary: 'none', observationHistory: [], actionQueue: [],
    lastSelfHealth: null, lastOpponentHealth: null,
    rewardComponents: { opponentDamage: 0, selfDamage: 0, total: 0 }
  }

  const reflected = {}
  API.addEventListener('load', initialize)
  API.addEventListener('update', update)
  API.addEventListener('sendchatmessage', chatCommand)
  API.addEventListener('key', emergencyKey)

  function initialize() {
    try {
      reflected.BlockPos = API.reflect.getClassById('net.minecraft.util.math.BlockPos')
        || API.reflect.getClassById('net.minecraft.util.BlockPos')
      reflected.BlockPosConstructor = reflected.BlockPos.constructors.find(constructor => constructor.length === 3)
        || reflected.BlockPos.constructors[0]
    } catch (error) {
      state.error = `Block observation unavailable: ${error.message}`
    }
    try {
      const controllerClass = API.reflect.getClassById('net.minecraft.client.multiplayer.PlayerControllerMP')
      const handClass = API.reflect.getClassById('net.minecraft.util.EnumHand')
      reflected.ProcessRightClick = controllerClass.methods.processRightClick.method
      reflected.OffHand = handClass.staticVariables.OFF_HAND
    } catch (error) {
      state.error = `${state.error ? `${state.error}; ` : ''}explicit offhand use unavailable: ${error.message}`
    }
    document.addEventListener('keydown', browserEmergencyKey, true)
    installOverlay()
    API.displayToChat('MCAI loaded disabled. Use .mcai on, .mcai record, or F8 emergency stop.')
  }

  function update() {
    state.tick++
    if (!API.player || !API.world) return
    const player = corrected(API.player)
    if (!player) return
    const currentObservation = buildObservation(player)
    updateRewardComponents(currentObservation)
    const observation = delayedObservation(currentObservation)
    if (state.recording) recordHumanAction(currentObservation, player)
    if (config.enabled) {
      if (state.flatPolicy) {
        const result = state.flatPolicy.step(observation)
        queuePolicyAction(result.action)
        state.value = result.value
      } else {
        ensureBridge()
        if (state.connected && !state.pending) sendBridgeStep(observation)
      }
      advanceActionQueue()
      applyAction(player, state.currentAction)
    } else {
      releaseControls()
    }
    updateOverlay(player)
    state.lastYaw = canonicalYaw(player.rotationYaw)
    state.lastPitch = canonicalPitch(player.rotationPitch)
    state.lastHotbar = number(player.inventory && player.inventory.currentItem, 0)
  }

  function buildObservation(player) {
    const selfPosition = position(player)
    const selfVelocity = velocity(player)
    const opponentEntity = nearestOpponent(player)
    const opponent = opponentEntity ? opponentState(player, opponentEntity, selfPosition, selfVelocity) : null
    const currentRaycast = raycast(player)
    return {
      schema_version: 1,
      match: {
        episode_id: config.episodeId, tick: state.tick, policy_version: state.policyVersion,
        arena_seed: config.arenaSeed, action_delay_ticks: config.actionDelayTicks,
        observation_delay_ticks: config.observationDelayTicks
      },
      self: {
        health: callNumber(player, 'getHealth', player.health || 0),
        absorption: callNumber(player, 'getAbsorptionAmount', 0),
        food: foodLevel(player), position: selfPosition, velocity: selfVelocity,
        yaw: canonicalYaw(player.rotationYaw), pitch: canonicalPitch(player.rotationPitch),
        on_ground: bool(player.onGround), sprinting: callBoolean(player, 'isSprinting', false),
        sneaking: callBoolean(player, 'isSneaking', false), hurt_time: number(player.hurtTime, 0),
        attack_cooldown: attackCooldown(player), active_hand: activeHand(player),
        use_ticks: state.activeHand === 'none' ? 0 : state.tick - state.useStartedTick,
        mining_progress: miningProgress(), selected_hotbar: number(player.inventory && player.inventory.currentItem, 0),
        hotbar: hotbar(player), offhand: itemState(call(player, 'getHeldItemOffhand')),
        armor: armor(player), raycast: currentRaycast
      },
      opponent,
      entities: entitySlots(player, selfPosition, selfVelocity),
      blocks: blockSlots(player, selfPosition, opponentEntity ? position(opponentEntity) : null),
      action_mask: actionMask(player)
    }
  }

  function opponentState(player, opponent, selfPosition, selfVelocity) {
    const opponentPosition = position(opponent)
    return {
      relative_position: egocentric(subtract(opponentPosition, selfPosition), canonicalYaw(player.rotationYaw)),
      relative_velocity: egocentric(subtract(velocity(opponent), selfVelocity), canonicalYaw(player.rotationYaw)),
      yaw: canonicalYaw(opponent.rotationYaw), pitch: canonicalPitch(opponent.rotationPitch),
      health: hasMethod(opponent, 'getHealth') ? callNumber(opponent, 'getHealth', 0) : null,
      hurt_time: number(opponent.hurtTime, 0), on_ground: bool(opponent.onGround),
      line_of_sight: canSee(player, opponent), mainhand: itemState(call(opponent, 'getHeldItemMainhand')),
      offhand: itemState(call(opponent, 'getHeldItemOffhand')), armor: armor(opponent)
    }
  }

  function entitySlots(player, selfPosition, selfVelocity) {
    const pointed = mouseOverEntity()
    const live = new Set()
    const slots = []
    for (const entity of javaList(corrected(API.world).loadedEntityList)) {
      if (!entity || referenceEquals(entity, player)) continue
      const kind = entityKind(entity)
      if (!/(crystal|arrow|projectile|pearl|snowball|fireball|egg)/i.test(kind)) continue
      const id = number(entity.entityId, number(call(entity, 'getEntityId'), -1))
      live.add(id)
      if (!state.entityBorn.has(id)) state.entityBorn.set(id, state.tick)
      const entityPosition = position(entity)
      slots.push({
        id,
        kind: kind.toLowerCase(),
        relative_position: egocentric(subtract(entityPosition, selfPosition), canonicalYaw(player.rotationYaw)),
        relative_velocity: egocentric(subtract(velocity(entity), selfVelocity), canonicalYaw(player.rotationYaw)),
        age_ticks: Math.max(0, state.tick - state.entityBorn.get(id)),
        distance: distance(selfPosition, entityPosition), raycastable: referenceEquals(pointed, entity)
      })
    }
    for (const id of state.entityBorn.keys()) if (!live.has(id)) state.entityBorn.delete(id)
    return slots.sort((a, b) => a.distance - b.distance || a.id - b.id).slice(0, 16).map(({ id, ...slot }) => slot)
  }

  function blockSlots(player, selfPosition, opponentPosition) {
    if (!reflected.BlockPosConstructor) return []
    if (state.tick - state.blockSampleTick < 5) {
      return state.blockCache.map(block => ({ ...block, sample_age_ticks: state.tick - state.blockSampleTick }))
    }
    state.blockSampleTick = state.tick
    const world = corrected(API.world)
    const centers = [selfPosition]
    if (opponentPosition && distance(selfPosition, opponentPosition) > 2) centers.push(opponentPosition)
    const candidates = []
    const seen = new Set()
    const pointed = mouseOverBlock()
    for (const center of centers) {
      const bx = Math.floor(center.x), by = Math.floor(center.y), bz = Math.floor(center.z)
      for (let dy = -2; dy <= 3; dy++) for (let dx = -5; dx <= 5; dx++) for (let dz = -5; dz <= 5; dz++) {
        if (dx * dx + dz * dz > 26) continue
        const coordinates = { x: bx + dx, y: by + dy, z: bz + dz }
        const key = `${coordinates.x},${coordinates.y},${coordinates.z}`
        if (seen.has(key)) continue
        seen.add(key)
        const block = blockAt(world, coordinates)
        if (!block) continue
        const replaceable = blockReplaceable(block)
        const faces = exposedFaces(world, coordinates)
        if (replaceable && faces === 0) continue
        const name = block.name
        const base = name === 'obsidian' || name === 'bedrock'
        const clearance = base && blockReplaceable(blockAt(world, offset(coordinates, 0, 1, 0)))
          && blockReplaceable(blockAt(world, offset(coordinates, 0, 2, 0)))
        const dist = distance(selfPosition, coordinates)
        const pointedAt = sameBlockPos(pointed, coordinates)
        candidates.push({
          name, relative_position: egocentric(subtract(coordinates, selfPosition), canonicalYaw(player.rotationYaw)),
          collision: block.collision, hardness: block.hardness, replaceable, break_progress: pointedAt ? miningProgress() : 0,
          crystal_clearance: clearance, exposed_faces: faces, distance: dist, within_reach: dist <= 5,
          raycastable: pointedAt, sample_age_ticks: 0,
          score: dist - (clearance ? 4 : base ? 2 : 0) - (pointedAt ? 8 : 0)
        })
      }
    }
    state.blockCache = candidates.sort((a, b) => a.score - b.score
      || a.relative_position.y - b.relative_position.y
      || a.relative_position.x - b.relative_position.x
      || a.relative_position.z - b.relative_position.z)
      .slice(0, 48).map(({ score, ...block }) => block)
    return state.blockCache
  }

  function blockAt(world, coordinates) {
    try {
      const rawPosition = reflected.BlockPosConstructor(coordinates.x, coordinates.y, coordinates.z)
      const blockState = corrected(world.getBlockState(rawPosition))
      const block = corrected(blockState.block || call(blockState, 'getBlock'))
      const material = corrected(blockState.material || call(blockState, 'getMaterial'))
      const name = registryName(block)
      const liquid = callBoolean(material, 'isLiquid', /water|lava/.test(name))
      const replaceable = callBoolean(material, 'isReplaceable', name === 'air')
      const solid = callBoolean(material, 'isSolid', !replaceable && !liquid)
      let hardness = 0
      try { hardness = number(block.getBlockHardness(world.getRef(), rawPosition), 0) } catch (_) {
        hardness = number(block.blockHardness, 0)
      }
      return { rawPosition, state: blockState, block, material, name,
        replaceable, collision: liquid ? 'liquid' : replaceable ? 'empty' : solid ? 'solid' : 'partial', hardness }
    } catch (_) { return null }
  }

  function exposedFaces(world, coordinates) {
    let result = 0
    for (const face of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) {
      if (blockReplaceable(blockAt(world, offset(coordinates, ...face)))) result++
    }
    return result
  }

  function blockReplaceable(block) { return !block || block.replaceable || block.collision === 'empty' }

  function actionMask(player) {
    const main = itemState(call(player, 'getHeldItemMainhand'))
    const offhand = itemState(call(player, 'getHeldItemOffhand'))
    return {
      attack: true, use_main: main.count > 0,
      use_offhand: offhand.count > 0 && !!reflected.ProcessRightClick,
      release_use: state.activeHand !== 'none', swap_offhand: main.count > 0 || offhand.count > 0,
      hotbar: Array.from({ length: 9 }, () => true)
    }
  }

  function applyAction(player, action) {
    const settings = corrected(API.settings)
    setKey(settings, 'keyBindForward', action.forward > 0)
    setKey(settings, 'keyBindBack', action.forward < 0)
    setKey(settings, 'keyBindLeft', action.strafe < 0)
    setKey(settings, 'keyBindRight', action.strafe > 0)
    setKey(settings, 'keyBindJump', action.jump)
    setKey(settings, 'keyBindSprint', action.sprint)
    setKey(settings, 'keyBindSneak', action.sneak)
    player.rotationYaw = number(player.rotationYaw, 0) + Coordinates.minecraftYawDelta(clamp(action.yaw_delta, -Math.PI, Math.PI))
    player.rotationPitch = clamp(number(player.rotationPitch, 0) + Coordinates.minecraftPitchDelta(action.pitch_delta), -90, 90)
    if (Number.isInteger(action.hotbar) && action.hotbar >= 0 && action.hotbar <= 8 && player.inventory) {
      player.inventory.currentItem = action.hotbar
    }
    const attack = action.primary === 'attack'
    const use = action.primary === 'use_main' || action.primary === 'use_offhand'
    setKey(settings, 'keyBindAttack', attack)
    setKey(settings, 'keyBindUseItem', use)
    if (attack) {
      API.clickMouse()
      state.lastAttackTick = state.tick
    }
    if (use && state.lastPrimary !== action.primary) {
      if (action.primary === 'use_offhand') explicitOffhandUse(player)
      else API.rightClickMouse()
      state.activeHand = action.primary === 'use_offhand' ? 'off' : 'main'
      state.useStartedTick = state.tick
    }
    if (action.release_use) {
      setKey(settings, 'keyBindUseItem', false)
      try { player.stopActiveHand() } catch (_) { }
      state.activeHand = 'none'
    }
    const swap = !!action.swap_offhand && !state.lastAction.swap_offhand
    setKey(settings, 'keyBindSwapHands', !!action.swap_offhand)
    if (swap) pulseKey(settings, 'keyBindSwapHands')
    state.lastPrimary = action.primary
    state.lastAction = action
  }

  function releaseControls() {
    const settings = corrected(API.settings)
    for (const key of ['keyBindForward', 'keyBindBack', 'keyBindLeft', 'keyBindRight', 'keyBindJump',
      'keyBindSprint', 'keyBindSneak', 'keyBindAttack', 'keyBindUseItem', 'keyBindSwapHands']) setKey(settings, key, false)
    try {
      const swap = corrected(settings && settings.keyBindSwapHands)
      if (swap) swap.pressTime = 0
    } catch (_) { }
    state.activeHand = 'none'
    state.lastPrimary = 'none'
    state.currentAction = noopAction()
    state.actionQueue = []
  }

  function setKey(settings, name, pressed) {
    try {
      const binding = corrected(settings && settings[name])
      if (binding) binding.pressed = pressed ? 1 : 0
    } catch (_) { }
  }

  function pulseKey(settings, name) {
    try {
      const binding = corrected(settings && settings[name])
      if (binding) binding.pressTime = Math.max(1, number(binding.pressTime, 0) + 1)
    } catch (_) { }
  }

  function explicitOffhandUse(player) {
    try {
      const controller = corrected(API.minecraft).playerController
      reflected.ProcessRightClick(
        controller.getRef ? controller.getRef() : controller,
        player.getRef ? player.getRef() : player,
        API.world.getRef ? API.world.getRef() : API.world,
        reflected.OffHand && reflected.OffHand.getRef ? reflected.OffHand.getRef() : reflected.OffHand
      )
    } catch (error) {
      state.error = `Offhand use failed safely: ${error.message}`
      setKey(corrected(API.settings), 'keyBindUseItem', false)
    }
  }

  function recordHumanAction(observation, player) {
    const settings = corrected(API.settings)
    const yaw = canonicalYaw(player.rotationYaw), pitch = canonicalPitch(player.rotationPitch)
    const attack = keyPressed(settings, 'keyBindAttack')
    const use = keyPressed(settings, 'keyBindUseItem')
    const selected = number(player.inventory && player.inventory.currentItem, 0)
    const action = {
      schema_version: 1,
      forward: axis(keyPressed(settings, 'keyBindForward'), keyPressed(settings, 'keyBindBack')),
      strafe: axis(keyPressed(settings, 'keyBindRight'), keyPressed(settings, 'keyBindLeft')),
      jump: keyPressed(settings, 'keyBindJump'), sprint: keyPressed(settings, 'keyBindSprint'),
      sneak: keyPressed(settings, 'keyBindSneak'), yaw_delta: normalizeAngle(yaw - state.lastYaw),
      pitch_delta: clamp(pitch - state.lastPitch, -Math.PI / 2, Math.PI / 2),
      primary: attack ? 'attack' : use ? (activeHand(player) === 'off' ? 'use_offhand' : 'use_main') : 'none',
      release_use: !use && state.activeHand !== 'none', hotbar: selected === state.lastHotbar ? -1 : selected,
      swap_offhand: keyPressed(settings, 'keyBindSwapHands')
    }
    state.demonstration.push({ match_id: state.matchId, recorded_at_ms: Date.now(), observation, action })
  }

  function ensureBridge() {
    if (state.socket && state.socket.readyState <= 1) return
    try {
      const socket = new WebSocket(config.bridgeUrl)
      state.socket = socket
      socket.onopen = () => {
        state.connected = true
        socket.send(JSON.stringify({ type: 'browser_hello', schema_version: 1, agent_id: playerName() }))
      }
      socket.onmessage = event => {
        const message = JSON.parse(event.data)
        if (message.type === 'browser_ready') state.policyVersion = number(message.policy_version, 0)
        if (message.type === 'browser_action') {
          queuePolicyAction(message.action)
          state.policyVersion = number(message.policy_version, state.policyVersion)
          state.value = number(message.value, 0)
          state.target = message.target
          state.latencyMs = performance.now() - state.lastRequestMs
          state.pending = false
        }
      }
      socket.onclose = () => { state.connected = false; state.pending = false }
      socket.onerror = () => { state.error = 'Local policy bridge unavailable'; state.pending = false }
    } catch (error) { state.error = error.message }
  }

  function sendBridgeStep(observation) {
    state.pending = true
    state.lastRequestMs = performance.now()
    state.socket.send(JSON.stringify({
      type: 'browser_step', schema_version: 1, sequence: ++state.sequence,
      observation, reward: 0, terminated: false, truncated: false
    }))
  }

  function delayedObservation(current) {
    state.observationHistory.push(current)
    const maximum = Math.max(8, config.observationDelayTicks + 2)
    while (state.observationHistory.length > maximum) state.observationHistory.shift()
    const index = Math.max(0, state.observationHistory.length - 1 - config.observationDelayTicks)
    const delayed = state.observationHistory[index]
    return {
      ...delayed,
      match: current.match,
      blocks: delayed.blocks.map(block => ({
        ...block, sample_age_ticks: block.sample_age_ticks + config.observationDelayTicks
      }))
    }
  }

  function queuePolicyAction(action) {
    state.actionQueue.push({ due: state.tick + config.actionDelayTicks, action })
    if (state.actionQueue.length > 128) state.actionQueue.splice(0, state.actionQueue.length - 128)
  }

  function advanceActionQueue() {
    while (state.actionQueue.length && state.actionQueue[0].due <= state.tick) {
      state.currentAction = state.actionQueue.shift().action
    }
  }

  function emergencyStop(reason) {
    config.enabled = false
    releaseControls()
    if (state.flatPolicy) state.flatPolicy.reset()
    state.actionQueue = []
    state.observationHistory = []
    state.lastSelfHealth = null
    state.lastOpponentHealth = null
    if (state.socket && state.socket.readyState === WebSocket.OPEN) {
      state.socket.send(JSON.stringify({ type: 'emergency_stop', schema_version: 1, reason }))
      state.socket.close()
    }
    state.connected = false
    state.pending = false
    API.displayToChat(`MCAI stopped: ${reason}`)
  }

  function emergencyKey(event) {
    if (number(event.key, 0) === 66) {
      event.preventDefault = true
      emergencyStop('F8 emergency stop')
    }
  }

  function browserEmergencyKey(event) {
    if (event.key === 'F8' || event.code === 'F8' || event.keyCode === 119) {
      event.preventDefault()
      event.stopImmediatePropagation()
      emergencyStop('F8 emergency stop')
    }
  }

  function chatCommand(event) {
    if (typeof event.message !== 'string' || !event.message.toLowerCase().startsWith('.mcai')) return
    event.preventDefault = true
    const parts = event.message.trim().split(/\s+/)
    const command = (parts[1] || 'status').toLowerCase()
    if (command === 'on') { config.enabled = true; API.displayToChat('MCAI enabled (structured state, unrestricted legal timing).') }
    else if (command === 'off' || command === 'stop') emergencyStop('chat command')
    else if (command === 'record') {
      config.enabled = false; releaseControls(); state.recording = true; state.demonstration = []
      state.matchId = parts[2] || makeId(); config.episodeId = state.matchId
      API.displayToChat(`Recording demonstration ${state.matchId}.`)
    } else if (command === 'recordstop') { state.recording = false; API.displayToChat(`Recorded ${state.demonstration.length} ticks.`) }
    else if (command === 'export') exportDemonstration()
    else if (command === 'delay') {
      config.actionDelayTicks = Math.round(clamp(parts[2], 0, 5))
      config.observationDelayTicks = Math.round(clamp(parts[3], 0, 5))
      state.actionQueue = []; state.observationHistory = []; state.currentAction = noopAction()
      API.displayToChat(`MCAI delays set to action=${config.actionDelayTicks}, observation=${config.observationDelayTicks} ticks.`)
    }
    else API.displayToChat(`MCAI ${config.enabled ? 'ON' : 'OFF'}, bridge=${state.connected}, recording=${state.recording}, ticks=${state.demonstration.length}`)
  }

  function exportDemonstration() {
    if (!state.demonstration.length) { API.displayToChat('No demonstration frames to export.'); return }
    const data = state.demonstration.map(record => JSON.stringify(record)).join('\n') + '\n'
    const url = URL.createObjectURL(new Blob([data], { type: 'application/x-ndjson' }))
    const anchor = document.createElement('a')
    anchor.href = url; anchor.download = `mcai-demo-${state.matchId}.jsonl`; anchor.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  function installOverlay() {
    const element = document.createElement('div')
    element.id = 'mcai-overlay'
    element.style.cssText = 'position:fixed;right:10px;top:10px;z-index:2147483647;padding:8px 10px;' +
      'background:rgba(8,12,18,.76);border:1px solid #63e6be;color:#e6fcf5;font:12px/1.45 monospace;' +
      'white-space:pre;pointer-events:none;text-shadow:1px 1px #000'
    document.body.appendChild(element)
    state.overlayElement = element
  }

  function updateOverlay(player) {
    if (!state.overlayElement) return
    state.overlayElement.style.display = config.overlay ? 'block' : 'none'
    const target = state.target ? `${number(state.target.x, 0).toFixed(1)},${number(state.target.y, 0).toFixed(1)},${number(state.target.z, 0).toFixed(1)}` : 'none'
    state.overlayElement.textContent = [
      `MCAI ${config.enabled ? 'ACTIVE' : 'SAFE/OFF'} ${state.flatPolicy ? 'browser-flat' : state.connected ? 'local-bridge' : 'disconnected'}`,
      `policy v${state.policyVersion}  tick ${state.tick}  ${state.latencyMs.toFixed(1)} ms`,
      `action ${state.currentAction.primary} move(${state.currentAction.forward},${state.currentAction.strafe})`,
      `target ${target}  value ${state.value.toFixed(3)}`,
      `reward ${state.rewardComponents.total.toFixed(3)}  dealt ${state.rewardComponents.opponentDamage.toFixed(3)}  taken ${state.rewardComponents.selfDamage.toFixed(3)}`,
      `health ${callNumber(player, 'getHealth', 0).toFixed(1)}  recording ${state.recording ? state.demonstration.length : 'off'}`,
      `F8 EMERGENCY STOP${state.error ? `\n${state.error}` : ''}`
    ].join('\n')
  }

  function updateRewardComponents(observation) {
    const selfHealth = number(observation.self.health, 0)
    const opponentHealth = observation.opponent && observation.opponent.health
    const selfDamage = state.lastSelfHealth === null ? 0 : -0.02 * Math.max(0, state.lastSelfHealth - selfHealth)
    const opponentDamage = state.lastOpponentHealth === null || opponentHealth === null
      ? 0 : 0.02 * Math.max(0, state.lastOpponentHealth - number(opponentHealth, state.lastOpponentHealth))
    state.rewardComponents = { opponentDamage, selfDamage, total: opponentDamage + selfDamage }
    state.lastSelfHealth = selfHealth
    state.lastOpponentHealth = opponentHealth === null ? null : number(opponentHealth, 0)
  }

  function nearestOpponent(player) {
    const self = position(player)
    return javaList(corrected(API.world).playerEntities)
      .filter(entity => entity && !referenceEquals(entity, player))
      .sort((a, b) => distance(self, position(a)) - distance(self, position(b)))[0] || null
  }

  function hotbar(player) {
    const inventory = player.inventory
    const main = inventory ? javaList(inventory.mainInventory) : []
    return Array.from({ length: 9 }, (_, index) => itemState(main[index]))
  }

  function armor(entity) {
    const inventory = entity.inventory
    const values = inventory ? javaList(inventory.armorInventory) : javaList(call(entity, 'getArmorInventoryList'))
    return [values[3], values[2], values[1], values[0]].map(itemState)
  }

  function itemState(stackValue) {
    const stack = corrected(stackValue)
    if (!stack || callBoolean(stack, 'isEmpty', false)) return { ...EMPTY_ITEM }
    const item = corrected(call(stack, 'getItem') || stack.item)
    const maximum = callNumber(stack, 'getMaxDamage', callNumber(item, 'getMaxDamage', 0))
    const used = callNumber(stack, 'getItemDamage', number(stack.itemDamage, 0))
    return {
      name: registryName(item), count: Math.max(0, Math.trunc(number(stack.count, number(stack.stackSize, 1)))),
      durability: maximum > 0 ? Math.max(0, maximum - used) : 0, max_durability: maximum,
      enchant_hash: knownKitEnchantHash(registryName(item))
    }
  }

  function knownKitEnchantHash(name) {
    let signature = ''
    if (name === 'diamond_sword') signature = 'knockback:1|sharpness:5'
    else if (name === 'diamond_pickaxe') signature = 'efficiency:5'
    else if (name.startsWith('diamond_') && /helmet|chestplate|leggings|boots/.test(name)) {
      signature = 'protection:4|unbreaking:3'
    }
    if (!signature) return 0
    let hash = 0x811c9dc5
    for (let index = 0; index < signature.length; index++) {
      hash = Math.imul(hash ^ signature.charCodeAt(index), 0x01000193)
    }
    return hash | 0
  }

  function raycast(player) {
    const hit = corrected(corrected(API.minecraft).objectMouseOver)
    if (!hit) return { kind: 'none', distance: 0, block_name: '', entity_kind: '' }
    const entity = corrected(hit.entityHit)
    if (entity) return { kind: 'entity', distance: distance(eyePosition(player), position(entity)), block_name: '', entity_kind: entityKind(entity) }
    const blockPosition = corrected(call(hit, 'getBlockPos') || hit.blockPos)
    if (blockPosition) {
      const block = blockAt(corrected(API.world), { x: blockPosition.x, y: blockPosition.y, z: blockPosition.z })
      return { kind: 'block', distance: distance(eyePosition(player), blockPosition), block_name: block ? block.name : '', entity_kind: '' }
    }
    return { kind: 'none', distance: 0, block_name: '', entity_kind: '' }
  }

  function mouseOverEntity() {
    const hit = corrected(corrected(API.minecraft).objectMouseOver)
    return hit ? corrected(hit.entityHit) : null
  }
  function mouseOverBlock() {
    const hit = corrected(corrected(API.minecraft).objectMouseOver)
    return hit ? corrected(call(hit, 'getBlockPos') || hit.blockPos) : null
  }
  function sameBlockPos(positionValue, expected) {
    return !!positionValue && number(positionValue.x, NaN) === expected.x
      && number(positionValue.y, NaN) === expected.y && number(positionValue.z, NaN) === expected.z
  }

  function activeHand(player) {
    if (!callBoolean(player, 'isHandActive', false)) { state.activeHand = 'none'; return 'none' }
    try {
      const value = javaString(call(player, 'getActiveHand')).toLowerCase()
      state.activeHand = value.includes('off') ? 'off' : 'main'
      return state.activeHand
    } catch (_) { return state.activeHand }
  }

  function attackCooldown(player) {
    try { return clamp(number(player.getCooledAttackStrength(0), 0), 0, 1) }
    catch (_) { return clamp((state.tick - state.lastAttackTick) / 13, 0, 1) }
  }
  function miningProgress() {
    const controller = corrected(corrected(API.minecraft).playerController)
    return clamp(number(controller && (controller.curBlockDamageMP || controller.curBlockDamage), 0), 0, 1)
  }
  function foodLevel(player) {
    const stats = corrected(call(player, 'getFoodStats') || player.foodStats)
    return callNumber(stats, 'getFoodLevel', 0)
  }

  function keyPressed(settings, name) {
    try { return bool(corrected(settings && settings[name]).pressed) } catch (_) { return false }
  }
  function axis(positive, negative) { return positive === negative ? 0 : positive ? 1 : -1 }
  function position(entity) { return { x: number(entity.posX, 0), y: number(entity.posY, 0), z: number(entity.posZ, 0) } }
  function eyePosition(entity) { const value = position(entity); value.y += callNumber(entity, 'getEyeHeight', 1.62); return value }
  function velocity(entity) { return { x: number(entity.motionX, 0), y: number(entity.motionY, 0), z: number(entity.motionZ, 0) } }
  function subtract(a, b) { return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z } }
  function offset(a, x, y, z) { return { x: a.x + x, y: a.y + y, z: a.z + z } }
  function distance(a, b) { return Math.hypot(a.x - number(b.x, 0), a.y - number(b.y, 0), a.z - number(b.z, 0)) }
  function egocentric(delta, yaw) {
    const sine = Math.sin(yaw), cosine = Math.cos(yaw)
    return { x: delta.x * cosine + delta.z * sine, y: delta.y, z: -delta.x * sine + delta.z * cosine }
  }
  function canSee(player, entity) { try { return bool(player.canEntityBeSeen(entity.getRef())) } catch (_) { return false } }
  function entityKind(entity) {
    try {
      const raw = (javaString(call(call(entity, 'getClass'), 'getSimpleName'))
        || javaString(call(entity, 'getName')) || 'unknown').toLowerCase()
      if (raw.includes('crystal')) return 'end_crystal'
      if (raw.includes('arrow')) return 'arrow'
      if (raw.includes('snowball')) return 'snowball'
      if (raw.includes('fireball')) return 'fireball'
      if (raw.includes('pearl')) return 'ender_pearl'
      if (raw.includes('player')) return 'player'
      if (raw.includes('egg')) return 'egg'
      if (raw.includes('projectile')) return 'projectile'
      return raw.replace(/^entity/, '').replace(/[^a-z0-9]+/g, '_') || 'unknown'
    } catch (_) { return 'unknown' }
  }
  function registryName(value) {
    if (!value) return ''
    try {
      const resource = corrected(call(value, 'getRegistryName'))
      const name = javaString(call(resource, 'toString') || resource).toLowerCase()
      return name.includes(':') ? name.split(':').pop() : name
    } catch (_) { return '' }
  }
  function playerName() { return javaString(call(corrected(API.player), 'getName')) || 'MCAI_BROWSER' }
  function corrected(value) { try { return value && value.getCorrective ? value.getCorrective() : value } catch (_) { return value } }
  function javaList(value) {
    const list = corrected(value)
    if (!list) return []
    if (Array.isArray(list)) return list.map(corrected)
    if (typeof list.length === 'number') return Array.from(list).map(corrected)
    try { return Array.from({ length: number(list.size(), 0) }, (_, index) => corrected(list.get(index))) } catch (_) { return [] }
  }
  function javaString(value) {
    if (value === null || value === undefined) return ''
    if (typeof value === 'string') return value
    try { return API.util.unstring(value.getRef ? value.getRef() : value) } catch (_) {
      try { return String(value) } catch (_) { return '' }
    }
  }
  function hasMethod(value, name) { return !!value && typeof value[name] === 'function' }
  function call(value, name, ...args) { try { return value && typeof value[name] === 'function' ? corrected(value[name](...args)) : null } catch (_) { return null } }
  function callNumber(value, name, fallback) { return number(call(value, name), fallback) }
  function callBoolean(value, name, fallback) { const result = call(value, name); return result === null ? fallback : bool(result) }
  function referenceEquals(a, b) {
    if (!a || !b) return false
    try { return (a.getRef ? a.getRef() : a) === (b.getRef ? b.getRef() : b) } catch (_) { return a === b }
  }
  function number(value, fallback) { const parsed = Number(value); return Number.isFinite(parsed) ? parsed : fallback }
  function bool(value) { return !!Number(value) || value === true }
  function clamp(value, low, high) { return Math.max(low, Math.min(high, number(value, 0))) }
  function canonicalYaw(value) { return Coordinates.canonicalYaw(value) }
  function canonicalPitch(value) { return Coordinates.canonicalPitch(value) }
  function normalizeAngle(value) { while (value > Math.PI) value -= Math.PI * 2; while (value < -Math.PI) value += Math.PI * 2; return value }
  function makeId() { return `browser-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}` }
  function noopAction() { return { schema_version: 1, forward: 0, strafe: 0, jump: false, sprint: false, sneak: false,
    yaw_delta: 0, pitch_delta: 0, primary: 'none', release_use: false, hotbar: -1, swap_offhand: false } }

  global.MCAI = Object.freeze({
    enable() { config.enabled = true },
    emergencyStop,
    status() { return { ...state, socket: undefined, flatPolicy: !!state.flatPolicy } },
    async loadWeights(manifestUrl, weightsUrl) {
      if (!global.MCAIFlatRuntime) throw new Error('flat-policy.js must be loaded first')
      state.flatPolicy = await global.MCAIFlatRuntime.fromUrls(manifestUrl, weightsUrl)
      state.policyVersion = 0
      return state.flatPolicy
    },
    async loadWeightsFromFiles(manifestFile, weightsFile) {
      if (!global.MCAIFlatRuntime) throw new Error('flat-policy.js must be loaded first')
      state.flatPolicy = await global.MCAIFlatRuntime.fromFiles(manifestFile, weightsFile)
      state.policyVersion = 0
      return state.flatPolicy
    }
  })
})(typeof globalThis !== 'undefined' ? globalThis : window)
