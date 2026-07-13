(function installMCAIFlatRuntime(global) {
  'use strict'

  const SIZES = Object.freeze({ self: 80, opponent: 48, entity: 18, block: 20, legal: 24 })
  const LIMITS = Object.freeze({ entities: 16, blocks: 48 })
  const HEADS = Object.freeze({
    forward: 3, strafe: 3, jump: 2, sprint: 2, sneak: 2,
    primary: 4, release_use: 2, hotbar: 10, swap_offhand: 2
  })
  const PRIMARY = Object.freeze(['none', 'attack', 'use_main', 'use_offhand'])

  class FlatPolicy {
    constructor(manifest, buffer) {
      if (!manifest || manifest.format !== 'mcai-flat-f32' || manifest.format_version !== 1) {
        throw new Error('unsupported MCAI flat-weight manifest')
      }
      if (!(buffer instanceof ArrayBuffer)) throw new Error('weights must be an ArrayBuffer')
      this.manifest = manifest
      this.weights = new Float32Array(buffer)
      this.tensors = Object.create(null)
      for (const entry of manifest.tensors) {
        this.tensors[entry.name] = this.weights.subarray(entry.offset_f32, entry.offset_f32 + entry.length)
      }
      if (this.weights.length !== manifest.parameter_count) throw new Error('weight package length mismatch')
      this.hidden = new Float32Array(128)
    }

    reset() {
      this.hidden.fill(0)
    }

    step(observation) {
      const feature = encodeObservation(observation)
      const self = this.mlp('self_encoder', feature.self_state, 128, 96)
      const opponent = this.mlp('opponent_encoder', feature.opponent, 96, 64)
      if (!feature.opponent_mask[0]) opponent.fill(0)
      const entityPool = this.pool('entity_encoder', feature.entities, feature.entity_mask, 18, 64)
      const blockPool = this.pool('block_encoder', feature.blocks, feature.block_mask, 20, 64)
      const legal = this.mlp('legal_encoder', feature.legal, 32, 24)
      const fusedInput = concatenate(self, opponent, entityPool, blockPool, legal)
      const fused = this.mlp('fusion', fusedInput, 192, 192)
      this.hidden = this.gru(fused, this.hidden)
      const logits = Object.create(null)
      for (const name of Object.keys(HEADS)) {
        logits[name] = this.linear(this.hidden, `categorical_heads.head_${name}.weight`,
          `categorical_heads.head_${name}.bias`, HEADS[name], 128)
      }
      const cameraMean = this.linear(this.hidden, 'camera_mean.weight', 'camera_mean.bias', 2, 128)
      const value = this.linear(this.hidden, 'value_head.weight', 'value_head.bias', 1, 128)[0]
      const action = selectLegalAction(logits, cameraMean, feature.legal)
      return { action, value, logits, camera_mean: cameraMean, hidden: this.hidden.slice() }
    }

    mlp(prefix, input, middle, output) {
      const first = tanhArray(this.linear(input, `${prefix}.layers.0.weight`, `${prefix}.layers.0.bias`, middle, input.length))
      return tanhArray(this.linear(first, `${prefix}.layers.2.weight`, `${prefix}.layers.2.bias`, output, middle))
    }

    linear(input, weightName, biasName, rows, columns) {
      const weight = this.required(weightName)
      const bias = this.required(biasName)
      if (weight.length !== rows * columns || bias.length !== rows) {
        throw new Error(`bad tensor shape for ${weightName}`)
      }
      const output = new Float32Array(rows)
      for (let row = 0; row < rows; row++) {
        let value = bias[row]
        const offset = row * columns
        for (let column = 0; column < columns; column++) {
          value = Math.fround(value + Math.fround(weight[offset + column] * input[column]))
        }
        output[row] = value
      }
      return output
    }

    pool(prefix, values, mask, featureSize, outputSize) {
      const mean = new Float32Array(outputSize)
      const maximum = new Float32Array(outputSize)
      maximum.fill(-Infinity)
      let count = 0
      for (let slot = 0; slot < mask.length; slot++) {
        if (!mask[slot]) continue
        count++
        const start = slot * featureSize
        const encoded = this.mlp(prefix, values.subarray(start, start + featureSize), 64, outputSize)
        for (let index = 0; index < outputSize; index++) {
          mean[index] = Math.fround(mean[index] + encoded[index])
          maximum[index] = Math.max(maximum[index], encoded[index])
        }
      }
      if (count === 0) maximum.fill(0)
      else for (let index = 0; index < outputSize; index++) mean[index] = Math.fround(mean[index] / count)
      return concatenate(mean, maximum)
    }

    gru(input, hidden) {
      const weightInput = this.required('memory.weight_ih_l0')
      const weightHidden = this.required('memory.weight_hh_l0')
      const biasInput = this.required('memory.bias_ih_l0')
      const biasHidden = this.required('memory.bias_hh_l0')
      const inputProjection = affineAll(input, weightInput, biasInput, 384, 192)
      const hiddenProjection = affineAll(hidden, weightHidden, biasHidden, 384, 128)
      const output = new Float32Array(128)
      for (let index = 0; index < 128; index++) {
        const reset = sigmoid(inputProjection[index] + hiddenProjection[index])
        const update = sigmoid(inputProjection[128 + index] + hiddenProjection[128 + index])
        const candidate = Math.tanh(inputProjection[256 + index] + reset * hiddenProjection[256 + index])
        output[index] = Math.fround((1 - update) * candidate + update * hidden[index])
      }
      return output
    }

    required(name) {
      const tensor = this.tensors[name]
      if (!tensor) throw new Error(`missing tensor: ${name}`)
      return tensor
    }
  }

  function encodeObservation(observation) {
    if (!observation || observation.schema_version !== 1) throw new Error('unsupported observation schema')
    const result = {
      self_state: new Float32Array(SIZES.self),
      opponent: new Float32Array(SIZES.opponent),
      opponent_mask: new Float32Array(1),
      entities: new Float32Array(LIMITS.entities * SIZES.entity),
      entity_mask: new Float32Array(LIMITS.entities),
      blocks: new Float32Array(LIMITS.blocks * SIZES.block),
      block_mask: new Float32Array(LIMITS.blocks),
      legal: new Float32Array(SIZES.legal)
    }
    encodeSelf(observation.self || {}, result.self_state)
    if (observation.opponent) {
      result.opponent_mask[0] = 1
      encodeOpponent(observation.opponent, result.opponent)
    }
    for (let i = 0; i < Math.min(LIMITS.entities, (observation.entities || []).length); i++) {
      result.entity_mask[i] = 1
      encodeEntity(observation.entities[i], result.entities.subarray(i * SIZES.entity, (i + 1) * SIZES.entity))
    }
    for (let i = 0; i < Math.min(LIMITS.blocks, (observation.blocks || []).length); i++) {
      result.block_mask[i] = 1
      encodeBlock(observation.blocks[i], result.blocks.subarray(i * SIZES.block, (i + 1) * SIZES.block))
    }
    encodeLegal(observation.action_mask || {}, result.legal)
    return result
  }

  function encodeSelf(state, output) {
    const values = [
      scale(state.health, 20), scale(state.absorption, 20), scale(state.food, 20),
      ...vector(state.velocity, 2), Math.sin(number(state.yaw)), Math.cos(number(state.yaw)),
      scale(state.pitch, Math.PI / 2), bool(state.on_ground), bool(state.sprinting), bool(state.sneaking),
      scale(state.hurt_time, 10), number(state.attack_cooldown), scale(state.use_ticks, 32),
      number(state.mining_progress), scale(state.selected_hotbar, 8)
    ]
    for (const name of ['none', 'main', 'off']) values.push(bool(state.active_hand === name))
    const raycast = state.raycast || {}
    for (const name of ['none', 'block', 'entity']) values.push(bool(raycast.kind === name))
    values.push(scale(raycast.distance, 6), ...item(state.offhand))
    for (const armor of (state.armor || []).slice(0, 4)) values.push(...item(armor).slice(0, 3))
    const hotbar = state.hotbar || []
    for (let index = 0; index < 9; index++) values.push(...item(hotbar[index]).slice(0, 4))
    write(output, values)
  }

  function encodeOpponent(state, output) {
    const hasHealth = state.health !== null && state.health !== undefined
    const values = [
      ...vector(state.relative_position, 12), ...vector(state.relative_velocity, 2),
      Math.sin(number(state.yaw)), Math.cos(number(state.yaw)), scale(state.pitch, Math.PI / 2),
      hasHealth ? scale(state.health, 20) : 0, bool(hasHealth), scale(state.hurt_time, 10),
      bool(state.on_ground), bool(state.line_of_sight), ...item(state.mainhand), ...item(state.offhand)
    ]
    for (const armor of (state.armor || []).slice(0, 4)) values.push(...item(armor).slice(0, 3))
    write(output, values)
  }

  function encodeEntity(state, output) {
    const kind = String(state.kind || '').toLowerCase()
    const values = [
      ...['end_crystal', 'arrow', 'snowball', 'egg', 'fireball'].map(name => bool(kind.includes(name))),
      ...vector(state.relative_position, 12), ...vector(state.relative_velocity, 2),
      scale(state.age_ticks, 200), scale(state.distance, 12), bool(state.raycastable), hashFeature(kind)
    ]
    write(output, values)
  }

  function encodeBlock(state, output) {
    const collision = String(state.collision || 'empty')
    const name = String(state.name || '')
    const values = [
      ...vector(state.relative_position, 6),
      ...['empty', 'solid', 'liquid', 'partial'].map(value => bool(collision === value)),
      scale(state.hardness, 50), bool(state.replaceable), number(state.break_progress),
      bool(state.crystal_clearance), scale(state.exposed_faces, 6), scale(state.distance, 8),
      bool(state.within_reach), bool(state.raycastable), scale(state.sample_age_ticks, 10),
      bool(name.includes('obsidian')), bool(name === 'bedrock' || name === 'obsidian'), hashFeature(name)
    ]
    write(output, values)
  }

  function encodeLegal(mask, output) {
    const values = [1, bool(mask.attack), bool(mask.use_main), bool(mask.use_offhand), 1,
      bool(mask.release_use), 1, bool(mask.swap_offhand)]
    const hotbar = mask.hotbar || []
    for (let index = 0; index < 9; index++) values.push(bool(hotbar[index]))
    values.push(1, 1, 1, 1, 1, 1, 1)
    write(output, values)
  }

  function item(value) {
    if (!value || typeof value !== 'object') return [0, 0, 0, 0, 0, 0]
    const name = String(value.name || '')
    const maximum = Math.max(number(value.max_durability), 1)
    return [
      bool(name), scale(value.count, 64), number(value.durability) / maximum,
      scale(value.max_durability, 2000), hashFeature(name),
      (number(value.enchant_hash) % 104729) / 104729
    ]
  }

  function selectLegalAction(logits, cameraMean, legal) {
    const primaryMask = [true, !!legal[1], !!legal[2], !!legal[3]]
    const primaryIndex = argmax(logits.primary, primaryMask)
    const hotbarMask = [true]
    for (let index = 8; index < 17; index++) hotbarMask.push(!!legal[index])
    return {
      schema_version: 1,
      forward: argmax(logits.forward) - 1,
      strafe: argmax(logits.strafe) - 1,
      jump: !!argmax(logits.jump),
      sprint: !!argmax(logits.sprint),
      sneak: !!argmax(logits.sneak),
      yaw_delta: Math.tanh(cameraMean[0]) * Math.PI,
      pitch_delta: Math.tanh(cameraMean[1]) * Math.PI / 2,
      primary: PRIMARY[primaryIndex],
      release_use: !!argmax(logits.release_use, [true, !!legal[5] && primaryIndex < 2]),
      hotbar: argmax(logits.hotbar, hotbarMask) - 1,
      swap_offhand: !!argmax(logits.swap_offhand, [true, !!legal[7]])
    }
  }

  function affineAll(input, weight, bias, rows, columns) {
    const output = new Float32Array(rows)
    for (let row = 0; row < rows; row++) {
      let value = bias[row]
      const offset = row * columns
      for (let column = 0; column < columns; column++) {
        value = Math.fround(value + Math.fround(weight[offset + column] * input[column]))
      }
      output[row] = value
    }
    return output
  }

  function concatenate(...arrays) {
    const length = arrays.reduce((sum, array) => sum + array.length, 0)
    const result = new Float32Array(length)
    let offset = 0
    for (const array of arrays) { result.set(array, offset); offset += array.length }
    return result
  }

  function tanhArray(input) {
    const result = new Float32Array(input.length)
    for (let index = 0; index < input.length; index++) result[index] = Math.tanh(input[index])
    return result
  }

  function argmax(values, mask) {
    let best = -Infinity
    let result = 0
    for (let index = 0; index < values.length; index++) {
      if (mask && !mask[index]) continue
      if (values[index] > best) { best = values[index]; result = index }
    }
    return result
  }

  function sigmoid(value) { return 1 / (1 + Math.exp(-value)) }
  function vector(value, divisor) {
    const source = value && typeof value === 'object' ? value : {}
    return ['x', 'y', 'z'].map(axis => scale(source[axis], divisor))
  }
  function bool(value) { return value ? 1 : 0 }
  function number(value) {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : 0
  }
  function scale(value, divisor) { return Math.max(-4, Math.min(4, number(value) / divisor)) }
  function write(output, values) {
    for (let index = 0; index < Math.min(output.length, values.length); index++) output[index] = values[index]
  }
  function hashFeature(value) {
    let hash = 2166136261 >>> 0
    const bytes = typeof TextEncoder !== 'undefined' ? new TextEncoder().encode(value) : asciiBytes(value)
    for (const byte of bytes) hash = Math.imul((hash ^ byte) >>> 0, 16777619) >>> 0
    return (hash / 0xFFFFFFFF) * 2 - 1
  }
  function asciiBytes(value) { return Array.from(unescape(encodeURIComponent(value))).map(character => character.charCodeAt(0)) }

  async function fromUrls(manifestUrl, weightsUrl) {
    const [manifestResponse, weightsResponse] = await Promise.all([fetch(manifestUrl), fetch(weightsUrl)])
    if (!manifestResponse.ok || !weightsResponse.ok) throw new Error('unable to fetch MCAI policy files')
    return new FlatPolicy(await manifestResponse.json(), await weightsResponse.arrayBuffer())
  }

  async function fromFiles(manifestFile, weightsFile) {
    return new FlatPolicy(JSON.parse(await manifestFile.text()), await weightsFile.arrayBuffer())
  }

  global.MCAIFlatRuntime = Object.freeze({ FlatPolicy, encodeObservation, fromUrls, fromFiles, SIZES, LIMITS, HEADS })
})(typeof globalThis !== 'undefined' ? globalThis : window)
