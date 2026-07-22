(function installMCAIFlatRuntime(global) {
  'use strict'

  const SIZES = Object.freeze({
    self: 80, opponent: 48, entity: 18, block: 20, legal: 24,
    tactical: 16, history: 12, survival: 12, threat: 12
  })
  const LIMITS = Object.freeze({ entities: 16, blocks: 48, tactical: 16, history: 8 })
  const HEADS = Object.freeze({
    intent: 9, target_index: 17,
    forward: 3, strafe: 3, jump: 2, sprint: 2, sneak: 2,
    primary: 4, release_use: 2, hotbar: 10, swap_offhand: 2
  })
  const PRIMARY = Object.freeze(['none', 'attack', 'use_main', 'use_offhand'])
  const CRYSTAL_CHAIN_REACH = 3.4
  const CRYSTAL_EYE_HEIGHT = 1.62
  const MAX_POLICY_CRYSTAL_TARGET_PITCH = Math.PI / 4

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
      this.hidden = new Float32Array(256)
    }

    reset() {
      this.hidden.fill(0)
    }

    step(observation) {
      const feature = encodeObservation(observation)
      const self = this.mlp('self_encoder', feature.self_state, 192, 128)
      const opponent = this.mlp('opponent_encoder', feature.opponent, 128, 128)
      if (!feature.opponent_mask[0]) opponent.fill(0)
      const query = this.mlp('attention_query', concatenate(self, opponent), 192, 128)
      const entityPool = this.pool('entity_encoder', feature.entities, feature.entity_mask, 18, 96, 96)
      const blockPool = this.pool('block_encoder', feature.blocks, feature.block_mask, 20, 96, 96)
      const legal = this.mlp('legal_encoder', feature.legal, 64, 64)
      const context = this.mlp('context_encoder', concatenate(feature.survival, feature.threat), 96, 96)
      const candidates = concatenate(
        this.candidateAttention('crystal_attention', feature.crystal_candidates,
          feature.crystal_candidate_mask, SIZES.tactical, query),
        this.candidateAttention('tactical_block_attention', feature.tactical_blocks,
          feature.tactical_block_mask, SIZES.tactical, query),
        this.candidateAttention('history_attention', feature.recent_history,
          feature.recent_history_mask, SIZES.history, query)
      )
      const fusedInput = concatenate(self, opponent, entityPool, blockPool, legal, context, candidates)
      const fused = this.mlp('fusion', fusedInput, 384, 384)
      this.hidden = this.gru(fused, this.hidden)
      const logits = Object.create(null)
      for (const name of Object.keys(HEADS)) {
        logits[name] = this.linear(this.hidden, `categorical_heads.head_${name}.weight`,
          `categorical_heads.head_${name}.bias`, HEADS[name], 256)
      }
      const cameraMean = this.linear(this.hidden, 'camera_mean.weight', 'camera_mean.bias', 2, 256)
      const value = this.linear(this.hidden, 'value_head.weight', 'value_head.bias', 1, 256)[0]
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

    pool(prefix, values, mask, featureSize, middleSize, outputSize) {
      const mean = new Float32Array(outputSize)
      const maximum = new Float32Array(outputSize)
      maximum.fill(-Infinity)
      let count = 0
      for (let slot = 0; slot < mask.length; slot++) {
        if (!mask[slot]) continue
        count++
        const start = slot * featureSize
        const encoded = this.mlp(prefix, values.subarray(start, start + featureSize), middleSize, outputSize)
        for (let index = 0; index < outputSize; index++) {
          mean[index] = Math.fround(mean[index] + encoded[index])
          maximum[index] = Math.max(maximum[index], encoded[index])
        }
      }
      if (count === 0) maximum.fill(0)
      else for (let index = 0; index < outputSize; index++) mean[index] = Math.fround(mean[index] / count)
      return concatenate(mean, maximum)
    }

    candidateAttention(prefix, values, mask, featureSize, query) {
      if (!mask.some(Boolean)) return new Float32Array(128)
      const encoded = new Float32Array(mask.length * 128)
      for (let slot = 0; slot < mask.length; slot++) {
        if (!mask[slot]) continue
        const start = slot * featureSize
        encoded.set(this.mlp(`${prefix}.encoder`, values.subarray(start, start + featureSize), 128, 128),
          slot * 128)
      }
      const projectedQuery = this.linear(query, `${prefix}.query.weight`, `${prefix}.query.bias`, 128, 128)
      const attended = this.multiheadAttention(`${prefix}.attention`, projectedQuery, encoded, mask)
      return this.mlp(`${prefix}.output`, attended, 256, 128)
    }

    multiheadAttention(prefix, query, encoded, mask) {
      const projectionWeight = this.required(`${prefix}.in_proj_weight`)
      const projectionBias = this.required(`${prefix}.in_proj_bias`)
      if (projectionWeight.length !== 384 * 128 || projectionBias.length !== 384) {
        throw new Error(`bad tensor shape for ${prefix}.in_proj_weight`)
      }
      const queryProjection = affineRows(query, projectionWeight, projectionBias, 0, 128, 128)
      const keys = new Float32Array(mask.length * 128)
      const values = new Float32Array(mask.length * 128)
      for (let slot = 0; slot < mask.length; slot++) {
        if (!mask[slot]) continue
        const input = encoded.subarray(slot * 128, (slot + 1) * 128)
        keys.set(affineRows(input, projectionWeight, projectionBias, 128, 128, 128), slot * 128)
        values.set(affineRows(input, projectionWeight, projectionBias, 256, 128, 128), slot * 128)
      }
      const combined = new Float32Array(128)
      const headSize = 32
      const inverseScale = 1 / Math.sqrt(headSize)
      for (let head = 0; head < 4; head++) {
        const scores = new Float64Array(mask.length)
        scores.fill(-Infinity)
        let maximum = -Infinity
        for (let slot = 0; slot < mask.length; slot++) {
          if (!mask[slot]) continue
          let score = 0
          for (let column = 0; column < headSize; column++) {
            const index = head * headSize + column
            score += queryProjection[index] * keys[slot * 128 + index]
          }
          score *= inverseScale
          scores[slot] = score
          maximum = Math.max(maximum, score)
        }
        let denominator = 0
        for (let slot = 0; slot < mask.length; slot++) {
          if (!mask[slot]) continue
          scores[slot] = Math.exp(scores[slot] - maximum)
          denominator += scores[slot]
        }
        for (let column = 0; column < headSize; column++) {
          let value = 0
          const index = head * headSize + column
          for (let slot = 0; slot < mask.length; slot++) {
            if (mask[slot]) value += (scores[slot] / denominator) * values[slot * 128 + index]
          }
          combined[index] = value
        }
      }
      return this.linear(combined, `${prefix}.out_proj.weight`, `${prefix}.out_proj.bias`, 128, 128)
    }

    gru(input, hidden) {
      const weightInput = this.required('memory.weight_ih_l0')
      const weightHidden = this.required('memory.weight_hh_l0')
      const biasInput = this.required('memory.bias_ih_l0')
      const biasHidden = this.required('memory.bias_hh_l0')
      const inputProjection = affineAll(input, weightInput, biasInput, 768, 384)
      const hiddenProjection = affineAll(hidden, weightHidden, biasHidden, 768, 256)
      const output = new Float32Array(256)
      for (let index = 0; index < 256; index++) {
        const reset = sigmoid(inputProjection[index] + hiddenProjection[index])
        const update = sigmoid(inputProjection[256 + index] + hiddenProjection[256 + index])
        const candidate = Math.tanh(inputProjection[512 + index] + reset * hiddenProjection[512 + index])
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
      legal: new Float32Array(SIZES.legal),
      crystal_candidates: new Float32Array(LIMITS.tactical * SIZES.tactical),
      crystal_candidate_mask: new Float32Array(LIMITS.tactical),
      tactical_blocks: new Float32Array(LIMITS.tactical * SIZES.tactical),
      tactical_block_mask: new Float32Array(LIMITS.tactical),
      recent_history: new Float32Array(LIMITS.history * SIZES.history),
      recent_history_mask: new Float32Array(LIMITS.history),
      survival: new Float32Array(SIZES.survival),
      threat: new Float32Array(SIZES.threat)
    }
    const selfState = observation.self || {}
    encodeSelf(selfState, result.self_state)
    const crystal = encodeCrystalContext(observation, selfState, result.self_state)
    if (observation.opponent) {
      result.opponent_mask[0] = 1
      encodeOpponent(observation.opponent, selfState, result.opponent)
    }
    const entities = (observation.entities || []).slice(0, LIMITS.entities)
    const crystalEntityTarget = crystal.capable ? selectCrystalEntity(entities) : -1
    for (let i = 0; i < entities.length; i++) {
      result.entity_mask[i] = 1
      encodeEntity(entities[i], selfState,
        result.entities.subarray(i * SIZES.entity, (i + 1) * SIZES.entity),
        i === crystalEntityTarget)
    }
    const blocks = (observation.blocks || []).slice(0, LIMITS.blocks)
    const crystalBlockTarget = crystal.capable && crystal.hasCrystal
      ? selectCrystalBase(blocks, observation.opponent, selfState) : -1
    const tacticalPlacementTarget = selectTacticalPlacementTarget(blocks)
    for (let i = 0; i < blocks.length; i++) {
      result.block_mask[i] = 1
      encodeBlock(blocks[i], selfState,
        result.blocks.subarray(i * SIZES.block, (i + 1) * SIZES.block),
        i === crystalBlockTarget, i === tacticalPlacementTarget)
    }
    encodeLegal(observation.action_mask || {}, observation.match || {}, result.legal)
    encodeTactical(observation.tactical, result)
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
    for (const armor of (state.armor || []).slice(0, 4)) values.push(...armorItem(armor))
    const hotbar = state.hotbar || []
    const selected = Math.max(0, Math.min(8, Math.trunc(number(state.selected_hotbar))))
    for (let index = 0; index < 9; index++) {
      let held = hotbar[index]
      if (index === selected && state.mainhand && typeof state.mainhand === 'object' && String(state.mainhand.name || '')) {
        held = state.mainhand
      }
      values.push(...hotbarItem(held))
    }
    write(output, values)
  }

  function encodeOpponent(state, selfState, output) {
    const hasHealth = state.health !== null && state.health !== undefined
    const values = [
      ...vector(state.relative_position, 12), ...vector(state.relative_velocity, 2),
      Math.sin(number(state.yaw)), Math.cos(number(state.yaw)), scale(state.pitch, Math.PI / 2),
      hasHealth ? scale(state.health, 20) : 0, bool(hasHealth), scale(state.hurt_time, 10),
      bool(state.on_ground), bool(state.line_of_sight), ...item(state.mainhand), ...item(state.offhand)
    ]
    for (const armor of (state.armor || []).slice(0, 4)) values.push(...armorItem(armor))
    write(output, values)
    const geometry = opponentGeometry(state, selfState)
    output[38] = scale(geometry.distance, 12)
    output[39] = scale(geometry.horizontalDistance, 12)
    output[40] = Math.sin(geometry.bearingError)
    output[41] = Math.cos(geometry.bearingError)
    output[42] = scale(geometry.pitchError, Math.PI / 2)
    output[43] = scale(geometry.closingSpeed, 2)
    output[44] = geometry.withinMeleeReach
    output[45] = clamp(geometry.aimAlignment, -1, 1)
    output[46] = clamp(geometry.facingTowardSelf, -1, 1)
    output[47] = pvpItemCategory(state.mainhand && typeof state.mainhand === 'object'
      ? state.mainhand.name : '')
  }

  function encodeEntity(state, selfState, output, crystalTarget = false) {
    const kind = String(state.kind || '')
    const lowered = kind.toLowerCase()
    const values = [
      ...['end_crystal', 'arrow', 'snowball', 'egg', 'fireball'].map(name => bool(lowered.includes(name))),
      ...vector(state.relative_position, 12), ...vector(state.relative_velocity, 2),
      scale(state.age_ticks, 200), scale(state.distance, 12), bool(state.raycastable), hashFeature(kind)
    ]
    write(output, values)
    output[15] = bool(crystalTarget)
    const bodyPosition = bodyRelativeVector(state, selfState, 'position')
    output[16] = scale(bodyPosition[0], 12)
    output[17] = scale(bodyPosition[2], 12)
  }

  function encodeBlock(state, selfState, output, crystalTarget = false, tacticalTarget = false) {
    const collision = String(state.collision || 'empty')
    const name = String(state.name || '')
    const bodyPosition = bodyRelativeVector(state, selfState, 'position')
    const bodyBearing = Math.hypot(bodyPosition[0], bodyPosition[2]) > 1e-6
      ? Math.atan2(-bodyPosition[0], -bodyPosition[2]) : 0
    const values = [
      ...vector(state.relative_position, 6),
      ...['empty', 'solid', 'liquid', 'partial'].map(value => bool(collision === value)),
      scale(state.hardness, 50), bool(state.replaceable), number(state.break_progress),
      bool(state.crystal_clearance), scale(state.exposed_faces, 6), scale(state.distance, 8),
      bool(state.within_reach), bool(state.raycastable), scale(state.sample_age_ticks, 10),
      bool(name.includes('obsidian')), bool(name === 'bedrock' || name === 'obsidian'),
      scale(bodyBearing, Math.PI)
    ]
    write(output, values)
    output[19] = bool(crystalTarget || tacticalTarget)
    if (tacticalTarget) output[17] = 0
  }

  function encodeLegal(mask, match, output) {
    const values = [1, bool(mask.attack), bool(mask.use_main), bool(mask.use_offhand), 1,
      bool(mask.release_use), 1, bool(mask.swap_offhand)]
    const hotbar = mask.hotbar || []
    for (let index = 0; index < 9; index++) values.push(bool(hotbar[index]))
    const mode = String(match.mode || '').toLowerCase()
    const lane = String(match.lane || '').toLowerCase()
    values.push(
      bool(mask.combat_attack_ready), bool(mask.crystal_place_ready),
      bool(mask.crystal_attack_ready), bool(mask.tactical_block_break_ready),
      scale(match.action_delay_ticks, 5), scale(match.observation_delay_ticks, 5),
      bool(mode === 'terrain' || lane === 'terrain')
    )
    write(output, values)
  }

  function encodeTactical(value, result) {
    const tactical = isRecord(value) ? value : {}
    const crystals = Array.isArray(tactical.crystal_candidates)
      ? tactical.crystal_candidates.slice(0, LIMITS.tactical) : []
    for (let index = 0; index < crystals.length; index++) {
      if (!isRecord(crystals[index])) continue
      result.crystal_candidate_mask[index] = 1
      result.crystal_candidates.set(crystalCandidateVector(crystals[index]), index * SIZES.tactical)
    }
    const blocks = Array.isArray(tactical.block_candidates)
      ? tactical.block_candidates.slice(0, LIMITS.tactical) : []
    for (let index = 0; index < blocks.length; index++) {
      if (!isRecord(blocks[index])) continue
      result.tactical_block_mask[index] = 1
      result.tactical_blocks.set(blockCandidateVector(blocks[index]), index * SIZES.tactical)
    }
    const history = Array.isArray(tactical.recent_history)
      ? tactical.recent_history.slice(-LIMITS.history) : []
    for (let index = 0; index < history.length; index++) {
      if (!isRecord(history[index])) continue
      result.recent_history_mask[index] = 1
      result.recent_history.set(tacticalVector(history[index], SIZES.history), index * SIZES.history)
    }
    result.survival.set(tacticalVector(tactical.survival, SIZES.survival))
    result.threat.set(tacticalVector(tactical.threat, SIZES.threat))
  }

  function tacticalVector(value, size) {
    const output = new Float32Array(size)
    if (!isRecord(value)) return output
    const preferred = [
      'distance', 'reach', 'visible', 'legal', 'opponent_damage', 'self_damage',
      'pop_probability', 'escape_x', 'escape_y', 'escape_z', 'closing_speed',
      'cover_value', 'follow_up_viability', 'attack_cooldown', 'item_class', 'age_ticks'
    ]
    for (let index = 0; index < Math.min(size, preferred.length); index++) {
      const raw = value[preferred[index]]
      if (typeof raw === 'boolean') output[index] = bool(raw)
      else if (typeof raw === 'number' && Number.isFinite(raw)) output[index] = clamp(raw, -10, 10)
    }
    return output
  }

  function crystalCandidateVector(value) {
    const output = new Float32Array(SIZES.tactical)
    const position = isRecord(value.body_relative_position) ? value.body_relative_position : {}
    const kind = String(value.kind || '').toLowerCase()
    output[0] = candidateNumber(value, 'distance')
    output[1] = candidateBool(value, 'reachable', 'reach')
    output[2] = candidateBool(value, 'visible', 'line_of_sight')
    output[3] = candidateBool(value, 'placement_legal', 'legal')
    output[4] = candidateNumber(value, 'estimated_opponent_damage', 'opponent_damage')
    output[5] = candidateNumber(value, 'estimated_self_damage', 'self_damage')
    output[6] = candidateNumber(value, 'pop_potential', 'pop_probability')
    output[7] = candidateNumber(value, 'escape_direction', 'escape_x')
    output[8] = candidateNumber(position, 'x')
    output[9] = candidateNumber(position, 'y')
    output[10] = candidateNumber(position, 'z')
    output[11] = candidateNumber(value, 'source_index')
    output[12] = kind === 'base' ? 1 : kind === 'crystal' ? -1 : 0
    return output
  }

  function blockCandidateVector(value) {
    const output = new Float32Array(SIZES.tactical)
    const position = isRecord(value.body_relative_position) ? value.body_relative_position : {}
    const purpose = String(value.purpose || '').toLowerCase()
    const purposeValue = { crystal_base: 1, cover: 0.5, mine_path: -0.5, high_ground: -1 }
    output[0] = candidateNumber(value, 'distance')
    output[1] = candidateBool(value, 'reachable', 'reach')
    output[2] = candidateBool(value, 'visible', 'line_of_sight')
    output[8] = candidateNumber(position, 'x')
    output[9] = candidateNumber(position, 'y')
    output[10] = candidateNumber(position, 'z')
    output[11] = candidateNumber(value, 'source_index')
    output[13] = candidateNumber(value, 'cover_value')
    output[14] = candidateNumber(value, 'followup_crystal_viability', 'follow_up_viability')
    output[15] = purposeValue[purpose] || 0
    return output
  }

  function candidateNumber(value, ...names) {
    for (const name of names) {
      const raw = value[name]
      if (typeof raw === 'number' && Number.isFinite(raw)) return clamp(raw, -10, 10)
    }
    return 0
  }

  function candidateBool(value, ...names) {
    for (const name of names) {
      if (Object.prototype.hasOwnProperty.call(value, name)) return bool(value[name])
    }
    return 0
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

  function hotbarItem(value) {
    if (!isRecord(value)) return [0, 0, 0, 0]
    const maximum = Math.max(number(value.max_durability), 1)
    return [
      bool(String(value.name || '')), scale(value.count, 64),
      number(value.durability) / maximum, pvpItemCategory(value.name)
    ]
  }

  function armorItem(value) {
    if (!isRecord(value)) return [0, 0, 0]
    const name = String(value.name || '').toLowerCase()
    const maximum = Math.max(number(value.max_durability), 1)
    return [bool(name), armorCategory(name), number(value.durability) / maximum]
  }

  function pvpItemCategory(value) {
    const name = String(value || '').toLowerCase()
    if (!name) return 0
    if (name.includes('sword')) return 1
    if (name.includes('pickaxe')) return 0.8
    if (name.includes('crystal')) return 0.6
    if (name.includes('obsidian') || name.includes('bedrock')) return 0.4
    if (name.includes('apple') || ['bread', 'carrot', 'potato'].some(food => name.includes(food))) return 0.2
    if (name.includes('totem')) return -0.2
    if (['bow', 'arrow', 'pearl', 'snowball', 'egg'].some(projectile => name.includes(projectile))) return -0.6
    if (['stone', 'cobblestone', 'planks', 'dirt', 'sand', 'netherrack'].some(block => name.includes(block))) return -0.4
    return -1
  }

  function armorCategory(name) {
    if (!name) return 0
    if (name.includes('netherite')) return 1
    if (name.includes('diamond')) return 0.8
    if (name.includes('iron')) return 0.6
    if (name.includes('chainmail') || name.includes('chain')) return 0.45
    if (name.includes('gold')) return 0.3
    if (name.includes('leather')) return 0.15
    return -1
  }

  function encodeCrystalContext(observation, state, output) {
    const hotbar = state.hotbar || []
    const hasCrystal = hotbar.some(held => isRecord(held) && number(held.count) > 0 &&
      String(held.name || '').toLowerCase().includes('crystal'))
    const match = isRecord(observation.match) ? observation.match : {}
    const mode = String(match.mode || '').toLowerCase()
    const lane = String(match.lane || '').toLowerCase()
    const declared = ['crystal', 'combined', 'terrain'].includes(mode) ||
      ['crystal_retention', 'combined', 'terrain'].includes(lane)
    const capable = declared || (hasCrystal && mode !== 'sword' && lane !== 'sword_retention')
    const retention = capable && (mode === 'crystal' || lane === 'crystal_retention')
    output[78] = bool(capable)
    output[79] = bool(retention)
    return { capable, hasCrystal }
  }

  function selectTacticalPlacementTarget(blocks) {
    for (let index = 0; index < blocks.length; index++) {
      if (isRecord(blocks[index]) && blocks[index].tactical_placement_target) return index
    }
    return -1
  }

  function opponentGeometry(state, selfState) {
    const position = bodyRelativeVector(state, selfState, 'position')
    const velocity = bodyRelativeVector(state, selfState, 'velocity')
    let distance = optionalNumber(state.distance)
    if (distance === null) distance = Math.hypot(...position)
    let horizontalDistance = optionalNumber(state.horizontal_distance)
    if (horizontalDistance === null) horizontalDistance = Math.hypot(position[0], position[2])
    let bearingError = optionalNumber(state.bearing_error)
    if (bearingError === null) bearingError = Math.atan2(-position[0], -position[2])
    let pitchError = optionalNumber(state.pitch_error)
    if (pitchError === null) {
      pitchError = Math.atan2(position[1], Math.max(horizontalDistance, 1e-6)) - number(selfState.pitch)
    }
    let closingSpeed = optionalNumber(state.closing_speed)
    if (closingSpeed === null) {
      closingSpeed = -(position[0] * velocity[0] + position[1] * velocity[1] + position[2] * velocity[2]) /
        Math.max(distance, 1e-6)
    }
    const withinMeleeReach = Object.prototype.hasOwnProperty.call(state, 'within_melee_reach')
      ? bool(state.within_melee_reach) : bool(distance <= 3.4)
    let aimAlignment = optionalNumber(state.aim_alignment)
    if (aimAlignment === null) aimAlignment = Math.cos(bearingError) * Math.cos(pitchError)
    let facingTowardSelf = optionalNumber(state.facing_toward_self)
    if (facingTowardSelf === null) {
      facingTowardSelf = opponentFacingAlignment(state, selfState, position)
    }
    return {
      distance: Math.max(0, distance), horizontalDistance: Math.max(0, horizontalDistance),
      bearingError, pitchError, closingSpeed, withinMeleeReach,
      aimAlignment, facingTowardSelf
    }
  }

  function bodyRelativeVector(state, selfState, suffix) {
    const explicit = rawVector(state[`body_relative_${suffix}`])
    if (explicit) return explicit
    const legacy = rawVector(state[`relative_${suffix}`]) || [0, 0, 0]
    const yaw = number(selfState.yaw)
    const sine = Math.sin(2 * yaw)
    const cosine = Math.cos(2 * yaw)
    return [
      cosine * legacy[0] - sine * legacy[2],
      legacy[1],
      sine * legacy[0] + cosine * legacy[2]
    ]
  }

  function opponentFacingAlignment(state, selfState, bodyPosition) {
    const horizontal = Math.hypot(bodyPosition[0], bodyPosition[2])
    if (horizontal <= 1e-6) return 0
    const selfYaw = number(selfState.yaw)
    const sine = Math.sin(selfYaw)
    const cosine = Math.cos(selfYaw)
    const worldX = bodyPosition[0] * cosine + bodyPosition[2] * sine
    const worldZ = -bodyPosition[0] * sine + bodyPosition[2] * cosine
    const opponentYaw = number(Object.prototype.hasOwnProperty.call(state, 'head_yaw')
      ? state.head_yaw : state.yaw)
    return (Math.sin(opponentYaw) * worldX + Math.cos(opponentYaw) * worldZ) / horizontal
  }

  function selectCrystalEntity(entities) {
    let selected = -1
    let bestDistance = Infinity
    for (let index = 0; index < entities.length; index++) {
      const entity = entities[index]
      if (!isRecord(entity) || !String(entity.kind || '').toLowerCase().includes('crystal')) continue
      const distance = Math.max(0, number(entity.distance))
      const relative = rawVector(entity.relative_position)
      if (relative && distance <= CRYSTAL_CHAIN_REACH &&
          normalCameraPitchReachable(relative, 1 - CRYSTAL_EYE_HEIGHT) && distance < bestDistance) {
        selected = index
        bestDistance = distance
      }
    }
    return selected
  }

  function selectCrystalBase(blocks, opponent, selfState) {
    const opponentVector = isRecord(opponent) ? rawVector(opponent.relative_position) : null
    const yaw = number(selfState.yaw)
    const yawSin = Math.sin(yaw)
    const yawCos = Math.cos(yaw)
    let selected = -1
    let bestScore = Infinity
    for (let index = 0; index < blocks.length; index++) {
      const block = blocks[index]
      if (!isRecord(block)) continue
      const name = String(block.name || '').toLowerCase()
      const distance = Math.max(0, number(block.distance))
      if (!['obsidian', 'bedrock'].includes(name) || !block.crystal_clearance ||
          !block.within_reach || distance > CRYSTAL_CHAIN_REACH) continue
      const relative = rawVector(block.relative_position)
      if (!relative) continue
      const centered = [
        relative[0] + 0.5 * (yawCos + yawSin),
        relative[1],
        relative[2] + 0.5 * (yawCos - yawSin)
      ]
      if (!normalCameraPitchReachable(centered, 1 - CRYSTAL_EYE_HEIGHT)) continue
      const opponentDistance = opponentVector
        ? Math.hypot(relative[0] - opponentVector[0], relative[1] - opponentVector[1],
          relative[2] - opponentVector[2])
        : distance
      if (distance < 1.35 || (opponentVector && opponentDistance < 1.35)) continue
      const closePenalty = distance < 2 ? 3 + (2 - distance) * 4 : 0
      const score = opponentDistance + distance * 0.15 + closePenalty
      if (score < bestScore) {
        selected = index
        bestScore = score
      }
    }
    return selected
  }

  function normalCameraPitchReachable(relative, verticalAdjustment) {
    const horizontal = Math.hypot(relative[0], relative[2])
    if (horizontal <= 1e-6) return false
    return Math.abs(Math.atan2(relative[1] + verticalAdjustment, horizontal)) <=
      MAX_POLICY_CRYSTAL_TARGET_PITCH
  }

  function rawVector(value) {
    if (!isRecord(value)) return null
    return ['x', 'y', 'z'].map(axis => number(value[axis]))
  }

  function optionalNumber(value) {
    if (value === null || value === undefined) return null
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : null
  }

  function isRecord(value) { return !!value && typeof value === 'object' && !Array.isArray(value) }
  function clamp(value, low, high) { return Math.max(low, Math.min(high, value)) }

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

  function affineRows(input, weight, bias, rowStart, rows, columns) {
    const output = new Float32Array(rows)
    for (let row = 0; row < rows; row++) {
      const sourceRow = rowStart + row
      let value = bias[sourceRow]
      const offset = sourceRow * columns
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
