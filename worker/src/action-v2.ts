import {
  SCHEMA_VERSION,
  type ActionV1,
  type ActionV2,
  type ObservationV2,
  validateActionV2
} from './contracts.js'

/** Resolve a conditional intent to one coherent, legal ordinary-player action. */
export function resolveActionV2(input: ActionV2, observation: ObservationV2): ActionV1 {
  const action = validateActionV2(input)
  const base: ActionV1 = {
    schema_version: SCHEMA_VERSION,
    forward: action.forward,
    strafe: action.strafe,
    jump: action.jump,
    sprint: action.sprint,
    sneak: action.sneak,
    yaw_delta: action.yaw_delta,
    pitch_delta: action.pitch_delta,
    primary: 'none',
    release_use: false,
    hotbar: -1,
    swap_offhand: false
  }
  const mask = observation.action_mask
  switch (action.intent) {
    case 'sword_engage':
      requireImplicitTarget(action)
      return { ...base, hotbar: itemSlot(observation, 'sword'), primary: mask.combat_attack_ready ? 'attack' : 'none' }
    case 'crystal_acquire':
      requireImplicitTarget(action)
      return { ...base, hotbar: itemSlot(observation, 'crystal') }
    case 'crystal_place': {
      const candidate = crystalTarget(action, observation, 'base')
      return {
        ...base,
        hotbar: itemSlot(observation, 'crystal'),
        primary: candidate.placement_legal && mask.crystal_place_ready ? usableHand(observation, 'crystal') : 'none'
      }
    }
    case 'crystal_detonate': {
      const candidate = crystalTarget(action, observation, 'crystal')
      return { ...base, primary: candidate.reachable && candidate.visible && mask.crystal_attack_ready ? 'attack' : 'none' }
    }
    case 'build_pad': {
      blockTarget(action, observation)
      return {
        ...base,
        hotbar: itemSlot(observation, 'obsidian'),
        primary: mask.tactical_block_place_ready ? usableHand(observation, 'obsidian') : 'none'
      }
    }
    case 'mine_path': {
      blockTarget(action, observation)
      return { ...base, hotbar: itemSlot(observation, 'pickaxe'), primary: mask.tactical_block_break_ready ? 'attack' : 'none' }
    }
    case 'heal_retotem': {
      requireImplicitTarget(action)
      if (!observation.tactical.survival.has_totem && observation.tactical.survival.spare_totems > 0) {
        return { ...base, hotbar: itemSlot(observation, 'totem'), swap_offhand: true }
      }
      return {
        ...base,
        hotbar: itemSlot(observation, 'golden_apple'),
        primary: observation.tactical.survival.heal_available ? usableHand(observation, 'golden_apple') : 'none'
      }
    }
    case 'disengage':
      requireImplicitTarget(action)
      return { ...base, forward: -1, sprint: true, strafe: action.strafe === 0 ? 1 : action.strafe }
    case 'reposition':
      requireImplicitTarget(action)
      return base
  }
}

function requireImplicitTarget(action: ActionV2): void {
  if (action.target_index !== -1) throw new Error(`${action.intent} does not accept a candidate target`)
}

function crystalTarget(action: ActionV2, observation: ObservationV2, kind: 'base' | 'crystal') {
  const candidate = observation.tactical.crystal_candidates[action.target_index]
  if (!candidate || candidate.kind !== kind) throw new Error(`${action.intent} target does not name a ${kind}`)
  return candidate
}

function blockTarget(action: ActionV2, observation: ObservationV2) {
  const candidate = observation.tactical.block_candidates[action.target_index]
  if (!candidate) throw new Error(`${action.intent} target does not name a block candidate`)
  return candidate
}

function itemSlot(observation: ObservationV2, needle: string): number {
  return observation.self.hotbar.findIndex(item => item.count > 0 && item.name.toLowerCase().includes(needle))
}

function usableHand(observation: ObservationV2, needle: string): 'use_main' | 'use_offhand' | 'none' {
  if (observation.self.offhand.count > 0 && observation.self.offhand.name.toLowerCase().includes(needle)) return 'use_offhand'
  return itemSlot(observation, needle) >= 0 ? 'use_main' : 'none'
}
