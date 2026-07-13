export const SCHEMA_VERSION = 1 as const
export const MAX_ENTITY_SLOTS = 16
export const MAX_BLOCK_SLOTS = 48

export type Vec3Value = { x: number; y: number; z: number }

export type ItemState = {
  name: string
  count: number
  durability: number
  max_durability: number
  enchant_hash: number
}

export type RaycastState = {
  kind: 'none' | 'block' | 'entity'
  distance: number
  block_name: string
  entity_kind: string
}

export type SelfState = {
  health: number
  absorption: number
  food: number
  position: Vec3Value
  velocity: Vec3Value
  yaw: number
  pitch: number
  on_ground: boolean
  sprinting: boolean
  sneaking: boolean
  hurt_time: number
  attack_cooldown: number
  active_hand: 'none' | 'main' | 'off'
  use_ticks: number
  mining_progress: number
  selected_hotbar: number
  hotbar: ItemState[]
  offhand: ItemState
  armor: ItemState[]
  raycast: RaycastState
}

export type OpponentState = {
  relative_position: Vec3Value
  relative_velocity: Vec3Value
  yaw: number
  pitch: number
  health: number | null
  hurt_time: number
  on_ground: boolean
  line_of_sight: boolean
  mainhand: ItemState
  offhand: ItemState
  armor: ItemState[]
}

export type EntitySlot = {
  kind: string
  relative_position: Vec3Value
  relative_velocity: Vec3Value
  age_ticks: number
  distance: number
  raycastable: boolean
}

export type BlockSlot = {
  name: string
  relative_position: Vec3Value
  collision: 'empty' | 'solid' | 'liquid' | 'partial'
  hardness: number
  replaceable: boolean
  break_progress: number
  crystal_clearance: boolean
  exposed_faces: number
  distance: number
  within_reach: boolean
  raycastable: boolean
  sample_age_ticks: number
}

export type ActionMask = {
  attack: boolean
  use_main: boolean
  use_offhand: boolean
  release_use: boolean
  swap_offhand: boolean
  hotbar: boolean[]
}

export type ObservationV1 = {
  schema_version: typeof SCHEMA_VERSION
  match: {
    episode_id: string
    tick: number
    policy_version: number
    arena_seed: number
    action_delay_ticks: number
    observation_delay_ticks: number
  }
  self: SelfState
  opponent: OpponentState | null
  entities: EntitySlot[]
  blocks: BlockSlot[]
  action_mask: ActionMask
}

export type PrimaryAction = 'none' | 'attack' | 'use_main' | 'use_offhand'

export type ActionV1 = {
  schema_version: typeof SCHEMA_VERSION
  forward: -1 | 0 | 1
  strafe: -1 | 0 | 1
  jump: boolean
  sprint: boolean
  sneak: boolean
  yaw_delta: number
  pitch_delta: number
  primary: PrimaryAction
  release_use: boolean
  hotbar: number
  swap_offhand: boolean
}

export type StepFeedback = {
  reward: number
  terminated: boolean
  truncated: boolean
  info: Record<string, unknown>
}

export type StepBatch = {
  schema_version: typeof SCHEMA_VERSION
  type: 'step_batch'
  sequence: number
  policy_version: number
  steps: Array<{
    agent_id: string
    observation: ObservationV1
    reward: number
    terminated: boolean
    truncated: boolean
    info: Record<string, unknown>
  }>
}

export type ActionBatch = {
  schema_version: typeof SCHEMA_VERSION
  type: 'action_batch'
  sequence: number
  policy_version: number
  actions: Array<{ agent_id: string; action: ActionV1 }>
}

export type HelloMessage = {
  schema_version: typeof SCHEMA_VERSION
  type: 'hello'
  sequence: number
  worker_id: string
  agents: string[]
  capabilities: string[]
}

export type ControlMessage = {
  schema_version: typeof SCHEMA_VERSION
  type: 'control'
  sequence: number
  command: string
  payload: Record<string, unknown>
}

export type WireMessage = HelloMessage | StepBatch | ActionBatch | ControlMessage

export const NOOP_ACTION: ActionV1 = Object.freeze({
  schema_version: SCHEMA_VERSION,
  forward: 0,
  strafe: 0,
  jump: false,
  sprint: false,
  sneak: false,
  yaw_delta: 0,
  pitch_delta: 0,
  primary: 'none',
  release_use: false,
  hotbar: -1,
  swap_offhand: false
})

export function validateAction(action: ActionV1): ActionV1 {
  if (action.schema_version !== SCHEMA_VERSION) throw new Error('unsupported action schema')
  if (![-1, 0, 1].includes(action.forward)) throw new Error('invalid forward control')
  if (![-1, 0, 1].includes(action.strafe)) throw new Error('invalid strafe control')
  if (!['none', 'attack', 'use_main', 'use_offhand'].includes(action.primary)) {
    throw new Error('invalid primary action')
  }
  if (!Number.isInteger(action.hotbar) || action.hotbar < -1 || action.hotbar > 8) {
    throw new Error('invalid hotbar slot')
  }
  if (!Number.isFinite(action.yaw_delta) || !Number.isFinite(action.pitch_delta)) {
    throw new Error('invalid camera delta')
  }
  return {
    ...action,
    yaw_delta: clamp(action.yaw_delta, -Math.PI, Math.PI),
    pitch_delta: clamp(action.pitch_delta, -Math.PI / 2, Math.PI / 2)
  }
}

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value))
}
