export const SCHEMA_VERSION = 1 as const
export const OBSERVATION_V2_SCHEMA_VERSION = 2 as const
export const ACTION_V2_SCHEMA_VERSION = 2 as const
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
  /** Item currently selected in the main hand, made explicit for the policy. */
  mainhand: ItemState
  hotbar: ItemState[]
  offhand: ItemState
  armor: ItemState[]
  raycast: RaycastState
}

export type OpponentState = {
  /** Legacy checkpoint-compatible coordinates. */
  relative_position: Vec3Value
  relative_velocity: Vec3Value
  /** Correct Mineflayer body coordinates: -Z forward and +X right. */
  body_relative_position: Vec3Value
  body_relative_velocity: Vec3Value
  distance: number
  horizontal_distance: number
  bearing_error: number
  pitch_error: number
  closing_speed: number
  within_melee_reach: boolean
  /** Cosine alignment of our camera toward the opponent's torso. */
  aim_alignment: number
  /** Cosine alignment of the opponent's head toward our torso. */
  facing_toward_self: number
  yaw: number
  head_yaw: number
  pitch: number
  health: number | null
  absorption: number
  /** Age of the matching authoritative arena snapshot, or null if unavailable. */
  server_state_age_ticks: number | null
  hurt_time: number
  on_ground: boolean
  line_of_sight: boolean
  mainhand: ItemState
  offhand: ItemState
  armor: ItemState[]
}

export type EntitySlot = {
  kind: string
  /** Legacy checkpoint-compatible coordinates. */
  relative_position: Vec3Value
  relative_velocity: Vec3Value
  body_relative_position: Vec3Value
  body_relative_velocity: Vec3Value
  age_ticks: number
  distance: number
  raycastable: boolean
}

export type BlockSlot = {
  name: string
  /** Legacy checkpoint-compatible coordinates. */
  relative_position: Vec3Value
  body_relative_position: Vec3Value
  /** Relative translational velocity of this stationary world block. */
  body_relative_velocity: Vec3Value
  collision: 'empty' | 'solid' | 'liquid' | 'partial'
  hardness: number
  replaceable: boolean
  break_progress: number
  crystal_clearance: boolean
  /** This solid block's top face supports the episode's one offensive pad. */
  tactical_placement_target: boolean
  exposed_faces: number
  distance: number
  within_reach: boolean
  raycastable: boolean
  sample_age_ticks: number
}

export type ActionMask = {
  attack: boolean
  combat_attack_ready: boolean
  crystal_place_ready: boolean
  crystal_attack_ready: boolean
  tactical_block_break_ready: boolean
  /** A useful marked support face is under the cursor and obsidian is available. */
  tactical_block_place_ready?: boolean
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
    /** Optional per-match curriculum context; older trainers may omit it. */
    mode?: 'sword' | 'crystal' | 'combined' | 'terrain'
    lane?: string
    arena_radius?: number
    curriculum_stage?: number
  }
  self: SelfState
  opponent: OpponentState | null
  entities: EntitySlot[]
  blocks: BlockSlot[]
  action_mask: ActionMask
}

export type PrimaryAction = 'none' | 'attack' | 'use_main' | 'use_offhand'

export type CombatIntent =
  | 'sword_engage' | 'crystal_acquire' | 'crystal_place' | 'crystal_detonate'
  | 'build_pad' | 'mine_path' | 'heal_retotem' | 'disengage' | 'reposition'

export type CrystalCandidate = {
  kind: 'base' | 'crystal'
  source_index: number
  body_relative_position: Vec3Value
  distance: number
  reachable: boolean
  visible: boolean
  placement_legal: boolean
  estimated_opponent_damage: number
  estimated_self_damage: number
  pop_potential: number
  escape_direction: -1 | 0 | 1
}

export type TacticalBlockCandidate = {
  source_index: number
  body_relative_position: Vec3Value
  distance: number
  purpose: 'crystal_base' | 'cover' | 'mine_path' | 'high_ground'
  reachable: boolean
  visible: boolean
  cover_value: number
  followup_crystal_viability: number
}

export type ActionHistoryEntry = {
  tick: number
  intent: CombatIntent | 'legacy' | 'safety' | `teacher_${string}`
  primary: PrimaryAction
  hotbar: number
  health_delta: number
  opponent_health_delta: number
}

export type TacticalStateV2 = {
  threat: {
    score: number
    opponent_attack_cooldown: number
    opponent_equipment_class: 'none' | 'sword' | 'crystal' | 'tool' | 'other'
    closing_direction: -1 | 0 | 1
    recent_item_changed: boolean
  }
  crystal_candidates: CrystalCandidate[]
  block_candidates: TacticalBlockCandidate[]
  recent_history: ActionHistoryEntry[]
  survival: {
    has_totem: boolean
    spare_totems: number
    heal_available: boolean
    nearby_explosion_risk: number
    cover: number
    arena_boundary_distance: number
    vertical_escape: -1 | 0 | 1
  }
}

/** V1 remains supported; V2 appends relational tactical state. */
export type ObservationV2 = Omit<ObservationV1, 'schema_version'> & {
  schema_version: typeof OBSERVATION_V2_SCHEMA_VERSION
  tactical: TacticalStateV2
}

export type AnyObservation = ObservationV1 | ObservationV2

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

/** Conditional V2 intent with explicit ordinary-player control outputs. */
export type ActionV2 = Omit<ActionV1, 'schema_version'> & {
  schema_version: typeof ACTION_V2_SCHEMA_VERSION
  intent: CombatIntent
  target_index: number
}

export type AnyAction = ActionV1 | ActionV2

export type ExecutionSource =
  | 'policy'
  | 'teacher_sword'
  | 'teacher_crystal'
  | 'teacher_block'
  | 'safety'

/** The control that actually reached the game client for a rollout step. */
export type StepExecution = {
  source: ExecutionSource
  action: ActionV1
  /** Correlates this execution with the trainer assignment it consumed. */
  action_id?: number
  /** State immediately before a teacher override, used only for imitation. */
  pre_execution_observation?: AnyObservation
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
    observation: AnyObservation
    reward: number
    terminated: boolean
    truncated: boolean
    info: Record<string, unknown>
    execution: StepExecution
  }>
}

export type ActionBatch = {
  schema_version: typeof SCHEMA_VERSION
  type: 'action_batch'
  sequence: number
  policy_version: number
  actions: Array<{ agent_id: string; action: AnyAction; action_id?: number }>
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

export function validateActionV2(action: ActionV2): ActionV2 {
  if (action.schema_version !== ACTION_V2_SCHEMA_VERSION) throw new Error('unsupported V2 action schema')
  const intents: CombatIntent[] = [
    'sword_engage', 'crystal_acquire', 'crystal_place', 'crystal_detonate',
    'build_pad', 'mine_path', 'heal_retotem', 'disengage', 'reposition'
  ]
  if (!intents.includes(action.intent)) throw new Error('invalid combat intent')
  if (!Number.isInteger(action.target_index) || action.target_index < -1 || action.target_index > 63) {
    throw new Error('invalid target index')
  }
  const lowLevel = validateAction({ ...action, schema_version: SCHEMA_VERSION })
  return { ...action, ...lowLevel, schema_version: ACTION_V2_SCHEMA_VERSION }
}

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value))
}
