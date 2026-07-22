# Combat AI wire protocol

The schemas in this directory are the public boundary between the Paper arena,
Mineflayer rollout workers, the Python trainer, and the EaglerForge adapter.

All angles are radians. Actions represent ordinary player controls; they never
contain a target entity ID or placement coordinate. The original
`relative_position`/`relative_velocity` fields retain their v1 transform for
checkpoint compatibility. New `body_relative_*` fields are the authoritative
Mineflayer body frame: local `-Z` is forward and local `+X` is right.

Opponent observations also expose direct PvP geometry (`distance`, horizontal
distance, bearing/pitch error, closing speed, melee reach, mutual aim), explicit
main/offhand and armor state, real block-occlusion visibility, and fresh
server-authoritative health/absorption/ground state when available. Only the
arena-assigned opponent may populate these fields; spectators and fighters in
other arenas are never observation targets.

Block slots contain solid tactical geometry only. Replaceable air/liquid does
not consume the fixed 48-slot budget; air is represented through support and
crystal-clearance affordances on nearby solid blocks. Current-cursor blocks,
legal crystal bases, exposed obstacles, and the assigned fight corridor are
reserved before ordinary floor geometry.

Protocol messages carry a `schema_version` field. A component must reject an
unknown major schema version instead of guessing.

Workers advertising `observation-v2` and `action-v2` send the V2 observation
inside the unchanged v1 batch envelope. V2 appends ranked crystal/block
candidates, survival/threat state, and eight executed-action history entries.
ActionV2 selects a conditional combat intent and candidate index while retaining
ordinary movement/camera inputs. The worker validates the intent/target family
and converts it to an executable ActionV1; invalid cross-family indices are
rejected rather than silently driving an unrelated click. V1 actions remain
accepted for checkpoint and trainer rollback compatibility.

Match context may include the per-episode `mode`, `lane`, `arena_radius`, and
curriculum stage. Workers must update their legal-control mode at match start;
the process-wide startup mode is only a compatibility fallback.

`step_batch.steps[].info.stats` contains cumulative, server-authoritative
combat counters for the current agent and episode. Consumers must difference
successive snapshots; raw clicks or item selections are not mechanic success.
The accompanying cumulative `point_breakdown` identifies counters that passed
the arena's anti-farming checks.

`step_batch.steps[].execution` optionally reports the action that actually
produced the next observation. Its `source` is `policy`, `teacher_sword`,
`teacher_crystal`, `teacher_block`, or `safety`. Trainers treat an absent field
as `policy` for compatibility; teacher actions are imitation labels and are not
credited to the sampled PPO action they replaced.

A teacher execution may also include `pre_execution_observation`, the complete
`ObservationV1` captured immediately before the override. The trainer uses that
state only as the supervised imitation input. It never turns the teacher step
into a PPO transition or transfers teacher-generated reward to a later policy
action.

`action_batch.actions[].action_id` identifies a sampled policy proposal.
Workers echo it as `step_batch.steps[].execution.action_id` only when that
proposal is actually consumed, including ticks where a teacher or safety
override replaces it. This keeps PPO credit aligned across configured action
delays and lets the trainer exclude exactly the overridden proposal. Both
fields are optional for compatibility with older workers.

`action_mask.tactical_block_break_ready` means the ordinary attack input is
currently aimed at a reachable, useful arena block and the combined kit's
pickaxe can execute the break. It carries no target coordinate and occupies a
previously reserved trainer feature slot.

Crystal-chain telemetry is cumulative and policy-owned:
`policy_crystal_chains_started`, `policy_crystal_chains_detonated`,
`policy_crystal_chains_damaging`, and `policy_crystal_chains_popping`. A chain
counts only when its placement and detonation are both policy actions within the
configured window; teacher activity remains visible in the attributed execution
counters but cannot satisfy crystal progression.

Obsidian telemetry separates generic placement from useful combat setup.
`tactical_obsidian_placed` counts a reachable base placed near the assigned
fighter corridor. `tactical_mine_place_sequences` requires the policy to mine
natural stone and replace that exact site with a useful obsidian base within the
configured window; mining by itself is neutral. `policy_built_crystal_chains_damaging`
counts only an autonomous crystal chain that damages the opponent from that exact
policy-built base. All three are exposed per execution source;
`rewarded_obsidian_combos` reports the episode-capped delayed claims. Generic
block placement, repeated coordinates, and teacher/safety actions earn no PPO
building credit.
