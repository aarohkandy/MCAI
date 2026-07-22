from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TacticalSnapshot:
    """Small, checkpoint-independent view of the state that produced an action."""

    distance: float | None
    line_of_sight: bool
    crosshair_on_opponent: bool
    pitch: float


@dataclass(frozen=True)
class TrainerShaping:
    total: float
    components: dict[str, float]


@dataclass
class _AgentState:
    episode_id: str
    stats: dict[str, float] = field(default_factory=dict)
    credited: dict[str, int] = field(default_factory=dict)
    extreme_pitch_streak: int = 0


class TrainerRewardShaper:
    """Adds small, bounded learning signals around verified arena events.

    The server remains authoritative for combat mechanics. This class only
    differences the cumulative counters included in ``step_feedback.info.stats``
    and compares two observations from the same episode. Consequently a click,
    slot selection, or guessed block interaction is never rewarded as success.
    """

    # Per-event bonuses are deliberately below the corresponding server reward.
    # The existing per-transition shaping clip remains the final safety bound.
    # These counters are emitted only after the server has verified a useful
    # policy-built base, an exact natural-stone mine -> useful-base sequence,
    # or a damaging crystal chain using that self-built base. Generic block and
    # obsidian placement deliberately earns no trainer-side success reward.
    TACTICAL_OBSIDIAN_SETUP = 0.006
    TACTICAL_MINE_PLACE = 0.010
    POLICY_BUILT_DAMAGING_COMBO = 0.020
    CRYSTAL_PLACED = 0.002
    CRYSTAL_DESTROYED = 0.002
    CRYSTAL_EXPLODED = 0.004

    APPROACH_PER_BLOCK = 0.001
    MAX_APPROACH_DELTA = 0.35
    PREFERRED_DISTANCE = 3.0
    VISIBILITY_ACQUIRED = 0.001
    VISIBILITY_LOST = -0.001

    EXTREME_PITCH = math.radians(65.0)
    EXTREME_PITCH_GRACE_STEPS = 20
    EXTREME_PITCH_PENALTY = -0.00025

    INVALID_INTERACTION = -0.001
    SPAM_ATTACK = -0.00075
    MISSED_ATTACK = -0.0005
    EXCESS_MECHANIC_SPAM = -0.0005

    # Episode budgets stop an agent from converting the arena floor into an
    # infinite reward source while leaving ample room for a real fight.
    CREDIT_LIMITS = {
        "tactical_obsidian_placed": 4,
        "tactical_mine_place_sequences": 3,
        "policy_built_crystal_chains_damaging": 4,
        "crystals_placed": 8,
        "crystals_destroyed": 8,
        "crystals_exploded": 8,
    }
    MAX_COUNTER_DELTA = 8
    NOMINAL_POINTS_PER_EVENT = {
        "crystal_placement": 2.5,
        "crystal_destruction": 1.2,
        "crystal_explosion": 3.0,
    }

    def __init__(self) -> None:
        self._agents: dict[str, _AgentState] = {}

    def shape(
        self,
        agent_id: str,
        episode_id: str,
        previous: TacticalSnapshot,
        observation: dict[str, Any],
        info: dict[str, Any] | None,
        *,
        same_episode: bool = True,
        done: bool = False,
    ) -> TrainerShaping:
        state = self._agents.get(agent_id)
        if state is None or state.episode_id != episode_id:
            state = _AgentState(episode_id=episode_id)
            self._agents[agent_id] = state

        components: dict[str, float] = {}
        if same_episode:
            current = tactical_snapshot(observation)
            self._shape_tactical(previous, current, state, components)
        else:
            state.extreme_pitch_streak = 0

        stats = info.get("stats") if isinstance(info, dict) else None
        if isinstance(stats, dict):
            deltas = self._counter_deltas(state, stats)
            point_deltas = self._point_deltas(state, stats.get("point_breakdown"))
            self._shape_mechanics(state, deltas, point_deltas, stats, components)

        raw_total = sum(components.values())
        # This is a second local bound. sanitize_reward still applies the shared
        # server+trainer shaping clip after terminal reward separation.
        total = max(-0.04, min(0.04, raw_total))
        if abs(total - raw_total) > 1e-12:
            components["local_clip"] = total - raw_total
        if done:
            self._agents.pop(agent_id, None)
        return TrainerShaping(total=total, components=components)

    def reset(self, agent_id: str) -> None:
        self._agents.pop(agent_id, None)

    @staticmethod
    def _shape_tactical(
        previous: TacticalSnapshot,
        current: TacticalSnapshot,
        state: _AgentState,
        components: dict[str, float],
    ) -> None:
        if previous.distance is not None and current.distance is not None:
            previous_gap = max(0.0, previous.distance - TrainerRewardShaper.PREFERRED_DISTANCE)
            current_gap = max(0.0, current.distance - TrainerRewardShaper.PREFERRED_DISTANCE)
            delta = max(
                -TrainerRewardShaper.MAX_APPROACH_DELTA,
                min(TrainerRewardShaper.MAX_APPROACH_DELTA, previous_gap - current_gap),
            )
            if abs(delta) > 1e-12:
                components["approach"] = delta * TrainerRewardShaper.APPROACH_PER_BLOCK

        if current.line_of_sight and not previous.line_of_sight:
            components["visibility"] = TrainerRewardShaper.VISIBILITY_ACQUIRED
        elif previous.line_of_sight and not current.line_of_sight:
            components["visibility"] = TrainerRewardShaper.VISIBILITY_LOST
        # Maintaining line of sight or simply holding the crosshair still is
        # intentionally free.  Both used to pay every policy tick, which made
        # passive stand-and-stare behavior more profitable than closing and
        # dealing damage.  Acquisition/loss remains a signed potential here;
        # the server remains authoritative for aim, attacks, hits, and damage.

        if abs(current.pitch) >= TrainerRewardShaper.EXTREME_PITCH:
            state.extreme_pitch_streak += 1
            if state.extreme_pitch_streak > TrainerRewardShaper.EXTREME_PITCH_GRACE_STEPS:
                severity = min(4.0, 1.0 + (
                    state.extreme_pitch_streak - TrainerRewardShaper.EXTREME_PITCH_GRACE_STEPS - 1
                ) / 20.0)
                components["extreme_pitch"] = TrainerRewardShaper.EXTREME_PITCH_PENALTY * severity
        else:
            state.extreme_pitch_streak = 0

    def _counter_deltas(self, state: _AgentState, stats: dict[str, Any]) -> dict[str, int]:
        names = (
            "blocks_placed",
            *self.CREDIT_LIMITS.keys(),
            "invalid_interactions", "spam_attack_swings", "missed_attack_swings",
        )
        execution = stats.get("execution")
        policy = execution.get("policy") if isinstance(execution, dict) else None
        policy = policy if isinstance(policy, dict) else None
        attributed_mechanics = {
            "blocks_placed", "blocks_mined", "crystals_placed",
            "crystals_destroyed", "crystals_exploded",
        }
        # These counters have no legacy/total fallback.  They are meaningful
        # only when the server attributes them to an actually executed policy
        # action; teacher-visible totals must never become PPO success.
        exact_policy_mechanics = {
            "tactical_obsidian_placed",
            "tactical_mine_place_sequences",
            "policy_built_crystal_chains_damaging",
        }
        deltas: dict[str, int] = {}
        for name in names:
            # New servers expose cumulative mechanic counters per execution
            # source. PPO shaping must difference POLICY counters: a passive
            # teacher completion can otherwise arrive on a later policy step,
            # receive no point credit, and be mislabeled as useless spam.
            if name in exact_policy_mechanics and (policy is None or name not in policy):
                deltas[name] = 0
                continue
            use_policy = (
                policy is not None
                and name in attributed_mechanics | exact_policy_mechanics
                and name in policy
            )
            source = policy if use_policy else stats
            state_key = f"execution:policy:{name}" if use_policy else name
            if name not in source:
                deltas[name] = 0
                continue
            current = _counter(source.get(name))
            previous = _counter(state.stats.get(state_key, 0.0))
            # Counters reset at episode boundaries. Within one episode a drop is
            # malformed telemetry, not a negative mechanic event.
            delta = max(0, current - previous)
            deltas[name] = min(self.MAX_COUNTER_DELTA, delta)
            state.stats[state_key] = float(current)
        return deltas

    def _point_deltas(self, state: _AgentState, breakdown: Any) -> dict[str, float]:
        breakdown = breakdown if isinstance(breakdown, dict) else {}
        deltas: dict[str, float] = {}
        for name in self.NOMINAL_POINTS_PER_EVENT:
            key = f"point_breakdown:{name}"
            if name not in breakdown:
                deltas[name] = 0.0
                continue
            current = _finite(breakdown.get(name))
            previous = _finite(state.stats.get(key, 0.0))
            deltas[name] = max(0.0, current - previous)
            state.stats[key] = current
        return deltas

    def _shape_mechanics(
        self,
        state: _AgentState,
        deltas: dict[str, int],
        point_deltas: dict[str, float],
        stats: dict[str, Any],
        components: dict[str, float],
    ) -> None:
        tactical_obsidian_events = deltas["tactical_obsidian_placed"]
        tactical_mine_place_events = deltas["tactical_mine_place_sequences"]
        policy_built_combo_events = deltas["policy_built_crystal_chains_damaging"]
        crystal_place_events = min(
            deltas["crystals_placed"], self._verified_events("crystal_placement", point_deltas),
        )
        crystal_destroy_events = min(
            deltas["crystals_destroyed"], self._verified_events("crystal_destruction", point_deltas),
        )
        crystal_explode_events = min(
            deltas["crystals_exploded"], self._verified_events("crystal_explosion", point_deltas),
        )

        tactical_obsidian = self._credit(
            state, "tactical_obsidian_placed", tactical_obsidian_events
        )
        tactical_mine_place = self._credit(
            state, "tactical_mine_place_sequences", tactical_mine_place_events
        )
        policy_built_combo = self._credit(
            state, "policy_built_crystal_chains_damaging", policy_built_combo_events
        )
        crystal_placed = self._credit(state, "crystals_placed", crystal_place_events)
        crystal_destroyed = self._credit(state, "crystals_destroyed", crystal_destroy_events)
        crystal_exploded = self._credit(state, "crystals_exploded", crystal_explode_events)

        if tactical_obsidian:
            components["tactical_obsidian_setup"] = (
                tactical_obsidian * self.TACTICAL_OBSIDIAN_SETUP
            )
        if tactical_mine_place:
            components["tactical_mine_place"] = (
                tactical_mine_place * self.TACTICAL_MINE_PLACE
            )
        if policy_built_combo:
            components["policy_built_damaging_combo"] = (
                policy_built_combo * self.POLICY_BUILT_DAMAGING_COMBO
            )
        if crystal_placed:
            components["crystal_place"] = crystal_placed * self.CRYSTAL_PLACED
        if crystal_destroyed:
            components["crystal_destroy"] = crystal_destroyed * self.CRYSTAL_DESTROYED
        if crystal_exploded:
            components["crystal_detonate"] = crystal_exploded * self.CRYSTAL_EXPLODED

        invalid = deltas["invalid_interactions"] * self.INVALID_INTERACTION
        attack_spam = deltas["spam_attack_swings"] * self.SPAM_ATTACK
        missed = deltas["missed_attack_swings"] * self.MISSED_ATTACK
        penalty = invalid + attack_spam + missed

        # Placing crystals without detonating them creates a verified cumulative
        # backlog. Only new placements beyond a generous allowance are penalized.
        execution = stats.get("execution")
        policy = execution.get("policy") if isinstance(execution, dict) else None
        policy = policy if isinstance(policy, dict) else None
        backlog_stats = policy if (
            policy is not None
            and "crystals_placed" in policy
            and "crystals_exploded" in policy
        ) else stats
        backlog = max(
            0,
            _counter(backlog_stats.get("crystals_placed"))
            - _counter(backlog_stats.get("crystals_exploded")),
        )
        if backlog > 4 and deltas["crystals_placed"]:
            penalty += min(deltas["crystals_placed"], backlog - 4) * self.EXCESS_MECHANIC_SPAM

        excess = sum(max(0, raw - credited) for raw, credited in (
            (deltas["blocks_placed"], tactical_obsidian),
            (deltas["crystals_placed"], crystal_placed),
        ))
        penalty += excess * self.EXCESS_MECHANIC_SPAM
        if penalty:
            components["useless_spam"] = penalty

    def _credit(self, state: _AgentState, name: str, delta: int) -> int:
        used = state.credited.get(name, 0)
        remaining = max(0, self.CREDIT_LIMITS[name] - used)
        credited = min(delta, remaining)
        state.credited[name] = used + credited
        return credited

    def _verified_events(self, name: str, point_deltas: dict[str, float]) -> int:
        points = point_deltas.get(name, 0.0)
        if points <= 1e-12:
            return 0
        nominal = self.NOMINAL_POINTS_PER_EVENT[name]
        return max(1, min(self.MAX_COUNTER_DELTA, int(round(points / nominal))))


def tactical_snapshot(observation: dict[str, Any]) -> TacticalSnapshot:
    opponent = observation.get("opponent")
    if isinstance(opponent, dict):
        relative = opponent.get("relative_position")
        if isinstance(relative, dict):
            values = [_finite(relative.get(axis)) for axis in ("x", "y", "z")]
            distance = math.sqrt(sum(value * value for value in values))
        else:
            distance = None
        line_of_sight = bool(opponent.get("line_of_sight"))
    else:
        distance = None
        line_of_sight = False

    self_state = observation.get("self")
    self_state = self_state if isinstance(self_state, dict) else {}
    action_mask = observation.get("action_mask")
    action_mask = action_mask if isinstance(action_mask, dict) else {}
    # This worker-computed bit is tied to the assigned arena opponent. A generic
    # player raycast could instead be the human spectator and must not earn reward.
    crosshair = bool(action_mask.get("combat_attack_ready"))
    return TacticalSnapshot(
        distance=distance,
        line_of_sight=line_of_sight,
        crosshair_on_opponent=crosshair,
        pitch=_finite(self_state.get("pitch")),
    )


def _counter(value: Any) -> int:
    return max(0, int(_finite(value)))


def _finite(value: Any) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return converted if math.isfinite(converted) else 0.0
