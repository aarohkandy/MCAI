from __future__ import annotations

import asyncio
import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROFILE_NAMES = ("damage", "crystal", "terminal_speed", "activity", "building")

# Deliberately much tighter than the server's protocol safety range of
# [0.25, 4.0].  The controller is a slow curriculum servo, not an optimizer
# which is allowed to rewrite the objective in one generation.
PROFILE_BOUNDS: dict[str, tuple[float, float]] = {
    "damage": (0.80, 1.60),
    "crystal": (0.75, 2.00),
    "terminal_speed": (1.00, 1.75),
    "activity": (0.80, 1.60),
    "building": (0.75, 1.80),
}


@dataclass(frozen=True)
class AdaptiveRewardConfig:
    minimum_transitions: int = 2_048
    required_streak: int = 2
    cooldown_updates: int = 3
    relative_step: float = 0.075
    ewma_alpha: float = 0.40
    max_categories_per_update: int = 2
    rollback_evaluation_updates: int = 2
    rollback_regression_tolerance: float = 0.12
    rollback_cooldown_updates: int = 8


@dataclass(frozen=True)
class RewardProfile:
    generation: int
    multipliers: dict[str, float]
    reason: str

    def payload(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "reason": self.reason,
            "multipliers": dict(self.multipliers),
        }


@dataclass(frozen=True)
class RewardAdaptationDecision:
    changed: bool
    skipped: bool
    generation: int
    multipliers: dict[str, float]
    signals: dict[str, float]
    directions: dict[str, int]
    reasons: dict[str, str]
    changes: dict[str, dict[str, float]] = field(default_factory=dict)
    rollback: bool = False
    health_reasons: tuple[str, ...] = ()


@dataclass
class _ControllerState:
    generation: int = 1
    observations: int = 0
    multipliers: dict[str, float] = field(
        default_factory=lambda: {name: 1.0 for name in PROFILE_NAMES}
    )
    ewma: dict[str, float] = field(default_factory=dict)
    streaks: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in PROFILE_NAMES}
    )
    cooldowns: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in PROFILE_NAMES}
    )
    last_reason: str = "initial neutral adaptive reward profile"
    pending_change: dict[str, Any] | None = None


class AdaptiveRewardController:
    """Slowly adjusts the server's reward profile from completed rollouts.

    A signal must remain outside its neutral band for several complete PPO
    generations before a weight moves.  Each move is small, each category then
    enters a cooldown, and hard bounds prevent reward drift.  State is written
    atomically after every observation so a trainer restart resumes its streaks
    and profile instead of starting a second independent tuning process.
    """

    STATE_FILE = "adaptive-reward-state.json"
    AUDIT_FILE = "adaptive-reward-audit.jsonl"
    SCHEMA_VERSION = 1

    def __init__(
        self,
        directory: Path,
        config: AdaptiveRewardConfig | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = config or AdaptiveRewardConfig()
        self.enabled = bool(enabled)
        self.state_path = self.directory / self.STATE_FILE
        self.audit_path = self.directory / self.AUDIT_FILE
        self._state = _ControllerState()
        self._load()

    @property
    def profile(self) -> RewardProfile:
        return RewardProfile(
            generation=self._state.generation,
            multipliers=dict(self._state.multipliers),
            reason=self._state.last_reason,
        )

    def telemetry(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "generation": self._state.generation,
            "observations": self._state.observations,
            "multipliers": dict(self._state.multipliers),
            "cooldowns": dict(self._state.cooldowns),
            "streaks": dict(self._state.streaks),
            "rollback_evaluation_pending": self._state.pending_change is not None,
        }

    def observe(
        self,
        metrics: dict[str, Any],
        *,
        policy_version: int,
        rollout_generation: int,
    ) -> RewardAdaptationDecision:
        signals = _signals(metrics)
        transitions = int(signals.get("transitions", 0.0))
        health_reasons = _health_freezes(
            metrics, transitions, self.config.minimum_transitions, self.enabled
        )
        skipped = bool(health_reasons)
        directions: dict[str, int] = {name: 0 for name in PROFILE_NAMES}
        reasons: dict[str, str] = {
            name: "; ".join(health_reasons) if health_reasons else "inside neutral hysteresis band"
            for name in PROFILE_NAMES
        }
        changes: dict[str, dict[str, float]] = {}
        before = dict(self._state.multipliers)
        rollback = False

        self._state.observations += 1
        if not skipped:
            self._update_ewma(signals)
            pending = self._state.pending_change
            if pending is not None:
                # Do not stack weight changes while measuring the previous
                # change. That preserves causal attribution for rollback.
                score = _composite_score(signals)
                if score is not None:
                    post_scores = list(pending.get("post_scores", []))
                    post_scores.append(score)
                    pending["post_scores"] = post_scores
                needed = max(1, self.config.rollback_evaluation_updates)
                if len(pending.get("post_scores", [])) >= needed:
                    post_score = sum(pending["post_scores"]) / len(pending["post_scores"])
                    baseline = float(pending.get("baseline_score", post_score))
                    if post_score < baseline - self.config.rollback_regression_tolerance:
                        previous = pending.get("previous_multipliers", {})
                        for name, old_raw in previous.items():
                            if name not in PROFILE_NAMES:
                                continue
                            old = self._state.multipliers[name]
                            restored = float(old_raw)
                            self._state.multipliers[name] = restored
                            self._state.streaks[name] = 0
                            self._state.cooldowns[name] = self.config.rollback_cooldown_updates
                            changes[name] = {"from": old, "to": restored}
                            reasons[name] = (
                                f"rolled back: composite score regressed from "
                                f"{baseline:.3f} to {post_score:.3f}"
                            )
                        if changes:
                            rollback = True
                            self._state.generation += 1
                            self._state.last_reason = "; ".join(
                                reasons[name] for name in changes
                            )[:512]
                    self._state.pending_change = None
                else:
                    reasons = {
                        name: "waiting for post-change composite rollback evidence"
                        for name in PROFILE_NAMES
                    }
            else:
                directions, reasons = _desired_directions(self._state.ewma, signals)
                self._advance_streaks(directions)
                eligible = [
                    name for name in PROFILE_NAMES
                    if directions[name] != 0
                    and abs(self._state.streaks[name]) >= self.config.required_streak
                    and self._state.cooldowns[name] <= 0
                ]
                eligible.sort(
                    key=lambda name: (-abs(self._state.streaks[name]), PROFILE_NAMES.index(name))
                )
                for name in eligible[: self.config.max_categories_per_update]:
                    old = self._state.multipliers[name]
                    factor = (
                        1.0 + self.config.relative_step
                        if directions[name] > 0
                        else 1.0 / (1.0 + self.config.relative_step)
                    )
                    lower, upper = PROFILE_BOUNDS[name]
                    new = max(lower, min(upper, old * factor))
                    new = round(new, 6)
                    self._state.streaks[name] = 0
                    self._state.cooldowns[name] = self.config.cooldown_updates
                    if abs(new - old) > 1e-12:
                        self._state.multipliers[name] = new
                        changes[name] = {"from": old, "to": new}

                if changes:
                    self._state.generation += 1
                    self._state.last_reason = "; ".join(
                        f"{name}: {reasons[name]}" for name in changes
                    )[:512]
                    baseline_score = _composite_score(signals)
                    if (
                        baseline_score is not None
                        and self.config.rollback_evaluation_updates > 0
                    ):
                        self._state.pending_change = {
                            "generation": self._state.generation,
                            "baseline_score": baseline_score,
                            "previous_multipliers": {
                                name: before[name] for name in changes
                            },
                            "post_scores": [],
                        }

        self._save()
        decision = RewardAdaptationDecision(
            changed=bool(changes),
            skipped=skipped,
            generation=self._state.generation,
            multipliers=dict(self._state.multipliers),
            signals=signals,
            directions=directions,
            reasons=reasons,
            changes=changes,
            rollback=rollback,
            health_reasons=tuple(health_reasons),
        )
        self._audit({
            "event": "adaptive_reward_evaluation",
            "policy_version": int(policy_version),
            "rollout_generation": int(rollout_generation),
            "skipped": skipped,
            "generation": self._state.generation,
            "signals": signals,
            "directions": directions,
            "reasons": reasons,
            "changes": changes,
            "rollback": rollback,
            "health_reasons": health_reasons,
            "before": before,
            "after": dict(self._state.multipliers),
        })
        return decision

    def note_update_failure(
        self, message: str, *, policy_version: int, rollout_generation: int,
    ) -> None:
        """Audit a learner failure; weights and hysteresis remain frozen."""
        self._audit({
            "event": "adaptive_reward_health_freeze",
            "policy_version": int(policy_version),
            "rollout_generation": int(rollout_generation),
            "reason": f"learner update failed: {message}"[:512],
            "generation": self._state.generation,
            "multipliers": dict(self._state.multipliers),
        })

    def _update_ewma(self, signals: dict[str, float]) -> None:
        alpha = max(0.0, min(1.0, self.config.ewma_alpha))
        for name, value in signals.items():
            if name == "transitions" or not math.isfinite(value):
                continue
            previous = self._state.ewma.get(name)
            self._state.ewma[name] = value if previous is None else (
                alpha * value + (1.0 - alpha) * previous
            )

    def _advance_streaks(self, directions: dict[str, int]) -> None:
        for name in PROFILE_NAMES:
            if self._state.cooldowns[name] > 0:
                self._state.cooldowns[name] -= 1
            desired = directions[name]
            previous = self._state.streaks[name]
            if desired == 0:
                self._state.streaks[name] = 0
            elif previous == 0 or (previous > 0) == (desired > 0):
                self._state.streaks[name] = previous + desired
            else:
                self._state.streaks[name] = desired

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if int(payload.get("schema_version", 0)) != self.SCHEMA_VERSION:
                raise ValueError("unsupported adaptive reward state schema")
            multipliers = _validated_multipliers(payload.get("multipliers"))
            generation = max(1, int(payload.get("generation", 1)))
            observations = max(0, int(payload.get("observations", 0)))
            ewma = {
                str(name): value
                for name, raw in dict(payload.get("ewma", {})).items()
                if (value := _number(raw)) is not None
            }
            streaks = {
                name: _bounded_int(dict(payload.get("streaks", {})).get(name), -1000, 1000)
                for name in PROFILE_NAMES
            }
            cooldowns = {
                name: _bounded_int(dict(payload.get("cooldowns", {})).get(name), 0, 1000)
                for name in PROFILE_NAMES
            }
            pending_change = _validated_pending_change(payload.get("pending_change"))
            self._state = _ControllerState(
                generation=generation,
                observations=observations,
                multipliers=multipliers,
                ewma=ewma,
                streaks=streaks,
                cooldowns=cooldowns,
                last_reason=str(payload.get("last_reason", "restored adaptive reward profile"))[:512],
                pending_change=pending_change,
            )
        except Exception as error:
            # A damaged tuning sidecar must never prevent the actor from serving
            # its valid model checkpoint. Leave the file for diagnosis and start
            # from a safe neutral profile, with the recovery in the audit trail.
            self._state = _ControllerState()
            self._audit({
                "event": "adaptive_reward_state_recovery",
                "message": str(error),
            })

    def _save(self) -> None:
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            **asdict(self._state),
        }
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, self.state_path)

    def _audit(self, payload: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        with self.audit_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")


async def publish_reward_profile(
    host: str,
    port: int,
    profile: RewardProfile,
    *,
    timeout_seconds: float = 3.0,
) -> dict[str, Any]:
    """Publish one idempotent profile over the localhost arena protocol."""
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(host, port), timeout=timeout_seconds
    )
    request_id = max(1, int(profile.generation))
    request = {
        "type": "command",
        "id": request_id,
        "command": "set_reward_multipliers",
        "payload": profile.payload(),
    }
    try:
        writer.write((json.dumps(request, separators=(",", ":")) + "\n").encode("utf-8"))
        await asyncio.wait_for(writer.drain(), timeout=timeout_seconds)
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0.0:
                raise TimeoutError("arena reward profile response timed out")
            line = await asyncio.wait_for(reader.readline(), timeout=remaining)
            if not line:
                raise ConnectionError("arena control closed before reward profile response")
            response = json.loads(line.decode("utf-8"))
            # Arena broadcasts may be interleaved with command responses on the
            # same connection. Ignore events and unrelated response IDs until
            # the exact request is acknowledged.
            if response.get("type") != "response" or response.get("id") != request_id:
                continue
            if response.get("ok") is not True:
                raise RuntimeError(str(response.get("error", "arena rejected reward profile")))
            payload = response.get("payload", {})
            return payload if isinstance(payload, dict) else {}
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except (ConnectionError, OSError):
            pass


def _signals(metrics: dict[str, Any]) -> dict[str, float]:
    transitions = max(0.0, _number(metrics.get("reward_transitions")) or 0.0)
    result = {"transitions": transitions}

    dealt = _number(metrics.get("server_policy_damage_dealt"))
    taken = _number(metrics.get("server_damage_taken"))
    self_damage = _number(metrics.get("server_self_damage"))
    hits = _number(metrics.get("server_policy_hits_landed"))
    if dealt is not None and taken is not None and self_damage is not None and hits is not None:
        result["hit_density"] = max(0.0, hits) / max(1.0, transitions)
        result["damage_efficiency"] = max(0.0, dealt) / max(
            1.0, max(0.0, taken) + max(0.0, self_damage)
        )

    started = _number(metrics.get("server_policy_crystal_chains_started_events"))
    damaging = _number(metrics.get("server_policy_crystal_chains_damaging_events"))
    if started is not None and damaging is not None:
        result["crystal_attempt_density"] = max(0.0, started) / max(1.0, transitions)
        result["crystal_conversion"] = max(0.0, damaging) / max(1.0, max(0.0, started))
        result["crystal_starts"] = max(0.0, started)

    kills = _number(metrics.get("policy_owned_kill_events"))
    terminals = _number(metrics.get("terminal_attached_events"))
    if kills is not None and terminals is not None:
        estimated_matches = max(1.0, max(0.0, terminals) / 2.0)
        result["fast_finish_rate"] = min(1.0, max(0.0, kills) / estimated_matches)
        result["terminal_evidence"] = max(0.0, terminals)

    inaction = _number(metrics.get("server_inaction_penalty_ticks"))
    if inaction is not None:
        result["inaction_density"] = max(0.0, inaction) / max(1.0, transitions)

    placed = _number(metrics.get("server_policy_blocks_placed"))
    mined = _number(metrics.get("server_policy_blocks_mined"))
    if placed is not None and mined is not None:
        result["building_density"] = (
            max(0.0, placed) + max(0.0, mined)
        ) / max(1.0, transitions)
    return result


def _desired_directions(
    smoothed: dict[str, float], raw: dict[str, float],
) -> tuple[dict[str, int], dict[str, str]]:
    directions = {name: 0 for name in PROFILE_NAMES}
    reasons = {name: "inside neutral hysteresis band" for name in PROFILE_NAMES}

    hit_density = smoothed.get("hit_density")
    efficiency = smoothed.get("damage_efficiency")
    if hit_density is None or efficiency is None:
        reasons["damage"] = "missing policy damage telemetry"
    elif hit_density < 0.002 or efficiency < 0.80:
        directions["damage"] = 1
        reasons["damage"] = "low autonomous hit density or damage efficiency"
    elif hit_density > 0.010 and efficiency > 1.35:
        directions["damage"] = -1
        reasons["damage"] = "damage retention gate is comfortably satisfied"

    starts = raw.get("crystal_starts")
    crystal_density = smoothed.get("crystal_attempt_density")
    crystal_conversion = smoothed.get("crystal_conversion")
    if starts is None or crystal_density is None or crystal_conversion is None:
        reasons["crystal"] = "missing autonomous crystal-chain telemetry"
    elif starts < 4.0 or crystal_density < 0.0005 or crystal_conversion < 0.55:
        directions["crystal"] = 1
        reasons["crystal"] = "crystal attempts or damaging conversion are below target"
    elif crystal_density > 0.002 and crystal_conversion > 0.78:
        directions["crystal"] = -1
        reasons["crystal"] = "crystal retention gate is comfortably satisfied"

    finish_rate = smoothed.get("fast_finish_rate")
    terminal_evidence = raw.get("terminal_evidence")
    if finish_rate is None or terminal_evidence is None or terminal_evidence < 8.0:
        reasons["terminal_speed"] = "not enough completed-match evidence"
    elif finish_rate < 0.25:
        directions["terminal_speed"] = 1
        reasons["terminal_speed"] = "too few autonomous fights end in policy-owned kills"
    elif finish_rate > 0.65:
        directions["terminal_speed"] = -1
        reasons["terminal_speed"] = "fast autonomous endings are retained"

    inaction = smoothed.get("inaction_density")
    if inaction is None:
        reasons["activity"] = "missing inaction telemetry"
    elif inaction > 0.20:
        directions["activity"] = 1
        reasons["activity"] = "inaction penalties occupy too much of the rollout"
    elif inaction < 0.03 and (hit_density or 0.0) > 0.005:
        directions["activity"] = -1
        reasons["activity"] = "activity and hit-density retention gates are satisfied"

    building = smoothed.get("building_density")
    if building is None:
        reasons["building"] = "missing policy building telemetry"
    elif building < 0.00075:
        directions["building"] = 1
        reasons["building"] = "policy-owned building/mining frequency is below target"
    elif building > 0.0035:
        directions["building"] = -1
        reasons["building"] = "terrain mechanic retention gate is satisfied"
    return directions, reasons


def _health_freezes(
    metrics: dict[str, Any], transitions: int, minimum_transitions: int, enabled: bool,
) -> list[str]:
    reasons: list[str] = []
    if not enabled:
        reasons.append("adaptive rewards disabled")
    if transitions < minimum_transitions:
        reasons.append(
            f"only {transitions} clean transitions; need {minimum_transitions}"
        )
    nonfinite = _number(metrics.get("reward_nonfinite_transitions"))
    if nonfinite is not None and nonfinite > 0.0:
        reasons.append("non-finite rewards were quarantined")
    quarantined = sum(
        max(0.0, _number(metrics.get(name)) or 0.0)
        for name in ("quarantined_sequences", "quarantined_samples")
    )
    if quarantined > 0.0:
        reasons.append("non-finite PPO sequences were quarantined")
    skipped_steps = _number(metrics.get("skipped_optimizer_steps"))
    if skipped_steps is not None and skipped_steps > 0.0:
        reasons.append("optimizer skipped a non-finite step")
    maximum_kl = max(
        _number(metrics.get("max_kl")) or 0.0,
        _number(metrics.get("approximate_kl")) or 0.0,
    )
    if maximum_kl > 0.015:
        reasons.append(f"PPO KL {maximum_kl:.4f} exceeds 0.015")
    p95 = _number(metrics.get("decision_round_trip_ms_p95"))
    if p95 is not None and p95 > 200.0:
        reasons.append(f"decision p95 {p95:.1f} ms exceeds 200 ms")
    effective_hz = _number(metrics.get("effective_decisions_hz_per_agent"))
    if effective_hz is not None and effective_hz < 4.0:
        reasons.append(f"effective decision rate {effective_hz:.2f} Hz is below 4 Hz")
    valid_samples = _number(metrics.get("valid_samples"))
    if valid_samples is not None and valid_samples <= 0.0:
        reasons.append("optimizer published no valid samples")
    return reasons


def _composite_score(signals: dict[str, float]) -> float | None:
    """Return a bounded outcome score used only for post-change rollback."""
    components: list[tuple[float, float]] = []
    if "hit_density" in signals and "damage_efficiency" in signals:
        damage = 0.5 * min(1.0, signals["hit_density"] / 0.010)
        damage += 0.5 * min(1.0, signals["damage_efficiency"] / 1.35)
        components.append((0.25, damage))
    if "crystal_attempt_density" in signals and "crystal_conversion" in signals:
        crystal = 0.4 * min(1.0, signals["crystal_attempt_density"] / 0.002)
        crystal += 0.6 * min(1.0, signals["crystal_conversion"] / 0.78)
        components.append((0.25, crystal))
    if "fast_finish_rate" in signals and signals.get("terminal_evidence", 0.0) >= 8.0:
        components.append((0.25, min(1.0, signals["fast_finish_rate"] / 0.65)))
    if "inaction_density" in signals:
        activity = 1.0 - min(1.0, signals["inaction_density"] / 0.25)
        components.append((0.15, activity))
    if "building_density" in signals:
        components.append((0.10, min(1.0, signals["building_density"] / 0.0035)))
    if len(components) < 3:
        return None
    total_weight = sum(weight for weight, _ in components)
    return sum(weight * value for weight, value in components) / total_weight


def _validated_pending_change(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("adaptive reward pending change must be an object")
    baseline = _number(value.get("baseline_score"))
    if baseline is None:
        raise ValueError("adaptive reward pending baseline is not finite")
    previous_raw = value.get("previous_multipliers")
    if not isinstance(previous_raw, dict) or not previous_raw:
        raise ValueError("adaptive reward pending change has no previous multipliers")
    previous: dict[str, float] = {}
    for name, raw in previous_raw.items():
        if name not in PROFILE_NAMES:
            continue
        number = _number(raw)
        if number is None:
            raise ValueError(f"pending multiplier {name!r} is not finite")
        lower, upper = PROFILE_BOUNDS[name]
        previous[name] = max(lower, min(upper, number))
    if not previous:
        raise ValueError("adaptive reward pending change has no known categories")
    post_scores = [
        score for raw in list(value.get("post_scores", []))[:32]
        if (score := _number(raw)) is not None
    ]
    return {
        "generation": max(1, _bounded_int(value.get("generation"), 1, 2**31 - 1)),
        "baseline_score": max(0.0, min(1.0, baseline)),
        "previous_multipliers": previous,
        "post_scores": post_scores,
    }


def _validated_multipliers(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError("adaptive reward multipliers must be an object")
    result: dict[str, float] = {}
    for name in PROFILE_NAMES:
        number = _number(value.get(name))
        if number is None:
            raise ValueError(f"adaptive reward multiplier {name!r} is not finite")
        lower, upper = PROFILE_BOUNDS[name]
        result[name] = max(lower, min(upper, number))
    return result


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _bounded_int(value: Any, lower: int, upper: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        result = 0
    return max(lower, min(upper, result))
