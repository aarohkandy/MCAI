from __future__ import annotations

import copy
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any


ELITE_EXECUTION_SOURCE = "elite_policy"
ELITE_KINDS = frozenset(("kill", "damaging_crystal"))


@dataclass
class _EliteEvent:
    event_id: str
    bucket: str
    kind: str
    quality: float
    ordinal: int
    records: list[dict[str, Any]]


class EliteReplayBuffer:
    """Bounded replay of verified autonomous successes.

    Policy controls remain in a short in-memory trace until the arena proves a
    useful outcome.  A hit, click, positive shaped reward, or teacher event is
    deliberately insufficient: admission requires either an explicitly
    policy-owned terminal win or the server's policy-owned damaging-crystal
    counter to advance.

    Capacity is measured in action records, but eviction is whole-event.  The
    most represented lane/mechanic bucket is evicted first and its weakest,
    oldest event goes first.  This keeps rare terrain/crystal layouts without
    allowing one easy arena to occupy the replay indefinitely.
    """

    def __init__(
        self,
        capacity: int = 2_048,
        trace_capacity: int = 128,
        kill_window: int = 64,
        crystal_window: int = 24,
        restored_records: list[dict[str, Any]] | None = None,
    ) -> None:
        self.capacity = max(1, int(capacity))
        self.trace_capacity = max(1, int(trace_capacity))
        self.kill_window = max(1, min(int(kill_window), self.trace_capacity))
        self.crystal_window = max(1, min(int(crystal_window), self.trace_capacity))
        self._traces: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self._events: list[_EliteEvent] = []
        self._event_ids: set[str] = set()
        self._ordinal = 0
        self.rejected_restored_records = 0
        if restored_records:
            self._restore(restored_records)

    def record_policy_action(
        self,
        *,
        agent_id: str,
        episode_id: str,
        action_id: int,
        policy_version: int,
        observation: dict[str, Any],
        action: dict[str, Any],
    ) -> None:
        """Remember an executed main-policy action until its outcome is known."""
        if not episode_id or episode_id == "waiting":
            return
        if not _finite_tree(observation) or not _finite_tree(action):
            return
        match = observation.get("match")
        match = match if isinstance(match, dict) else {}
        try:
            tick = int(match.get("tick", 0))
        except (TypeError, ValueError, OverflowError):
            tick = 0
        key = (str(agent_id), str(episode_id))
        trace = self._traces.setdefault(key, [])
        trace.append({
            "match_id": str(episode_id),
            "agent_id": str(agent_id),
            "tick": tick,
            "policy_version": int(policy_version),
            "observation": copy.deepcopy(observation),
            "action": copy.deepcopy(action),
            "execution_source": ELITE_EXECUTION_SOURCE,
            "elite_action_id": int(action_id),
        })
        overflow = len(trace) - self.trace_capacity
        if overflow > 0:
            del trace[:overflow]

    def promote(
        self,
        agent_id: str,
        episode_id: str,
        kind: str,
        *,
        event_token: str,
        quality: float = 1.0,
    ) -> int:
        """Admit the recent policy sequence for one verified success event."""
        if kind not in ELITE_KINDS:
            raise ValueError(f"unsupported elite replay kind: {kind}")
        key = (str(agent_id), str(episode_id))
        trace = self._traces.get(key, [])
        if not trace:
            return 0
        event_id = f"{episode_id}|{agent_id}|{kind}|{event_token}"
        if event_id in self._event_ids:
            return 0
        window = self.kill_window if kind == "kill" else self.crystal_window
        selected = trace[-min(window, self.capacity):]
        if not selected:
            return 0
        bounded_quality = _bounded_quality(quality)
        bucket = _event_bucket(kind, selected[-1].get("observation"))
        self._ordinal += 1
        records = []
        for record in selected:
            records.append({
                **record,
                "elite_kind": kind,
                "elite_event_id": event_id,
                "elite_bucket": bucket,
                "elite_quality": bounded_quality,
            })
        self._append_event(_EliteEvent(
            event_id=event_id,
            bucket=bucket,
            kind=kind,
            quality=bounded_quality,
            ordinal=self._ordinal,
            records=records,
        ))
        return len(records)

    def clear_episode(self, agent_id: str, episode_id: str) -> None:
        self._traces.pop((str(agent_id), str(episode_id)), None)

    def clear_agent(self, agent_id: str) -> None:
        identifier = str(agent_id)
        self._traces = {
            key: value for key, value in self._traces.items() if key[0] != identifier
        }

    def clear_traces(self) -> None:
        self._traces.clear()

    @property
    def records(self) -> list[dict[str, Any]]:
        return [record for event in self._events for record in event.records]

    def metrics(self) -> dict[str, int]:
        kinds = Counter(event.kind for event in self._events)
        buckets = Counter(event.bucket for event in self._events)
        return {
            "elite_replay_records": len(self),
            "elite_replay_events": len(self._events),
            "elite_replay_buckets": len(buckets),
            "elite_replay_kill_events": kinds.get("kill", 0),
            "elite_replay_crystal_events": kinds.get("damaging_crystal", 0),
            "elite_replay_rejected_restored_records": self.rejected_restored_records,
        }

    def __len__(self) -> int:
        return sum(len(event.records) for event in self._events)

    def _append_event(self, event: _EliteEvent) -> None:
        if event.event_id in self._event_ids:
            return
        self._events.append(event)
        self._event_ids.add(event.event_id)
        while len(self) > self.capacity and self._events:
            counts = Counter(value.bucket for value in self._events for _ in value.records)
            largest = max(counts.values())
            overrepresented = {bucket for bucket, count in counts.items() if count == largest}
            candidates = [
                value for value in self._events if value.bucket in overrepresented
            ]
            victim = min(candidates, key=lambda value: (value.quality, value.ordinal))
            self._events.remove(victim)
            self._event_ids.discard(victim.event_id)

    def _restore(self, records: list[dict[str, Any]]) -> None:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for raw in records:
            if not _valid_restored_record(raw):
                self.rejected_restored_records += 1
                continue
            event_id = str(raw["elite_event_id"])
            grouped.setdefault(event_id, []).append(copy.deepcopy(raw))
        for event_id, values in grouped.items():
            first = values[0]
            self._ordinal += 1
            self._append_event(_EliteEvent(
                event_id=event_id,
                bucket=str(first["elite_bucket"]),
                kind=str(first["elite_kind"]),
                quality=_bounded_quality(first.get("elite_quality", 1.0)),
                ordinal=self._ordinal,
                records=values[-self.capacity:],
            ))


def _event_bucket(kind: str, observation: Any) -> str:
    match = observation.get("match") if isinstance(observation, dict) else None
    match = match if isinstance(match, dict) else {}
    lane = str(match.get("lane") or "unknown")
    mode = str(match.get("mode") or "unknown")
    stage = str(match.get("stage") if match.get("stage") is not None else "unknown")
    return f"{kind}:{lane}:{mode}:stage-{stage}"


def _valid_restored_record(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("execution_source") != ELITE_EXECUTION_SOURCE:
        return False
    if str(value.get("elite_kind", "")) not in ELITE_KINDS:
        return False
    if not all(isinstance(value.get(name), dict) for name in ("observation", "action")):
        return False
    if not str(value.get("elite_event_id", "")) or not str(value.get("elite_bucket", "")):
        return False
    return _finite_tree(value)


def _finite_tree(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(isinstance(key, str) and _finite_tree(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    return False


def _bounded_quality(value: Any) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.25
    if not math.isfinite(converted):
        return 0.25
    return max(0.25, min(1.0, converted))
