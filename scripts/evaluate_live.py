from __future__ import annotations

import argparse
import json
import select
import socket
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


LANES = ("sword_retention", "crystal_retention", "combined", "terrain")


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0


def integer(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return 0


def policy_stats(payload: dict[str, Any]) -> dict[str, Any]:
    stats = payload.get("stats") if isinstance(payload.get("stats"), dict) else {}
    execution = stats.get("execution") if isinstance(stats.get("execution"), dict) else {}
    value = execution.get("policy") if isinstance(execution.get("policy"), dict) else {}
    return value


def lane_name(payload: dict[str, Any]) -> str:
    lane = str(payload.get("lane", ""))
    if lane in LANES:
        return lane
    return {
        "sword": "sword_retention",
        "crystal": "crystal_retention",
        "terrain": "terrain",
    }.get(str(payload.get("mode", "combined")), "combined")


def summarize_episode(events: list[dict[str, Any]]) -> dict[str, Any]:
    payloads = [event.get("payload", {}) for event in events]
    policies = [policy_stats(payload) for payload in payloads]
    first_hits = [
        integer(policy.get("first_hit_tick"))
        for policy in policies
        if integer(policy.get("first_hit_tick", -1)) >= 0
    ]
    first_damage = [
        integer(policy.get("first_damage_tick"))
        for policy in policies
        if integer(policy.get("first_damage_tick", -1)) >= 0
    ]
    summed = lambda key: sum(number(policy.get(key)) for policy in policies)
    return {
        "episode_id": str(payloads[0].get("episode_id", "")),
        "arena_seed": integer(payloads[0].get("arena_seed")),
        "lane": lane_name(payloads[0]),
        "reason": str(payloads[0].get("reason", "")),
        "match_ticks": max(integer(payload.get("match_ticks")) for payload in payloads),
        "policy_hits": int(summed("hits_landed")),
        "policy_damage": summed("damage_dealt"),
        "policy_crystals_placed": int(summed("crystals_placed")),
        "policy_crystals_exploded": int(summed("crystals_exploded")),
        "policy_crystal_damage_events": int(summed("crystal_damage_events")),
        "policy_crystal_totems": int(summed("crystal_totems_forced")),
        "policy_blocks_placed": int(summed("blocks_placed")),
        "policy_blocks_mined": int(summed("blocks_mined")),
        "first_policy_hit_tick": min(first_hits) if first_hits else -1,
        "first_policy_damage_tick": min(first_damage) if first_damage else -1,
    }


def lane_summary(lane: str, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    hit_episodes = sum(episode["policy_hits"] > 0 for episode in episodes)
    damaging = sum(
        episode["policy_damage"] > 0
        or episode["policy_crystal_damage_events"] > 0
        or episode["policy_crystal_totems"] > 0
        for episode in episodes
    )
    crystal_cycles = sum(
        episode["policy_crystals_placed"] > 0 and episode["policy_crystals_exploded"] > 0
        for episode in episodes
    )
    damaging_crystals = sum(
        episode["policy_crystal_damage_events"] > 0 or episode["policy_crystal_totems"] > 0
        for episode in episodes
    )
    block_cycles = sum(
        episode["policy_blocks_placed"] > 0 and episode["policy_blocks_mined"] > 0
        for episode in episodes
    )
    hit_ticks = [episode["first_policy_hit_tick"] for episode in episodes if episode["first_policy_hit_tick"] >= 0]
    result = {
        "episodes": len(episodes),
        "hit_episodes": hit_episodes,
        "damaging_engagement_episodes": damaging,
        "crystal_cycle_episodes": crystal_cycles,
        "damaging_crystal_episodes": damaging_crystals,
        "block_cycle_episodes": block_cycles,
        "death_endings": sum(episode["reason"] == "death" for episode in episodes),
        "median_first_hit_ticks": statistics.median(hit_ticks) if hit_ticks else None,
    }
    count = len(episodes)
    if lane == "sword_retention":
        result["passed"] = count >= 8 and hit_episodes >= 6 and bool(hit_ticks) and statistics.median(hit_ticks) <= 160
    elif lane == "crystal_retention":
        result["passed"] = count >= 8 and crystal_cycles >= 4 and damaging_crystals >= 2
    elif lane == "combined":
        result["passed"] = count >= 8 and damaging >= 6
    else:
        result["passed"] = count >= 8 and damaging >= 6 and block_cycles >= 4
    return result


def collect(host: str, port: int, episodes_per_lane: int, timeout_seconds: float) -> dict[str, Any]:
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    completed: dict[str, list[dict[str, Any]]] = {lane: [] for lane in LANES}
    deadline = time.monotonic() + timeout_seconds
    with socket.create_connection((host, port), timeout=10) as connection:
        connection.setblocking(False)
        connection.sendall((json.dumps({"type": "command", "id": 1, "command": "ping", "payload": {}}) + "\n").encode())
        buffered = b""
        while time.monotonic() < deadline and any(len(completed[lane]) < episodes_per_lane for lane in LANES):
            readable, _, _ = select.select([connection], [], [], 1.0)
            if not readable:
                continue
            chunk = connection.recv(64 * 1024)
            if not chunk:
                raise RuntimeError("arena control connection closed")
            buffered += chunk
            while b"\n" in buffered:
                raw, buffered = buffered.split(b"\n", 1)
                try:
                    event = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if event.get("type") != "event" or event.get("event") != "match_ended":
                    continue
                payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
                episode_id = str(payload.get("episode_id", ""))
                if not episode_id:
                    continue
                entries = by_episode[episode_id]
                if any(item.get("agent_id") == event.get("agent_id") for item in entries):
                    continue
                entries.append(event)
                if len(entries) < 2:
                    continue
                episode = summarize_episode(entries)
                lane = episode["lane"]
                if lane in completed and len(completed[lane]) < episodes_per_lane:
                    completed[lane].append(episode)
                del by_episode[episode_id]
    summaries = {lane: lane_summary(lane, values) for lane, values in completed.items()}
    return {
        "episodes_per_lane_requested": episodes_per_lane,
        "complete": all(len(completed[lane]) >= episodes_per_lane for lane in LANES),
        "passed": all(value["passed"] for value in summaries.values()),
        "lanes": summaries,
        "episodes": completed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Observe a frozen four-lane MCAI evaluation without changing live state")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--episodes-per-lane", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = collect(arguments.host, arguments.port, max(1, arguments.episodes_per_lane), arguments.timeout_seconds)
    rendered = json.dumps(report, indent=2)
    if arguments.output:
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if report["complete"] and report["passed"] else 1)


if __name__ == "__main__":
    main()
