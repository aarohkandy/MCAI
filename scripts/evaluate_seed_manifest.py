from __future__ import annotations

import argparse
import json
import os
import select
import socket
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluate_live import LANES, lane_summary, summarize_episode


LANE_MODES = {
    "sword_retention": "sword",
    "crystal_retention": "crystal",
    "combined": "combined",
    "terrain": "terrain",
}


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Reject manifests that cannot provide a fair, reproducible paired run."""
    if manifest.get("format_version") != 1:
        raise ValueError("evaluation manifest format_version must be 1")

    target_count = manifest.get("episodes_per_lane")
    if type(target_count) is not int or target_count <= 0:
        raise ValueError("episodes_per_lane must be a positive integer")

    lanes = manifest.get("lanes")
    if lanes != list(LANES):
        raise ValueError(f"manifest lanes must be exactly {list(LANES)!r}")

    players = manifest.get("players")
    if (
        not isinstance(players, list)
        or len(players) != 8
        or any(not isinstance(player, str) or not player for player in players)
        or len(set(players)) != 8
    ):
        raise ValueError("manifest must contain exactly eight distinct non-empty player names")

    batches = manifest.get("batches")
    if not isinstance(batches, list) or len(batches) != target_count:
        raise ValueError("manifest batch count must equal episodes_per_lane")

    for batch_number, batch in enumerate(batches, start=1):
        if not isinstance(batch, dict) or batch.get("index") != batch_number:
            raise ValueError(f"manifest batch {batch_number} has an invalid index")
        matches = batch.get("matches")
        if not isinstance(matches, list) or len(matches) != len(LANES):
            raise ValueError(f"manifest batch {batch_number} must contain one match per lane")

        seen_lanes: set[str] = set()
        seen_players: list[str] = []
        for match in matches:
            if not isinstance(match, dict):
                raise ValueError(f"manifest batch {batch_number} contains a non-object match")
            lane = match.get("lane")
            if lane not in LANE_MODES or lane in seen_lanes:
                raise ValueError(f"manifest batch {batch_number} has an invalid or duplicate lane {lane!r}")
            seen_lanes.add(lane)
            if match.get("mode") != LANE_MODES[lane]:
                raise ValueError(f"manifest lane {lane!r} has the wrong mode")
            if match.get("held_out") is not True:
                raise ValueError(f"manifest lane {lane!r} is not marked held-out")
            for field in ("seed", "action_delay_ticks", "observation_delay_ticks"):
                value = match.get(field)
                if type(value) is not int or value < 0:
                    raise ValueError(f"manifest lane {lane!r} has an invalid {field}")
            seen_players.extend((match.get("player_a"), match.get("player_b")))

        if set(seen_players) != set(players) or len(seen_players) != len(set(seen_players)):
            raise ValueError(f"manifest batch {batch_number} must use every player exactly once")


def write_json_atomic(output: Path, value: dict[str, Any]) -> None:
    """Keep dashboard/readers from observing a partially written JSON report."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


class ArenaClient:
    def __init__(self, host: str, port: int):
        self.connection = socket.create_connection((host, port), timeout=10)
        self.connection.setblocking(False)
        self.buffered = b""
        self.next_id = 1
        self.events: list[dict[str, Any]] = []

    def close(self) -> None:
        self.connection.close()

    def command(self, command: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        request_id = self.next_id
        self.next_id += 1
        request = {"type": "command", "id": request_id, "command": command, "payload": payload or {}}
        self.connection.sendall((json.dumps(request) + "\n").encode("utf-8"))
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            message = self.read_message(deadline - time.monotonic())
            if message.get("type") == "event":
                self.events.append(message)
                continue
            if message.get("type") != "response" or message.get("id") != request_id:
                continue
            if not message.get("ok"):
                raise RuntimeError(f"arena command {command!r} failed: {message.get('error')}")
            value = message.get("payload")
            return value if isinstance(value, dict) else {}
        raise TimeoutError(f"arena command {command!r} timed out")

    def read_message(self, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            if b"\n" in self.buffered:
                raw, self.buffered = self.buffered.split(b"\n", 1)
                try:
                    return json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("arena event timed out")
            readable, _, _ = select.select([self.connection], [], [], min(1.0, remaining))
            if not readable:
                continue
            chunk = self.connection.recv(64 * 1024)
            if not chunk:
                raise RuntimeError("arena control connection closed")
            self.buffered += chunk


def generate_manifest(client: ArenaClient, players: list[str], episodes_per_lane: int) -> dict[str, Any]:
    if len(players) != 8 or len(set(players)) != 8:
        raise ValueError("exactly eight distinct bot usernames are required")
    batches: list[dict[str, Any]] = []
    for batch_index in range(episodes_per_lane):
        # Rotate the eight bots so every lane sees different agents/opponents,
        # while the persisted manifest makes both policies use the exact same pairing.
        offset = batch_index % len(players)
        rotated = players[offset:] + players[:offset]
        matches = []
        for lane_index, lane in enumerate(LANES):
            mode = LANE_MODES[lane]
            sampled = client.command("sample_evaluation", {"mode": mode})
            matches.append({
                "lane": lane,
                "mode": mode,
                "seed": int(sampled["arena_seed"]),
                "action_delay_ticks": int(sampled["action_delay_ticks"]),
                "observation_delay_ticks": int(sampled["observation_delay_ticks"]),
                "held_out": bool(sampled.get("held_out")),
                "player_a": rotated[lane_index * 2],
                "player_b": rotated[lane_index * 2 + 1],
            })
        batches.append({"index": batch_index + 1, "matches": matches})
    manifest = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "episodes_per_lane": episodes_per_lane,
        "lanes": list(LANES),
        "players": players,
        "batches": batches,
    }
    validate_manifest(manifest)
    return manifest


def evaluate_manifest(
    client: ArenaClient,
    manifest: dict[str, Any],
    output: Path,
    policy_label: str,
    batch_timeout_seconds: float,
) -> dict[str, Any]:
    validate_manifest(manifest)
    completed: dict[str, list[dict[str, Any]]] = {lane: [] for lane in LANES}
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        for batch in manifest["batches"]:
            refresh_all_cooldowns(client)
            # Explicit starts are allowed above the auto-pair cap. Keeping the
            # cap at one means that, after the first explicit start, the normal
            # matcher cannot claim any of the six bots needed by the other
            # three starts even if their refreshed cooldowns expire.
            client.command("set_max_pairs", {"pairs": 1})
            client.command("resume")
            expected: dict[str, dict[str, Any]] = {}
            for match in batch["matches"]:
                started = client.command("start_match", {
                    "player_a": match["player_a"],
                    "player_b": match["player_b"],
                    "mode": match["mode"],
                    "evaluation": True,
                    "seed": int(match["seed"]),
                })
                if str(started.get("lane")) != str(match["lane"]):
                    raise RuntimeError(f"lane mismatch: expected {match['lane']}, got {started.get('lane')}")
                if int(started.get("arena_seed", -1)) != int(match["seed"]):
                    raise RuntimeError(f"seed mismatch for {match['lane']}")
                for delay_field in ("action_delay_ticks", "observation_delay_ticks"):
                    if started.get(delay_field) != match[delay_field]:
                        raise RuntimeError(
                            f"{delay_field} mismatch for {match['lane']}: "
                            f"expected {match[delay_field]}, got {started.get(delay_field)}"
                        )
                if started.get("held_out") is not True:
                    raise RuntimeError(f"server did not start held-out evaluation for {match['lane']}")
                expected[str(started["episode_id"])] = match

            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            deadline = time.monotonic() + batch_timeout_seconds
            while expected and time.monotonic() < deadline:
                if client.events:
                    event = client.events.pop(0)
                else:
                    event = client.read_message(deadline - time.monotonic())
                if event.get("type") != "event" or event.get("event") != "match_ended":
                    continue
                payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
                episode_id = str(payload.get("episode_id", ""))
                if episode_id not in expected:
                    continue
                entries = grouped[episode_id]
                if any(value.get("agent_id") == event.get("agent_id") for value in entries):
                    continue
                entries.append(event)
                if len(entries) < 2:
                    continue
                episode = summarize_episode(entries)
                match = expected.pop(episode_id)
                if int(episode["arena_seed"]) != int(match["seed"]):
                    raise RuntimeError(f"ended seed mismatch for {match['lane']}")
                completed[match["lane"]].append(episode)
                write_report(output, policy_label, manifest, completed, started_at, complete=False)
            if expected:
                raise TimeoutError(f"batch {batch['index']} did not finish: {sorted(expected)}")
            print(json.dumps({
                "event": "evaluation_batch_complete",
                "policy": policy_label,
                "batch": batch["index"],
                "episodes": {lane: len(values) for lane, values in completed.items()},
            }), flush=True)
    finally:
        try:
            client.command("stop_all")
            client.command("set_max_pairs", {"pairs": 4})
            client.command("resume")
        except Exception as error:
            print(json.dumps({"event": "arena_restore_failed", "error": str(error)}), flush=True)
    return write_report(output, policy_label, manifest, completed, started_at, complete=True)


def refresh_all_cooldowns(client: ArenaClient) -> None:
    """Ensure auto-pairing cannot steal a bot during four explicit starts.

    Fighters from the first match to end can outlive their normal cooldown by
    the time the slowest of four evaluation matches finishes. Briefly allowing
    the normal matcher to bind all eight bots and immediately stopping those
    throwaway matches refreshes every cooldown under one paused boundary.
    """
    client.command("stop_all")
    client.command("set_max_pairs", {"pairs": 4})
    client.command("resume")
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        status = client.command("status")
        if int(status.get("active_pairs", 0)) >= 4:
            client.command("stop_all")
            return
        time.sleep(0.05)
    client.command("stop_all")
    raise TimeoutError("could not bind all eight bots to refresh matchmaking cooldowns")


def write_report(
    output: Path,
    policy_label: str,
    manifest: dict[str, Any],
    completed: dict[str, list[dict[str, Any]]],
    started_at: str,
    *,
    complete: bool,
) -> dict[str, Any]:
    target_count = int(manifest["episodes_per_lane"])
    summaries = {lane: lane_summary(lane, values) for lane, values in completed.items()}
    actually_complete = complete and all(len(completed[lane]) >= target_count for lane in LANES)
    report = {
        "format_version": 1,
        "policy": policy_label,
        "manifest_created_at": manifest.get("created_at"),
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat() if actually_complete else None,
        "episodes_per_lane_requested": target_count,
        "complete": actually_complete,
        "passed": actually_complete and all(value["passed"] for value in summaries.values()),
        "lanes": summaries,
        "episodes": completed,
    }
    write_json_atomic(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or replay a paired four-lane MCAI evaluation manifest")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--generate-manifest", type=Path)
    parser.add_argument("--players", default=",".join(f"MCAI_{index:03d}" for index in range(1, 9)))
    parser.add_argument("--episodes-per-lane", type=int, default=8)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--policy-label")
    parser.add_argument("--batch-timeout-seconds", type=float, default=120.0)
    arguments = parser.parse_args()
    client = ArenaClient(arguments.host, arguments.port)
    try:
        if arguments.generate_manifest:
            players = [value.strip() for value in arguments.players.split(",") if value.strip()]
            manifest = generate_manifest(client, players, max(1, arguments.episodes_per_lane))
            write_json_atomic(arguments.generate_manifest, manifest)
            print(json.dumps({"manifest": str(arguments.generate_manifest), "batches": len(manifest["batches"])}))
            return
        if not arguments.manifest or not arguments.output or not arguments.policy_label:
            parser.error("evaluation requires --manifest, --output, and --policy-label")
        manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
        validate_manifest(manifest)
        report = evaluate_manifest(
            client, manifest, arguments.output, arguments.policy_label, arguments.batch_timeout_seconds
        )
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["complete"] and report["passed"] else 1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
