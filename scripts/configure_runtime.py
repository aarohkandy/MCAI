from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from pathlib import Path


def offline_uuid(username: str) -> str:
    digest = bytearray(hashlib.md5(f"OfflinePlayer:{username}".encode("utf-8")).digest())
    digest[6] = (digest[6] & 0x0F) | 0x30
    digest[8] = (digest[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(digest)))


def update_properties(path: Path, updates: dict[str, str]) -> None:
    existing: list[str] = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    seen: set[str] = set()
    output: list[str] = []
    for line in existing:
        key = line.split("=", 1)[0] if "=" in line and not line.lstrip().startswith("#") else None
        if key in updates:
            output.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            output.append(line)
    output.extend(f"{key}={value}" for key, value in updates.items() if key not in seen)
    path.write_text("\n".join(output) + "\n", encoding="utf-8")


BUKKIT_LIMITS = {
    "spawn-limits": {"monsters": "0", "animals": "0", "water-animals": "0", "ambient": "0"},
    "ticks-per": {"animal-spawns": "-1", "monster-spawns": "-1"},
}


def _write_bukkit_limits(path: Path) -> None:
    """Belt-and-braces mob suppression. server.properties already disables natural spawning; these
    per-category limits stop anything that slips through from accumulating AI-ticking entities.

    The server writes its own bukkit.yml on first boot, so this MERGES into an existing file
    (rewriting only the keys we own) instead of skipping it."""
    if not path.exists():
        lines = ["spawn-limits:", "  monsters: 70", "  animals: 15",
                 "  water-animals: 5", "  ambient: 15",
                 "ticks-per:", "  animal-spawns: 400", "  monster-spawns: 1"]
    else:
        lines = path.read_text(encoding="utf-8").splitlines()

    section: str | None = None
    output: list[str] = []
    seen: dict[str, set[str]] = {name: set() for name in BUKKIT_LIMITS}
    for line in lines:
        stripped = line.strip()
        if stripped and not line.startswith((" ", "\t")) and stripped.endswith(":"):
            section = stripped[:-1]
        elif section in BUKKIT_LIMITS and ":" in stripped and not stripped.startswith("#"):
            key = stripped.split(":", 1)[0].strip()
            if key in BUKKIT_LIMITS[section]:
                seen[section].add(key)
                output.append(f"  {key}: {BUKKIT_LIMITS[section][key]}")
                continue
        output.append(line)

    for name, values in BUKKIT_LIMITS.items():
        missing = [k for k in values if k not in seen[name]]
        if not missing:
            continue
        if any(line.strip() == f"{name}:" for line in output):
            index = next(i for i, line in enumerate(output) if line.strip() == f"{name}:")
            for offset, key in enumerate(missing, start=1):
                output.insert(index + offset, f"  {key}: {values[key]}")
        else:
            output.append(f"{name}:")
            output.extend(f"  {key}: {values[key]}" for key in missing)
    path.write_text("\n".join(output) + "\n", encoding="utf-8")


# Ticks before a spent arrow despawns. Vanilla/Paper default is 1200 (60s), but the kit's bow has
# Infinity, so a bot that spams it can leave dozens of arrows alive at once. The observation only
# exposes the 16 nearest combat entities, so those arrows would evict end crystals from the policy's
# view — exactly the entities crystal PvP depends on. 100 ticks (5s) keeps arrows visible while they
# are still in flight and relevant, without letting them accumulate.
# NOTE: player-fired arrows (which is all of them here) are governed by spigot.yml's
# `arrow-despawn-rate`; paper.yml only carries the non-player variant. Both are set.
ARROW_LIMITS = {"arrow-despawn-rate": "100", "non-player-arrow-despawn-rate": "100"}


def _write_arrow_limits(path: Path) -> None:
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    for line in lines:
        key = line.strip().split(":", 1)[0].strip()
        if key in ARROW_LIMITS and line.strip().endswith(tuple("0123456789")):
            indent = line[: len(line) - len(line.lstrip())]
            output.append(f"{indent}{key}: {ARROW_LIMITS[key]}")
            continue
        output.append(line)
    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime", type=Path)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--bots", type=int, default=4)
    parser.add_argument("--prefix", default="MCAI_")
    arguments = parser.parse_args()
    runtime = arguments.runtime.resolve()
    update_properties(runtime / "server.properties", {
        "server-ip": arguments.bind,
        "server-port": "25565",
        "online-mode": "false",
        "white-list": "true",
        "spawn-protection": "0",
        "view-distance": "6",
        "max-players": str(max(12, arguments.bots + 4)),
        # Every CPU cycle spent on vanilla world simulation is a cycle not spent on arenas.
        # Mob AI is also actively dangerous here: passive mobs in the unused default world
        # run PathfinderGoalRandomStroll, which force-loads chunks and can stall the main
        # thread past Paper's 60s watchdog, killing the server mid-training.
        "spawn-animals": "false",
        "spawn-monsters": "false",
        "spawn-npcs": "false",
        "generate-structures": "false",
        "level-type": "FLAT",
        "allow-nether": "false",
        "enable-command-block": "false",
        "announce-player-achievements": "false",
        "difficulty": "2",
    })
    _write_bukkit_limits(runtime / "bukkit.yml")
    _write_arrow_limits(runtime / "paper.yml")
    _write_arrow_limits(runtime / "spigot.yml")
    names = [f"{arguments.prefix}{index:03d}" for index in range(1, arguments.bots + 1)]
    names.append(f"{arguments.prefix}BROWSER")
    names.append("AIWatcher")
    whitelist = [{"uuid": offline_uuid(name), "name": name} for name in names]
    (runtime / "whitelist.json").write_text(json.dumps(whitelist, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runtime": str(runtime), "bind": arguments.bind, "whitelisted": names}, indent=2))


if __name__ == "__main__":
    main()
