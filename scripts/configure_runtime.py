from __future__ import annotations

import argparse
import hashlib
import json
import re
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


def update_yaml_scalar(path: Path, section_path: tuple[str, ...], value: str) -> None:
    """Update an existing scalar in the template's indentation-based YAML."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = 0
    end = len(lines)
    parent_indent = -2
    for depth, key in enumerate(section_path):
        expected_indent = parent_indent + 2
        pattern = re.compile(rf"^\s{{{expected_indent}}}{re.escape(key)}\s*:")
        found = None
        for index in range(start, end):
            line = lines[index]
            stripped = line.lstrip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(line) - len(stripped)
            if indent < expected_indent:
                break
            if indent == expected_indent and pattern.match(line):
                found = index
                break
        if found is None:
            raise RuntimeError(f"Missing expected YAML key {'.'.join(section_path)} in {path}")
        if depth == len(section_path) - 1:
            lines[found] = " " * expected_indent + f"{key}: {value}"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return
        start = found + 1
        end = len(lines)
        for index in range(start, len(lines)):
            stripped = lines[index].lstrip()
            if not stripped or stripped.startswith("#"):
                continue
            if len(lines[index]) - len(stripped) <= expected_indent:
                end = index
                break
        parent_indent = expected_indent


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
        # Eaglercraft's first-run profile chooses a random username. The server
        # is bound to localhost, so a whitelist would only lock out the local
        # spectator on a fresh profile without adding network isolation.
        "white-list": "false",
        "spawn-protection": "0",
        "allow-flight": "true",
        "view-distance": "5",
        "max-players": str(max(16, arguments.bots + 4)),
        "spawn-animals": "false",
        "spawn-monsters": "false",
    })
    update_yaml_scalar(runtime / "spigot.yml", ("world-settings", "default", "view-distance"), "5")
    update_yaml_scalar(runtime / "paper.yml", ("world-settings", "default", "keep-spawn-loaded"), "false")
    update_yaml_scalar(runtime / "paper.yml", ("world-settings", "default", "keep-spawn-loaded-range"), "0")
    update_yaml_scalar(runtime / "paper.yml", ("world-settings", "default", "optimize-explosions"), "true")
    update_yaml_scalar(runtime / "paper.yml", ("timings", "enabled"), "false")
    via_version_config = runtime / "plugins" / "ViaVersion" / "config.yml"
    if via_version_config.exists():
        # Orbiting spectators receive frequent server teleports and answer with
        # movement/teleport-confirm packets, while bot packets can arrive in a
        # burst after a server stall. On this loopback-only server ViaVersion's
        # aggregate PPS heuristics can mistake either case for network abuse.
        update_yaml_scalar(via_version_config, ("max-pps",), "-1")
        update_yaml_scalar(via_version_config, ("tracking-period",), "-1")
    names = [f"{arguments.prefix}{index:03d}" for index in range(1, arguments.bots + 1)]
    names.append(f"{arguments.prefix}BROWSER")
    names.append("AIWatcher")
    whitelist = [{"uuid": offline_uuid(name), "name": name} for name in names]
    (runtime / "whitelist.json").write_text(json.dumps(whitelist, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runtime": str(runtime), "bind": arguments.bind, "whitelisted": names}, indent=2))


if __name__ == "__main__":
    main()
