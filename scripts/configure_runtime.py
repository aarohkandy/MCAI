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
        "view-distance": "8",
        "max-players": str(max(12, arguments.bots + 4)),
    })
    names = [f"{arguments.prefix}{index:03d}" for index in range(1, arguments.bots + 1)]
    names.append(f"{arguments.prefix}BROWSER")
    names.append("AIWatcher")
    whitelist = [{"uuid": offline_uuid(name), "name": name} for name in names]
    (runtime / "whitelist.json").write_text(json.dumps(whitelist, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runtime": str(runtime), "bind": arguments.bind, "whitelisted": names}, indent=2))


if __name__ == "__main__":
    main()
