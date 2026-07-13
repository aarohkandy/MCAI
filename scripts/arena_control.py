from __future__ import annotations

import argparse
import json
import socket


def main() -> None:
    parser = argparse.ArgumentParser(description="Send one localhost-only command to MCAIArena")
    parser.add_argument("command")
    parser.add_argument("payload", nargs="?", default="{}", help="JSON object")
    parser.add_argument("--port", type=int, default=8765)
    arguments = parser.parse_args()
    payload = json.loads(arguments.payload)
    if not isinstance(payload, dict):
        raise SystemExit("payload must be a JSON object")
    request = json.dumps({"type": "command", "id": 1, "command": arguments.command, "payload": payload}) + "\n"
    with socket.create_connection(("127.0.0.1", arguments.port), timeout=10) as connection:
        connection.sendall(request.encode("utf-8"))
        with connection.makefile("r", encoding="utf-8") as response:
            line = response.readline()
    if not line:
        raise SystemExit("arena control closed without a response")
    result = json.loads(line)
    print(json.dumps(result, indent=2))
    if not result.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
