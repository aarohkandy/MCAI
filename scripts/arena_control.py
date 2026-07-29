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
    request_id = 1
    request = json.dumps({"type": "command", "id": request_id, "command": arguments.command,
                          "payload": payload}) + "\n"
    result = None
    with socket.create_connection(("127.0.0.1", arguments.port), timeout=10) as connection:
        connection.sendall(request.encode("utf-8"))
        with connection.makefile("r", encoding="utf-8") as response:
            for line in response:
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # The server also broadcasts async {"type":"event"} lines to every client, so
                # skip anything that is not this command's response.
                if message.get("type") == "response" and message.get("id") == request_id:
                    result = message
                    break
    if result is None:
        raise SystemExit("arena control closed without a response")
    print(json.dumps(result, indent=2))
    if not result.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
