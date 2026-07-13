#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNTIME="${MCAI_RUNTIME:-$ROOT/server-runtime}"
JAVA_MEMORY="${MCAI_JAVA_MEMORY:-3G}"

set -a
if [ -f "$ROOT/.env.surface" ]; then source "$ROOT/.env.surface"; fi
set +a

cleanup() {
  if [ -n "${SERVER_PID:-}" ]; then kill "$SERVER_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

(cd "$RUNTIME" && java -Xms1G -Xmx"$JAVA_MEMORY" -jar paper-1.12.2.jar nogui) &
SERVER_PID=$!
for _ in $(seq 1 120); do
  if (exec 3<>/dev/tcp/127.0.0.1/8765) 2>/dev/null; then exec 3>&-; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then wait "$SERVER_PID"; exit 1; fi
  sleep 1
done
(cd "$ROOT/worker" && exec npm start)
