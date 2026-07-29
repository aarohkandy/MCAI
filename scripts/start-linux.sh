#!/usr/bin/env bash
# All-in-one Linux launcher: trainer + Paper server + rollout worker + dashboard, all on loopback.
# This is the headless-training counterpart to scripts/start-windows.ps1. It is what the Azure
# container entrypoint execs, but it also runs directly on any Linux/WSL box.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNTIME="${MCAI_RUNTIME:-$ROOT/server-runtime}"
PYTHON="${MCAI_VENV_PYTHON:-$ROOT/trainer/.venv/bin/python}"
CHECKPOINTS="${MCAI_CHECKPOINT_DIR:-$ROOT/checkpoints}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${MCAI_RUN_DIR:-$ROOT/runs/linux-$STAMP}"

# Loopback-only defaults; every port stays on 127.0.0.1 for headless training.
export MCAI_SERVER_HOST="${MCAI_SERVER_HOST:-127.0.0.1}"
export MCAI_SERVER_PORT="${MCAI_SERVER_PORT:-25565}"
export MCAI_ARENA_HOST="${MCAI_ARENA_HOST:-127.0.0.1}"
export MCAI_ARENA_PORT="${MCAI_ARENA_PORT:-8765}"
export MCAI_TRAINER_URL="${MCAI_TRAINER_URL:-ws://127.0.0.1:8766}"
export MCAI_BOT_COUNT="${MCAI_BOT_COUNT:-4}"
export MCAI_USERNAME_PREFIX="${MCAI_USERNAME_PREFIX:-MCAI_}"
export MCAI_DASHBOARD_PORT="${MCAI_DASHBOARD_PORT:-8788}"
export MCAI_ROLLOUT_STEPS="${MCAI_ROLLOUT_STEPS:-8192}"
export MCAI_RUN_DIR="$RUN_DIR"
export MCAI_CHECKPOINT_DIR="$CHECKPOINTS"

JAVA_MEMORY="${MCAI_JAVA_MEMORY:-3G}"
JAVA_INITIAL="${MCAI_JAVA_INITIAL_MEMORY:-1G}"
MODE="${MCAI_MODE:-sword}"
case "$MODE" in sword|crystal|combined) ;; *) echo "MCAI_MODE must be sword, crystal, or combined" >&2; exit 1 ;; esac

for required in "$PYTHON" "$RUNTIME/paper-1.12.2.jar" "$ROOT/worker/dist/src/index.js"; do
  [ -e "$required" ] || { echo "Missing $required. Run the bootstrap first." >&2; exit 1; }
done

mkdir -p "$RUN_DIR" "$CHECKPOINTS"
PIDS=()

cleanup() {
  echo "Shutting down MCAI..."
  "$PYTHON" "$ROOT/scripts/arena_control.py" stop_all >/dev/null 2>&1 || true
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "trainer -> $RUN_DIR/trainer.log"
"$PYTHON" -m combat_ai.cli serve --host 127.0.0.1 --port 8766 \
  --checkpoints "$CHECKPOINTS" --rollout-steps "$MCAI_ROLLOUT_STEPS" \
  --cpu-threads "${MCAI_TORCH_THREADS:-0}" \
  ${MCAI_IMITATION_DATA:+--imitation-data "$MCAI_IMITATION_DATA"} \
  >"$RUN_DIR/trainer.log" 2>&1 &
PIDS+=($!)

echo "paper -> $RUN_DIR/paper.log"
( cd "$RUNTIME" && exec java -Xms"$JAVA_INITIAL" -Xmx"$JAVA_MEMORY" -jar paper-1.12.2.jar nogui ) \
  >"$RUN_DIR/paper.log" 2>&1 &
PIDS+=($!)

echo "waiting for the arena control socket on 127.0.0.1:${MCAI_ARENA_PORT}..."
arena_ready=false
for _ in $(seq 1 240); do
  if (exec 3<>"/dev/tcp/127.0.0.1/${MCAI_ARENA_PORT}") 2>/dev/null; then exec 3>&-; arena_ready=true; break; fi
  for pid in "${PIDS[@]}"; do kill -0 "$pid" 2>/dev/null || { echo "A process exited before the arena was ready. See $RUN_DIR." >&2; exit 1; }; done
  sleep 1
done
[ "$arena_ready" = true ] || { echo "Arena did not become ready; see $RUN_DIR/paper.log." >&2; exit 1; }

"$PYTHON" "$ROOT/scripts/arena_control.py" set_mode "{\"mode\":\"$MODE\"}" >/dev/null
"$PYTHON" "$ROOT/scripts/arena_control.py" resume >/dev/null

echo "dashboard -> $RUN_DIR/dashboard.log"
( cd "$ROOT" && exec node dashboard/server.mjs ) >"$RUN_DIR/dashboard.log" 2>&1 &
PIDS+=($!)

echo "worker -> $RUN_DIR/worker.log"
( cd "$ROOT/worker" && exec node dist/src/index.js ) >"$RUN_DIR/worker.log" 2>&1 &
PIDS+=($!)

echo "MCAI is running in '$MODE' mode. Dashboard: http://127.0.0.1:${MCAI_DASHBOARD_PORT} (reach it via SSH tunnel). Logs: $RUN_DIR"
while true; do
  for pid in "${PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "A training process exited; stopping the stack. Check the logs in $RUN_DIR." >&2
      exit 1
    fi
  done
  sleep 2
done
