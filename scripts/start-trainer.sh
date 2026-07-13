#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${MCAI_TRAINER_HOST:-0.0.0.0}"
PORT="${MCAI_TRAINER_PORT:-8766}"
CHECKPOINTS="${MCAI_CHECKPOINT_DIR:-$ROOT/checkpoints}"
exec "$ROOT/trainer/.venv/bin/mcai-trainer" serve --host "$HOST" --port "$PORT" --checkpoints "$CHECKPOINTS"
