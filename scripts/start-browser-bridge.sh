#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHECKPOINT="${1:-$ROOT/checkpoints/latest.pt}"
exec "$ROOT/trainer/.venv/bin/mcai-trainer" browser --checkpoint "$CHECKPOINT" --host 127.0.0.1 --port 8767
