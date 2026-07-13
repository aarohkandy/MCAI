#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export UV_INSTALL_DIR="${UV_INSTALL_DIR:-$HOME/.local/bin}"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$UV_INSTALL_DIR:$PATH"
fi

uv python install 3.12
if [ -x "$ROOT/trainer/.venv/bin/python" ]; then
  VENV_VERSION="$($ROOT/trainer/.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [ "$VENV_VERSION" != "3.12" ]; then
    echo "trainer/.venv uses Python $VENV_VERSION; move it aside, then rerun this bootstrap." >&2
    exit 1
  fi
else
  uv venv --python 3.12 "$ROOT/trainer/.venv"
fi
uv pip install --python "$ROOT/trainer/.venv/bin/python" -e "$ROOT/trainer[dev,export]"
"$ROOT/trainer/.venv/bin/pytest" -q "$ROOT/trainer/tests"
"$ROOT/trainer/.venv/bin/python" "$ROOT/scripts/verify_model_parity.py"
echo "Mac development checks passed. Production training remains on the Windows Surface."
