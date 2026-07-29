#!/usr/bin/env bash
# Idempotent container startup: (re)build the components from the mounted repo, make sure the
# Paper server runtime exists on the persistent volume, then launch the full headless stack.
# Running `docker compose restart` after a `git pull` re-applies code changes and resumes
# training from the latest checkpoint on /data.
set -euo pipefail

APP=/app
DATA=/data
RUNTIME="${MCAI_RUNTIME:-$DATA/server-runtime}"
CHECKPOINTS="${MCAI_CHECKPOINT_DIR:-$DATA/checkpoints}"
RUN_BASE="${MCAI_RUN_DIR_BASE:-$DATA/runs}"
BOTS="${MCAI_BOT_COUNT:-4}"
MAVEN_REPO="$DATA/.m2"

mkdir -p "$DATA" "$CHECKPOINTS" "$RUN_BASE" "$RUNTIME/plugins" "$MAVEN_REPO"

echo "[entrypoint] registering the trainer package (editable)..."
python3 -c "import combat_ai" 2>/dev/null || pip install -e "$APP/trainer" --no-deps

echo "[entrypoint] building the rollout worker..."
cd "$APP/worker"
[ -d node_modules ] || npm ci
npm run build

echo "[entrypoint] building the arena plugin..."
cd "$APP"
mvn -q -Dmaven.repo.local="$MAVEN_REPO" -DskipTests -f server-plugin/pom.xml package
ARENA_JAR="$(find server-plugin/target -maxdepth 1 -name 'mcai-arena-*.jar' ! -name 'original-*.jar' | sort | head -n1)"
[ -n "$ARENA_JAR" ] || { echo "[entrypoint] no built arena jar found" >&2; exit 1; }
cp "$ARENA_JAR" "$RUNTIME/plugins/MCAIArena.jar"

# Official Paper 1.12.2 server jar. Fetched at runtime into the volume; never baked into the image
# or committed, so no game binaries are redistributed. Headless self-play needs only Paper + the
# MCAIArena plugin (Mineflayer bots connect over the vanilla protocol).
if [ ! -f "$RUNTIME/paper-1.12.2.jar" ]; then
  # PaperMC's v2 API returns 410 Gone for legacy versions; the v3 ("fill") API still serves
  # 1.12.2 and publishes a sha256 we verify before trusting the jar.
  echo "[entrypoint] downloading Paper 1.12.2 (official PaperMC v3 API, checksum-verified)..."
  RUNTIME="$RUNTIME" python3 - <<'PY'
import hashlib, json, os, pathlib, urllib.request
def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "mcai-setup"}))
builds = json.load(get("https://fill.papermc.io/v3/projects/paper/versions/1.12.2/builds"))
download = builds[0]["downloads"]["server:default"]
data = get(download["url"]).read()
digest = hashlib.sha256(data).hexdigest()
if digest != download["checksums"]["sha256"]:
    raise SystemExit("Paper jar checksum mismatch; refusing to install")
pathlib.Path(os.environ["RUNTIME"], "paper-1.12.2.jar").write_bytes(data)
print(f"[entrypoint] installed {download['name']} (build {builds[0]['id']})")
PY
fi

# Minecraft EULA: headless servers require explicit acceptance by the operator.
if [ "${MCAI_ACCEPT_EULA:-}" = "true" ]; then
  echo "eula=true" > "$RUNTIME/eula.txt"
elif [ ! -f "$RUNTIME/eula.txt" ]; then
  echo "[entrypoint] Minecraft EULA not accepted. Re-run with MCAI_ACCEPT_EULA=true." >&2
  exit 1
fi

echo "[entrypoint] configuring server.properties + whitelist for $BOTS bots..."
python3 scripts/configure_runtime.py "$RUNTIME" --bind 127.0.0.1 --bots "$BOTS"

# Write the arena plugin config, mirroring scripts/bootstrap-windows.ps1. Without this the plugin
# keeps its bundled default of 2 concurrent pairs no matter how large the VM is, so most of the
# paid-for cores would sit idle.
MODE="${MCAI_MODE:-sword}"
case "$MODE" in sword|crystal|combined) ;; *) echo "MCAI_MODE must be sword, crystal, or combined" >&2; exit 1 ;; esac
MAX_PAIRS="${MCAI_MAX_PAIRS:-$(( BOTS / 2 ))}"
[ "$MAX_PAIRS" -ge 1 ] 2>/dev/null || MAX_PAIRS=1
mkdir -p "$RUNTIME/plugins/MCAIArena"
cat > "$RUNTIME/plugins/MCAIArena/config.yml" <<CONFIG
world-name: mcai_training
control-port: 8765
max-concurrent-pairs: $MAX_PAIRS
bot-name-prefix: MCAI_
match-timeout-seconds: 120
auto-pair-bots: true
default-mode: $MODE
arena-spacing: 96
shaping-scale: 1.0
CONFIG
echo "[entrypoint] arena config: mode=$MODE max-concurrent-pairs=$MAX_PAIRS bots=$BOTS"

export MCAI_VENV_PYTHON=python3
export MCAI_RUN_DIR="$RUN_BASE/linux-$(date +%Y%m%d-%H%M%S)"
echo "[entrypoint] launching the training stack..."
exec "$APP/scripts/start-linux.sh"
