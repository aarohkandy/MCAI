#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNTIME="${MCAI_RUNTIME:-$ROOT/server-runtime}"
BIND_ADDRESS="${MCAI_BIND_ADDRESS:-127.0.0.1}"
BOT_COUNT="${MCAI_BOT_COUNT:-4}"
MAVEN_VERSION="3.9.9"
MAVEN_HOME="$ROOT/.tools/apache-maven-$MAVEN_VERSION"

for command in git curl node npm python3 java; do
  command -v "$command" >/dev/null || { echo "Missing required command: $command" >&2; exit 1; }
done

JAVA_MAJOR="$(java -version 2>&1 | awk -F'[\".]' '/version/ {print ($2 == "1" ? $3 : $2); exit}')"
if [ "${JAVA_MAJOR:-0}" -lt 17 ]; then
  echo "Current EaglerXServer requires Java 17 or newer." >&2
  exit 1
fi

if [ ! -d "$RUNTIME/.git" ]; then
  if [ -e "$RUNTIME" ]; then
    echo "$RUNTIME exists but is not the expected template checkout; refusing to overwrite it." >&2
    exit 1
  fi
  git clone --depth 1 https://github.com/Eaglercraft-Templates/Eaglercraft-Server-Paper.git "$RUNTIME"
fi

mkdir -p "$RUNTIME/plugins" "$ROOT/.tools"
python3 - "$RUNTIME/plugins/EaglerXServer.jar" <<'PY'
import json, pathlib, sys, urllib.request
target = pathlib.Path(sys.argv[1])
with urllib.request.urlopen("https://api.github.com/repos/lax1dude/eaglerxserver/releases/latest") as response:
    release = json.load(response)
asset = next(value for value in release["assets"] if value["name"] == "EaglerXServer.jar")
with urllib.request.urlopen(asset["browser_download_url"]) as response:
    target.write_bytes(response.read())
print(f"Installed EaglerXServer {release['tag_name']} at {target}")
PY

if [ ! -x "$MAVEN_HOME/bin/mvn" ]; then
  curl -fsSL "https://archive.apache.org/dist/maven/maven-3/$MAVEN_VERSION/binaries/apache-maven-$MAVEN_VERSION-bin.tar.gz" -o "$ROOT/.tools/maven.tar.gz"
  tar -xzf "$ROOT/.tools/maven.tar.gz" -C "$ROOT/.tools"
fi

"$MAVEN_HOME/bin/mvn" -q -f "$ROOT/server-plugin/pom.xml" package
# Match the built jar by glob so a pom <version> bump does not silently break the copy. Prefer the
# shaded artifact (mcai-arena-<version>.jar), excluding Maven's original-*.jar.
ARENA_JAR="$(find "$ROOT/server-plugin/target" -maxdepth 1 -name 'mcai-arena-*.jar' ! -name 'original-*.jar' | sort | head -n1)"
if [ -z "$ARENA_JAR" ]; then echo "No built mcai-arena jar found in server-plugin/target" >&2; exit 1; fi
cp "$ARENA_JAR" "$RUNTIME/plugins/MCAIArena.jar"
python3 "$ROOT/scripts/configure_runtime.py" "$RUNTIME" --bind "$BIND_ADDRESS" --bots "$BOT_COUNT"

# Paper will not start until the Minecraft EULA is accepted (the Windows launcher writes this after
# the operator accepts). On this dev/WSL path, opt in explicitly with MCAI_ACCEPT_EULA=true.
if [ "${MCAI_ACCEPT_EULA:-}" = "true" ]; then
  echo "eula=true" > "$RUNTIME/eula.txt"
elif [ ! -f "$RUNTIME/eula.txt" ]; then
  echo "Note: set MCAI_ACCEPT_EULA=true to accept the Minecraft EULA before start-surface.sh (writes $RUNTIME/eula.txt)." >&2
fi

mkdir -p "$RUNTIME/plugins-disabled"
find "$RUNTIME/plugins" -maxdepth 1 -iname 'AuthMe*.jar' -exec mv {} "$RUNTIME/plugins-disabled/" \;

(cd "$ROOT/worker" && npm ci && npm run build && npm test -- --run)
echo "Linux/WSL alternate rollout setup ready. The supported Surface deployment is scripts/bootstrap-windows.ps1."
