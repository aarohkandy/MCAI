#!/usr/bin/env bash
# Open the loopback-only SSH tunnels you need to WATCH the fights on the Azure training VM:
#   - the Minecraft server, so a real 1.12.2 client on your machine can spectate in 3D
#   - the dashboard, for the top-down live arena view and PPO metrics
#
# Nothing is ever exposed publicly: every forward binds 127.0.0.1 on both ends, and the VM's
# only open inbound port stays SSH.
#
# Why the Minecraft side needs a relay: deploy/azure runs Paper with server-ip=127.0.0.1 INSIDE
# the container (scripts/configure_runtime.py --bind 127.0.0.1) and docker-compose publishes only
# the dashboard port. Docker's port publishing forwards to a container's eth0, never to its
# loopback, so there is nothing on the VM host to forward to. This script starts a tiny TCP relay
# inside the container that listens on the container's own address and forwards to 127.0.0.1:25565,
# then tunnels to that. The relay is reachable from the VM host only (unpublished container ports
# are not routable from the internet), and it disappears when the container restarts.
#
# See docs/SPECTATING.md for the full walkthrough, including the one-time operator setup.
set -euo pipefail

SSH_HOST=""
SSH_USER="azureuser"
SSH_PORT="22"
IDENTITY=""
CONTAINER="mcai"
LOCAL_MC_PORT="25565"
LOCAL_DASHBOARD_PORT="8788"
REMOTE_DASHBOARD_PORT="8788"
RELAY_PORT="25566"
SERVER_PORT="25565"
REMOTE_MC=""
WANT_MINECRAFT=true
WANT_DASHBOARD=true
LOCAL_ONLY=false

usage() {
  cat <<'USAGE'
Usage: scripts/watch-fights.sh --host <vm-ip> [options]
       scripts/watch-fights.sh --local

Opens SSH tunnels to the MCAI training VM so you can watch the fights, then prints exactly
what to connect to and which in-game commands to run.

Options:
  -H, --host HOST            Public IP or hostname of the training VM (required unless --local)
  -u, --user USER            SSH user (default: azureuser)
  -p, --ssh-port PORT        SSH port (default: 22)
  -i, --identity FILE        SSH private key to use
  -c, --container NAME       Docker container running the stack (default: mcai)
      --mc-port PORT         Local port for the Minecraft tunnel (default: 25565)
      --dashboard-port PORT  Local port for the dashboard tunnel (default: 8788)
      --remote-dashboard PORT  Dashboard port on the VM host (default: 8788)
      --relay-port PORT      Port the in-container relay listens on (default: 25566)
      --remote-mc HOST:PORT  Skip the container relay and forward straight to this address as
                             seen from the VM (use when the stack runs on the VM without Docker,
                             or when you publish the port yourself)
      --no-minecraft         Forward the dashboard only
      --no-dashboard         Forward Minecraft only
      --local                No SSH: the stack is on this machine; just check ports and print
                             the connection instructions
  -h, --help                 Show this help

Examples:
  scripts/watch-fights.sh --host 20.1.2.3
  scripts/watch-fights.sh --host mcai.example.net --identity ~/.ssh/mcai_ed25519
  scripts/watch-fights.sh --host 20.1.2.3 --mc-port 25566   # if 25565 is already taken locally
  scripts/watch-fights.sh --local
USAGE
}

fail() { printf 'watch-fights: %s\n' "$1" >&2; exit 1; }

while [ "$#" -gt 0 ]; do
  case "$1" in
    -H|--host) [ "$#" -ge 2 ] || fail "$1 needs a value"; SSH_HOST="$2"; shift 2 ;;
    -u|--user) [ "$#" -ge 2 ] || fail "$1 needs a value"; SSH_USER="$2"; shift 2 ;;
    -p|--ssh-port) [ "$#" -ge 2 ] || fail "$1 needs a value"; SSH_PORT="$2"; shift 2 ;;
    -i|--identity) [ "$#" -ge 2 ] || fail "$1 needs a value"; IDENTITY="$2"; shift 2 ;;
    -c|--container) [ "$#" -ge 2 ] || fail "$1 needs a value"; CONTAINER="$2"; shift 2 ;;
    --mc-port) [ "$#" -ge 2 ] || fail "$1 needs a value"; LOCAL_MC_PORT="$2"; shift 2 ;;
    --dashboard-port) [ "$#" -ge 2 ] || fail "$1 needs a value"; LOCAL_DASHBOARD_PORT="$2"; shift 2 ;;
    --remote-dashboard) [ "$#" -ge 2 ] || fail "$1 needs a value"; REMOTE_DASHBOARD_PORT="$2"; shift 2 ;;
    --relay-port) [ "$#" -ge 2 ] || fail "$1 needs a value"; RELAY_PORT="$2"; shift 2 ;;
    --remote-mc) [ "$#" -ge 2 ] || fail "$1 needs a value"; REMOTE_MC="$2"; shift 2 ;;
    --no-minecraft) WANT_MINECRAFT=false; shift ;;
    --no-dashboard) WANT_DASHBOARD=false; shift ;;
    --local) LOCAL_ONLY=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; fail "unknown argument: $1" ;;
  esac
done

[ "$WANT_MINECRAFT" = true ] || [ "$WANT_DASHBOARD" = true ] \
  || fail "--no-minecraft and --no-dashboard together leave nothing to do"

# bash's /dev/tcp is enough for a liveness probe and avoids depending on nc/lsof being installed.
port_open() {
  local host="$1" port="$2"
  (exec 3<>"/dev/tcp/${host}/${port}") 2>/dev/null || return 1
  exec 3>&-
  return 0
}

wait_for_port() {
  local host="$1" port="$2" attempts="$3"
  local index=1
  while [ "$index" -le "$attempts" ]; do
    if port_open "$host" "$port"; then return 0; fi
    sleep 1
    index=$((index + 1))
  done
  return 1
}

print_next_steps() {
  local mc_target="$1"
  printf '\n'
  printf '=====================================================================\n'
  printf ' WATCH THE FIGHTS\n'
  printf '=====================================================================\n'
  if [ "$WANT_MINECRAFT" = true ]; then
    printf '\n1. Live 3D (your own Minecraft client does the rendering — the VM has no GPU)\n'
    printf '   Client version : Minecraft 1.12.2 (vanilla; not 1.12, not 1.13+)\n'
    printf '   Username       : AIWatcher   (exact spelling — it is the whitelisted name)\n'
    printf '   Server address : %s\n' "$mc_target"
    printf '   Then in chat, in this order:\n'
    printf '     /aiwatch arena-1 orbit   circling camera; also moves you into the arena world\n'
    printf '     /aiwatch arena-1 pov     lock the camera to a fighter\n'
    printf '     /ainext                  cycle to the next active arena, same mode\n'
    printf '   Arena ids are arena-1, arena-2, ... Only ACTIVE arenas can be watched.\n'
    printf '   NOTE: /aistop is NOT "stop spectating" — it halts every match and kicks the bots.\n'
    printf '         Leave it alone unless you mean it (resume with the dashboard Resume button).\n'
  fi
  if [ "$WANT_DASHBOARD" = true ]; then
    printf '\n2. Dashboard (top-down live arena, health bars, PPO losses)\n'
    printf '   http://127.0.0.1:%s\n' "$LOCAL_DASHBOARD_PORT"
  fi
  printf '\nTroubleshooting and the one-time operator setup: docs/SPECTATING.md\n'
  if [ "$LOCAL_ONLY" = false ]; then
    printf 'Press Ctrl-C to close the tunnel(s).\n'
  fi
  printf '\n'
}

if [ "$LOCAL_ONLY" = true ]; then
  [ -z "$SSH_HOST" ] || fail "--local and --host are mutually exclusive"
  missing=false
  if [ "$WANT_MINECRAFT" = true ] && ! port_open 127.0.0.1 "$LOCAL_MC_PORT"; then
    printf 'watch-fights: nothing is listening on 127.0.0.1:%s (Minecraft). Is the stack running?\n' \
      "$LOCAL_MC_PORT" >&2
    missing=true
  fi
  if [ "$WANT_DASHBOARD" = true ] && ! port_open 127.0.0.1 "$LOCAL_DASHBOARD_PORT"; then
    printf 'watch-fights: nothing is listening on 127.0.0.1:%s (dashboard). Is the stack running?\n' \
      "$LOCAL_DASHBOARD_PORT" >&2
    missing=true
  fi
  [ "$missing" = false ] || fail "start the stack first (scripts/start-linux.sh, or deploy/azure)"
  print_next_steps "127.0.0.1:${LOCAL_MC_PORT}"
  exit 0
fi

[ -n "$SSH_HOST" ] || { usage >&2; fail "--host is required (or use --local)"; }

SSH_OPTIONS=(-o "BatchMode=yes" -o "ExitOnForwardFailure=yes" -o "ServerAliveInterval=30"
             -o "ConnectTimeout=15" -p "$SSH_PORT")
[ -z "$IDENTITY" ] || SSH_OPTIONS+=(-i "$IDENTITY" -o "IdentitiesOnly=yes")
TARGET="${SSH_USER}@${SSH_HOST}"

command -v ssh >/dev/null 2>&1 || fail "ssh is not installed"

echo "watch-fights: checking SSH to ${TARGET}..."
ssh "${SSH_OPTIONS[@]}" "$TARGET" true \
  || fail "cannot SSH to ${TARGET}. Check the IP, --user, and --identity (BatchMode is on, so a key is required)."

# Refuse to start a forward on a local port that is already taken; ExitOnForwardFailure would
# abort the whole tunnel and the reason would be buried in ssh's output.
if [ "$WANT_MINECRAFT" = true ] && port_open 127.0.0.1 "$LOCAL_MC_PORT"; then
  fail "local port ${LOCAL_MC_PORT} is already in use. Pick another with --mc-port (then connect the client to that port)."
fi
if [ "$WANT_DASHBOARD" = true ] && port_open 127.0.0.1 "$LOCAL_DASHBOARD_PORT"; then
  fail "local port ${LOCAL_DASHBOARD_PORT} is already in use. Pick another with --dashboard-port."
fi

FORWARDS=()
MC_TARGET="(not forwarded)"

if [ "$WANT_MINECRAFT" = true ] && [ -n "$REMOTE_MC" ]; then
  case "$REMOTE_MC" in
    *:*) : ;;
    *) fail "--remote-mc wants HOST:PORT, for example 127.0.0.1:25565" ;;
  esac
  echo "watch-fights: forwarding Minecraft straight to ${REMOTE_MC} on the VM (relay skipped)"
  FORWARDS+=(-L "127.0.0.1:${LOCAL_MC_PORT}:${REMOTE_MC}")
  MC_TARGET="127.0.0.1:${LOCAL_MC_PORT}"
elif [ "$WANT_MINECRAFT" = true ]; then
  echo "watch-fights: preparing the in-container Minecraft relay on ${SSH_HOST}..."
  # The remote half runs as one bash script fed over stdin, so nothing has to survive a second
  # round of shell quoting. It prints the container address as its final line.
  REMOTE_SCRIPT="$(mktemp "${TMPDIR:-/tmp}/mcai-watch-remote.XXXXXX")"
  trap 'rm -f "$REMOTE_SCRIPT"' EXIT
  cat >"$REMOTE_SCRIPT" <<'REMOTE'
set -euo pipefail
container="$1"; relay_port="$2"; server_port="$3"

docker_cmd="docker"
if ! docker info >/dev/null 2>&1; then
  if sudo -n docker info >/dev/null 2>&1; then
    docker_cmd="sudo -n docker"
  else
    echo "remote: cannot talk to the Docker daemon as $(id -un). Add yourself to the docker group ('sudo usermod -aG docker \$USER' then log out and back in) or enable passwordless sudo." >&2
    exit 1
  fi
fi

if [ "$($docker_cmd inspect -f '{{.State.Running}}' "$container" 2>/dev/null || echo missing)" != "true" ]; then
  echo "remote: container '$container' is not running. Start it with 'cd MCAI/deploy/azure && docker compose up -d'." >&2
  exit 1
fi

address="$($docker_cmd inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}' "$container" | awk '{print $1}')"
[ -n "$address" ] || { echo "remote: container '$container' has no network address" >&2; exit 1; }

probe() { (exec 3<>"/dev/tcp/$1/$2") 2>/dev/null && exec 3>&- ; }

if ! probe "$address" "$relay_port"; then
  # Written fresh every time: the relay lives only for the container's lifetime, and rewriting
  # it keeps a stale copy from an older revision of this script out of the way.
  $docker_cmd exec -i "$container" sh -c 'cat > /tmp/mcai-spectate-relay.py' <<'PY'
"""Forward <listen_port> on every container interface to Paper on the container's loopback.

Paper deliberately binds 127.0.0.1 inside the container, which Docker's port publishing cannot
reach. This relay is the smallest thing that lets an SSH tunnel get to it. It touches only TCP
bytes -- the game protocol is untouched, so the client is an ordinary vanilla client.
"""
import socket
import sys
import threading

listen_port = int(sys.argv[1])
target_port = int(sys.argv[2])


def pump(source, sink):
    try:
        while True:
            chunk = source.recv(65536)
            if not chunk:
                break
            sink.sendall(chunk)
    except OSError:
        pass
    finally:
        try:
            sink.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def serve(client):
    try:
        upstream = socket.create_connection(("127.0.0.1", target_port), timeout=10)
    except OSError:
        client.close()
        return
    threading.Thread(target=pump, args=(client, upstream), daemon=True).start()
    pump(upstream, client)
    client.close()
    upstream.close()


listener = socket.socket()
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(("0.0.0.0", listen_port))
listener.listen(32)
while True:
    connection, _ = listener.accept()
    threading.Thread(target=serve, args=(connection,), daemon=True).start()
PY
  $docker_cmd exec -d "$container" python3 /tmp/mcai-spectate-relay.py "$relay_port" "$server_port"
  attempt=1
  while [ "$attempt" -le 10 ]; do
    probe "$address" "$relay_port" && break
    sleep 1
    attempt=$((attempt + 1))
  done
fi

if ! probe "$address" "$relay_port"; then
  echo "remote: the relay did not come up on ${address}:${relay_port}. Check 'docker logs $container'." >&2
  exit 1
fi

# Prove Paper itself is accepting connections, not just the relay.
if ! $docker_cmd exec "$container" python3 -c "import socket,sys; s=socket.socket(); s.settimeout(5); sys.exit(s.connect_ex(('127.0.0.1', int('$server_port'))))"; then
  echo "remote: the relay is up but Paper is not accepting connections on 127.0.0.1:${server_port} yet. Give the stack a minute after startup." >&2
  exit 1
fi

echo "MCAI_RELAY_ENDPOINT=${address}:${relay_port}"
REMOTE

  relay_endpoint="$(ssh "${SSH_OPTIONS[@]}" "$TARGET" bash -s -- \
    "$CONTAINER" "$RELAY_PORT" "$SERVER_PORT" <"$REMOTE_SCRIPT")" \
    || fail "could not prepare the Minecraft relay on the VM (see the remote: message above). If the relay started but the VM cannot reach the container's address, forward to a port you publish yourself with --remote-mc HOST:PORT."
  rm -f "$REMOTE_SCRIPT"
  trap - EXIT

  relay_endpoint="${relay_endpoint##*MCAI_RELAY_ENDPOINT=}"
  case "$relay_endpoint" in
    *:*) : ;;
    *) fail "unexpected response from the VM: '${relay_endpoint}'" ;;
  esac
  echo "watch-fights: relay ready at ${relay_endpoint} (inside the VM only)"
  FORWARDS+=(-L "127.0.0.1:${LOCAL_MC_PORT}:${relay_endpoint}")
  MC_TARGET="127.0.0.1:${LOCAL_MC_PORT}"
fi

if [ "$WANT_DASHBOARD" = true ]; then
  FORWARDS+=(-L "127.0.0.1:${LOCAL_DASHBOARD_PORT}:127.0.0.1:${REMOTE_DASHBOARD_PORT}")
fi

echo "watch-fights: opening the tunnel..."
ssh -N "${SSH_OPTIONS[@]}" "${FORWARDS[@]}" "$TARGET" &
TUNNEL_PID=$!
trap 'kill "$TUNNEL_PID" 2>/dev/null || true; wait "$TUNNEL_PID" 2>/dev/null || true' EXIT INT TERM

if [ "$WANT_DASHBOARD" = true ] && ! wait_for_port 127.0.0.1 "$LOCAL_DASHBOARD_PORT" 15; then
  fail "the dashboard tunnel never came up. Is the stack running on the VM ('docker compose logs -f')?"
fi
if [ "$WANT_MINECRAFT" = true ] && ! wait_for_port 127.0.0.1 "$LOCAL_MC_PORT" 15; then
  fail "the Minecraft tunnel never came up."
fi

print_next_steps "$MC_TARGET"

# Hold the tunnel open in the foreground so Ctrl-C (or closing the terminal) tears it down.
wait "$TUNNEL_PID"
