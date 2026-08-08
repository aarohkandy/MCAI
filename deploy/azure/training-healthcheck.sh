#!/usr/bin/env bash
# Hourly health check for an MCAI training run on homebase.
#
# "The container is up" is NOT health. Every failure mode that actually matters here leaves the
# container running and the logs flowing while the model learns nothing:
#   - bots never pair, so no experience is generated at all
#   - the trainer collects forever but never completes a PPO update
#   - Paper's TPS collapses, so the observations being learned from are laggy and wrong
#   - entropy collapses to ~0 (policy degenerate) or KL explodes (diverging)
# This checks for those directly, from the trainer's own JSONL log and the arena control socket.
#
# Exit codes: 0 healthy, 1 warning, 2 critical. Designed to be run from cron/launchd hourly.
#
# Usage:
#   ./training-healthcheck.sh [--json] [--container mcai] [--stall-minutes 30]
set -uo pipefail

CONTAINER="mcai"
STALL_MINUTES=30
MIN_TPS=17.0
AS_JSON=false
STATE_DIR="${MCAI_HEALTH_STATE:-$HOME/logs/mcai-health}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --json) AS_JSON=true; shift ;;
    --container) CONTAINER="$2"; shift 2 ;;
    --stall-minutes) STALL_MINUTES="$2"; shift 2 ;;
    -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$STATE_DIR"
PREV_TICKS_FILE="$STATE_DIR/last_total_ticks"
STATUS=0
FINDINGS=()
note() { FINDINGS+=("ok|$1"); }
warn() { FINDINGS+=("WARN|$1"); [ "$STATUS" -lt 1 ] && STATUS=1; return 0; }
crit() { FINDINGS+=("CRIT|$1"); STATUS=2; return 0; }

# --- container ------------------------------------------------------------------------------
if ! docker ps --filter "name=^${CONTAINER}$" --format '{{.Names}}' | grep -q .; then
  crit "container '$CONTAINER' is not running"
  # Everything below reads from inside it, so stop here rather than emit misleading all-clears.
  FINAL_TICKS=""; UPDATES=""; TPS=""; PAIRS=""
else
  note "container up ($(docker ps --filter "name=^${CONTAINER}$" --format '{{.Status}}'))"

  RUN_DIR="$(docker exec "$CONTAINER" sh -lc 'ls -1dt /data/runs/*/ 2>/dev/null | head -1' 2>/dev/null | tr -d '\r')"
  if [ -z "$RUN_DIR" ]; then
    crit "no run directory under /data/runs -- the stack never started"
  else
    LOG="$RUN_DIR/trainer.log"

    # --- is data flowing? ---------------------------------------------------------------------
    FINAL_TICKS="$(docker exec "$CONTAINER" sh -lc \
      "grep -h '\"total_agent_ticks\"' '$LOG' 2>/dev/null | tail -1 | sed -n 's/.*\"total_agent_ticks\": *\([0-9]*\).*/\1/p'" 2>/dev/null | tr -d '\r')"
    FINAL_TICKS="${FINAL_TICKS:-0}"
    if [ -f "$PREV_TICKS_FILE" ]; then
      PREV="$(cat "$PREV_TICKS_FILE" 2>/dev/null || echo 0)"
      if [ "$FINAL_TICKS" -le "$PREV" ] 2>/dev/null; then
        crit "total_agent_ticks stuck at $FINAL_TICKS since the last check -- no experience is being collected"
      else
        note "experience flowing: $PREV -> $FINAL_TICKS agent-ticks"
      fi
    else
      note "first run, baseline total_agent_ticks=$FINAL_TICKS"
    fi
    echo "$FINAL_TICKS" > "$PREV_TICKS_FILE"

    # --- are PPO updates completing? ----------------------------------------------------------
    UPDATES="$(docker exec "$CONTAINER" sh -lc "grep -hc '\"event\": \"ppo_update\"' '$LOG' 2>/dev/null" 2>/dev/null | tr -d '\r')"
    UPDATES="${UPDATES:-0}"
    LAST_UPDATE_AGE="$(docker exec "$CONTAINER" sh -lc \
      "awk '/ppo_update/{n=NR} END{print NR-n}' '$LOG' 2>/dev/null" 2>/dev/null | tr -d '\r')"
    if [ "$UPDATES" -eq 0 ] 2>/dev/null; then
      warn "no PPO update yet (updates=0) -- normal only for the first rollout; investigate if it persists"
    else
      note "PPO updates completed: $UPDATES"
    fi

    # --- policy sanity: entropy collapse / KL explosion ---------------------------------------
    METRICS="$(docker exec "$CONTAINER" sh -lc \
      "grep -h '\"event\": \"ppo_update\"' '$LOG' 2>/dev/null | tail -1" 2>/dev/null | tr -d '\r')"
    if [ -n "$METRICS" ]; then
      ENT="$(printf '%s' "$METRICS" | sed -n 's/.*"entropy": *\([0-9.eE+-]*\).*/\1/p')"
      KL="$(printf '%s' "$METRICS" | sed -n 's/.*"approximate_kl": *\([0-9.eE+-]*\).*/\1/p')"
      # Entropy near zero means every action head has collapsed to one choice: the policy has
      # stopped exploring and will not improve. KL above ~0.5 per update means the step size is
      # destroying the policy rather than refining it.
      awk -v e="${ENT:-1}" 'BEGIN{exit !(e < 0.05)}' && warn "entropy collapsed to $ENT -- policy is degenerate" \
        || note "entropy healthy ($ENT)"
      awk -v k="${KL:-0}" 'BEGIN{exit !(k > 0.5)}' && warn "approximate_kl $KL is very high -- training may be diverging" \
        || note "KL stable ($KL)"
    fi
  fi

  # --- arena: are bots actually fighting, and is the server keeping up? ----------------------
  ARENA="$(docker exec "$CONTAINER" sh -lc \
    'python3 /app/scripts/arena_control.py status 2>/dev/null' 2>/dev/null | tr -d '\r')"
  if [ -z "$ARENA" ]; then
    crit "arena control socket did not answer -- the Minecraft server is down or wedged"
  else
    PAIRS="$(printf '%s' "$ARENA" | sed -n 's/.*"active_pairs": *\([0-9]*\).*/\1/p' | head -1)"
    TPS="$(printf '%s' "$ARENA" | sed -n 's/.*"estimated_tps": *\([0-9.]*\).*/\1/p' | head -1)"
    [ "${PAIRS:-0}" -ge 1 ] 2>/dev/null && note "active arena pairs: $PAIRS" \
      || crit "no active arena pairs -- bots are not fighting, so nothing is being learned"
    awk -v t="${TPS:-0}" -v m="$MIN_TPS" 'BEGIN{exit !(t < m)}' \
      && warn "server TPS ${TPS} is below ${MIN_TPS} -- observations are laggy and training data quality is degraded" \
      || note "server TPS healthy (${TPS})"
  fi
fi

if [ "$AS_JSON" = true ]; then
  printf '{"status":%d,"total_agent_ticks":%s,"ppo_updates":%s,"active_pairs":%s,"tps":%s,"findings":[' \
    "$STATUS" "${FINAL_TICKS:-0}" "${UPDATES:-0}" "${PAIRS:-0}" "${TPS:-0}"
  sep=""
  for f in "${FINDINGS[@]}"; do
    printf '%s{"level":"%s","message":"%s"}' "$sep" "${f%%|*}" "$(printf '%s' "${f#*|}" | sed 's/"/\\"/g')"
    sep=","
  done
  printf ']}\n'
else
  echo "MCAI training health -- $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  for f in "${FINDINGS[@]}"; do printf '  %-5s %s\n' "${f%%|*}" "${f#*|}"; done
  case "$STATUS" in
    0) echo "  -- training healthy --" ;;
    1) echo "  -- WARNINGS present --" ;;
    2) echo "  -- CRITICAL: training is not progressing --" ;;
  esac
fi
exit "$STATUS"
