# MCAI on homebase

Training runs on the existing **homebase** box, not a new VM. No provisioning was needed and no
new Azure spend was created — homebase already runs 24/7.

| | |
|---|---|
| Box | `homebase` · RG `homebase-rg` · westus2 · 8 vCPU / 32 GB · Ubuntu 24.04 |
| Reach it | `ssh -i ~/.ssh/homebase_ed25519 azureuser@100.108.213.121` |
| Repo on box | `~/work/mcai` (branch `fix/verified-training-stack`) |
| Service | `~/services/mcai/compose.yml`, managed by `svc` |
| Data | Docker volume `mcai_mcai-data` → `/data` (checkpoints, runs, server-runtime) |
| Control repo | `~/az_serv` (the operator/agent entry point for the box) |

## Service, not an `hb` job — and why

homebase draws a hard line between **jobs** (`hb`, work that ends, memory *reserved*, may be
stopped to honour someone's reservation) and **services** (`svc`, work that never ends, memory
*capped*, never stopped, survives reboot).

MCAI training is a service here, despite training normally being job-shaped, for three reasons:

1. It is docker-compose supervised, which is exactly what `svc` manages.
2. A broker job wrapping `docker compose` would **double-count memory** — the broker reserves for
   the systemd unit while the real memory lives in the container cgroup, which it already counts
   as `unmanaged`.
3. `hb` caps `max_hours` at 72; this run is meant to continue for weeks.

**The cost, stated plainly:** the broker measures *real* memory, not the cap, so the pool shrinks
by what MCAI actually uses. **Measured: 2.1 GB** of its 11 GB ceiling while training 4 bots — far
less than the 8–11 GB estimated up front. The 11 GB `mem_limit` is a safety ceiling to contain a
leak, not a reservation. `svc mem` shows the split. If the box is needed for something else,
`svc down mcai` stops it cleanly (checkpoints are atomic).

For context, the pool sits around 13.6 GB free; the largest consumer by far is **ollama at ~9.8 GB**,
not MCAI.

## Sizing

4 bots / 2 arena pairs on 8 shared vCPU — roughly one pair per 4 vCPU. Measured **TPS 19.99** at
this size. More bots would push TPS down, and below ~19 the observations the policy trains on go
laggy, which makes the training data *worse*, not more plentiful.

## Operating it

```bash
svc ls                       # state + memory cap vs actual
svc logs mcai -f             # live output
svc down mcai                # stop cleanly (60s grace so checkpoints finish writing)
svc up mcai                  # start again; resumes from /data/checkpoints/latest.pt
```

Change the code and restart:
```bash
cd ~/work/mcai && git pull && svc down mcai && svc up mcai
```

Switch curriculum stage (only once sword is competent — edit `MCAI_MODE` in the compose):
`sword` → `crystal` → `combined`.

## Hourly health check

`~/bin/mcai-hourly-check.sh` runs at **:17 past every hour** via cron **on the box** — not on the
laptop, which is frequently closed. It wraps `deploy/azure/training-healthcheck.sh`.

```bash
tail -40 ~/logs/mcai-health/history.log   # what it found, hour by hour
cat ~/logs/mcai-health/last_status        # 0 healthy · 1 warning · 2 critical
~/bin/mcai-hourly-check.sh                # run it now
```

It checks the things that actually fail *while the container still looks fine*:

| Check | Catches |
|---|---|
| `total_agent_ticks` advanced since last hour | trainer alive but collecting nothing |
| PPO updates completing | rollouts gathered but never learned from |
| `active_pairs ≥ 1` | bots not pairing, so zero experience |
| TPS ≥ 17 | server overloaded → laggy, low-quality observations |
| entropy > 0.05 | policy collapsed to one action, stopped exploring |
| approx KL < 0.5 | training diverging |

## Watching the fights

Ports are published to the box's **loopback only**. Verified with `ss -tlnp` and
`iptables -t nat -L DOCKER`: `docker-proxy` listens on `127.0.0.1` and the DNAT destination is
`127.0.0.1`, so nothing is reachable from the internet. This matters because Docker writes its own
iptables rules and an unqualified `-p 8788:8788` would be world-reachable *regardless of the Azure
NSG*.

```bash
ssh -i ~/.ssh/homebase_ed25519 -L 8788:127.0.0.1:8788 -L 25565:127.0.0.1:25565 \
    azureuser@100.108.213.121
```
Then open `http://127.0.0.1:8788` for the dashboard, or connect a Minecraft 1.12.2 client to
`127.0.0.1:25565` as **AIWatcher** and run `/aiwatch arena-1 orbit` (or `pov`), `/ainext` to cycle.
EaglerXServer v1.1.1 is installed, so an Eaglercraft browser client can join the same server.

## Checking the neurons

```bash
docker exec mcai python -m combat_ai.cli introspect /data/checkpoints/latest.pt
```
Reports tanh saturation, GRU memory use, which observation fields drive each decision, per-head
entropy against maximum, and behavioural probes. Run it now for a baseline and again later — the
probe signals should move from ~0 toward positive as the policy learns.
