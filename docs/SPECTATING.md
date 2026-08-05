# Watching the fights

Three ways to see what the policy is doing, in order of how much they show you:

| | what you get | needs |
|---|---|---|
| **Live 3D** | the real game, from a circling camera or from a fighter's eyes | a Minecraft 1.12.2 client on your machine + an SSH tunnel |
| **Dashboard** | top-down animated arena, health, damage, PPO losses | a browser + an SSH tunnel |
| **Eaglercraft** | fighting the bot yourself, in a browser | your own legitimate Eaglercraft build (see [EAGLER.md](EAGLER.md)) |

Both tunnels come from one command:

```bash
scripts/watch-fights.sh --host <vm-public-ip>
```

It forwards Minecraft to `127.0.0.1:25565` and the dashboard to `127.0.0.1:8788`, checks that both
are actually listening, prints what to connect to, and holds the tunnel open until Ctrl-C.
Everything binds loopback on both ends — nothing is ever exposed to the internet, and the VM's only
open inbound port stays SSH. `scripts/watch-fights.sh --help` lists the options
(`--user`, `--identity`, `--mc-port`, `--dashboard-port`, `--local`, …).

If the stack is running on the machine you are sitting at, use `scripts/watch-fights.sh --local`:
it skips SSH, verifies the ports, and prints the same instructions.

---

## One-time setup

Do this once per VM. It survives restarts and evictions (it lives on the `mcai-data` volume).

### 1. Make `AIWatcher` an operator

`/aiwatch` and `/ainext` require the `mcai.spectate` permission, which defaults to op
(`server-plugin/src/main/resources/plugin.yml`). The headless stack runs Paper with no attached
console, so there is no place to type `/op` — write `ops.json` directly instead. The server reads
it at startup, so do it while the container is stopped:

```bash
cd ~/MCAI/deploy/azure
docker compose stop
docker compose run --rm -T --entrypoint python3 mcai - <<'PY'
import json, pathlib, sys
sys.path.insert(0, "/app/scripts")
from configure_runtime import offline_uuid   # same offline UUID the whitelist uses

ops = pathlib.Path("/data/server-runtime/ops.json")
existing = json.loads(ops.read_text()) if ops.exists() and ops.read_text().strip() else []
if not any(entry.get("name") == "AIWatcher" for entry in existing):
    existing.append({"uuid": offline_uuid("AIWatcher"), "name": "AIWatcher",
                     "level": 4, "bypassesPlayerLimit": False})
ops.write_text(json.dumps(existing, indent=2) + "\n")
print(ops.read_text())
PY
docker compose start
```

It is idempotent — re-running it changes nothing. Expect `AIWatcher` with uuid
`e92a7a53-ad1a-308b-9f01-abbab03ada66` (the offline UUID for that exact name).

Running the stack directly (`scripts/start-linux.sh`, no Docker)? Same file, at
`server-runtime/ops.json`, then restart the stack.

### 2. Point a Minecraft **1.12.2** client at it, named `AIWatcher`

The server runs `online-mode=false` with `white-list=true`, and
`scripts/configure_runtime.py` regenerates `whitelist.json` on **every container start** with
exactly: `MCAI_001…MCAI_00N`, `MCAI_BROWSER`, and `AIWatcher`. So your client must present the
username `AIWatcher` — anything else is kicked as not-whitelisted, and hand-editing the whitelist
is undone at the next restart.

Use a launcher that lets you choose the in-game name for an offline session (Prism Launcher,
MultiMC, …) on a copy of Minecraft you own, create a 1.12.2 instance, and set the name to
`AIWatcher` (capital A, capital W, no spaces).

---

## 1. Live 3D — the good one

```bash
scripts/watch-fights.sh --host <vm-public-ip>
```

Then in the client: **Multiplayer → Direct Connect → `127.0.0.1:25565`** (or whatever
`--mc-port` you passed). In chat:

```
/aiwatch arena-1 orbit     camera circling the arena — start with this one
/aiwatch arena-1 pov       camera locked to a fighter's eyes
/ainext                    next active arena, keeping the current mode
```

- **Start with `orbit`.** Orbit teleports you into the arena world (`mcai_training`); you spawn in
  the ordinary default world, and `pov` alone only re-points the camera, it does not move you
  between worlds. Once orbit has put you in the arena world, `pov` works.
- Orbit sits 12 blocks out and 9 blocks up, pitched 25° down, and takes ~21 s per lap
  (`SpectatorController.tick()`).
- `pov` attaches to the first fighter in the arena. Run `/ainext` twice to come back around if you
  want the other one, or watch from orbit and read names off the dashboard.
- Arena ids are `arena-1`, `arena-2`, … up to `max-concurrent-pairs`. Only **active** arenas can be
  watched; an idle id answers `That arena is not active.` List the live ones with:
  ```bash
  ssh <user>@<vm-ip> 'cd ~/MCAI/deploy/azure && docker compose exec -T mcai python3 scripts/arena_control.py status'
  ```
- Each arena is a 21×21 stone floor at y=63 walled with barrier blocks, in an otherwise empty void
  world, and arenas are 96 blocks apart — you will not see a neighbouring fight by accident.
- **`/aistop` is not "stop spectating".** It halts every match, releases all AI control, and kicks
  every `MCAI_*` account (`MCAIPlugin.onCommand`). To stop watching, just disconnect. If you do hit
  it, press **Resume fights** on the dashboard.
- When a match ends, the camera stops tracking. `/ainext` re-locks onto a live arena.

**Why this beats RDP/VNC into the VM:** the VM is a headless CPU box with no GPU — running the game
there would render in software over a remote-desktop link, and every frame it drew would be stolen
from the arenas you are paying for. Here the VM ships you nothing but the ordinary Minecraft
protocol and *your* machine does the rendering, so watching costs the training run essentially
nothing.

## 2. Dashboard — numbers plus a top-down replay

Same command (the tunnel is already open), then open **<http://127.0.0.1:8788>**.

What it actually draws, from `dashboard/public/app.js` fed by 10 Hz arena snapshots
(`ArenaManager.tick()` emits every 2 ticks):

- a top-down canvas of the selected arena: fighters as coloured circles with a line showing camera
  yaw and their name above; **gold diamonds are end crystals**; purple squares are placed obsidian
  and grey squares any other block the fighters changed (untouched floor is not drawn);
- per-fighter cards: current HP, a health bar, cumulative damage dealt, and invalid-interaction
  count;
- arena title with mode (`SWORD`/`CRYSTAL`/`COMBINED`), seconds remaining, and the episode seed;
- an arena picker when more than one pair is live;
- training state: policy version, phase, agent ticks, parameter count, server TPS and p95 tick time,
  active pairs, progress to the next PPO update, and the last update's policy loss, value loss,
  entropy and approximate KL;
- **Emergency stop fights** / **Resume fights** buttons (the same `stop_all` / `resume` commands as
  `/aistop`). The stop button really does stop training — treat it as the big red one.

This is the view to leave open on a second monitor: it is cheap, it survives reconnects, and it is
the fastest way to tell whether crystal placement is actually happening.

## 3. Eaglercraft — fighting it yourself, later

`docs/EAGLER.md` covers the browser adapter. Two things to know before you plan on it:

- **This repo ships no Eaglercraft client, no Minecraft assets and no patched binaries.** You must
  supply your own legitimate build (1.12 u2 offline, processed with EaglerForgeInjector) — see
  `docs/EAGLER.md` for the exact expected form.
- The Azure deployment is headless by design; serving a browser client from it would mean opening a
  port on the VM and is deliberately out of scope of `deploy/azure`. Until then, the way to fight
  the bot yourself is the normal 1.12.2 client above with a human-versus-`MCAI_BROWSER` evaluation
  match, started with `scripts/arena_control.py start_match` (see `docs/EAGLER.md`) — and your
  username must be whitelisted, which today means the `AIWatcher` slot.

---

## Troubleshooting

**"You are not white-listed on this server!"** — your client is presenting the wrong username. It
must be exactly `AIWatcher`. Editing `whitelist.json` on the VM does not help: the entrypoint
rewrites it from `scripts/configure_runtime.py` at every container start.

**"I'm sorry, but you do not have permission to perform this command"** — the op
step above was skipped, or the server has not restarted since. Re-run step 1 and check
`docker compose logs mcai | tail`.

**"Outdated server!" / "Outdated client!"** — wrong protocol. Paper here is **1.12.2** and accepts
1.12.2 clients only; 1.12, 1.12.1 and anything 1.13+ are rejected.

**`/aiwatch` says "That arena is not active."** — nothing is fighting in that id yet. Check
`arena_control.py status` (above) or the dashboard's arena picker, and remember bots take a minute
to connect and pair after a restart.

**Black screen / stuck in the void after `/aiwatch … pov`** — you are still in the default world.
Run `/aiwatch <arena> orbit` first, then switch to `pov`.

**`local port 25565 is already in use`** — you have another Minecraft server or an older tunnel
running. Close it, or `--mc-port 25566` and Direct Connect to `127.0.0.1:25566`.

**`cannot SSH to …`** — the script uses `BatchMode=yes`, so a key must work without a prompt. Pass
`--identity ~/.ssh/<key>`, and check `--user` (Azure default `azureuser`).

**`remote: cannot talk to the Docker daemon`** — the SSH user is not in the `docker` group on the
VM: `sudo usermod -aG docker $USER`, then log out and back in.

**`remote: the relay did not come up`** — the script starts a small TCP relay inside the container
because Paper binds `127.0.0.1` *inside* the container (`--bind 127.0.0.1` in
`deploy/azure/entrypoint.sh`) and Docker's port publishing can only reach a container's external
interface, never its loopback. If your VM cannot route to the container's address, publish a port
yourself and bypass the relay with `--remote-mc HOST:PORT` (as seen from the VM).

**Tunnel up, client hangs on "Connecting to the server"** — Paper may still be starting or the
stack may have died. `docker compose logs -f mcai` on the VM, and look at
`/data/runs/linux-*/paper.log`.

**The fights look frozen** — check the dashboard's TPS. If it is well below 20 the VM is
oversubscribed; lower `MCAI_BOT_COUNT` / `MCAI_MAX_PAIRS`.
