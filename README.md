# MCAI

MCAI is a structured-state, self-play combat agent for native Minecraft/Eaglercraft 1.12.2 mechanics. It sees only state already tracked by the normal client and acts only through ordinary movement, rotation, click/use, hotbar, offhand, placement, and mining paths.

The Windows Surface is the training machine. It runs the server, rollout clients, and CPU trainer together. The Mac is only a development/verification machine and is not required once the repository has been copied to the Surface.

- `server-plugin/` — deterministic Paper 1.12.2 arenas, exact kits, rewards, reset tracking, metrics, spectator cameras, and a localhost-only NDJSON control socket.
- `worker/` — TypeScript Mineflayer rollout clients, portable `ObservationV1`/`ActionV1` adapters, delayed actions/observations, scripted teachers, and automatic rollout-concurrency control.
- `trainer/` — a 296k-parameter structured MLP + GRU policy, whole-match behavior cloning, recurrent PPO, Windows CPU training, atomic checkpoints, and ONNX/flat exports.
- `eagler-mod/` — EaglerForgeInjector 1.12 adapter, human demonstration recorder, local debug bridge, decision overlay, F8 emergency stop, and a dependency-free typed-array policy runtime.
- `protocol/` — the versioned JSON Schemas shared by every adapter.

## What “not hacking” means here

The bot does not set position or velocity, extend reach, invent unloaded state, or construct arbitrary gameplay packets. It can use off-camera entities that the normal client still tracks, and its reaction/click/rotation timing is unrestricted. That makes it stronger than a human-equivalent bot, so the Eagler account and overlay identify it as AI during invited human matches.

Use it only on a server you own. Public-server botting, anti-cheat evasion, leaked client source, and redistribution of Minecraft/Eaglercraft binaries or assets are deliberately outside this repository.

## Download and start on Windows

On the Windows Surface:

1. Download this repository with GitHub's **Code → Download ZIP**, then extract the ZIP. A normal `git clone` works too.
2. Double-click **`START_MCAI.cmd`**.
3. Review and accept the Minecraft server EULA when asked. If Windows needs Git, Node.js, Java, or the isolated Python manager, the launcher lists them and asks before Windows Package Manager installs them.
4. Leave the training window open. The live dashboard opens automatically when the first arena is ready.

That is the complete normal setup. The first start downloads and builds the private Paper environment, Python 3.12/PyTorch trainer, EaglerXServer plugin, arena plugin, and Mineflayer worker. It can take a while; later starts reuse everything. Hardware-safe bot count and memory defaults are selected from the Surface's actual RAM and CPU.

- Double-click **`WATCH_TRAINING.cmd`** to reopen the dashboard.
- Double-click **`STOP_MCAI.cmd`** for an orderly emergency stop.
- Double-click **`START_MCAI.cmd`** later to resume from the latest complete checkpoint.

The dashboard is available only at `http://127.0.0.1:8788`. It shows an animated top-down arena, fighter health and damage, rollout progress toward the next PPO update, policy metrics, TPS, and memory. It does not contain or redistribute Minecraft/Eaglercraft art or client code.

The launcher uses the documented Windows Package Manager agreement flags only after the user types `INSTALL`; see [Microsoft's install-command documentation](https://learn.microsoft.com/windows/package-manager/winget/install). It keeps all gameplay and training ports on loopback by default.

The safe default binds Minecraft/Eaglercraft to `127.0.0.1`. Change `MCAI_BIND_ADDRESS` in `config.windows.ps1` to the Surface's private LAN IP only when you want an Eagler spectator or invited player to connect, and restrict TCP 25565 with Windows Firewall. Ports 8765–8767 stay local.

CPU-only training will be much slower than GPU training. The load controller backs down immediately under load and only adds a pair after five healthy minutes. The generated `config.windows.ps1` is the advanced settings file; edit it only while MCAI is stopped. Training begins in sword mode because that is the first curriculum stage. Change `MCAI_MODE` to `crystal` or `combined` only after an appropriate checkpoint exists.

The non-Windows shell scripts are retained for development and WSL/Linux experimentation. `scripts/bootstrap-mac.sh` only runs trainer tests and parity checks; it does not make the Mac part of the training deployment.

The trainer freezes a policy version while collecting each 8,192-agent-tick batch, rejects stale versions, saves `checkpoints/latest.pt` atomically, and snapshots every 100,000 accepted ticks. Per-run logs are kept under `runs/windows-*`; model checkpoints are kept separately under `checkpoints/`.

## Eagler browser client

This repository never downloads or contains an Eaglercraft client. Supply your own legitimate, unminified, unobfuscated, unsigned 1.12 u2 offline build, process it with the official EaglerForgeInjector, and load `eagler-mod/build/mcai.bundle.js` as a mod.

For the initial debug path:

```bash
./scripts/start-browser-bridge.sh checkpoints/latest.pt
cd eagler-mod && npm run build
```

In game:

- `.mcai on` enables the frozen policy.
- `.mcai off` or F8 releases every control and disconnects the bridge.
- `.mcai record MATCH_NAME` records your exact sword inputs and structured observations.
- `.mcai recordstop` ends a recording.
- `.mcai export` saves whole-match JSONL for behavior cloning.

For helper-free deployment, export a checkpoint and load the generated manifest/binary through `MCAI.loadWeights(...)` or `MCAI.loadWeightsFromFiles(...)`. The typed-array runtime matches PyTorch and ONNX within `1e-5` on fixed fixtures.

```powershell
.\trainer\.venv\Scripts\mcai-trainer.exe export .\checkpoints\latest.pt --directory .\exports
```

## Verification

```powershell
Set-Location worker; npm test -- --run; npm run typecheck; npm run build; Set-Location ..
.\trainer\.venv\Scripts\python.exe -m pytest -q .\trainer\tests
.\.tools\apache-maven-3.9.9\bin\mvn.cmd -q -f .\server-plugin\pom.xml test
Set-Location eagler-mod; npm run check; npm run build; Set-Location ..
.\trainer\.venv\Scripts\python.exe .\scripts\verify_model_parity.py
```

The server template, Paper jar, EaglerXServer jar, generated worlds, checkpoints, Eagler client, and exported model weights are all ignored and are not distributed by this repository.

See [architecture](docs/ARCHITECTURE.md), [training curriculum](docs/TRAINING.md), [Eagler setup](docs/EAGLER.md), and [deployment safety](docs/SAFETY.md) for the complete operating guide.
