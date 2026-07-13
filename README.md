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

## Quick start on the Windows Surface

Install Git for Windows, Node.js LTS, and Microsoft OpenJDK 17. Open PowerShell in the repository, then run:

```powershell
Copy-Item config.windows.example.ps1 config.windows.ps1
# Review the Minecraft EULA, then set MCAI_ACCEPT_EULA to true in config.windows.ps1.
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap-windows.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\start-windows.ps1
```

The bootstrap installs an isolated Python 3.12 environment, CPU PyTorch, the server template, the current official EaglerXServer release, the arena plugin, worker dependencies, and runs the test suites. The starter launches the CPU trainer and Paper first, waits for the arena control port, then starts Mineflayer.

The safe default binds Minecraft/Eaglercraft to `127.0.0.1`. Change `MCAI_BIND_ADDRESS` in `config.windows.ps1` to the Surface's private LAN IP only when you want an Eagler spectator or invited player to connect, and restrict TCP 25565 with Windows Firewall. Ports 8765–8767 stay local.

CPU-only training will be much slower than GPU training. Start with the default four bots/two pairs; the load controller backs down immediately under load and only adds a pair after five healthy minutes. If the Surface has 4 GB RAM, lower `MCAI_JAVA_MEMORY` to `1G` before starting.

The non-Windows shell scripts are retained for development and WSL/Linux experimentation. `scripts/bootstrap-mac.sh` only runs trainer tests and parity checks; it does not make the Mac part of the training deployment.

The trainer freezes a policy version while collecting each 8,192-agent-tick batch, rejects stale versions, saves `checkpoints/latest.pt` atomically, and snapshots every 100,000 accepted ticks.

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
