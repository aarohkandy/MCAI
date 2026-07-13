# Eaglercraft browser adapter

## Supported input

MCAI targets a user-supplied legitimate, unminified, unobfuscated, unsigned Eaglercraft 1.12 u2 offline build processed with EaglerForgeInjector. The project contains no client, Minecraft assets, or patched binaries. u3/WASM is isolated as a future adapter port.

Use the disclosed `MCAI_BROWSER` account for the AI client; the Windows setup whitelists it automatically. Add invited human usernames to the private server whitelist yourself.

Build the mod with `cd eagler-mod && npm run build`, then add `build/mcai.bundle.js` through EaglerForge's Mods UI. The adapter requires `player`, `world`, `network`, and `resolution` and refreshes the player/world proxies on every `update` event.

The adapter reads normal client-known state, including tracked opponents outside the camera frustum. It samples blocks every five ticks and marks sample age. It never writes player position/motion. Actions set ordinary key bindings, update camera yaw/pitch, choose a hotbar slot, use the normal click/right-click functions, or press the vanilla swap-hands binding.

## Debug bridge

`scripts/start-browser-bridge.sh` serves a frozen checkpoint on loopback port 8767. The mod exchanges JSON because this path is for inspection, not bulk rollout throughput. The training worker uses MessagePack separately.

## Helper-free inference

Export `policy.manifest.json` and `policy.weights.bin`. The runtime implements every Linear/Tanh layer, masked mean/max pool, PyTorch-order GRU gates, legal argmax, and camera squashing with typed arrays. Call one of:

```javascript
await MCAI.loadWeights('/models/policy.manifest.json', '/models/policy.weights.bin')
await MCAI.loadWeightsFromFiles(manifestFile, weightsFile)
MCAI.enable()
```

Weights must be from a frozen reproducible checkpoint. Run `scripts/verify_model_parity.py` for every architecture/export change.

## Spectating

Join with the whitelisted `AIWatcher` account. On the server, use `/aiwatch arena-1 pov`, `/aiwatch arena-1 orbit`, and `/ainext`. The in-browser overlay shows policy version, action, target vector, value, health, recorder status, and inference latency.

First sample a held-out tuple from PowerShell on the Surface:

```powershell
.\trainer\.venv\Scripts\python.exe .\scripts\arena_control.py sample_evaluation `
  '{"mode":"combined"}'
```

In the browser, apply the returned delays with `.mcai delay ACTION OBSERVATION`. Then start the disclosed browser-versus-human evaluation using the returned seed:

```powershell
.\trainer\.venv\Scripts\python.exe .\scripts\arena_control.py start_match `
  '{"player_a":"MCAI_BROWSER","player_b":"INVITED_NAME","mode":"combined","evaluation":true,"seed":RETURNED_SEED}'
```

Evaluation mode also chooses a held-out tuple automatically if the seed is omitted, but sampling first lets the browser reproduce the same delay contract from tick one. `/aistop` is the operator emergency stop; after checking the cause, `arena_control.py resume` explicitly re-enables automatic pairing.
