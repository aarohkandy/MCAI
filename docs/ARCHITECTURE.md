# Architecture

## Data flow

At 20 decisions per second, each Mineflayer or Eagler adapter produces `ObservationV1`. The trainer batches observations, advances the shared GRU, masks impossible actions before sampling, and returns `ActionV1`. Adapters translate that action into ordinary client operations. The Paper plugin independently scores server-observed outcomes.

```text
Windows Surface 5
  ├─ Paper 1.12.2 + EaglerXServer
  │    └─ MCAIArena plugin ── loopback NDJSON ─┐
  ├─ Mineflayer rollout clients                  ├─ rollout worker
  │    └─ ObservationV1 / ActionV1 ──────────┘
  ├─ CPU trainer ── loopback MessagePack WebSocket
  │    ├─ structured encoders + GRU128
  │    ├─ recurrent PPO / behavior cloning / league
  │    └─ checkpoint → ONNX + flat float32 weights
  └─ Eagler spectator/browser ── native server connection

Mac development checkout (optional)
  └─ builds, unit tests, and PyTorch/ONNX/browser parity verification
```

## Contracts

`ObservationV1` contains match delay/version metadata; self/opponent state; up to 16 deterministically sorted combat entities; up to 48 deterministically sorted nearby blocks; explicit masks; and legal-action masks. Empty/unloaded slots are zeroed and masked rather than retaining stale tensors.

Yaw and pitch use Mineflayer's canonical radians in both adapters: yaw zero faces north/negative Z and positive pitch looks upward. The Eagler adapter converts Minecraft's degree/sign convention at its boundary, so recorded demonstrations and headless rollouts share the same tensors and camera actions.

`ActionV1` contains two three-way movement axes, jump/sprint/sneak, bounded yaw/pitch deltas, attack/use/release, hotbar selection, and offhand swap. It intentionally has no target-coordinate, teleport, velocity, reach, or arbitrary-packet operation.

The schemas live in `protocol/schema/`. The TypeScript and Python feature encoders are fixture-tested, and `scripts/verify_model_parity.py` checks PyTorch, ONNX Runtime, and the exact typed-array browser implementation.

## Policy

Self, opponent, entity-slot, block-slot, and legal-mask MLPs feed masked mean/max slot pooling. A fusion MLP feeds a 128-unit GRU. Independent categorical heads control discrete actions; a tanh-squashed Gaussian controls camera deltas; a scalar head estimates state value. The current network has roughly one third of the one-million-parameter ceiling.

PPO defaults match the design: discount `0.995`, GAE `0.95`, clip `0.2`, learning rate `3e-4`, entropy `0.01`, value coefficient `0.5`, gradient clip `0.5`, 8,192 agent-ticks, 32-tick sequences, 512 samples/minibatch, and four epochs.

## Arena control

The plugin binds the control socket to the OS loopback address regardless of server binding. Supported commands include `ping`, `status`, `register_agent`, `start_match`, `set_mode`, `set_max_pairs`, `set_shaping_scale`, `stop_all`, and explicit `resume`. `stop_all` leaves automatic pairing paused; it cannot silently restart a stopped bot. Events include `match_started`, `step_feedback`, `match_ended`, and `emergency_stop`.

The worker starts with two duel pairs (or fewer if fewer clients exist). Every five minutes of healthy samples it adds exactly one pair. It immediately removes one pair when TPS is below 19.5, p95 tick interval exceeds 55 ms, memory exceeds 80%, or p95 Node event-loop delay exceeds 10 ms. The trainer defaults to half of the Surface's logical CPU threads so Paper and Node retain headroom; `MCAI_TORCH_THREADS` overrides this after measurement.
