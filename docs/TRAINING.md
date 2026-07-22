# Training curriculum

## 1. Infrastructure and scripted teachers

Run the adapter and arena reset tests first. Use the worker's scripted styles for turning, approach/follow, raycast alignment, reach, jumping/sprinting, item selection, one-block placement, and one-block mining. Do not promote until each held-out task reaches 95% over 500 seeded episodes.

The Paper plugin generates deterministic layouts and resets all touched blocks plus non-player entities without restarting. Before long training runs, execute 1,000 reset cycles and verify the server event counters show no leftover crystals, blocks, inventories, effects, or cross-arena damage.

## 2. Sword imitation and drills

Use the exact target client/server. Record two to four hours in whole matches:

```text
.mcai record sword-001
...fight...
.mcai recordstop
.mcai export
```

Combine the JSONL files without shuffling frames between matches, then run:

```powershell
.\trainer\.venv\Scripts\mcai-trainer.exe clone .\demonstrations\sword.jsonl `
  --output .\checkpoints\imitation.pt --epochs 30
```

The loader splits 80/10/10 by match ID and early-stops on validation loss. During the first five million PPO ticks, retain a 10% auxiliary imitation objective and decay it linearly to zero. Exercise rush, strafe, retreat, jump-critical, defensive, and erratic teachers.

## 3. Sword self-play

Set the arena mode to `sword`. It uses flat 21×21 layouts and progressively randomized facing/delay. Promote only at 90% wins against every scripted baseline over 500 held-out matches, with no regression against frozen checkpoints. The user gate is 60 wins in 100 held-out matches.

## 4. Crystal mechanics

Use scripted teachers and deterministic seeds for obsidian placement, 1.12 crystal clearance, crystal aim/break, self-damage avoidance, hit-crystal sequences, eating, retoteming, mining cover, escaping crystals, and using obsidian as cover. Crystal video is useful for ideas, not structured labels.

Each drill gate is 90%. Retotem must occur within two ticks in 95% of legal cases, and avoidable self-kills must remain under 10%.

## 5. Combined league

Switch to `combined`, and reduce shaping with the control command `set_shaping_scale` to `0.2`. Terminal reward remains ±1; nonterminal damage/pop reward is clipped to ±0.05 per tick. Draws give both sides −0.05.

The target opponent mix is 40% latest mirrors, 40% frozen checkpoints near 50% expected win rate, and 20% scripted/style-randomized opponents. Snapshot every 100,000 accepted ticks, retain the best 20 by held-out Elo plus the latest 10, and create an exploiter after five flat evaluations. Held-back seeds and delay combinations never enter training.

For a fresh combined run, set `MCAI_INITIALIZE_FROM` in `config.windows.ps1` to the sword and crystal checkpoints separated by a semicolon. The service averages compatible weights only when `checkpoints/latest.pt` does not already exist. Set `MCAI_IMITATION_DATA` to the sword JSONL to retain the decaying auxiliary loss.

Every arena seed/action-delay/observation-delay tuple is assigned permanently to training or the 20% held-out set by the same SHA-256 split in Java and Python. Automatic matches reject held-out tuples. Explicit evaluation matches must send `evaluation: true` through the loopback arena control API.

Evaluation results are machine-gated rather than promoted by hand. A result file contains the current stage and named metrics, for example:

```json
{
  "stage": "infrastructure",
  "metrics": {
    "turning": {"episodes": 500, "rate": 0.97},
    "approaching": {"episodes": 500, "rate": 0.96}
  }
}
```

Include every named gate for that stage, then run:

```powershell
.\trainer\.venv\Scripts\mcai-trainer.exe curriculum `
  --state .\checkpoints\curriculum.json --results .\evaluation\results.json
```

Missing gates fail closed. Record held-out Elo after each league evaluation with `mcai-trainer league-eval ELO`. A flat five-evaluation window may request an exploiter only after the main policy passes the diversity gauntlet recorded in `checkpoints/league.json`: at least four matches in 10 of 12 crazy styles, a 55% score in at least 8 styles, and a 60% aggregate score. Before that gate, match assignment is 10% mirror, 15% historical, and 75% adaptive crazy-strategy play; weak and under-tested styles are sampled more often. After passing, historical specialization may rise to 40%, while mirror play remains capped at 15%.

To act on that request, stop the stack, set `MCAI_EXPLOITER_TARGET` to a frozen main snapshot, and restart. The Windows launcher uses an isolated `checkpoints/exploiter-active` run initialized from that target; every opponent is the frozen main policy while the exploiter learns. After the desired exploiter budget/evaluation, stop it and promote the frozen result:

```powershell
.\trainer\.venv\Scripts\mcai-trainer.exe promote-exploiter `
  .\checkpoints\exploiter-active\latest.pt --checkpoints .\checkpoints
```

Clear `MCAI_EXPLOITER_TARGET` before restarting main training. Promotion validates the checkpoint, copies it atomically under a `policy-exploiter-*` name, seeds its rating near the main policy, and makes it eligible for the 40% historical-opponent pool.

## Automatic reward adaptation

Online training adjusts five bounded reward groups automatically after complete, successfully published PPO generations:

- `damage`: policy hits, opponent damage, damage taken, and totem advantage.
- `crystal`: verified placement/detonation/damaging chains and matching self-crystal penalties.
- `terminal_speed`: the early-kill bonus, timeout/disengagement losses, and fight-time pressure. The base kill/death objective is never changed.
- `activity`: approach, legal attacks, inaction, missed/spam attacks, and invalid interactions.
- `building`: tactical obsidian, mine-to-place sequences, and useful mining.

The controller uses only accepted main-policy transitions. It watches hit density, damage efficiency, autonomous crystal conversion, policy-owned endings, inactivity, building frequency, PPO KL, and worker latency. A signal must persist across multiple rollout generations before a multiplier moves; changes are small, bounded, and followed by a cooldown. Numerically bad or overloaded generations do not tune rewards.

Verified autonomous wins and damaging crystal chains also enter a bounded elite replay buffer. Later auxiliary imitation batches reserve a small, capped quota for these exact executed policy actions, weighted toward faster kills and balanced across lanes/mechanics. Teacher, safety, scripted-opponent, malformed, and non-finite actions are never admitted.

League matchmaking is competence-gated. Below 25% held-out scripted score it emphasizes mirrors and attainable-frontier scripts so the learner sees enough successful endings; from 25–50% it adds more history and exploiters; at 50% it restores the full hard PFSP population. Hard and under-covered styles retain probability in every stage, and the stage is reconstructed from persisted results after restart.

State survives restarts in `checkpoints/adaptive-reward-state.json`, and every evaluation/change is appended to `checkpoints/adaptive-reward-audit.jsonl`. The trainer republishes the saved profile whenever Paper reconnects. Paper snapshots each profile at episode start, so a live match never changes objectives halfway through. The current multipliers and the reason for the last change are visible on the dashboard.

Adaptive rewards are enabled by default. For a controlled ablation only, start the trainer with `--disable-adaptive-rewards`.

## Evaluation gates

- Sword: ≥90% against every script and ≥60% against the user over 100 matches.
- Crystal: ≥90% per drill, ≥95% timely retotems, <10% avoidable self-kills.
- Combined local: ≥90% against scripts and positive frozen-pool Elo trend across three evaluations.
- Experienced humans: ≥50% across at least three invited volunteers and 90 unseen held-out matches.

Elite play is open-ended. Continue only while held-out Elo or invited-human win rate improves; local hardware cannot guarantee an elite result.
