# Preserved checkpoints

`mcai-v3098-full.pt` is the exact full recovery checkpoint that was serving
the four live arenas when training was paused on 2026-07-22. It is stored with
Git LFS because the checkpoint includes much more than inference weights.

It contains the policy, Adam optimizer state, rollout counters, online
imitation buffer, and elite replay records. The two adjacent JSON files retain
the matching PFSP league and adaptive-reward controller state.

To resume this exact training state, install Git LFS, fetch the files, and copy
them before starting the stack:

```text
models/mcai-v3098-full.pt             -> checkpoints/latest.pt
models/mcai-v3098-league.json         -> checkpoints/league.json
models/mcai-v3098-adaptive-reward.json -> checkpoints/adaptive-reward-state.json
```

Verify the file against `mcai-v3098-full.json` before restoring it.
