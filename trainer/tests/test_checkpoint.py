from pathlib import Path

import torch

from combat_ai.checkpoint import CheckpointManager, CheckpointState
from combat_ai.config import PPOConfig
from combat_ai.model import CombatPolicy


def test_atomic_snapshot_persists_the_advanced_boundary(tmp_path: Path):
    policy = CombatPolicy()
    optimizer = torch.optim.Adam(policy.parameters())
    manager = CheckpointManager(tmp_path, 100)
    state = CheckpointState(policy_version=3, total_agent_ticks=100, next_snapshot_tick=100)
    manager.save(policy, optimizer, state, PPOConfig(), {"loss": 1.0})
    assert (tmp_path / "latest.pt").exists()
    assert (tmp_path / "policy-000000000100.pt").exists()
    restored_policy = CombatPolicy()
    restored_optimizer = torch.optim.Adam(restored_policy.parameters())
    restored = manager.restore(restored_policy, restored_optimizer, torch.device("cpu"))
    assert restored.next_snapshot_tick == 200
