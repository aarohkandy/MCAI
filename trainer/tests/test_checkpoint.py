from pathlib import Path

import torch

from combat_ai.checkpoint import (
    FEATURE_CONTRACT_VERSION,
    CheckpointManager,
    CheckpointState,
)
from combat_ai.config import PPOConfig
from combat_ai.model import CombatPolicy
from combat_ai.export import load_policy
from combat_ai.service import _initialize_policy


def test_checkpoint_persists_generation_and_online_imitation(tmp_path: Path):
    manager = CheckpointManager(tmp_path, 100_000)
    policy = CombatPolicy()
    optimizer = torch.optim.Adam(policy.parameters())
    records = [{"execution_source": "teacher_crystal", "tick": 4}]
    manager.save(
        policy, optimizer, CheckpointState(rollout_generation=7), PPOConfig(), {}, records,
    )
    restored_policy = CombatPolicy()
    restored_optimizer = torch.optim.Adam(restored_policy.parameters())
    restored = manager.restore(restored_policy, restored_optimizer, torch.device("cpu"))
    assert restored.rollout_generation == 7
    assert manager.restored_imitation_records == records


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
    payload = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=False)
    assert payload["feature_contract_version"] == FEATURE_CONTRACT_VERSION
    historical = torch.load(
        tmp_path / "policy-000000000100.pt", map_location="cpu", weights_only=False,
    )
    assert historical["feature_contract_version"] == FEATURE_CONTRACT_VERSION
    assert "optimizer" not in historical
    assert "online_imitation_records" not in historical
    assert "elite_replay_records" not in historical
    assert set(historical["policy"]) == set(payload["policy"])


def test_legacy_checkpoint_neutralizes_repurposed_feature_columns_and_optimizer(tmp_path: Path):
    policy = CombatPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    # Create optimizer moments so the migration proves they are discarded.
    sum(parameter.sum() for parameter in policy.parameters()).backward()
    optimizer.step()
    expected_opponent_bias = (
        policy.opponent_encoder.layers[0].bias
        + policy.opponent_encoder.layers[0].weight[:, 12]
    ).detach().clone()
    payload = {
        "format_version": 1,
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "policy_version": 460,
        "total_agent_ticks": 123,
        "next_snapshot_tick": 200,
    }
    torch.save(payload, tmp_path / "latest.pt")

    restored_policy = CombatPolicy()
    restored_optimizer = torch.optim.Adam(restored_policy.parameters(), lr=1e-4)
    restored = CheckpointManager(tmp_path, 100).restore(
        restored_policy, restored_optimizer, torch.device("cpu")
    )

    assert restored.policy_version == 460
    assert restored_optimizer.state == {}
    assert torch.count_nonzero(
        restored_policy.opponent_encoder.layers[0].weight[:, 38:48]
    ) == 0
    for index in (9, 10, 12, 13):
        assert torch.count_nonzero(
            restored_policy.opponent_encoder.layers[0].weight[:, index]
        ) == 0
    assert torch.allclose(
        restored_policy.opponent_encoder.layers[0].bias, expected_opponent_bias
    )
    assert torch.count_nonzero(
        restored_policy.entity_encoder.layers[0].weight[:, 16:18]
    ) == 0
    assert torch.count_nonzero(
        restored_policy.block_encoder.layers[0].weight[:, 18]
    ) == 0
    assert torch.count_nonzero(
        restored_policy.legal_encoder.layers[0].weight[:, 21:24]
    ) == 0
    for index in (31, 34, 37, 40, 45, 49, 53, 57, 61, 65, 69, 73, 77):
        assert torch.count_nonzero(restored_policy.self_encoder.layers[0].weight[:, index]) == 0

    historical = load_policy(tmp_path / "latest.pt")
    assert torch.count_nonzero(historical.opponent_encoder.layers[0].weight[:, 38:48]) == 0
    for index in (9, 10, 12, 13):
        assert torch.count_nonzero(
            historical.opponent_encoder.layers[0].weight[:, index]
        ) == 0
    assert torch.count_nonzero(historical.block_encoder.layers[0].weight[:, 18]) == 0
    assert torch.count_nonzero(historical.legal_encoder.layers[0].weight[:, 21:24]) == 0


def test_contract_two_checkpoint_neutralizes_only_new_candidate_inputs(tmp_path: Path):
    policy = CombatPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    sum(parameter.sum() for parameter in policy.parameters()).backward()
    optimizer.step()
    with torch.no_grad():
        policy.crystal_attention.encoder.layers[0].weight.fill_(7.0)
        policy.tactical_block_attention.encoder.layers[0].weight.fill_(5.0)
        # These columns became meaningful in contract 2 and must not be erased
        # again when only the candidate contract advances from 2 to 3.
        policy.opponent_encoder.layers[0].weight[:, 38:48].fill_(3.0)
    payload = {
        "format_version": 2,
        "architecture_version": policy.architecture_version,
        "feature_contract_version": 2,
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(payload, tmp_path / "latest.pt")

    restored_policy = CombatPolicy()
    restored_optimizer = torch.optim.Adam(restored_policy.parameters(), lr=1e-4)
    CheckpointManager(tmp_path, 100).restore(
        restored_policy, restored_optimizer, torch.device("cpu")
    )

    for layer, stable_value in (
        (restored_policy.crystal_attention.encoder.layers[0], 7.0),
        (restored_policy.tactical_block_attention.encoder.layers[0], 5.0),
    ):
        assert torch.all(layer.weight[:, 0] == stable_value)
        assert torch.all(layer.weight[:, 2] == stable_value)
        reassigned = [index for index in range(16) if index not in (0, 2)]
        assert torch.count_nonzero(layer.weight[:, reassigned]) == 0
    assert torch.all(restored_policy.opponent_encoder.layers[0].weight[:, 38:48] == 3.0)
    assert restored_optimizer.state == {}

    # Export/load follows the same version-aware path.
    historical = load_policy(tmp_path / "latest.pt")
    assert torch.all(historical.crystal_attention.encoder.layers[0].weight[:, 0] == 7.0)
    assert torch.count_nonzero(
        historical.crystal_attention.encoder.layers[0].weight[:, [1, *range(3, 16)]]
    ) == 0


def test_initialize_from_migrates_each_legacy_policy_before_averaging(tmp_path: Path):
    legacy = CombatPolicy()
    with torch.no_grad():
        legacy.opponent_encoder.layers[0].weight[:, 38:48].fill_(7.0)
        legacy.opponent_encoder.layers[0].weight[:, [9, 10, 13]].fill_(6.0)
        legacy.opponent_encoder.layers[0].weight[:, 12].fill_(4.0)
        legacy.entity_encoder.layers[0].weight[:, 16:18].fill_(5.0)
        legacy.block_encoder.layers[0].weight[:, 18].fill_(3.0)
    expected_opponent_bias = (
        legacy.opponent_encoder.layers[0].bias
        + legacy.opponent_encoder.layers[0].weight[:, 12]
    ).detach().clone()
    checkpoint = tmp_path / "legacy-initialize-from.pt"
    torch.save({"policy": legacy.state_dict()}, checkpoint)

    initialized = CombatPolicy()
    _initialize_policy(initialized, [checkpoint], torch.device("cpu"))

    assert torch.count_nonzero(
        initialized.opponent_encoder.layers[0].weight[:, 38:48]
    ) == 0
    assert torch.count_nonzero(
        initialized.entity_encoder.layers[0].weight[:, 16:18]
    ) == 0
    assert torch.count_nonzero(
        initialized.opponent_encoder.layers[0].weight[:, [9, 10, 12, 13]]
    ) == 0
    assert torch.allclose(
        initialized.opponent_encoder.layers[0].bias, expected_opponent_bias
    )
    assert torch.count_nonzero(initialized.block_encoder.layers[0].weight[:, 18]) == 0
