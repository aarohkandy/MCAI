import torch
import pytest

from combat_ai.buffer import SequenceBatch
from combat_ai.config import PPOConfig
from combat_ai.distribution import ActionTensor
from combat_ai.features import FeatureBatch
from combat_ai.model import CombatPolicy
from combat_ai.ppo import (
    PPOTrainer,
    _length_bucketed_microbatches,
    _sample_aware_minibatches,
)


def _batch(valid_lengths: list[int], sequence_length: int = 32) -> SequenceBatch:
    count = len(valid_lengths)
    valid = torch.zeros((count, sequence_length), dtype=torch.float32)
    for index, length in enumerate(valid_lengths):
        valid[index, :length] = 1.0
    feature_values = {
        name: torch.zeros((count, sequence_length, 1), dtype=torch.float32)
        for name in (
            "self_state", "opponent", "opponent_mask", "entities",
            "entity_mask", "blocks", "block_mask", "legal",
            "crystal_candidates", "crystal_candidate_mask", "tactical_blocks",
            "tactical_block_mask", "recent_history", "recent_history_mask",
            "survival", "threat",
        )
    }
    return SequenceBatch(
        features=FeatureBatch(**feature_values),
        hidden=torch.zeros((1, count, 1), dtype=torch.float32),
        actions=ActionTensor(
            categorical={},
            camera=torch.zeros((count, sequence_length, 2), dtype=torch.float32),
        ),
        old_log_probability=torch.zeros((count, sequence_length)),
        old_value=torch.zeros((count, sequence_length)),
        advantage=torch.zeros((count, sequence_length)),
        returns=torch.zeros((count, sequence_length)),
        done=torch.zeros((count, sequence_length)),
        valid=valid,
    )


def test_short_fragments_pack_by_valid_samples_instead_of_padded_length(monkeypatch):
    batch = _batch([1] * 2048)
    trainer = PPOTrainer(
        CombatPolicy(),
        PPOConfig(optimization_epochs=1, minibatch_samples=512, target_kl=0),
        torch.device("cpu"),
    )
    calls: list[tuple[int, int]] = []

    def fake_minibatch(minibatch: SequenceBatch) -> tuple[float, ...]:
        valid = int(minibatch.valid.sum().item())
        calls.append((valid, minibatch.sequence_length))
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(valid)

    monkeypatch.setattr(trainer, "_minibatch", fake_minibatch)
    metrics = trainer.update(batch)

    assert calls == [(512, 1)] * 4
    assert metrics.optimizer_updates == 4
    assert metrics.valid_samples == 512
    assert metrics.optimizer_sample_exposures == 2048
    assert metrics.rollout_sample_coverage == 1.0


def test_sample_aware_batches_are_balanced_when_sequence_lengths_vary():
    batch = _batch(([32, 17, 8, 3, 1] * 32))

    minibatches = _sample_aware_minibatches(batch, target_samples=512)
    valid_counts = [int(value.valid.sum().item()) for value in minibatches]

    assert sum(valid_counts) == int(batch.valid.sum().item())
    assert max(valid_counts) - min(valid_counts) <= batch.sequence_length
    assert len(minibatches) == round(int(batch.valid.sum().item()) / 512)


def test_length_buckets_bound_padding_for_mixed_short_and_full_sequences():
    batch = _batch([1] * 32 + [32] * 15)
    minibatch = _sample_aware_minibatches(batch, target_samples=512)[0]

    microbatches = _length_bucketed_microbatches(minibatch)
    valid = sum(int(value.valid.sum().item()) for value in microbatches)
    evaluated_cells = sum(
        value.sequence_count * value.sequence_length for value in microbatches
    )

    assert valid == 512
    assert evaluated_cells == 512
    assert sorted(value.sequence_length for value in microbatches) == [1, 32]


def test_nonfinite_recurrent_sequence_is_quarantined_without_discarding_valid_ones(monkeypatch):
    batch = _batch([3, 3, 3], sequence_length=3)
    batch.features.self_state[1, 1, 0] = float("nan")
    trainer = PPOTrainer(
        CombatPolicy(),
        PPOConfig(optimization_epochs=1, minibatch_samples=32, target_kl=0),
        torch.device("cpu"),
    )

    def finite_minibatch(value: SequenceBatch) -> tuple[float, ...]:
        assert value.sequence_count == 2
        assert all(torch.isfinite(tensor).all() for tensor in vars(value.features).values())
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(value.valid.sum())

    monkeypatch.setattr(trainer, "_minibatch", finite_minibatch)
    metrics = trainer.update(batch)

    assert metrics.quarantined_sequences == 1
    assert metrics.quarantined_samples == 3
    assert metrics.rollout_valid_samples == 6


def test_all_nonfinite_sequences_fail_without_publishing_an_update():
    batch = _batch([2], sequence_length=2)
    batch.actions.camera[0, 0, 0] = float("inf")
    trainer = PPOTrainer(CombatPolicy(), PPOConfig(), torch.device("cpu"))

    with pytest.raises(ValueError, match="all rollout sequences"):
        trainer.update(batch)
