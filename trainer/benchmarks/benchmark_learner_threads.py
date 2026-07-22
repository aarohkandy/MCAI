from __future__ import annotations

import argparse
import json
import time

import torch

from combat_ai.buffer import SequenceBatch
from combat_ai.config import PPOConfig
from combat_ai.distribution import ActionTensor
from combat_ai.features import (
    BLOCK_SIZE,
    ENTITY_SIZE,
    HISTORY_SIZE,
    LEGAL_SIZE,
    MAX_BLOCKS,
    MAX_ENTITIES,
    MAX_RECENT_HISTORY,
    MAX_TACTICAL_CANDIDATES,
    OPPONENT_SIZE,
    SELF_SIZE,
    SURVIVAL_SIZE,
    TACTICAL_CANDIDATE_SIZE,
    THREAT_SIZE,
    FeatureBatch,
)
from combat_ai.model import CATEGORICAL_SIZES, CombatPolicy, HIDDEN_SIZE
from combat_ai.ppo import PPOTrainer


def representative_minibatch(
    sequence_count: int = 32, sequence_length: int = 32,
) -> SequenceBatch:
    """Create a full 1,024-sample V2 recurrent minibatch for CPU timing.

    Entity, block, and tactical masks are populated so the benchmark exercises
    every encoder and attention path. It intentionally times only optimizer
    work; process startup and rollout preparation are outside the measurement.
    """
    generator = torch.Generator().manual_seed(7)
    shape = (sequence_count, sequence_length)

    def values(*tail: int) -> torch.Tensor:
        return torch.randn((*shape, *tail), generator=generator) * 0.1

    features = FeatureBatch(
        self_state=values(SELF_SIZE),
        opponent=values(OPPONENT_SIZE),
        opponent_mask=torch.ones((*shape, 1)),
        entities=values(MAX_ENTITIES, ENTITY_SIZE),
        entity_mask=torch.ones((*shape, MAX_ENTITIES)),
        blocks=values(MAX_BLOCKS, BLOCK_SIZE),
        block_mask=torch.ones((*shape, MAX_BLOCKS)),
        legal=torch.ones((*shape, LEGAL_SIZE)),
        crystal_candidates=values(
            MAX_TACTICAL_CANDIDATES, TACTICAL_CANDIDATE_SIZE,
        ),
        crystal_candidate_mask=torch.ones(
            (*shape, MAX_TACTICAL_CANDIDATES),
        ),
        tactical_blocks=values(
            MAX_TACTICAL_CANDIDATES, TACTICAL_CANDIDATE_SIZE,
        ),
        tactical_block_mask=torch.ones(
            (*shape, MAX_TACTICAL_CANDIDATES),
        ),
        recent_history=values(MAX_RECENT_HISTORY, HISTORY_SIZE),
        recent_history_mask=torch.ones((*shape, MAX_RECENT_HISTORY)),
        survival=values(SURVIVAL_SIZE),
        threat=values(THREAT_SIZE),
    )
    actions = ActionTensor(
        categorical={
            name: torch.zeros(shape, dtype=torch.long)
            for name in CATEGORICAL_SIZES
        },
        camera=torch.zeros((*shape, 2)),
    )
    return SequenceBatch(
        features=features,
        hidden=torch.zeros((1, sequence_count, HIDDEN_SIZE)),
        actions=actions,
        old_log_probability=torch.zeros(shape),
        old_value=torch.zeros(shape),
        advantage=torch.randn(shape, generator=generator),
        returns=torch.randn(shape, generator=generator),
        done=torch.zeros(shape),
        valid=torch.ones(shape),
    )


def benchmark(thread_count: int, repeats: int) -> dict[str, object]:
    torch.set_num_threads(thread_count)
    policy = CombatPolicy()
    trainer = PPOTrainer(
        policy,
        PPOConfig(
            optimization_epochs=1,
            minibatch_samples=1024,
            target_kl=0,
        ),
        torch.device("cpu"),
    )
    batch = representative_minibatch()
    durations: list[float] = []
    for _ in range(repeats + 1):
        started = time.perf_counter()
        trainer._minibatch(batch)
        durations.append(time.perf_counter() - started)
    measured = durations[1:]
    return {
        "threads": thread_count,
        "warmup_seconds": durations[0],
        "seconds": measured,
        "median_seconds": sorted(measured)[len(measured) // 2],
        "samples_per_second": 1024 / (sum(measured) / len(measured)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--repeats", type=int, default=1)
    arguments = parser.parse_args()
    for thread_count in arguments.threads:
        print(json.dumps(benchmark(thread_count, arguments.repeats)), flush=True)


if __name__ == "__main__":
    main()
