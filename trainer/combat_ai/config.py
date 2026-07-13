from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    learning_rate: float = 3e-4
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_gradient_norm: float = 0.5
    rollout_agent_ticks: int = 8192
    recurrent_sequence_length: int = 32
    minibatch_samples: int = 512
    optimization_epochs: int = 4
    checkpoint_every_ticks: int = 100_000
    imitation_start_weight: float = 0.10
    imitation_decay_ticks: int = 5_000_000

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ServiceConfig:
    host: str = "0.0.0.0"
    port: int = 8766
    checkpoint_dir: Path = Path("checkpoints")
    deterministic_inference: bool = False
    seed: int = 7
    cpu_threads: int = 0
