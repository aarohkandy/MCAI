from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    learning_rate: float = 1e-4
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    # Combat returns contain rare +/- terminal floods and their scale changes
    # as the adaptive reward profile moves. A Huber critic keeps those
    # outliers from consuming the shared actor/critic gradient budget.
    value_huber_delta: float = 1.0
    max_gradient_norm: float = 0.5
    rollout_agent_ticks: int = 4096
    recurrent_sequence_length: int = 32
    minibatch_samples: int = 1024
    optimization_epochs: int = 4
    # The learner is isolated from the websocket actor in its own process.
    # Two Torch threads are the measured sweet spot on the 12-thread i7-1265U;
    # higher counts oversubscribe its heterogeneous cores and become slower.
    learner_cpu_threads: int = 2
    target_kl: float = 0.01
    adaptive_lr_low_kl: float = 0.002
    adaptive_lr_high_kl: float = 0.015
    adaptive_lr_low_updates: int = 3
    adaptive_lr_increase: float = 1.5
    adaptive_lr_decrease: float = 0.5
    minimum_learning_rate: float = 2.5e-5
    maximum_learning_rate: float = 3e-4
    checkpoint_every_ticks: int = 100_000
    imitation_start_weight: float = 0.10
    imitation_decay_ticks: int = 5_000_000
    # Autonomous self-imitation is deliberately smaller than bootstrapping
    # teacher loss.  It preserves proven main-policy tactics without anchoring
    # the actor to every behavior from its current generation.
    elite_imitation_weight: float = 0.015
    elite_replay_capacity: int = 2_048
    elite_trace_actions: int = 128
    elite_kill_window: int = 64
    elite_crystal_window: int = 24
    # Arena shaping is normally emitted once per 50 ms action step. During a
    # synchronous CPU PPO update the worker deliberately holds its last action,
    # so several server ticks can be attributed to one transition afterward.
    # Bound only that shaping component; terminal outcomes are kept separate.
    shaping_reward_clip: float = 0.25
    # A real kill is intentionally much more valuable than any one dense
    # shaping event. Keep a generous safety ceiling so an arena-side terminal
    # bonus is not silently flattened back to the historical +/-1 outcome.
    terminal_reward_clip: float = 64.0

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
    arena_host: str = "127.0.0.1"
    arena_port: int = 8765
    adaptive_rewards: bool = True
