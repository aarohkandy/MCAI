from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


STAGES = ("infrastructure", "sword", "crystal", "combined", "experienced_human")


@dataclass(frozen=True)
class Gate:
    metric: str
    minimum_episodes: int = 0
    minimum_rate: float | None = None
    maximum_rate: float | None = None
    minimum_value: float | None = None

    def passes(self, metrics: dict[str, dict[str, Any]]) -> bool:
        result = metrics.get(self.metric, {})
        if int(result.get("episodes", 0)) < self.minimum_episodes:
            return False
        rate = float(result.get("rate", 0.0))
        value = float(result.get("value", 0.0))
        if self.minimum_rate is not None and rate < self.minimum_rate:
            return False
        if self.maximum_rate is not None and rate >= self.maximum_rate:
            return False
        if self.minimum_value is not None and value < self.minimum_value:
            return False
        return True


INFRASTRUCTURE_TASKS = (
    "turning", "approaching", "following", "raycasting", "attack_reach",
    "jumping", "sprinting", "block_placement", "block_mining", "item_selection",
)
CRYSTAL_TASKS = (
    "obsidian_placement", "crystal_clearance", "crystal_breaking", "lower_self_damage",
    "hit_crystal", "safe_eating", "retotem", "mine_cover", "escape_crystal", "obsidian_cover",
)

GATES: dict[str, tuple[Gate, ...]] = {
    "infrastructure": tuple(Gate(task, 500, minimum_rate=0.95) for task in INFRASTRUCTURE_TASKS),
    "sword": (
        *(Gate(f"script_{style}", 500, minimum_rate=0.90) for style in
          ("rush", "strafe", "retreat", "jump_critical", "defensive", "erratic")),
        Gate("user_sword", 100, minimum_rate=0.60),
        Gate("frozen_no_regression", 3, minimum_rate=1.0),
    ),
    "crystal": (
        *(Gate(task, 100, minimum_rate=0.90) for task in CRYSTAL_TASKS),
        Gate("timely_retotem", 100, minimum_rate=0.95),
        Gate("avoidable_self_kill", 100, maximum_rate=0.10),
    ),
    "combined": (
        Gate("scripted_combined", 500, minimum_rate=0.90),
        Gate("frozen_pool_elo_trend", 3, minimum_value=1e-9),
    ),
    "experienced_human": (
        Gate("experienced_human", 90, minimum_rate=0.50),
        Gate("experienced_volunteers", minimum_value=3.0),
    ),
}


@dataclass
class CurriculumState:
    current_stage: str = "infrastructure"
    completed: list[str] = field(default_factory=list)
    evaluations: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "CurriculumState":
        if not path.exists():
            return cls()
        value = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            current_stage=str(value.get("current_stage", "infrastructure")),
            completed=[str(stage) for stage in value.get("completed", [])],
            evaluations=list(value.get("evaluations", [])),
        )

    def evaluate(self, results: dict[str, Any]) -> dict[str, Any]:
        stage = str(results.get("stage", self.current_stage))
        if stage != self.current_stage:
            raise ValueError(f"results are for {stage}, but current stage is {self.current_stage}")
        metrics = dict(results.get("metrics", {}))
        failed = [asdict(gate) for gate in GATES[stage] if not gate.passes(metrics)]
        promoted = not failed
        self.evaluations.append({"stage": stage, "metrics": metrics, "passed": promoted})
        if promoted and stage not in self.completed:
            self.completed.append(stage)
            index = STAGES.index(stage)
            if index + 1 < len(STAGES):
                self.current_stage = STAGES[index + 1]
        return {"stage": stage, "promoted": promoted, "next_stage": self.current_stage, "failed_gates": failed}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)


def is_held_out(arena_seed: int, action_delay: int, observation_delay: int, fraction: float = 0.20) -> bool:
    """Stable split; a seed/delay tuple can never drift between train and evaluation."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between zero and one")
    digest = hashlib.sha256(f"{arena_seed}:{action_delay}:{observation_delay}".encode("ascii")).digest()
    buckets = round(fraction * 100)
    return int.from_bytes(digest[:4], "big") % 100 < buckets
