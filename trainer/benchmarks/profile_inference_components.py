from __future__ import annotations

import asyncio
import json
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import torch


TRAINER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINER_ROOT))
sys.path.insert(0, str(TRAINER_ROOT / "tests"))

import combat_ai.league as league_module  # noqa: E402
import combat_ai.service as service_module  # noqa: E402
from combat_ai.config import PPOConfig, ServiceConfig  # noqa: E402
from combat_ai.model import CombatPolicy  # noqa: E402
from fixtures import observation  # noqa: E402


def main() -> None:
    samples: dict[str, list[float]] = defaultdict(list)

    def wrap(module: object, name: str, label: str) -> None:
        original = getattr(module, name)

        def measured(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                samples[label].append((time.perf_counter() - started) * 1000.0)

        setattr(module, name, measured)

    wrap(service_module, "encode_observation", "main_encode_for_replay")
    wrap(service_module, "batch_observations", "main_batch_encode")
    wrap(service_module, "sample_actions", "main_sample")
    wrap(league_module, "batch_observations", "league_batch_encode")
    wrap(league_module, "sample_actions", "league_sample")
    original_forward = CombatPolicy.forward

    def measured_forward(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            samples["policy_forward"].append((time.perf_counter() - started) * 1000.0)

    CombatPolicy.forward = measured_forward
    with tempfile.TemporaryDirectory(prefix="mcai-profile-") as raw_directory:
        directory = Path(raw_directory)
        historical = directory / "historical.pt"
        torch.save({"policy": CombatPolicy().state_dict()}, historical)
        service = service_module.PolicyService(
            PPOConfig(rollout_agent_ticks=100_000),
            ServiceConfig(checkpoint_dir=directory, deterministic_inference=False, cpu_threads=1),
        )
        service.league.force_frozen_opponent(historical)
        service._finish_pending = lambda _step, _bootstrap: None
        for tick in range(20):
            message = batch_message(tick)
            started = time.perf_counter()
            asyncio.run(service.handle_message(message))
            samples["total"].append((time.perf_counter() - started) * 1000.0)
    print(json.dumps({
        name: {
            "calls": len(values),
            "median_ms": statistics.median(values),
            "total_ms": sum(values),
        }
        for name, values in samples.items()
    }, indent=2))


def batch_message(tick: int) -> dict:
    return {
        "schema_version": 1,
        "type": "step_batch",
        "sequence": tick,
        "steps": [
            {
                "agent_id": f"agent-{lane * 2 + side}",
                "observation": observation(f"profile-lane-{lane}", tick),
                "reward": 0.0,
                "terminated": False,
                "truncated": False,
                "info": {},
            }
            for lane in range(4)
            for side in range(2)
        ],
    }


if __name__ == "__main__":
    main()
