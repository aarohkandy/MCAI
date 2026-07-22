from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch


TRAINER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINER_ROOT))
sys.path.insert(0, str(TRAINER_ROOT / "tests"))

from combat_ai.config import PPOConfig, ServiceConfig  # noqa: E402
from combat_ai.model import CombatPolicy  # noqa: E402
from combat_ai.service import PolicyService  # noqa: E402
from fixtures import observation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark one realistic 8-agent/4-lane historical league batch"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--cpu-threads", type=int, default=1)
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="mcai-league-bench-") as raw_directory:
        directory = Path(raw_directory)
        historical = directory / "historical.pt"
        torch.save({"policy": CombatPolicy().state_dict()}, historical)
        service = PolicyService(
            PPOConfig(rollout_agent_ticks=100_000),
            ServiceConfig(
                checkpoint_dir=directory,
                deterministic_inference=True,
                cpu_threads=max(1, arguments.cpu_threads),
            ),
        )
        service.league.force_frozen_opponent(historical)
        # Keep the benchmark scoped to observation-to-action latency. Replay,
        # reward attribution and rollout bookkeeping are measured separately.
        service._finish_pending = lambda _step, _bootstrap: None
        durations = []
        total = max(0, arguments.warmup) + max(1, arguments.iterations)
        for tick in range(1, total + 1):
            message = batch_message(tick)
            started = time.perf_counter()
            response = asyncio.run(service.handle_message(message))
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if len(response["actions"]) != 8:
                raise RuntimeError("benchmark batch did not return eight actions")
            if tick > arguments.warmup:
                durations.append(elapsed_ms)
        ordered = sorted(durations)
        p95_index = max(0, min(len(ordered) - 1, int(len(ordered) * 0.95) - 1))
        print(json.dumps({
            "scenario": "8_agents_4_lanes_4_main_4_historical",
            "iterations": len(durations),
            "cpu_threads": max(1, arguments.cpu_threads),
            "median_ms": statistics.median(durations),
            "p95_ms": ordered[p95_index],
            "mean_ms": statistics.mean(durations),
        }, indent=2))


def batch_message(tick: int) -> dict:
    steps = []
    for lane in range(4):
        episode = f"benchmark-lane-{lane}"
        for side in range(2):
            agent_id = f"agent-{lane * 2 + side}"
            steps.append({
                "agent_id": agent_id,
                "observation": observation(episode, tick),
                "reward": 0.0,
                "terminated": False,
                "truncated": False,
                "info": {},
            })
    return {"schema_version": 1, "type": "step_batch", "sequence": tick, "steps": steps}


if __name__ == "__main__":
    main()
