from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import PPOConfig
from .model import CombatPolicy


@dataclass
class CheckpointState:
    policy_version: int = 0
    total_agent_ticks: int = 0
    next_snapshot_tick: int = 100_000


class CheckpointManager:
    def __init__(self, directory: Path, snapshot_interval: int):
        self.directory = directory
        self.snapshot_interval = snapshot_interval
        self.directory.mkdir(parents=True, exist_ok=True)

    def restore(
        self,
        policy: CombatPolicy,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> CheckpointState:
        latest = self.directory / "latest.pt"
        if not latest.exists():
            return CheckpointState(next_snapshot_tick=self.snapshot_interval)
        payload = torch.load(latest, map_location=device, weights_only=False)
        policy.load_state_dict(payload["policy"])
        optimizer.load_state_dict(payload["optimizer"])
        return CheckpointState(
            policy_version=int(payload.get("policy_version", 0)),
            total_agent_ticks=int(payload.get("total_agent_ticks", 0)),
            next_snapshot_tick=int(payload.get("next_snapshot_tick", self.snapshot_interval)),
        )

    def save(
        self,
        policy: CombatPolicy,
        optimizer: torch.optim.Optimizer,
        state: CheckpointState,
        config: PPOConfig,
        metrics: dict[str, Any],
    ) -> None:
        snapshot: Path | None = None
        if state.total_agent_ticks >= state.next_snapshot_tick:
            snapshot = self.directory / f"policy-{state.total_agent_ticks:012d}.pt"
            while state.total_agent_ticks >= state.next_snapshot_tick:
                state.next_snapshot_tick += self.snapshot_interval
        payload = {
            "format_version": 1,
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "policy_version": state.policy_version,
            "total_agent_ticks": state.total_agent_ticks,
            "next_snapshot_tick": state.next_snapshot_tick,
            "config": config.to_dict(),
            "metrics": metrics,
        }
        temporary = self.directory / "latest.pt.tmp"
        torch.save(payload, temporary)
        os.replace(temporary, self.directory / "latest.pt")
        if snapshot is not None:
            snapshot_temporary = snapshot.with_suffix(".pt.tmp")
            torch.save(payload, snapshot_temporary)
            os.replace(snapshot_temporary, snapshot)
        with (self.directory / "metrics.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps({"ticks": state.total_agent_ticks, "version": state.policy_version, **metrics}) + "\n")
