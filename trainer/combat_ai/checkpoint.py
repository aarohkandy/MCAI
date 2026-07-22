from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import PPOConfig
from .model import CombatPolicy


# Contract 2 activated legacy observation geometry/inventory columns. Contract
# 3 activates canonical V2 tactical candidate columns that were previously
# zero or carried different aliases. Both migrations are one-time and keep all
# tensor shapes stable.
FEATURE_CONTRACT_VERSION = 3


def load_policy_weights(policy: CombatPolicy, payload: dict[str, Any]) -> bool:
    """Load V2 exactly or migrate every shape-compatible V1 parameter."""
    architecture = int(payload.get("architecture_version", 1))
    if architecture == getattr(policy, "architecture_version", 1):
        policy.load_state_dict(payload["policy"])
        return True
    current = policy.state_dict()
    current.update({
        name: value for name, value in payload["policy"].items()
        if name in current and current[name].shape == value.shape
    })
    policy.load_state_dict(current)
    return False

_SELF_REPURPOSED_COLUMNS = (
    31, 34, 37, 40,  # armor count -> semantic armor category
    45, 49, 53, 57, 61, 65, 69, 73, 77,  # hotbar max durability -> item category
)
_OPPONENT_REPURPOSED_COLUMNS = (
    9, 10,  # formerly-null opponent health/known flag now server-authoritative
    27, 30, 33, 36,  # armor count -> semantic armor category
    13,  # formerly-dead LOS now performs real block occlusion checks
    *range(38, 48),  # previously-zero direct combat geometry
)
_OPPONENT_FORMER_CONSTANT_COLUMNS = (12,)  # stale on_ground was effectively true
_ENTITY_REPURPOSED_COLUMNS = (16, 17)  # previously-zero corrected x/z
_BLOCK_REPURPOSED_COLUMNS = (18,)  # arbitrary name hash -> corrected bearing
_LEGAL_CONTEXT_COLUMNS = (21, 22, 23)  # constant ones -> delay/delay/terrain
_TACTICAL_STABLE_COLUMNS = (0, 2)  # distance and visibility retained their meanings
_TACTICAL_REPURPOSED_COLUMNS = tuple(
    index for index in range(16) if index not in _TACTICAL_STABLE_COLUMNS
)


@dataclass
class CheckpointState:
    policy_version: int = 0
    total_agent_ticks: int = 0
    next_snapshot_tick: int = 100_000
    rollout_generation: int = 0


class CheckpointManager:
    def __init__(self, directory: Path, snapshot_interval: int):
        self.directory = directory
        self.snapshot_interval = snapshot_interval
        self.directory.mkdir(parents=True, exist_ok=True)
        self.restored_imitation_records: list[dict[str, Any]] = []
        self.restored_elite_replay_records: list[dict[str, Any]] = []

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
        architecture = int(payload.get("architecture_version", 1))
        if architecture == getattr(policy, "architecture_version", 1):
            load_policy_weights(policy, payload)
            optimizer.load_state_dict(payload["optimizer"])
        else:
            # Checkpoint-breaking V2: retain every compatible tensor (notably
            # feature-side biases/priors) and start new relational/recurrent
            # modules clean. This is also the distillation initialization path.
            load_policy_weights(policy, payload)
            optimizer.state.clear()
        self.restored_imitation_records = list(payload.get("online_imitation_records", []))
        self.restored_elite_replay_records = list(payload.get("elite_replay_records", []))
        # The sidecar is written after actor publication and therefore can
        # include successes collected while the learner process was busy.  A
        # corrupt/missing sidecar is harmless because latest.pt contains the
        # last complete generation snapshot as a fallback.
        replay_sidecar = self.directory / "elite-replay.pt"
        if replay_sidecar.exists():
            try:
                replay_payload = torch.load(
                    replay_sidecar, map_location="cpu", weights_only=False,
                )
                replay_records = replay_payload.get("records")
                sidecar_version = int(replay_payload.get("policy_version", -1))
                checkpoint_version = int(payload.get("policy_version", 0))
                if isinstance(replay_records, list) and sidecar_version == checkpoint_version:
                    self.restored_elite_replay_records = replay_records
            except Exception:
                pass
        from_version = int(payload.get("feature_contract_version", 1))
        if from_version < FEATURE_CONTRACT_VERSION:
            migrate_feature_contract(policy, from_version)
            # Adam's moments refer to the old meanings of the repurposed
            # inputs. Starting those moments clean prevents the first update
            # from reintroducing the stale slot-based behavior.
            optimizer.state.clear()
        return CheckpointState(
            policy_version=int(payload.get("policy_version", 0)),
            total_agent_ticks=int(payload.get("total_agent_ticks", 0)),
            next_snapshot_tick=int(payload.get("next_snapshot_tick", self.snapshot_interval)),
            rollout_generation=int(payload.get("rollout_generation", 0)),
        )

    def save(
        self,
        policy: CombatPolicy,
        optimizer: torch.optim.Optimizer,
        state: CheckpointState,
        config: PPOConfig,
        metrics: dict[str, Any], online_imitation_records: list[dict[str, Any]] | None = None,
        elite_replay_records: list[dict[str, Any]] | None = None,
    ) -> None:
        snapshot: Path | None = None
        if state.total_agent_ticks >= state.next_snapshot_tick:
            snapshot = self.directory / f"policy-{state.total_agent_ticks:012d}.pt"
            while state.total_agent_ticks >= state.next_snapshot_tick:
                state.next_snapshot_tick += self.snapshot_interval
        payload = {
            "format_version": 3,
            "architecture_version": getattr(policy, "architecture_version", 1),
            "feature_contract_version": FEATURE_CONTRACT_VERSION,
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "policy_version": state.policy_version,
            "total_agent_ticks": state.total_agent_ticks,
            "next_snapshot_tick": state.next_snapshot_tick,
            "rollout_generation": state.rollout_generation,
            "online_imitation_records": list(online_imitation_records or []),
            "elite_replay_records": list(elite_replay_records or []),
            "config": config.to_dict(),
            "metrics": metrics,
        }
        temporary = self.directory / "latest.pt.tmp"
        torch.save(payload, temporary)
        os.replace(temporary, self.directory / "latest.pt")
        self._save_elite_replay_sidecar(
            list(elite_replay_records or []), state.policy_version,
        )
        if snapshot is not None:
            # Historical opponents are inference artifacts, not recovery
            # checkpoints. Keeping Adam moments and thousands of imitation
            # observations in every numbered policy inflated each file above
            # 150 MB; PFSP then stalled live control while loading a new one.
            # latest.pt remains the complete atomic recovery checkpoint.
            historical_payload = {
                "format_version": payload["format_version"],
                "architecture_version": payload["architecture_version"],
                "feature_contract_version": payload["feature_contract_version"],
                "policy": payload["policy"],
                "policy_version": payload["policy_version"],
                "total_agent_ticks": payload["total_agent_ticks"],
                "config": payload["config"],
                "metrics": payload["metrics"],
            }
            snapshot_temporary = snapshot.with_suffix(".pt.tmp")
            torch.save(historical_payload, snapshot_temporary)
            os.replace(snapshot_temporary, snapshot)
        with (self.directory / "metrics.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps({"ticks": state.total_agent_ticks, "version": state.policy_version, **metrics}) + "\n")

    def stage(
        self,
        policy: CombatPolicy,
        optimizer: torch.optim.Optimizer,
        state: CheckpointState,
        config: PPOConfig,
        metrics: dict[str, Any],
        online_imitation_records: list[dict[str, Any]] | None = None,
        elite_replay_records: list[dict[str, Any]] | None = None,
    ) -> tuple[Path, Path | None, Path | None]:
        """Write a generation without making it a recoverable checkpoint.

        The learner process may finish before the actor process can publish its
        weights.  Staging prevents a crash in that interval from restoring a
        policy generation that was never served.
        """
        snapshot: Path | None = None
        if state.total_agent_ticks >= state.next_snapshot_tick:
            snapshot = self.directory / f"policy-{state.total_agent_ticks:012d}.pt"
            while state.total_agent_ticks >= state.next_snapshot_tick:
                state.next_snapshot_tick += self.snapshot_interval
        payload = {
            "format_version": 3,
            "architecture_version": getattr(policy, "architecture_version", 1),
            "feature_contract_version": FEATURE_CONTRACT_VERSION,
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "policy_version": state.policy_version,
            "total_agent_ticks": state.total_agent_ticks,
            "next_snapshot_tick": state.next_snapshot_tick,
            "rollout_generation": state.rollout_generation,
            "online_imitation_records": list(online_imitation_records or []),
            "elite_replay_records": list(elite_replay_records or []),
            "config": config.to_dict(),
            "metrics": metrics,
        }
        suffix = f"{os.getpid()}-{state.policy_version}-{state.rollout_generation}"
        latest_stage = self.directory / f"latest.{suffix}.stage"
        torch.save(payload, latest_stage)
        snapshot_stage: Path | None = None
        if snapshot is not None:
            historical_payload = {
                "format_version": payload["format_version"],
                "architecture_version": payload["architecture_version"],
                "feature_contract_version": payload["feature_contract_version"],
                "policy": payload["policy"],
                "policy_version": payload["policy_version"],
                "total_agent_ticks": payload["total_agent_ticks"],
                "config": payload["config"],
                "metrics": payload["metrics"],
            }
            snapshot_stage = self.directory / f"{snapshot.name}.{suffix}.stage"
            torch.save(historical_payload, snapshot_stage)
        return latest_stage, snapshot_stage, snapshot

    def promote_staged(
        self,
        latest_stage: Path,
        snapshot_stage: Path | None,
        snapshot: Path | None,
        state: CheckpointState,
        metrics: dict[str, Any],
        elite_replay_records: list[dict[str, Any]] | None = None,
    ) -> None:
        """Atomically expose artifacts after the matching actor is published."""
        os.replace(latest_stage, self.directory / "latest.pt")
        if elite_replay_records is not None:
            self._save_elite_replay_sidecar(
                list(elite_replay_records), state.policy_version,
            )
        if snapshot_stage is not None and snapshot is not None:
            os.replace(snapshot_stage, snapshot)
        with (self.directory / "metrics.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps({"ticks": state.total_agent_ticks, "version": state.policy_version, **metrics}) + "\n")

    def _save_elite_replay_sidecar(
        self, records: list[dict[str, Any]], policy_version: int,
    ) -> None:
        temporary = self.directory / "elite-replay.pt.tmp"
        torch.save({
            "format_version": 1,
            "policy_version": int(policy_version),
            "records": records,
        }, temporary)
        os.replace(temporary, self.directory / "elite-replay.pt")


def migrate_feature_contract(policy: CombatPolicy, from_version: int) -> None:
    """Neutralize new/repurposed feature inputs in a legacy policy.

    The action priors can use the new geometry and semantic inventory
    immediately. The learned encoders begin at zero influence and acquire the
    new signals through ordinary PPO gradients without a random logit jump at
    process restart.
    """
    version = max(1, int(from_version))
    with torch.no_grad():
        if version < 2:
            policy.self_encoder.layers[0].weight[:, _SELF_REPURPOSED_COLUMNS] = 0
            opponent_layer = policy.opponent_encoder.layers[0]
            opponent_layer.weight[:, _OPPONENT_REPURPOSED_COLUMNS] = 0
            # Mineflayer's other-player onGround flag was effectively a constant
            # true in the old feed. Keep that learned offset in the bias before the
            # server-authoritative grounded signal becomes live.
            opponent_layer.bias.add_(
                opponent_layer.weight[:, _OPPONENT_FORMER_CONSTANT_COLUMNS].sum(dim=1)
            )
            opponent_layer.weight[:, _OPPONENT_FORMER_CONSTANT_COLUMNS] = 0
            policy.entity_encoder.layers[0].weight[:, _ENTITY_REPURPOSED_COLUMNS] = 0
            policy.block_encoder.layers[0].weight[:, _BLOCK_REPURPOSED_COLUMNS] = 0
            legal_layer = policy.legal_encoder.layers[0]
            # These three inputs were always exactly one. Preserve their complete
            # legacy contribution in the bias before assigning them live context.
            legal_layer.bias.add_(legal_layer.weight[:, _LEGAL_CONTEXT_COLUMNS].sum(dim=1))
            legal_layer.weight[:, _LEGAL_CONTEXT_COLUMNS] = 0
        if version < 3:
            # Canonical candidate aliases make these inputs live for the first
            # time. Preserve only distance and visibility, whose meanings were
            # already stable in contract 2; learn every other candidate signal
            # from a neutral influence instead of stale random attention weights.
            policy.crystal_attention.encoder.layers[0].weight[
                :, _TACTICAL_REPURPOSED_COLUMNS
            ] = 0
            policy.tactical_block_attention.encoder.layers[0].weight[
                :, _TACTICAL_REPURPOSED_COLUMNS
            ] = 0
