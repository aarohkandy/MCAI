from __future__ import annotations

import asyncio
import concurrent.futures
import json
import math
import multiprocessing
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import torch
import websockets

from .adaptive_rewards import AdaptiveRewardController, publish_reward_profile
from .buffer import RolloutBuffer, Transition, prepare_sequences
from .checkpoint import (
    FEATURE_CONTRACT_VERSION, CheckpointManager, CheckpointState, load_policy_weights,
    migrate_feature_contract,
)
from .config import PPOConfig, ServiceConfig
from .distribution import ActionTensor, actions_from_wire, sample_actions
from .elite_replay import EliteReplayBuffer
from .features import (
    PRIMARY_NAMES, batch_encoded_observations, batch_observations,
    categorical_masks, encode_observation,
)
from .model import CombatPolicy
from .imitation import classify_crystal_teacher_action, load_demonstrations, split_matches
from .league import LeagueManager
from .ppo import PPOTrainer, choose_device
from .reward_shaping import TacticalSnapshot, TrainerRewardShaper, TrainerShaping, tactical_snapshot


@dataclass
class PendingStep:
    action_id: int
    recurrent_parent_action_id: int | None
    agent_id: str
    episode_id: str
    policy_version: int
    features: dict[str, np.ndarray]
    hidden: np.ndarray
    categorical_action: dict[str, int]
    camera_action: np.ndarray
    log_probability: float
    value: float
    reward_snapshot: TacticalSnapshot
    observation: dict[str, Any]
    wire_action: dict[str, Any]


@dataclass(frozen=True)
class SanitizedReward:
    server_reward: float
    trainer_shaping_reward: float
    raw_reward: float
    training_reward: float
    raw_shaping_reward: float
    shaping_reward: float
    raw_terminal_reward: float
    terminal_reward: float
    clipped: bool
    nonfinite: bool


@dataclass
class RewardTelemetry:
    transitions: int = 0
    raw_sum: float = 0.0
    training_sum: float = 0.0
    clipped_transitions: int = 0
    nonfinite_transitions: int = 0
    terminal_transitions: int = 0
    max_abs_raw: float = 0.0
    server_sum: float = 0.0
    trainer_shaping_sum: float = 0.0
    trainer_component_sums: dict[str, float] = field(default_factory=dict)
    trainer_component_events: dict[str, int] = field(default_factory=dict)
    policy_crystal_counter_deltas: dict[str, int] = field(default_factory=dict)
    rollout_counter_deltas: dict[str, float] = field(default_factory=dict)
    policy_owned_kill_events: int = 0
    terminal_attached_events: int = 0
    rejected_nonpolicy_terminals: int = 0

    def record(self, reward: SanitizedReward, shaping: TrainerShaping | None = None) -> None:
        self.transitions += 1
        self.server_sum += reward.server_reward
        self.trainer_shaping_sum += reward.trainer_shaping_reward
        self.raw_sum += reward.raw_reward
        self.training_sum += reward.training_reward
        self.clipped_transitions += int(reward.clipped)
        self.nonfinite_transitions += int(reward.nonfinite)
        self.terminal_transitions += int(abs(reward.raw_terminal_reward) > 1e-12)
        self.max_abs_raw = max(self.max_abs_raw, abs(reward.raw_reward))
        if shaping is not None:
            for name, value in shaping.components.items():
                self.trainer_component_sums[name] = self.trainer_component_sums.get(name, 0.0) + value
                self.trainer_component_events[name] = self.trainer_component_events.get(name, 0) + int(
                    abs(value) > 1e-12
                )

    def metrics(self) -> dict[str, float | int]:
        denominator = max(1, self.transitions)
        metrics: dict[str, float | int] = {
            "reward_transitions": self.transitions,
            "reward_mean_server_raw": self.server_sum / denominator,
            "reward_mean_trainer_shaping": self.trainer_shaping_sum / denominator,
            "reward_mean_raw": self.raw_sum / denominator,
            "reward_mean_training": self.training_sum / denominator,
            "reward_clipped_transitions": self.clipped_transitions,
            "reward_nonfinite_transitions": self.nonfinite_transitions,
            "reward_terminal_transitions": self.terminal_transitions,
            "policy_owned_kill_events": self.policy_owned_kill_events,
            "terminal_attached_events": self.terminal_attached_events,
            "rejected_nonpolicy_terminals": self.rejected_nonpolicy_terminals,
            "reward_max_abs_raw": self.max_abs_raw,
        }
        for name in sorted(self.trainer_component_sums):
            metrics[f"trainer_reward_{name}_sum"] = self.trainer_component_sums[name]
            metrics[f"trainer_reward_{name}_events"] = self.trainer_component_events[name]
        for name in sorted(self.policy_crystal_counter_deltas):
            metrics[f"server_{name}_events"] = self.policy_crystal_counter_deltas[name]
        for name in sorted(self.rollout_counter_deltas):
            metrics[f"server_{name}"] = self.rollout_counter_deltas[name]
        return metrics


def _learn_generation_job(
    actor_state: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any],
    transitions: list[Transition],
    config: PPOConfig,
    imitation_records: list[dict[str, Any]],
    online_imitation_records: list[dict[str, Any]],
    elite_replay_records: list[dict[str, Any]],
    imitation_weight: float,
    reward_metrics: dict[str, Any],
    checkpoint_directory: Path,
    snapshot_interval: int,
    proposed_state: CheckpointState,
    low_kl_updates: int,
) -> tuple[dict, dict, CheckpointState, int, str, str | None, str | None]:
    """CPU learner-process entrypoint; never touches the live actor."""
    torch.set_num_threads(max(1, config.learner_cpu_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device("cpu")
    learner_policy = CombatPolicy()
    learner_policy.load_state_dict(actor_state)
    learner = PPOTrainer(learner_policy, config, device)
    learner.optimizer.load_state_dict(optimizer_state)
    learner.low_kl_updates = int(low_kl_updates)
    batch = prepare_sequences(
        transitions, config.recurrent_sequence_length, config.gamma,
        config.gae_lambda, device,
    )
    metrics = learner.update(batch)
    metrics.imitation_loss = learner.auxiliary_imitation_update(
        imitation_records, imitation_weight
    )
    imitation = learner.last_imitation_metrics
    metrics.imitation_updates = int(imitation.get("updates", 0))
    metrics.imitation_samples = int(imitation.get("samples", 0))
    metrics.imitation_crystal_samples = int(imitation.get("crystal_samples", 0))
    metrics.imitation_crystal_fraction = float(imitation.get("crystal_fraction", 0.0))
    metrics.imitation_elite_samples = int(imitation.get("elite_samples", 0))
    metrics.imitation_elite_fraction = float(imitation.get("elite_fraction", 0.0))
    metrics.imitation_elite_events_sampled = int(
        imitation.get("elite_events_sampled", 0)
    )
    metrics.imitation_crystal_buffer = sum(
        str(record.get("execution_source", "")) == "teacher_crystal"
        for record in online_imitation_records
    )
    metrics.imitation_sword_buffer = sum(
        str(record.get("execution_source", "")) == "teacher_sword"
        for record in online_imitation_records
    )
    metrics.imitation_block_buffer = sum(
        str(record.get("execution_source", "")) == "teacher_block"
        for record in online_imitation_records
    )
    metrics.imitation_elite_buffer = len(elite_replay_records)
    metrics.imitation_elite_events = len({
        str(record.get("elite_event_id", "")) for record in elite_replay_records
        if record.get("elite_event_id")
    })
    metrics.imitation_elite_buckets = len({
        str(record.get("elite_bucket", "")) for record in elite_replay_records
        if record.get("elite_bucket")
    })
    metrics_dict = vars(metrics)
    checkpoints = CheckpointManager(checkpoint_directory, snapshot_interval)
    latest_stage, snapshot_stage, snapshot = checkpoints.stage(
        learner.policy, learner.optimizer, proposed_state, config, metrics_dict,
        online_imitation_records, elite_replay_records,
    )
    # Do not return model/Adam tensors through ProcessPool IPC. PyTorch's
    # tensor reducers use a second resource-sharing channel, and a live
    # Windows learner can wedge indefinitely after successfully writing the
    # checkpoint while that large result is reconstructed in the actor.
    # The actor asynchronously preloads the already crash-safe stage instead.
    return (
        metrics_dict, reward_metrics, proposed_state, learner.low_kl_updates,
        str(latest_stage),
        str(snapshot_stage) if snapshot_stage is not None else None,
        str(snapshot) if snapshot is not None else None,
    )


def _load_staged_training_state(
    latest_stage: Path, device: torch.device, expected_state: CheckpointState,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load publication tensors in the actor process without using tensor IPC."""
    payload = torch.load(latest_stage, map_location=device, weights_only=False)
    if int(payload.get("policy_version", -1)) != expected_state.policy_version:
        raise RuntimeError("staged policy version does not match learner result")
    if int(payload.get("rollout_generation", -1)) != expected_state.rollout_generation:
        raise RuntimeError("staged rollout generation does not match learner result")
    policy_state = payload.get("policy")
    optimizer_state = payload.get("optimizer")
    if not isinstance(policy_state, dict) or not isinstance(optimizer_state, dict):
        raise RuntimeError("staged learner checkpoint is missing training state")
    return policy_state, optimizer_state


def _save_elite_replay_sidecar_job(
    checkpoint_directory: Path,
    records: list[dict[str, Any]],
    policy_version: int,
) -> tuple[int, int]:
    """Persist post-learner successes away from the live actor event loop."""
    checkpoints = CheckpointManager(checkpoint_directory, 1)
    checkpoints._save_elite_replay_sidecar(records, policy_version)
    return int(policy_version), len(records)


class PolicyService:
    ONLINE_IMITATION_CAPACITY = 8_192
    ONLINE_IMITATION_MIN_WEIGHT = 0.02
    TEACHER_SOURCES = frozenset(("teacher_sword", "teacher_crystal", "teacher_block"))
    MAX_PENDING_ACTIONS_PER_AGENT = 32

    def __init__(
        self, ppo_config: PPOConfig, service_config: ServiceConfig,
        imitation_data: Path | None = None, initialize_from: list[Path] | None = None,
        exploiter_target: Path | None = None, freeze_policy: bool = False,
    ):
        mps = getattr(torch.backends, "mps", None)
        if service_config.cpu_threads > 0:
            torch.set_num_threads(service_config.cpu_threads)
        elif not torch.cuda.is_available() and not (mps is not None and mps.is_available()):
            # Leave roughly half the Surface CPU available to Paper and rollout clients.
            torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        torch.manual_seed(service_config.seed)
        np.random.seed(service_config.seed)
        self.config = ppo_config
        self.service_config = service_config
        self.freeze_policy = bool(freeze_policy)
        self.device = choose_device()
        self.policy = CombatPolicy()
        if not 1_000_000 <= self.policy.parameter_count <= 2_000_000:
            raise RuntimeError(f"V2 policy must stay near 1-2M parameters: {self.policy.parameter_count:,}")
        self.trainer = PPOTrainer(self.policy, ppo_config, self.device)
        self.checkpoints = CheckpointManager(service_config.checkpoint_dir, ppo_config.checkpoint_every_ticks)
        self.state = self.checkpoints.restore(self.policy, self.trainer.optimizer, self.device)
        # Adam checkpoints include their original parameter-group learning rate.
        # Reapply today's safer runtime setting after restore so an older run
        # cannot silently put online PPO back at its former aggressive rate.
        self.trainer.reapply_configured_learning_rate()
        if self.state.total_agent_ticks == 0 and initialize_from and not (service_config.checkpoint_dir / "latest.pt").exists():
            _initialize_policy(self.policy, initialize_from, self.device)
        self.buffer = RolloutBuffer(ppo_config.rollout_agent_ticks)
        self.hidden: dict[str, torch.Tensor] = {}
        self.pending: dict[str, dict[int, PendingStep]] = {}
        # Tracks proposal ancestry, not accepted PPO samples. It advances for
        # every recurrent policy inference so a later teacher/safety exclusion
        # remains visible as a hard replay boundary even when reports arrive
        # asynchronously.
        self.last_recurrent_proposal: dict[str, tuple[str, int]] = {}
        # An explicit execution override can occur without consuming a queued
        # action ID. Attach that environmental off-policy boundary to the next
        # accepted policy transition, independently of proposal ancestry.
        self.execution_gap_pending: set[str] = set()
        self.next_action_id = 1
        self.update_lock = asyncio.Lock()
        self.buffer_ready_grace_remaining: int | None = None
        self.league = LeagueManager(service_config.checkpoint_dir, self.device)
        self.league.force_frozen_opponent(exploiter_target)
        self.imitation_records = []
        self.online_imitation_records: list[dict[str, Any]] = list(
            self.checkpoints.restored_imitation_records
        )
        self.elite_replay = EliteReplayBuffer(
            capacity=ppo_config.elite_replay_capacity,
            trace_capacity=ppo_config.elite_trace_actions,
            kill_window=ppo_config.elite_kill_window,
            crystal_window=ppo_config.elite_crystal_window,
            restored_records=self.checkpoints.restored_elite_replay_records,
        )
        self.progress_interval = max(128, ppo_config.rollout_agent_ticks // 32)
        self.next_progress_tick = self.progress_interval
        self.reward_telemetry = RewardTelemetry()
        self.server_crystal_counter_state: dict[tuple[str, str, str], int] = {}
        self.server_rollout_counter_state: dict[tuple[str, str, str], float] = {}
        self.reward_shaper = TrainerRewardShaper()
        self.adaptive_rewards = AdaptiveRewardController(
            service_config.checkpoint_dir,
            enabled=service_config.adaptive_rewards,
        )
        # The server can restart independently of the trainer, so always
        # republish the persisted idempotent profile once control port 8765 is
        # reachable. Later successful policy generations mark it pending again
        # only when the controller actually changes a weight.
        self.adaptive_reward_sync_pending = bool(service_config.adaptive_rewards)
        self.adaptive_reward_last_error_log = 0.0
        self.adaptive_reward_quarantine_until = 0.0
        self.adaptive_reward_quarantine_generation = 0
        self.adaptive_reward_quarantined_steps = 0
        self.training_task: asyncio.Future | None = None
        self.learner_executor: concurrent.futures.Executor | None = None
        self.sidecar_tasks: set[asyncio.Future] = set()
        self.training_transitions: list[Transition] = []
        self.training_ticks = 0
        self.training_retry_not_before = 0.0
        self.worker_metrics: dict[str, float | int] = {}
        if imitation_data is not None:
            self.imitation_records = split_matches(load_demonstrations(imitation_data))[0]

    async def handle(self, websocket: Any) -> None:
        async for packed in websocket:
            try:
                message = msgpack.unpackb(packed, raw=False)
                response = await self.handle_message(message)
                if response is not None:
                    await websocket.send(msgpack.packb(response, use_bin_type=True))
            except Exception as error:
                response = {"schema_version": 1, "type": "control", "sequence": 0,
                            "command": "error", "payload": {"message": str(error)}}
                await websocket.send(msgpack.packb(response, use_bin_type=True))

    async def handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        if int(message.get("schema_version", -1)) != 1:
            raise ValueError("unsupported wire schema_version")
        if message.get("type") == "hello":
            # A hello begins a new websocket control session. Policy proposals
            # from an earlier socket can no longer exist in the worker queue,
            # so their trainer-side state must not survive the reconnect.
            for raw_agent_id in message.get("agents") or []:
                agent_id = str(raw_agent_id)
                self.pending.pop(agent_id, None)
                self.hidden.pop(agent_id, None)
                self.last_recurrent_proposal.pop(agent_id, None)
                self.execution_gap_pending.discard(agent_id)
                self.reward_shaper.reset(agent_id)
                self.elite_replay.clear_agent(agent_id)
            return {"schema_version": 1, "type": "control", "sequence": message.get("sequence", 0),
                    "command": "hello_ack", "payload": {"device": str(self.device),
                    "policy_version": self.state.policy_version,
                    "freeze_policy": self.freeze_policy,
                    "parameters": self.policy.parameter_count}}
        if message.get("type") != "step_batch":
            return None
        return await self._step_batch(message)

    async def _step_batch(self, message: dict[str, Any]) -> dict[str, Any]:
        if self.training_task is not None and self.training_task.done():
            await self._publish_background_generation()
        reward_cutover_quarantine = self._refresh_adaptive_reward_quarantine()
        steps = list(message.get("steps") or [])
        if not steps:
            return self._action_response(message, [])
        if reward_cutover_quarantine:
            self.adaptive_reward_quarantined_steps += sum(
                str(step.get("observation", {}).get("match", {}).get("episode_id", ""))
                != "waiting"
                for step in steps
            )
        # A frozen evaluation must measure the selected current checkpoint on
        # every bot. League assignment can otherwise silently replace one side
        # with a historical or scripted opponent while the worker attributes
        # both controls to policy execution.
        if not self.freeze_policy:
            self.league.assign_batch(steps)
        league_requests: list[tuple[int, Any, str, dict[str, Any]]] = []
        league_controlled: set[int] = set()
        if not self.freeze_policy:
            for index, step in enumerate(steps):
                assignment = self.league.assignment_for(str(step["observation"]["match"]["episode_id"]))
                agent_id = str(step["agent_id"])
                if assignment is not None and assignment.opponent_agent == agent_id and assignment.mode != "mirror":
                    league_controlled.add(index)
                    episode_id = str(step["observation"]["match"]["episode_id"])
                    done = bool(step.get("terminated") or step.get("truncated"))
                    if episode_id != "waiting" and not done:
                        league_requests.append((index, assignment, agent_id, step["observation"]))

        # League-controlled opponents never create a main-policy proposal or
        # PPO transition. Terminal/waiting rows need a zero bootstrap. Avoiding
        # their feature encoding, recurrent forward, and action sampling cuts
        # the common four-lane batch from eight main-policy rows to four.
        policy_indices = [
            index for index, step in enumerate(steps)
            if index not in league_controlled
            and str(step["observation"]["match"]["episode_id"]) != "waiting"
            and not bool(step.get("terminated") or step.get("truncated"))
        ]
        policy_offsets = {index: offset for offset, index in enumerate(policy_indices)}
        policy_observations = [steps[index]["observation"] for index in policy_indices]
        encoded = [encode_observation(observation) for observation in policy_observations]
        # Pending PPO records need the NumPy encoding, so stack that exact
        # representation for inference instead of encoding every row twice.
        features = batch_encoded_observations(encoded, self.device) if policy_indices else None
        hidden = torch.cat([
            self.hidden.get(
                str(steps[index]["agent_id"]),
                self.policy.initial_hidden(1, self.device),
            )
            for index in policy_indices
        ], dim=1) if policy_indices else None

        bootstrap_output = None
        if policy_indices:
            self.policy.eval()
            with torch.inference_mode():
                bootstrap_output = self.policy(features, hidden)
        if not self.freeze_policy:
            for index, step in enumerate(steps):
                offset = policy_offsets.get(index)
                bootstrap_value = (
                    float(bootstrap_output.value[offset])
                    if offset is not None and bootstrap_output is not None else 0.0
                )
                self._finish_pending(step, bootstrap_value)

        train_after_terminal_grace = False
        if not self.freeze_policy and self.buffer.ready:
            if self.buffer_ready_grace_remaining is None:
                # A policy action's damage/death feedback can trail its
                # execution report by a couple of localhost batches. Do not
                # drain the rollout on the exact batch where it becomes ready;
                # keep the terminal candidate patchable across two more
                # feedback opportunities.
                self.buffer_ready_grace_remaining = 2
            else:
                self.buffer_ready_grace_remaining -= 1
                train_after_terminal_grace = self.buffer_ready_grace_remaining <= 0
        if (
            train_after_terminal_grace
            and self.training_task is None
            and time.monotonic() >= self.training_retry_not_before
        ):
            if "_train_update" in self.__dict__:  # test/debug instrumentation hook
                self._train_update()
            else:
                self._start_background_update()

        wire_actions = [_noop_action() for _ in steps]
        action_tensor = None
        log_probability = None
        if policy_indices and bootstrap_output is not None and features is not None:
            with torch.inference_mode():
                sampled_actions, action_tensor, log_probability, _ = sample_actions(
                    # features was constructed directly on self.device, which
                    # is also the trainer/actor device. Calling FeatureBatch.to
                    # here dispatched 15 redundant tensor transfers per batch.
                    bootstrap_output, features,
                    self.service_config.deterministic_inference, compute_entropy=False,
                )
            for offset, index in enumerate(policy_indices):
                wire_actions[index] = sampled_actions[offset]

        current_version = self.state.policy_version
        action_ids: list[int | None] = [None] * len(steps)
        league_actions = self.league.opponent_actions(league_requests) if league_requests else {}
        for index, step in enumerate(steps):
            agent_id = str(step["agent_id"])
            episode_id = str(step["observation"]["match"]["episode_id"])
            if episode_id == "waiting":
                wire_actions[index] = _noop_action()
                self.hidden.pop(agent_id, None)
                self.pending.pop(agent_id, None)
                self.last_recurrent_proposal.pop(agent_id, None)
                self.execution_gap_pending.discard(agent_id)
                self.reward_shaper.reset(agent_id)
                self.elite_replay.clear_agent(agent_id)
                continue
            done = bool(step.get("terminated") or step.get("truncated"))
            if done:
                wire_actions[index] = _noop_action()
                self.hidden.pop(agent_id, None)
                self.pending.pop(agent_id, None)
                self.last_recurrent_proposal.pop(agent_id, None)
                self.execution_gap_pending.discard(agent_id)
                continue
            action_id = self.next_action_id
            self.next_action_id += 1
            action_ids[index] = action_id
            if not self.freeze_policy:
                if index in league_actions:
                    wire_actions[index] = league_actions[index]
                    self.hidden.pop(agent_id, None)
                    self.pending.pop(agent_id, None)
                    self.last_recurrent_proposal.pop(agent_id, None)
                    self.execution_gap_pending.discard(agent_id)
                    continue
            offset = policy_offsets[index]
            assert bootstrap_output is not None and hidden is not None
            assert action_tensor is not None and log_probability is not None
            self.hidden[agent_id] = bootstrap_output.hidden[:, offset:offset + 1].detach()
            previous_proposal = self.last_recurrent_proposal.get(agent_id)
            if previous_proposal is not None and previous_proposal[0] != episode_id:
                self.execution_gap_pending.discard(agent_id)
            recurrent_parent_action_id = (
                previous_proposal[1]
                if previous_proposal is not None and previous_proposal[0] == episode_id
                else None
            )
            self.last_recurrent_proposal[agent_id] = (episode_id, action_id)
            if self.freeze_policy or reward_cutover_quarantine:
                self.pending.pop(agent_id, None)
                continue
            pending = PendingStep(
                action_id=action_id,
                recurrent_parent_action_id=recurrent_parent_action_id,
                agent_id=agent_id,
                episode_id=episode_id,
                policy_version=current_version,
                features=encoded[offset],
                hidden=hidden[:, offset].detach().cpu().numpy()[0],
                categorical_action={name: int(value[offset]) for name, value in action_tensor.categorical.items()},
                camera_action=action_tensor.camera[offset].detach().cpu().numpy().astype(np.float32),
                log_probability=float(log_probability[offset]),
                value=float(bootstrap_output.value[offset]),
                reward_snapshot=tactical_snapshot(step["observation"]),
                observation=step["observation"],
                wire_action=wire_actions[index],
            )
            agent_pending = self.pending.setdefault(agent_id, {})
            agent_pending[action_id] = pending
            while len(agent_pending) > self.MAX_PENDING_ACTIONS_PER_AGENT:
                agent_pending.pop(next(iter(agent_pending)))
        actions = [
            {"agent_id": step["agent_id"], "action_id": action_ids[index],
             "action": wire_actions[index]}
            for index, step in enumerate(steps)
            if action_ids[index] is not None
        ]
        return self._action_response(message, actions)

    def _finish_pending(self, step: dict[str, Any], bootstrap_value: float) -> None:
        agent_id = str(step["agent_id"])
        terminal = bool(step.get("terminated") or step.get("truncated"))
        episode_id = str(step["observation"]["match"]["episode_id"])
        if episode_id == "waiting":
            # Waiting is a worker lifecycle sentinel, never an RL episode. A
            # stale worker must not turn it into safety rollouts, imitation, or
            # recurrent state that carries into the next real match.
            self.pending.pop(agent_id, None)
            self.hidden.pop(agent_id, None)
            self.last_recurrent_proposal.pop(agent_id, None)
            self.execution_gap_pending.discard(agent_id)
            self.reward_shaper.reset(agent_id)
            self.elite_replay.clear_agent(agent_id)
            return
        info = step.get("info") if isinstance(step.get("info"), dict) else {}
        reported_metrics = info.get("worker_metrics")
        if isinstance(reported_metrics, dict):
            self.worker_metrics = {
                str(name): value for name, value in reported_metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value))
            }
        source_hint, referenced_action_id = _execution_reference(step)
        agent_pending = self.pending.get(agent_id)
        pending: PendingStep | None = None
        if agent_pending:
            if referenced_action_id is not None:
                # An execution ID names the exact proposal which reached the
                # game. Pop it even when the accompanying source/action is
                # malformed so a poisoned report cannot leak pending entries.
                pending = agent_pending.pop(referenced_action_id, None)
            elif source_hint == "policy":
                # Backward compatibility for workers predating action IDs:
                # retain the former immediate-action behavior by consuming the
                # newest proposal.
                newest_action_id = next(reversed(agent_pending))
                pending = agent_pending.pop(newest_action_id)
            elif source_hint in self.TEACHER_SOURCES or source_hint == "safety":
                # A teacher/safety control without an ID did not prove that it
                # consumed a queued policy proposal. Use the newest observation
                # only as shaping/imitation context and leave the proposal live.
                pending = agent_pending[next(reversed(agent_pending))]

        known_episode_change = False
        if pending is not None and pending.episode_id != episode_id:
            known_episode_change = True
        if agent_pending and any(item.episode_id != episode_id for item in agent_pending.values()):
            known_episode_change = True
        if terminal or known_episode_change:
            # No proposal may cross an episode boundary. The worker also clears
            # its delayed queue on terminal/reset, so retaining these IDs could
            # only create false future matches.
            self.pending.pop(agent_id, None)
        elif agent_pending is not None and not agent_pending:
            self.pending.pop(agent_id, None)

        if pending is None:
            # An explicitly reported control which cannot become an on-policy
            # transition is still part of environment chronology. The next
            # accepted policy action must begin a fresh GAE/replay run.
            if "execution" in step:
                self.execution_gap_pending.add(agent_id)
            if terminal:
                # Even with no newly queued proposal, terminate the last
                # on-policy sample. Only a safety-delayed outcome whose last
                # real owner was policy receives reward; teacher/invalid
                # terminals merely close the value bootstrap.
                self._close_delayed_terminal(
                    agent_id, episode_id, step, source_hint,
                )
            if terminal or known_episode_change:
                self.reward_shaper.reset(agent_id)
                self.elite_replay.clear_agent(agent_id)
            return

        same_episode = episode_id == pending.episode_id
        done = terminal or not same_episode
        info_episode = info.get("episode_id")
        shaping_info = info if info_episode is None or str(info_episode) == pending.episode_id else {}
        source, executed_action, _ = _execution(step, pending.wire_action)
        policy_execution_valid = (
            source == "policy"
            and executed_action is not None
            and (
                "execution" not in step
                or _actions_match(executed_action, pending.wire_action, pending.observation)
            )
        )
        trainer_shaping = self.reward_shaper.shape(
            agent_id,
            pending.episode_id,
            pending.reward_snapshot,
            step["observation"],
            shaping_info,
            same_episode=same_episode,
            done=done,
        )
        if policy_execution_valid:
            # Store the hierarchical policy proposal, not the resolved V1
            # Minecraft controls.  The former retains intent/target ownership
            # and is admitted only if a verified autonomous outcome follows.
            self.elite_replay.record_policy_action(
                agent_id=agent_id,
                episode_id=pending.episode_id,
                action_id=pending.action_id,
                policy_version=pending.policy_version,
                observation=pending.observation,
                action=pending.wire_action,
            )
        crystal_deltas = self._record_policy_crystal_counters(
            agent_id, pending.episode_id, shaping_info
        )
        if crystal_deltas.get("policy_crystal_chains_damaging", 0) > 0:
            self.elite_replay.promote(
                agent_id,
                pending.episode_id,
                "damaging_crystal",
                event_token=f"action-{pending.action_id}",
                quality=1.0,
            )
        kill_quality = _verified_policy_kill_quality(
            step, self.config.terminal_reward_clip,
        )
        delayed_policy_delivery = (
            not policy_execution_valid
            and source == "safety"
            and _terminal_training_eligible(step, delayed=True)
        )
        if kill_quality is not None and (policy_execution_valid or delayed_policy_delivery):
            self.elite_replay.promote(
                agent_id,
                pending.episode_id,
                "kill",
                event_token="terminal",
                quality=kill_quality,
            )
        if done:
            self.elite_replay.clear_episode(agent_id, pending.episode_id)
        if done:
            self.server_crystal_counter_state = {
                key: value for key, value in self.server_crystal_counter_state.items()
                if key[:2] != (agent_id, pending.episode_id)
            }
            self.server_rollout_counter_state = {
                key: value for key, value in self.server_rollout_counter_state.items()
                if key[:2] != (agent_id, pending.episode_id)
            }
        if done:
            self.league.record_result(
                pending.episode_id,
                agent_id,
                float(step.get("reward", 0.0)),
                info.get("outcome"),
                info,
            )

        if done and not policy_execution_valid:
            self._close_delayed_terminal(
                agent_id, pending.episode_id, step, source,
            )

        # Teacher overrides are supervised examples, never on-policy PPO
        # samples. Safety/invalid overrides are neither: learning from them
        # would assign the policy a control it did not choose.
        if source in self.TEACHER_SOURCES:
            self.execution_gap_pending.add(agent_id)
            teacher_observation = _pre_execution_observation(step, pending.observation)
            teacher_phase = (
                classify_crystal_teacher_action(executed_action)
                if source == "teacher_crystal" else None
            )
            # Crystal waits are useful completion telemetry but harmful
            # imitation targets: they break the place->detonate chain.
            useful = source != "teacher_crystal" or teacher_phase is not None
            if (
                useful and executed_action is not None
                and _action_is_legal(executed_action, teacher_observation)
            ):
                match = teacher_observation.get("match")
                match = match if isinstance(match, dict) else {}
                try:
                    teacher_tick = int(match.get("tick", 0))
                except (TypeError, ValueError, OverflowError):
                    teacher_tick = 0
                self._append_online_imitation({
                    "match_id": pending.episode_id,
                    "agent_id": agent_id,
                    "tick": teacher_tick,
                    "observation": teacher_observation,
                    "action": executed_action,
                    "execution_source": source,
                    "teacher_phase": teacher_phase,
                })
            return
        if not policy_execution_valid:
            self.execution_gap_pending.add(agent_id)
            return
        # Explicit policy attribution must agree with the exact action the
        # trainer supplied. Any mismatch is an unlabelled override and is
        # conservatively excluded from PPO.
        terminal_eligible = _terminal_training_eligible(step, delayed=False)
        if done and _has_explicit_terminal_attribution(step) and not terminal_eligible:
            self.reward_telemetry.rejected_nonpolicy_terminals += 1
        reward = sanitize_reward(
            step,
            self.config.shaping_reward_clip,
            self.config.terminal_reward_clip,
            trainer_shaping.total,
            include_terminal=terminal_eligible,
        )
        self.reward_telemetry.record(reward, trainer_shaping)
        self._record_terminal_delivery(step, reward)
        self.buffer.append(Transition(
            agent_id=agent_id, episode_id=pending.episode_id, policy_version=pending.policy_version,
            action_id=pending.action_id,
            recurrent_parent_action_id=pending.recurrent_parent_action_id,
            execution_gap_before=agent_id in self.execution_gap_pending,
            features=pending.features, hidden=pending.hidden,
            categorical_action=pending.categorical_action, camera_action=pending.camera_action,
            old_log_probability=pending.log_probability, old_value=pending.value,
            reward=reward.training_reward, done=done,
            next_value=0.0 if done else bootstrap_value,
        ))
        self.execution_gap_pending.discard(agent_id)
        if len(self.buffer) >= self.next_progress_tick and not self.buffer.ready:
            print(json.dumps({"event": "rollout_progress", "policy_version": self.state.policy_version,
                              "collected_agent_ticks": len(self.buffer),
                              "target_agent_ticks": self.config.rollout_agent_ticks,
                              "total_agent_ticks": self.state.total_agent_ticks,
                              "worker_metrics": self.worker_metrics,
                              **self.reward_telemetry.metrics()}), flush=True)
            self.next_progress_tick += self.progress_interval

    def _close_delayed_terminal(
        self, agent_id: str, episode_id: str, step: dict[str, Any], source: str,
    ) -> None:
        """Attach an asynchronous outcome to the last executed policy action.

        The terminal safety no-op is never itself learned. It can only deliver
        an outcome backward to an existing on-policy transition when the server
        explicitly attributes a policy-owned kill, or an avoidable negative
        self/environment death. A teacher/invalid terminal still marks that
        transition done, preventing a value bootstrap across the episode
        boundary, but receives no reward.
        """
        transition = next((
            item for item in reversed(self.buffer.transitions)
            if item.agent_id == agent_id
            and item.episode_id == episode_id
            and item.policy_version == self.state.policy_version
        ), None)
        if transition is None:
            return
        transition.done = True
        transition.next_value = 0.0
        if source != "safety" or not _terminal_training_eligible(step, delayed=True):
            if _has_explicit_terminal_attribution(step):
                self.reward_telemetry.rejected_nonpolicy_terminals += 1
            return
        kill_quality = _verified_policy_kill_quality(
            step, self.config.terminal_reward_clip,
        )
        if kill_quality is not None:
            self.elite_replay.promote(
                agent_id,
                episode_id,
                "kill",
                event_token="terminal",
                quality=kill_quality,
            )
        # Server damage shaping is already execution-attributed and therefore
        # safe to carry with this delayed terminal. Do not add trainer shaping
        # again: it was evaluated on the safety observation and is not owned by
        # the policy action being credited.
        reward = sanitize_reward(
            step,
            self.config.shaping_reward_clip,
            self.config.terminal_reward_clip,
            trainer_shaping=0.0,
            include_terminal=True,
        )
        transition.reward += reward.training_reward
        self.reward_telemetry.record(reward)
        self._record_terminal_delivery(step, reward)

    def _record_terminal_delivery(
        self, step: dict[str, Any], reward: SanitizedReward,
    ) -> None:
        if abs(reward.terminal_reward) <= 1e-12:
            return
        self.reward_telemetry.terminal_attached_events += 1
        info = step.get("info") if isinstance(step.get("info"), dict) else {}
        if (
            reward.terminal_reward > 0.0
            and info.get("terminal_source") == "policy"
            and info.get("policy_owned_kill") is True
        ):
            self.reward_telemetry.policy_owned_kill_events += 1

    def _append_online_imitation(self, record: dict[str, Any]) -> None:
        self.online_imitation_records.append(record)
        source = str(record.get("execution_source", ""))
        source_capacity = {
            "teacher_crystal": self.ONLINE_IMITATION_CAPACITY // 2,
            "teacher_sword": self.ONLINE_IMITATION_CAPACITY // 4,
            "teacher_block": self.ONLINE_IMITATION_CAPACITY // 4,
        }.get(source, self.ONLINE_IMITATION_CAPACITY)
        source_indices = [
            index for index, value in enumerate(self.online_imitation_records)
            if str(value.get("execution_source", "")) == source
        ]
        if len(source_indices) > source_capacity:
            del self.online_imitation_records[source_indices[0]]
        overflow = len(self.online_imitation_records) - self.ONLINE_IMITATION_CAPACITY
        if overflow > 0:
            # Preserve scarce crystal chains when another teacher dominates.
            for _ in range(overflow):
                remove = next((
                    index for index, value in enumerate(self.online_imitation_records)
                    if str(value.get("execution_source", "")) != "teacher_crystal"
                ), 0)
                del self.online_imitation_records[remove]

    def _record_policy_crystal_counters(
        self, agent_id: str, episode_id: str, info: dict[str, Any],
    ) -> dict[str, int]:
        stats = info.get("stats") if isinstance(info, dict) else None
        if not isinstance(stats, dict):
            return {}
        crystal_deltas: dict[str, int] = {}
        names = (
            "policy_crystal_chains_started",
            "policy_crystal_chains_detonated",
            "policy_crystal_chains_damaging",
            "policy_crystal_chains_popping",
            "rewarded_crystal_combos",
        )
        for name in names:
            try:
                current = max(0, int(float(stats.get(name, 0))))
            except (TypeError, ValueError, OverflowError):
                continue
            key = (agent_id, episode_id, name)
            baseline_known = key in self.server_crystal_counter_state
            previous = self.server_crystal_counter_state.get(key, 0)
            delta = max(0, current - previous)
            if delta:
                # Telemetry may adopt a non-zero cumulative counter after a
                # reconnect, but elite replay must see both sides of the
                # increment or it could label unrelated actions as the chain.
                if baseline_known:
                    crystal_deltas[name] = delta
                self.reward_telemetry.policy_crystal_counter_deltas[name] = (
                    self.reward_telemetry.policy_crystal_counter_deltas.get(name, 0) + delta
                )
            self.server_crystal_counter_state[key] = current

        execution = stats.get("execution")
        policy = execution.get("policy") if isinstance(execution, dict) else None
        policy = policy if isinstance(policy, dict) else {}
        counters = (
            ("policy_damage_dealt", policy, "damage_dealt"),
            ("policy_hits_landed", policy, "hits_landed"),
            ("policy_blocks_placed", policy, "blocks_placed"),
            ("policy_blocks_mined", policy, "blocks_mined"),
            ("damage_taken", stats, "damage_taken"),
            ("self_damage", stats, "self_damage"),
            ("inaction_penalty_ticks", stats, "inaction_penalty_ticks"),
        )
        for output_name, source, source_name in counters:
            if source_name not in source:
                continue
            try:
                current = max(0.0, float(source[source_name]))
            except (TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(current):
                continue
            key = (agent_id, episode_id, output_name)
            previous = self.server_rollout_counter_state.get(key, 0.0)
            delta = max(0.0, current - previous)
            if delta > 0.0:
                self.reward_telemetry.rollout_counter_deltas[output_name] = (
                    self.reward_telemetry.rollout_counter_deltas.get(output_name, 0.0) + delta
                )
            self.server_rollout_counter_state[key] = current
        return crystal_deltas

    def _train_update(self) -> None:
        if self.freeze_policy:
            return
        transitions = self.buffer.drain(self.state.policy_version)
        self.buffer_ready_grace_remaining = None
        if not transitions:
            return
        self.next_progress_tick = self.progress_interval
        # Freeze the control-health sample with the rollout it describes.
        # Reading self.worker_metrics after synchronous optimization would
        # judge the rollout using optimizer-era CPU contention instead.
        reward_metrics = {
            **self.reward_telemetry.metrics(),
            **self.worker_metrics,
            **self.elite_replay.metrics(),
        }
        print(json.dumps({"event": "ppo_training_started", "policy_version": self.state.policy_version,
                          "batch_agent_ticks": len(transitions),
                          "total_agent_ticks": self.state.total_agent_ticks,
                          **reward_metrics}), flush=True)
        self.policy.train()
        batch = prepare_sequences(
            transitions, self.config.recurrent_sequence_length, self.config.gamma,
            self.config.gae_lambda, self.trainer.device,
        )
        metrics = self.trainer.update(batch)
        elite_records = self.elite_replay.records
        imitation_weight = self.config.imitation_start_weight * max(
            0.0, 1.0 - self.state.total_agent_ticks / max(1, self.config.imitation_decay_ticks)
        )
        if self.online_imitation_records:
            imitation_weight = max(imitation_weight, self.ONLINE_IMITATION_MIN_WEIGHT)
        if elite_records:
            imitation_weight = max(imitation_weight, self.config.elite_imitation_weight)
        imitation_records = [
            *self.imitation_records, *self.online_imitation_records, *elite_records,
        ]
        metrics.imitation_loss = self.trainer.auxiliary_imitation_update(
            imitation_records, imitation_weight
        )
        imitation_metrics = self.trainer.last_imitation_metrics
        metrics.imitation_updates = int(imitation_metrics.get("updates", 0))
        metrics.imitation_samples = int(imitation_metrics.get("samples", 0))
        metrics.imitation_crystal_samples = int(imitation_metrics.get("crystal_samples", 0))
        metrics.imitation_crystal_fraction = float(imitation_metrics.get("crystal_fraction", 0.0))
        metrics.imitation_elite_samples = int(imitation_metrics.get("elite_samples", 0))
        metrics.imitation_elite_fraction = float(imitation_metrics.get("elite_fraction", 0.0))
        metrics.imitation_elite_events_sampled = int(
            imitation_metrics.get("elite_events_sampled", 0)
        )
        metrics.imitation_crystal_buffer = sum(
            str(record.get("execution_source", "")) == "teacher_crystal"
            for record in self.online_imitation_records
        )
        metrics.imitation_sword_buffer = sum(
            str(record.get("execution_source", "")) == "teacher_sword"
            for record in self.online_imitation_records
        )
        metrics.imitation_block_buffer = sum(
            str(record.get("execution_source", "")) == "teacher_block"
            for record in self.online_imitation_records
        )
        metrics.imitation_elite_buffer = len(elite_records)
        metrics.imitation_elite_events = len({
            str(record.get("elite_event_id", "")) for record in elite_records
            if record.get("elite_event_id")
        })
        metrics.imitation_elite_buckets = len({
            str(record.get("elite_bucket", "")) for record in elite_records
            if record.get("elite_bucket")
        })
        self.device = self.trainer.device
        self.league.set_device(self.device)
        self.state.policy_version += 1
        self.state.total_agent_ticks += len(transitions)
        # A policy-version response makes the worker discard its delayed action
        # queues. Mirror that boundary here so stale IDs from the previous
        # policy can never be accepted after the update.
        self.pending.clear()
        self.last_recurrent_proposal.clear()
        self.execution_gap_pending.clear()
        self.elite_replay.clear_traces()
        metrics_dict = vars(metrics)
        self.state.rollout_generation += 1
        self.checkpoints.save(
            self.policy, self.trainer.optimizer, self.state, self.config, metrics_dict,
            self.online_imitation_records, elite_records,
        )
        adaptive_reward = self._adapt_completed_rollout(metrics_dict, reward_metrics)
        self.league.prune_pool()
        print(json.dumps({"event": "ppo_update", "policy_version": self.state.policy_version,
                          "total_agent_ticks": self.state.total_agent_ticks,
                          "worker_metrics": self.worker_metrics,
                          "adaptive_reward": adaptive_reward,
                          **metrics_dict, **reward_metrics}), flush=True)
        self.reward_telemetry = RewardTelemetry()

    def _start_background_update(self) -> None:
        """Train in a separate process while the frozen actor keeps answering."""
        transitions = self.buffer.drain(self.state.policy_version)
        self.buffer_ready_grace_remaining = None
        if not transitions:
            return
        self.training_ticks = len(transitions)
        self.training_transitions = transitions
        self.next_progress_tick = self.progress_interval
        # state_dict() containers are cheap views. The actor and optimizer are
        # immutable until publication, so the process-pool feeder can serialize
        # them without a multi-second deepcopy on the websocket event loop.
        actor_state = self.policy.state_dict()
        optimizer_state = self.trainer.optimizer.state_dict()
        elite_records = self.elite_replay.records
        imitation_records = [
            *self.imitation_records, *self.online_imitation_records, *elite_records,
        ]
        imitation_weight = self.config.imitation_start_weight * max(
            0.0, 1.0 - self.state.total_agent_ticks / max(1, self.config.imitation_decay_ticks)
        )
        if self.online_imitation_records:
            imitation_weight = max(imitation_weight, self.ONLINE_IMITATION_MIN_WEIGHT)
        if elite_records:
            imitation_weight = max(imitation_weight, self.config.elite_imitation_weight)
        # The learner returns this exact snapshot at publication. Do not use
        # whatever latency happens to be current after the CPU update finishes.
        reward_metrics = {
            **self.reward_telemetry.metrics(),
            **self.worker_metrics,
            **self.elite_replay.metrics(),
        }
        self.reward_telemetry = RewardTelemetry()
        print(json.dumps({
            "event": "ppo_training_started", "policy_version": self.state.policy_version,
            "rollout_generation": self.state.rollout_generation,
            "batch_agent_ticks": len(transitions), **reward_metrics,
        }), flush=True)

        proposed_state = CheckpointState(
            policy_version=self.state.policy_version + 1,
            total_agent_ticks=self.state.total_agent_ticks + len(transitions),
            next_snapshot_tick=self.state.next_snapshot_tick,
            rollout_generation=self.state.rollout_generation + 1,
        )

        if self.learner_executor is None:
            # A thread still shares the GIL and Torch CPU pool with inference.
            # One persistent spawned learner process gives the actor its own
            # scheduler/GIL. The learner's small, separately configurable Torch
            # pool bounds its CPU use while the websocket actor remains live.
            self.learner_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=1, mp_context=multiprocessing.get_context("spawn")
            )
        loop = asyncio.get_running_loop()
        learner_future = loop.run_in_executor(
            self.learner_executor,
            _learn_generation_job,
            actor_state,
            optimizer_state,
            transitions,
            self.config,
            imitation_records,
            list(self.online_imitation_records),
            elite_records,
            imitation_weight,
            reward_metrics,
            self.checkpoints.directory,
            self.checkpoints.snapshot_interval,
            proposed_state,
            self.trainer.low_kl_updates,
        )
        # The process result contains only small metadata and file paths. Load
        # the staged tensors on a background actor thread before exposing a
        # done task to _step_batch; publication itself therefore remains a
        # short atomic generation swap and never parks the worker websocket.
        self.training_task = asyncio.create_task(
            self._prepare_background_generation(learner_future)
        )

    async def _prepare_background_generation(
        self, learner_future: asyncio.Future,
    ) -> tuple[dict, dict, dict, dict, CheckpointState, int, str, str | None, str | None]:
        (
            metrics, reward_metrics, saved_state, low_kl_updates,
            latest_stage, snapshot_stage, snapshot,
        ) = await learner_future
        policy_state, optimizer_state = await asyncio.to_thread(
            _load_staged_training_state,
            Path(latest_stage), self.device, saved_state,
        )
        return (
            policy_state, optimizer_state, metrics, reward_metrics,
            saved_state, low_kl_updates, latest_stage, snapshot_stage, snapshot,
        )

    async def _publish_background_generation(self) -> None:
        """Atomically publish a completed learner only between generations."""
        task = self.training_task
        if task is None:
            return
        self.training_task = None
        try:
            (
                policy_state, optimizer_state, metrics, reward_metrics,
                saved_state, low_kl_updates, latest_stage, snapshot_stage, snapshot,
            ) = await task
        except Exception as error:
            # Keep serving the old frozen actor and make the drained generation
            # eligible for a retry. No version/checkpoint boundary was published.
            restored = [
                *self.training_transitions, *self.buffer.transitions,
            ]
            # A deterministic numerical failure must not spin a new child on
            # every incoming batch or retain an unbounded failed generation.
            # Keep the freshest configured rollout and wait before retrying.
            dropped = max(0, len(restored) - self.buffer.capacity)
            self.buffer.transitions = restored[-self.buffer.capacity:]
            self.training_retry_not_before = time.monotonic() + 60.0
            self.training_transitions = []
            self.training_ticks = 0
            try:
                self.adaptive_rewards.note_update_failure(
                    str(error),
                    policy_version=self.state.policy_version,
                    rollout_generation=self.state.rollout_generation,
                )
            except OSError:
                pass
            print(json.dumps({
                "event": "ppo_update_failed",
                "policy_version": self.state.policy_version,
                "message": str(error),
                "retry_after_seconds": 60,
                "quarantined_oldest_transitions": dropped,
            }), flush=True)
            return
        async with self.update_lock:
            self.policy.load_state_dict(policy_state)
            self.trainer.optimizer.load_state_dict(optimizer_state)
            self.trainer.low_kl_updates = int(low_kl_updates)
            self.state = saved_state
            # Every sample accumulated during learning belongs to the prior
            # generation. Drop it at the atomic boundary instead of mixing PPO
            # importance ratios across policy versions.
            self.buffer.transitions.clear()
            self.pending.clear()
            self.hidden.clear()
            self.last_recurrent_proposal.clear()
            self.execution_gap_pending.clear()
            self.elite_replay.clear_traces()
            self.training_transitions = []
            self.training_ticks = 0
            self.training_retry_not_before = 0.0
            # The actor now serves this exact generation. Exposing the staged
            # checkpoint after publication prevents recovery from ever getting
            # ahead of live inference.
            self.checkpoints.promote_staged(
                Path(latest_stage),
                Path(snapshot_stage) if snapshot_stage is not None else None,
                Path(snapshot) if snapshot is not None else None,
                self.state,
                metrics,
                # The full replay sidecar is tens of megabytes. Serializing it
                # here used to block the websocket event loop for several
                # seconds after every PPO generation. ``latest.pt`` already
                # contains a complete crash-safe replay fallback, so publish
                # the actor/checkpoint atomically first and persist successes
                # collected during learning in the idle learner process.
                None,
            )
            post_learner_elite_records = self.elite_replay.records
        self._schedule_elite_replay_sidecar(
            post_learner_elite_records, self.state.policy_version,
        )
        adaptive_reward = self._adapt_completed_rollout(metrics, reward_metrics)
        print(json.dumps({
            "event": "ppo_update", "policy_version": self.state.policy_version,
            "rollout_generation": self.state.rollout_generation,
            "total_agent_ticks": self.state.total_agent_ticks,
            "worker_metrics": self.worker_metrics,
            "adaptive_reward": adaptive_reward,
            **metrics, **reward_metrics,
        }), flush=True)

    def _schedule_elite_replay_sidecar(
        self, records: list[dict[str, Any]], policy_version: int,
    ) -> None:
        """Queue replay persistence without delaying live inference.

        Jobs share the single learner executor, which makes sidecar publication
        ordered by policy version and prevents two writers from racing on the
        atomic temporary path. A crash before completion is harmless because
        the just-promoted recovery checkpoint embeds the rollout-start replay.
        """
        if self.learner_executor is None:
            return
        loop = asyncio.get_running_loop()
        task = loop.run_in_executor(
            self.learner_executor,
            _save_elite_replay_sidecar_job,
            self.checkpoints.directory,
            records,
            int(policy_version),
        )
        self.sidecar_tasks.add(task)
        task.add_done_callback(self._elite_replay_sidecar_finished)

    def _elite_replay_sidecar_finished(self, task: asyncio.Future) -> None:
        self.sidecar_tasks.discard(task)
        try:
            saved_version, saved_records = task.result()
        except Exception as error:
            # Recovery still has the replay embedded in latest.pt. Keep the
            # actor serving and retry naturally after the next generation.
            print(json.dumps({
                "event": "elite_replay_sidecar_failed",
                "policy_version": self.state.policy_version,
                "message": str(error),
            }), flush=True)
            return
        print(json.dumps({
            "event": "elite_replay_sidecar_saved",
            "policy_version": int(saved_version),
            "records": saved_records,
        }), flush=True)

    def _adapt_completed_rollout(
        self, optimizer_metrics: dict[str, Any], reward_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Observe only a successfully published rollout generation."""
        if self.adaptive_reward_sync_pending:
            # The controller's current profile has not reached the arena yet.
            # Training may safely continue under the old objective, but those
            # rollouts cannot evaluate or roll back an unapplied change.
            return {
                **self.adaptive_rewards.telemetry(),
                "changed": False,
                "skipped": True,
                "health_reasons": ("awaiting arena reward-profile acknowledgement",),
            }
        try:
            decision = self.adaptive_rewards.observe(
                {**reward_metrics, **optimizer_metrics},
                policy_version=self.state.policy_version,
                rollout_generation=self.state.rollout_generation,
            )
        except Exception as error:
            # Reward tuning is an auxiliary control plane. A sidecar I/O error
            # must not invalidate an actor/checkpoint generation which was
            # already published successfully.
            print(json.dumps({
                "event": "adaptive_reward_update_failed",
                "policy_version": self.state.policy_version,
                "message": str(error),
            }), flush=True)
            return {"enabled": self.adaptive_rewards.enabled, "error": str(error)}
        if decision.changed:
            self.adaptive_reward_sync_pending = True
        result = {
            "enabled": self.adaptive_rewards.enabled,
            "changed": decision.changed,
            "skipped": decision.skipped,
            "generation": decision.generation,
            "multipliers": decision.multipliers,
            "changes": decision.changes,
            "signals": decision.signals,
            "rollback": decision.rollback,
            "health_reasons": decision.health_reasons,
        }
        print(json.dumps({
            "event": "adaptive_reward_update",
            "policy_version": self.state.policy_version,
            "rollout_generation": self.state.rollout_generation,
            **result,
        }), flush=True)
        return result

    def _begin_adaptive_reward_quarantine(
        self, generation: int, duration_seconds: float = 40.0,
    ) -> None:
        """Drop mixed-objective samples while old arena episodes age out."""
        self.adaptive_reward_quarantine_until = max(
            self.adaptive_reward_quarantine_until,
            time.monotonic() + max(40.0, float(duration_seconds)),
        )
        self.adaptive_reward_quarantine_generation = int(generation)
        self.adaptive_reward_quarantined_steps = 0
        self.buffer.transitions.clear()
        self.buffer_ready_grace_remaining = None
        self.next_progress_tick = self.progress_interval
        self.pending.clear()
        self.hidden.clear()
        self.last_recurrent_proposal.clear()
        self.execution_gap_pending.clear()
        self.reward_shaper = TrainerRewardShaper()
        self.elite_replay.clear_traces()
        self.reward_telemetry = RewardTelemetry()
        self.server_crystal_counter_state.clear()
        self.server_rollout_counter_state.clear()
        print(json.dumps({
            "event": "adaptive_reward_cutover_started",
            "generation": self.adaptive_reward_quarantine_generation,
            "quarantine_seconds": 40,
            "inference_continues": True,
        }), flush=True)

    def _refresh_adaptive_reward_quarantine(self) -> bool:
        until = self.adaptive_reward_quarantine_until
        if until <= 0.0:
            return False
        remaining = until - time.monotonic()
        if remaining > 0.0:
            return True
        generation = self.adaptive_reward_quarantine_generation
        quarantined_steps = self.adaptive_reward_quarantined_steps
        self.adaptive_reward_quarantine_until = 0.0
        self.pending.clear()
        self.hidden.clear()
        self.last_recurrent_proposal.clear()
        self.execution_gap_pending.clear()
        self.reward_shaper = TrainerRewardShaper()
        self.server_crystal_counter_state.clear()
        self.server_rollout_counter_state.clear()
        print(json.dumps({
            "event": "adaptive_reward_cutover_completed",
            "generation": generation,
            "quarantined_agent_steps": quarantined_steps,
        }), flush=True)
        return False

    def adaptive_reward_cutover_telemetry(self) -> dict[str, Any]:
        return {
            "active": self.adaptive_reward_quarantine_until > time.monotonic(),
            "generation": self.adaptive_reward_quarantine_generation,
            "remaining_seconds": max(
                0.0, self.adaptive_reward_quarantine_until - time.monotonic()
            ),
            "quarantined_agent_steps": self.adaptive_reward_quarantined_steps,
        }

    async def _sync_adaptive_reward_profile(self) -> None:
        if not self.adaptive_rewards.enabled or not self.adaptive_reward_sync_pending:
            return
        if self.training_task is not None:
            # Never change the objective beneath an already-drained learner
            # generation. The retry loop will publish immediately afterward.
            return
        profile = self.adaptive_rewards.profile
        try:
            response = await publish_reward_profile(
                self.service_config.arena_host,
                self.service_config.arena_port,
                profile,
            )
        except Exception as error:
            now = time.monotonic()
            if now - self.adaptive_reward_last_error_log >= 60.0:
                self.adaptive_reward_last_error_log = now
                print(json.dumps({
                    "event": "adaptive_reward_profile_sync_failed",
                    "generation": profile.generation,
                    "message": str(error),
                    "retrying": True,
                }), flush=True)
            return
        # A new decision may have landed while this socket request was in
        # flight. Only clear pending when the acknowledged generation is still
        # the controller's current generation.
        if self.adaptive_rewards.profile.generation == profile.generation:
            self.adaptive_reward_sync_pending = False
        self._begin_adaptive_reward_quarantine(profile.generation)
        self.adaptive_reward_last_error_log = 0.0
        print(json.dumps({
            "event": "adaptive_reward_profile_applied",
            "generation": profile.generation,
            "multipliers": profile.multipliers,
            "arena_response": response,
        }), flush=True)

    async def adaptive_reward_sync_loop(self, stop: asyncio.Future) -> None:
        """Retry profile publication without ever blocking policy inference."""
        while not stop.done():
            await self._sync_adaptive_reward_profile()
            try:
                await asyncio.wait_for(asyncio.shield(stop), timeout=5.0)
            except asyncio.TimeoutError:
                pass

    def _action_response(self, message: dict[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
        return {"schema_version": 1, "type": "action_batch", "sequence": message.get("sequence", 0),
                "policy_version": self.state.policy_version, "actions": actions}


async def serve(
    ppo_config: PPOConfig, service_config: ServiceConfig, imitation_data: Path | None = None,
    initialize_from: list[Path] | None = None, exploiter_target: Path | None = None,
    freeze_policy: bool = False,
) -> None:
    initial = list(initialize_from or [])
    if exploiter_target is not None and not initial:
        initial.append(exploiter_target)
    service = PolicyService(
        ppo_config, service_config, imitation_data, initial, exploiter_target, freeze_policy
    )
    stop = asyncio.Future()
    loop = asyncio.get_running_loop()
    for signal_name in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signal_name, stop.set_result, None)
        except NotImplementedError:
            pass
    reward_sync_task = asyncio.create_task(service.adaptive_reward_sync_loop(stop))
    # PPO updates are intentionally synchronous and can occupy the event loop
    # longer than websockets' default 20-second pong deadline on a CPU-only
    # machine. Keep sending pings, but don't tear down healthy localhost workers
    # solely because an update is in progress.
    try:
        async with websockets.serve(
            service.handle, service_config.host, service_config.port,
            max_size=32 * 1024 * 1024, ping_timeout=None,
            # Step batches are already compact MessagePack and never leave the
            # laptop. Deflate was spending CPU compressing eight large structured
            # observations on every combat decision, directly extending latency.
            compression=None,
        ):
            print(json.dumps(_trainer_ready_payload(service, service_config)), flush=True)
            await stop
    finally:
        reward_sync_task.cancel()
        try:
            await reward_sync_task
        except asyncio.CancelledError:
            pass
        if service.learner_executor is not None:
            service.learner_executor.shutdown(wait=False, cancel_futures=True)


def _trainer_ready_payload(
    service: PolicyService, service_config: ServiceConfig,
) -> dict[str, Any]:
    """Report the effective PPO settings, including CLI/runtime overrides."""
    config = service.config
    return {
        "event": "trainer_ready",
        "host": service_config.host,
        "port": service_config.port,
        "device": str(service.device),
        "policy_version": service.state.policy_version,
        "total_agent_ticks": service.state.total_agent_ticks,
        "freeze_policy": service.freeze_policy,
        "parameters": service.policy.parameter_count,
        "rollout_agent_ticks": config.rollout_agent_ticks,
        "recurrent_sequence_length": config.recurrent_sequence_length,
        "learning_rate": config.learning_rate,
        "minibatch_samples": config.minibatch_samples,
        "optimization_epochs": config.optimization_epochs,
        "learner_cpu_threads": config.learner_cpu_threads,
        "target_kl": config.target_kl,
        "elite_imitation_weight": config.elite_imitation_weight,
        "elite_replay_capacity": config.elite_replay_capacity,
        "elite_replay": service.elite_replay.metrics(),
        "adaptive_reward": service.adaptive_rewards.telemetry(),
        "adaptive_reward_cutover": service.adaptive_reward_cutover_telemetry(),
        "adaptive_reward_arena": {
            "host": service_config.arena_host,
            "port": service_config.arena_port,
        },
    }


def _noop_action() -> dict[str, Any]:
    return {"schema_version": 1, "forward": 0, "strafe": 0, "jump": False, "sprint": False,
            "sneak": False, "yaw_delta": 0.0, "pitch_delta": 0.0, "primary": "none",
            "release_use": False, "hotbar": -1, "swap_offhand": False}


def _execution(
    step: dict[str, Any], proposed_action: dict[str, Any]
) -> tuple[str, dict[str, Any] | None, int | None]:
    """Return a conservative execution label and validated actual action.

    Older workers omit ``execution`` entirely; those steps retain the original
    on-policy behavior. Once a worker sends the field, malformed/unknown values
    are treated as invalid instead of silently contaminating PPO.
    """
    source, action_id = _execution_reference(step)
    if "execution" not in step:
        return "policy", dict(proposed_action), None
    execution = step.get("execution")
    if source == "invalid" or not isinstance(execution, dict):
        return "invalid", None, action_id
    action = _canonical_action(execution.get("action"))
    if action is None:
        return "invalid", None, action_id
    return source, action, action_id


def _execution_reference(step: dict[str, Any]) -> tuple[str, int | None]:
    """Read execution ownership and its optional queued proposal ID.

    The ID is deliberately parsed separately from the actual action. This lets
    the service retire an explicitly referenced proposal even when the rest of
    the execution report is invalid, while never guessing which proposal an
    uncorrelated override consumed.
    """
    if "execution" not in step:
        return "policy", None
    execution = step.get("execution")
    if not isinstance(execution, dict):
        return "invalid", None
    source = str(execution.get("source", "invalid"))
    valid_sources = {"policy", "teacher_sword", "teacher_crystal", "teacher_block", "safety"}
    if source not in valid_sources:
        source = "invalid"
    action_id: int | None = None
    if "action_id" in execution:
        raw_action_id = execution.get("action_id")
        if isinstance(raw_action_id, bool) or not isinstance(raw_action_id, int) or raw_action_id < 1:
            return "invalid", None
        action_id = raw_action_id
    return source, action_id


def _pre_execution_observation(
    step: dict[str, Any], fallback: dict[str, Any],
) -> dict[str, Any]:
    """Prefer the exact state a teacher saw before taking its control.

    Delayed policy queues make the proposal observation an unreliable teacher
    target, while the ordinary step observation is built after the teacher has
    already aimed/clicked. New workers attach the pre-control observation to
    the execution report. Both field spellings are accepted during rollout.
    """
    execution = step.get("execution")
    if not isinstance(execution, dict):
        return fallback
    fallback_match = fallback.get("match")
    fallback_match = fallback_match if isinstance(fallback_match, dict) else {}
    fallback_episode = str(fallback_match.get("episode_id", ""))
    for name in ("pre_execution_observation", "teacher_observation"):
        candidate = execution.get(name)
        if not isinstance(candidate, dict):
            continue
        try:
            schema_version = int(candidate.get("schema_version", -1))
        except (TypeError, ValueError, OverflowError):
            continue
        if schema_version not in (1, 2):
            continue
        if not all(
            key in candidate
            for key in ("match", "self", "opponent", "entities", "blocks", "action_mask")
        ):
            continue
        match = candidate.get("match")
        if not isinstance(match, dict):
            continue
        if fallback_episode and str(match.get("episode_id", "")) != fallback_episode:
            continue
        return candidate
    return fallback


def _canonical_action(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    try:
        forward = int(value["forward"])
        strafe = int(value["strafe"])
        hotbar = int(value["hotbar"])
        yaw_delta = float(value["yaw_delta"])
        pitch_delta = float(value["pitch_delta"])
        primary = str(value["primary"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if forward not in (-1, 0, 1) or strafe not in (-1, 0, 1):
        return None
    if hotbar < -1 or hotbar > 8 or primary not in PRIMARY_NAMES:
        return None
    if not math.isfinite(yaw_delta) or not math.isfinite(pitch_delta):
        return None
    if abs(yaw_delta) > math.pi or abs(pitch_delta) > math.pi / 2:
        return None
    return {
        "schema_version": 1,
        "forward": forward,
        "strafe": strafe,
        "jump": bool(value.get("jump", False)),
        "sprint": bool(value.get("sprint", False)),
        "sneak": bool(value.get("sneak", False)),
        "yaw_delta": yaw_delta,
        "pitch_delta": pitch_delta,
        "primary": primary,
        "release_use": bool(value.get("release_use", False)),
        "hotbar": hotbar,
        "swap_offhand": bool(value.get("swap_offhand", False)),
    }


def _actions_match(
    actual: dict[str, Any], proposed: dict[str, Any], observation: dict[str, Any] | None = None,
) -> bool:
    # V1 observations cannot resolve V2 candidate intent; retain the historical
    # echoed-wire behavior for synthetic/legacy reporters. Real V2 workers send
    # ObservationV2 and ordinary schema-1 Minecraft inputs.
    expected = (
        _canonical_action(proposed)
        if not isinstance(observation, dict) or int(observation.get("schema_version", 1)) != 2
        else _resolve_proposed_action(proposed, observation)
    )
    if expected is None:
        return False
    discrete = (
        "forward", "strafe", "jump", "sprint", "sneak", "primary",
        "release_use", "hotbar", "swap_offhand",
    )
    return all(actual[name] == expected[name] for name in discrete) and all(
        math.isclose(actual[name], expected[name], rel_tol=0.0, abs_tol=1e-6)
        for name in ("yaw_delta", "pitch_delta")
    )


def _resolve_proposed_action(
    proposed: dict[str, Any], observation: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Mirror the worker's deterministic ActionV2 legality resolution.

    ActionV2's item/use heads are conditional on intent and candidate geometry,
    so the ordinary Minecraft input reported by the worker intentionally differs
    from the unresolved wire heads.  Correlate PPO with that resolved input,
    while retaining exact V1 matching and rejecting malformed targets.
    """
    expected = _canonical_action(proposed)
    if expected is None or int(proposed.get("schema_version", 1)) != 2:
        return expected
    if not isinstance(observation, dict):
        return None
    intent = str(proposed.get("intent", ""))
    try:
        target_index = int(proposed.get("target_index", -2))
    except (TypeError, ValueError, OverflowError):
        return None
    self_state = observation.get("self")
    tactical = observation.get("tactical")
    mask = observation.get("action_mask")
    if not isinstance(self_state, dict) or not isinstance(tactical, dict) or not isinstance(mask, dict):
        return None
    expected.update({"primary": "none", "release_use": False, "hotbar": -1, "swap_offhand": False})

    def item_slot(needle: str) -> int:
        hotbar = self_state.get("hotbar")
        if not isinstance(hotbar, list):
            return -1
        for index, item in enumerate(hotbar):
            if not isinstance(item, dict):
                continue
            try:
                count = int(item.get("count", 0))
            except (TypeError, ValueError, OverflowError):
                count = 0
            if count > 0 and needle in str(item.get("name", "")).lower():
                return index
        return -1

    def usable_hand(needle: str) -> str:
        offhand = self_state.get("offhand")
        if isinstance(offhand, dict):
            try:
                count = int(offhand.get("count", 0))
            except (TypeError, ValueError, OverflowError):
                count = 0
            if count > 0 and needle in str(offhand.get("name", "")).lower():
                return "use_offhand"
        return "use_main" if item_slot(needle) >= 0 else "none"

    implicit = {"sword_engage", "crystal_acquire", "heal_retotem", "disengage", "reposition"}
    if intent in implicit and target_index != -1:
        return None
    if intent == "sword_engage":
        expected["hotbar"] = item_slot("sword")
        expected["primary"] = "attack" if bool(mask.get("combat_attack_ready")) else "none"
    elif intent == "crystal_acquire":
        expected["hotbar"] = item_slot("crystal")
    elif intent in {"crystal_place", "crystal_detonate"}:
        candidates = tactical.get("crystal_candidates")
        if not isinstance(candidates, list) or target_index < 0 or target_index >= len(candidates):
            return None
        candidate = candidates[target_index]
        required_kind = "base" if intent == "crystal_place" else "crystal"
        if not isinstance(candidate, dict) or candidate.get("kind") != required_kind:
            return None
        if intent == "crystal_place":
            expected["hotbar"] = item_slot("crystal")
            if bool(candidate.get("placement_legal")) and bool(mask.get("crystal_place_ready")):
                expected["primary"] = usable_hand("crystal")
        elif bool(candidate.get("reachable")) and bool(candidate.get("visible")) and bool(mask.get("crystal_attack_ready")):
            expected["primary"] = "attack"
    elif intent in {"build_pad", "mine_path"}:
        candidates = tactical.get("block_candidates")
        if not isinstance(candidates, list) or target_index < 0 or target_index >= len(candidates):
            return None
        if not isinstance(candidates[target_index], dict):
            return None
        if intent == "build_pad":
            expected["hotbar"] = item_slot("obsidian")
            if bool(mask.get("tactical_block_place_ready")):
                expected["primary"] = usable_hand("obsidian")
        else:
            expected["hotbar"] = item_slot("pickaxe")
            if bool(mask.get("tactical_block_break_ready")):
                expected["primary"] = "attack"
    elif intent == "heal_retotem":
        survival = tactical.get("survival")
        if not isinstance(survival, dict):
            return None
        if not bool(survival.get("has_totem")) and int(survival.get("spare_totems", 0)) > 0:
            expected["hotbar"] = item_slot("totem")
            expected["swap_offhand"] = True
        else:
            expected["hotbar"] = item_slot("golden_apple")
            if bool(survival.get("heal_available")):
                expected["primary"] = usable_hand("golden_apple")
    elif intent == "disengage":
        expected["forward"] = -1
        expected["sprint"] = True
        expected["strafe"] = 1 if expected["strafe"] == 0 else expected["strafe"]
    elif intent != "reposition":
        return None
    return expected


def _action_is_legal(action: dict[str, Any], observation: dict[str, Any]) -> bool:
    """Reject supervised targets that the policy mask could never emit."""
    try:
        features = batch_observations([observation], "cpu")
        actions = actions_from_wire([action], "cpu")
        masks = categorical_masks(features)
        for name, target in actions.categorical.items():
            if name in ("intent", "target_index"):
                continue
            index = int(target[0])
            if index < 0 or index >= masks[name].shape[1] or not bool(masks[name][0, index]):
                return False
        if bool(action["release_use"]) and str(action["primary"]) in ("use_main", "use_offhand"):
            return False
        return True
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def sanitize_reward(
    step: dict[str, Any], shaping_clip: float = 0.25, terminal_clip: float = 64.0,
    trainer_shaping: float = 0.0, *, include_terminal: bool = True,
) -> SanitizedReward:
    """Separate sparse outcomes from dense shaping and make both finite.

    A worker can accumulate many server feedback events while the trainer is
    performing a CPU update. Treating that delayed sum as one unbounded reward
    creates a large, poorly attributed PPO target. Shaping is therefore capped
    per consumed transition, while the separately configured match outcome can
    remain orders of magnitude larger than one dense event.
    """
    server_reward, reward_nonfinite = _finite_number(step.get("reward", 0.0))
    trainer_reward, trainer_nonfinite = _finite_number(trainer_shaping)
    raw_reward = server_reward + trainer_reward
    explicitly_done = bool(step.get("terminated") or step.get("truncated"))
    info = step.get("info") if isinstance(step.get("info"), dict) else {}
    terminal_present = explicitly_done and "reward" in info
    if terminal_present:
        raw_terminal, terminal_nonfinite = _finite_number(info.get("reward"))
        raw_shaping = server_reward - raw_terminal if not reward_nonfinite else 0.0
        raw_shaping += trainer_reward
    elif explicitly_done:
        # Client-local fallback deaths do not carry the arena's info.reward.
        raw_terminal, terminal_nonfinite = server_reward, reward_nonfinite
        raw_shaping = trainer_reward
    else:
        raw_terminal, terminal_nonfinite = 0.0, False
        raw_shaping = server_reward + trainer_reward
    if not include_terminal:
        raw_terminal = 0.0
    shaping_limit = max(0.0, float(shaping_clip))
    terminal_limit = max(0.0, float(terminal_clip))
    shaping = max(-shaping_limit, min(shaping_limit, raw_shaping))
    terminal = max(-terminal_limit, min(terminal_limit, raw_terminal))
    clipped = abs(shaping - raw_shaping) > 1e-12 or abs(terminal - raw_terminal) > 1e-12
    return SanitizedReward(
        server_reward=server_reward,
        trainer_shaping_reward=trainer_reward,
        raw_reward=raw_reward,
        training_reward=shaping + terminal,
        raw_shaping_reward=raw_shaping,
        shaping_reward=shaping,
        raw_terminal_reward=raw_terminal,
        terminal_reward=terminal,
        clipped=clipped,
        nonfinite=reward_nonfinite or trainer_nonfinite or terminal_nonfinite,
    )


def _has_explicit_terminal_attribution(step: dict[str, Any]) -> bool:
    info = step.get("info") if isinstance(step.get("info"), dict) else {}
    return "terminal_source" in info or "policy_owned_kill" in info


def _verified_policy_kill_quality(
    step: dict[str, Any], terminal_clip: float,
) -> float | None:
    """Return speed-sensitive replay quality for an autonomous winning kill."""
    if not bool(step.get("terminated") or step.get("truncated")):
        return None
    info = step.get("info") if isinstance(step.get("info"), dict) else {}
    terminal, nonfinite = _finite_number(info.get("reward", 0.0))
    if (
        nonfinite
        or terminal <= 0.0
        or info.get("terminal_source") != "policy"
        or info.get("policy_owned_kill") is not True
        or str(info.get("outcome", "")) != "win"
    ):
        return None
    limit = max(1e-6, float(terminal_clip))
    return max(0.25, min(1.0, terminal / limit))


def _terminal_training_eligible(step: dict[str, Any], *, delayed: bool) -> bool:
    """Accept autonomous outcomes, including the cost of failing to finish.

    Timeout and disengagement are arena-wide failures rather than actions
    owned by the last controller tick. They must reach the preceding on-policy
    transition even when the terminal report arrives on the worker's safety
    no-op. Death rewards remain causal: policy kills teach both the winner and
    victim, self/environment deaths teach avoidance, and teacher/safety kills
    never enter PPO.
    """
    if not bool(step.get("terminated") or step.get("truncated")):
        return True
    info = step.get("info") if isinstance(step.get("info"), dict) else {}
    terminal_source = info.get("terminal_source")
    if terminal_source is None:
        return not delayed
    raw_terminal, _ = _finite_number(info.get("reward", 0.0))
    reason = str(info.get("reason", ""))
    if reason in ("timeout", "disengaged"):
        return raw_terminal < 0.0
    if terminal_source == "policy":
        return info.get("policy_owned_kill") is True or raw_terminal < 0.0
    if terminal_source in ("self", "environment"):
        return raw_terminal < 0.0
    if terminal_source == "none":
        return not delayed and reason != "death"
    return False


def _finite_number(value: Any) -> tuple[float, bool]:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0, True
    if not np.isfinite(converted):
        return 0.0, True
    return converted, False


def _initialize_policy(policy: CombatPolicy, checkpoints: list[Path], device: torch.device) -> None:
    states = []
    for checkpoint in checkpoints:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        source = CombatPolicy()
        load_policy_weights(source, payload)
        from_version = int(payload.get("feature_contract_version", 1))
        if from_version < FEATURE_CONTRACT_VERSION:
            migrate_feature_contract(source, from_version)
        states.append(source.state_dict())
    if not states:
        return
    merged = {
        name: torch.stack([state[name].float() for state in states]).mean(dim=0).to(value.dtype)
        for name, value in states[0].items()
    }
    policy.load_state_dict(merged)
    policy.to(device)
