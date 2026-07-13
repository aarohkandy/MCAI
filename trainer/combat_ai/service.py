from __future__ import annotations

import asyncio
import json
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import torch
import websockets

from .buffer import RolloutBuffer, Transition, prepare_sequences
from .checkpoint import CheckpointManager
from .config import PPOConfig, ServiceConfig
from .distribution import ActionTensor, sample_actions
from .features import batch_observations, encode_observation
from .model import CombatPolicy
from .imitation import load_demonstrations, split_matches
from .league import LeagueManager
from .ppo import PPOTrainer, choose_device


@dataclass
class PendingStep:
    agent_id: str
    episode_id: str
    policy_version: int
    features: dict[str, np.ndarray]
    hidden: np.ndarray
    categorical_action: dict[str, int]
    camera_action: np.ndarray
    log_probability: float
    value: float


class PolicyService:
    def __init__(
        self, ppo_config: PPOConfig, service_config: ServiceConfig,
        imitation_data: Path | None = None, initialize_from: list[Path] | None = None,
        exploiter_target: Path | None = None,
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
        self.device = choose_device()
        self.policy = CombatPolicy()
        if self.policy.parameter_count >= 1_000_000:
            raise RuntimeError(f"policy is too large: {self.policy.parameter_count:,} parameters")
        self.trainer = PPOTrainer(self.policy, ppo_config, self.device)
        self.checkpoints = CheckpointManager(service_config.checkpoint_dir, ppo_config.checkpoint_every_ticks)
        self.state = self.checkpoints.restore(self.policy, self.trainer.optimizer, self.device)
        if self.state.total_agent_ticks == 0 and initialize_from and not (service_config.checkpoint_dir / "latest.pt").exists():
            _initialize_policy(self.policy, initialize_from, self.device)
        self.buffer = RolloutBuffer(ppo_config.rollout_agent_ticks)
        self.hidden: dict[str, torch.Tensor] = {}
        self.pending: dict[str, PendingStep] = {}
        self.update_lock = asyncio.Lock()
        self.league = LeagueManager(service_config.checkpoint_dir, self.device)
        self.league.force_frozen_opponent(exploiter_target)
        self.imitation_records = []
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
            return {"schema_version": 1, "type": "control", "sequence": message.get("sequence", 0),
                    "command": "hello_ack", "payload": {"device": str(self.device),
                    "policy_version": self.state.policy_version,
                    "parameters": self.policy.parameter_count}}
        if message.get("type") != "step_batch":
            return None
        return await self._step_batch(message)

    async def _step_batch(self, message: dict[str, Any]) -> dict[str, Any]:
        steps = list(message.get("steps") or [])
        if not steps:
            return self._action_response(message, [])
        observations = [step["observation"] for step in steps]
        self.league.assign_batch(steps)
        encoded = [encode_observation(observation) for observation in observations]
        features = batch_observations(observations, self.device)
        hidden = torch.cat([
            self.hidden.get(step["agent_id"], self.policy.initial_hidden(1, self.device))
            for step in steps
        ], dim=1)

        self.policy.eval()
        with torch.no_grad():
            bootstrap_output = self.policy(features, hidden)
        for index, step in enumerate(steps):
            self._finish_pending(step, float(bootstrap_output.value[index]))

        if self.buffer.ready:
            async with self.update_lock:
                if self.buffer.ready:
                    self._train_update()
                    hidden = self.policy.initial_hidden(len(steps), self.device)
                    for agent_id in list(self.hidden):
                        self.hidden[agent_id] = self.policy.initial_hidden(1, self.device)
                    with torch.no_grad():
                        bootstrap_output = self.policy(features.to(self.device), hidden)

        with torch.no_grad():
            wire_actions, action_tensor, log_probability, _ = sample_actions(
                bootstrap_output, features.to(self.trainer.device), self.service_config.deterministic_inference
            )
        current_version = self.state.policy_version
        for index, step in enumerate(steps):
            agent_id = str(step["agent_id"])
            done = bool(step.get("terminated") or step.get("truncated"))
            if done:
                wire_actions[index] = _noop_action()
                self.hidden[agent_id] = self.policy.initial_hidden(1, self.trainer.device)
                self.pending.pop(agent_id, None)
                continue
            assignment = self.league.assignment_for(str(step["observation"]["match"]["episode_id"]))
            if assignment is not None and assignment.opponent_agent == agent_id and assignment.mode != "mirror":
                wire_actions[index] = self.league.opponent_action(assignment, agent_id, step["observation"])
                self.hidden[agent_id] = self.policy.initial_hidden(1, self.trainer.device)
                self.pending.pop(agent_id, None)
                continue
            self.hidden[agent_id] = bootstrap_output.hidden[:, index:index + 1].detach()
            self.pending[agent_id] = PendingStep(
                agent_id=agent_id,
                episode_id=str(step["observation"]["match"]["episode_id"]),
                policy_version=current_version,
                features=encoded[index],
                hidden=hidden[:, index].detach().cpu().numpy()[0],
                categorical_action={name: int(value[index]) for name, value in action_tensor.categorical.items()},
                camera_action=action_tensor.camera[index].detach().cpu().numpy().astype(np.float32),
                log_probability=float(log_probability[index]),
                value=float(bootstrap_output.value[index]),
            )
        actions = [
            {"agent_id": step["agent_id"], "action": wire_actions[index]}
            for index, step in enumerate(steps)
        ]
        return self._action_response(message, actions)

    def _finish_pending(self, step: dict[str, Any], bootstrap_value: float) -> None:
        agent_id = str(step["agent_id"])
        pending = self.pending.pop(agent_id, None)
        if pending is None:
            return
        done = bool(step.get("terminated") or step.get("truncated"))
        episode_id = str(step["observation"]["match"]["episode_id"])
        if episode_id != pending.episode_id:
            done = True
        self.buffer.append(Transition(
            agent_id=agent_id, episode_id=pending.episode_id, policy_version=pending.policy_version,
            features=pending.features, hidden=pending.hidden,
            categorical_action=pending.categorical_action, camera_action=pending.camera_action,
            old_log_probability=pending.log_probability, old_value=pending.value,
            reward=float(step.get("reward", 0.0)), done=done,
            next_value=0.0 if done else bootstrap_value,
        ))
        if done:
            info = step.get("info") if isinstance(step.get("info"), dict) else {}
            self.league.record_result(
                pending.episode_id, agent_id, float(step.get("reward", 0.0)), info.get("outcome")
            )

    def _train_update(self) -> None:
        transitions = self.buffer.drain(self.state.policy_version)
        if not transitions:
            return
        self.policy.train()
        batch = prepare_sequences(
            transitions, self.config.recurrent_sequence_length, self.config.gamma,
            self.config.gae_lambda, self.trainer.device,
        )
        metrics = self.trainer.update(batch)
        imitation_weight = self.config.imitation_start_weight * max(
            0.0, 1.0 - self.state.total_agent_ticks / max(1, self.config.imitation_decay_ticks)
        )
        metrics.imitation_loss = self.trainer.auxiliary_imitation_update(
            self.imitation_records, imitation_weight
        )
        self.device = self.trainer.device
        self.league.set_device(self.device)
        self.state.policy_version += 1
        self.state.total_agent_ticks += len(transitions)
        metrics_dict = vars(metrics)
        self.checkpoints.save(self.policy, self.trainer.optimizer, self.state, self.config, metrics_dict)
        self.league.prune_pool()
        print(json.dumps({"event": "ppo_update", "policy_version": self.state.policy_version,
                          "total_agent_ticks": self.state.total_agent_ticks, **metrics_dict}), flush=True)

    def _action_response(self, message: dict[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
        return {"schema_version": 1, "type": "action_batch", "sequence": message.get("sequence", 0),
                "policy_version": self.state.policy_version, "actions": actions}


async def serve(
    ppo_config: PPOConfig, service_config: ServiceConfig, imitation_data: Path | None = None,
    initialize_from: list[Path] | None = None, exploiter_target: Path | None = None,
) -> None:
    initial = list(initialize_from or [])
    if exploiter_target is not None and not initial:
        initial.append(exploiter_target)
    service = PolicyService(ppo_config, service_config, imitation_data, initial, exploiter_target)
    stop = asyncio.Future()
    loop = asyncio.get_running_loop()
    for signal_name in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signal_name, stop.set_result, None)
        except NotImplementedError:
            pass
    async with websockets.serve(service.handle, service_config.host, service_config.port, max_size=32 * 1024 * 1024):
        print(json.dumps({"event": "trainer_ready", "host": service_config.host,
                          "port": service_config.port, "device": str(service.device),
                          "policy_version": service.state.policy_version,
                          "parameters": service.policy.parameter_count}), flush=True)
        await stop


def _noop_action() -> dict[str, Any]:
    return {"schema_version": 1, "forward": 0, "strafe": 0, "jump": False, "sprint": False,
            "sneak": False, "yaw_delta": 0.0, "pitch_delta": 0.0, "primary": "none",
            "release_use": False, "hotbar": -1, "swap_offhand": False}


def _initialize_policy(policy: CombatPolicy, checkpoints: list[Path], device: torch.device) -> None:
    states = []
    for checkpoint in checkpoints:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        states.append(payload["policy"])
    if not states:
        return
    merged = {
        name: torch.stack([state[name].float() for state in states]).mean(dim=0).to(value.dtype)
        for name, value in states[0].items()
    }
    policy.load_state_dict(merged)
    policy.to(device)
