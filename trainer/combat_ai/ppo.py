from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .buffer import SequenceBatch, actions_at, features_at
from .config import PPOConfig
from .distribution import evaluate_actions
from .model import CombatPolicy


@dataclass
class UpdateMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    gradient_norm: float
    valid_samples: int
    imitation_loss: float = 0.0


class PPOTrainer:
    def __init__(self, policy: CombatPolicy, config: PPOConfig, device: torch.device):
        self.policy = policy.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)

    def update(self, batch: SequenceBatch) -> UpdateMetrics:
        try:
            return self._update(batch)
        except RuntimeError as error:
            if self.device.type != "mps" or not _is_mps_unsupported(error):
                raise
            self.device = torch.device("cpu")
            self.policy.to(self.device)
            _optimizer_to(self.optimizer, self.device)
            return self._update(_batch_to(batch, self.device))

    def _update(self, batch: SequenceBatch) -> UpdateMetrics:
        sequence_batch_size = max(1, self.config.minibatch_samples // batch.sequence_length)
        aggregate = torch.zeros(7, dtype=torch.float64)
        updates = 0
        for _ in range(self.config.optimization_epochs):
            order = torch.randperm(batch.sequence_count, device=self.device)
            for start in range(0, batch.sequence_count, sequence_batch_size):
                minibatch = batch.index(order[start:start + sequence_batch_size])
                metrics = self._minibatch(minibatch)
                aggregate += torch.tensor(metrics, dtype=torch.float64)
                updates += 1
        averaged = (aggregate / max(updates, 1)).tolist()
        return UpdateMetrics(
            policy_loss=averaged[0], value_loss=averaged[1], entropy=averaged[2],
            approximate_kl=averaged[3], clip_fraction=averaged[4], gradient_norm=averaged[5],
            valid_samples=int(averaged[6]),
        )

    def _minibatch(self, batch: SequenceBatch) -> tuple[float, ...]:
        hidden = batch.hidden.detach()
        log_probabilities: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for time_index in range(batch.sequence_length):
            output = self.policy(features_at(batch.features, time_index), hidden)
            log_probability, entropy = evaluate_actions(
                output, features_at(batch.features, time_index), actions_at(batch.actions, time_index)
            )
            log_probabilities.append(log_probability)
            entropies.append(entropy)
            values.append(output.value)
            hidden = output.hidden * (1.0 - batch.done[:, time_index]).view(1, -1, 1)
        new_log_probability = torch.stack(log_probabilities, dim=1)
        entropy = torch.stack(entropies, dim=1)
        value = torch.stack(values, dim=1)
        valid_count = batch.valid.sum().clamp_min(1.0)
        ratio = torch.exp(new_log_probability - batch.old_log_probability)
        unclipped = ratio * batch.advantage
        clipped = ratio.clamp(1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * batch.advantage
        policy_loss = -(torch.minimum(unclipped, clipped) * batch.valid).sum() / valid_count

        clipped_value = batch.old_value + (value - batch.old_value).clamp(
            -self.config.clip_ratio, self.config.clip_ratio
        )
        value_error = torch.maximum((value - batch.returns).square(), (clipped_value - batch.returns).square())
        value_loss = 0.5 * (value_error * batch.valid).sum() / valid_count
        entropy_mean = (entropy * batch.valid).sum() / valid_count
        loss = policy_loss + self.config.value_coefficient * value_loss - self.config.entropy_coefficient * entropy_mean

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_gradient_norm)
        self.optimizer.step()
        with torch.no_grad():
            log_ratio = new_log_probability - batch.old_log_probability
            approximate_kl = (((torch.exp(log_ratio) - 1) - log_ratio) * batch.valid).sum() / valid_count
            clip_fraction = (((ratio - 1).abs() > self.config.clip_ratio).float() * batch.valid).sum() / valid_count
        return (
            float(policy_loss.detach()), float(value_loss.detach()), float(entropy_mean.detach()),
            float(approximate_kl), float(clip_fraction), float(gradient_norm), float(valid_count),
        )

    def auxiliary_imitation_update(self, records: list[dict], weight: float, batch_size: int = 512) -> float:
        if not records or weight <= 0:
            return 0.0
        import random
        from .imitation import imitation_loss
        batch = random.sample(records, min(batch_size, len(records)))
        self.policy.train()
        loss = imitation_loss(self.policy, batch, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        (loss * weight).backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_gradient_norm)
        self.optimizer.step()
        return float(loss.detach())


def choose_device() -> torch.device:
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available() and mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _is_mps_unsupported(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "mps" in message and any(word in message for word in ("unsupported", "not implemented", "placeholder"))


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _batch_to(batch: SequenceBatch, device: torch.device) -> SequenceBatch:
    return SequenceBatch(
        features=batch.features.to(device), hidden=batch.hidden.to(device),
        actions=type(batch.actions)(
            categorical={name: value.to(device) for name, value in batch.actions.categorical.items()},
            camera=batch.actions.camera.to(device),
        ),
        old_log_probability=batch.old_log_probability.to(device), old_value=batch.old_value.to(device),
        advantage=batch.advantage.to(device), returns=batch.returns.to(device), done=batch.done.to(device),
        valid=batch.valid.to(device),
    )
