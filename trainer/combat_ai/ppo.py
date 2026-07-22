from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

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
    rollout_valid_samples: int = 0
    optimizer_sample_exposures: int = 0
    rollout_sample_coverage: float = 0.0
    max_kl: float = 0.0
    optimizer_updates: int = 0
    early_stopped: bool = False
    imitation_loss: float = 0.0
    imitation_updates: int = 0
    imitation_samples: int = 0
    imitation_crystal_samples: int = 0
    imitation_crystal_fraction: float = 0.0
    imitation_elite_samples: int = 0
    imitation_elite_fraction: float = 0.0
    imitation_elite_events_sampled: int = 0
    imitation_crystal_buffer: int = 0
    imitation_sword_buffer: int = 0
    imitation_block_buffer: int = 0
    imitation_elite_buffer: int = 0
    imitation_elite_events: int = 0
    imitation_elite_buckets: int = 0
    learning_rate: float = 0.0
    quarantined_sequences: int = 0
    quarantined_samples: int = 0
    skipped_optimizer_steps: int = 0
    optimizer_state_resets: int = 0


class PPOTrainer:
    def __init__(self, policy: CombatPolicy, config: PPOConfig, device: torch.device):
        self.policy = policy.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.low_kl_updates = 0
        self.last_imitation_metrics: dict[str, float | int] = {
            "updates": 0, "samples": 0, "crystal_samples": 0,
            "crystal_fraction": 0.0, "elite_samples": 0,
            "elite_fraction": 0.0, "elite_events_sampled": 0,
        }

    def reapply_configured_learning_rate(self) -> None:
        """Keep current runtime settings authoritative over restored optimizer metadata."""
        for group in self.optimizer.param_groups:
            group["lr"] = self.config.learning_rate

    def update(self, batch: SequenceBatch) -> UpdateMetrics:
        if not _module_is_finite(self.policy):
            raise ValueError("refusing to optimize a non-finite policy")
        batch, quarantined_sequences, quarantined_samples = _quarantine_nonfinite_sequences(batch)
        optimizer_state_resets = int(not _optimizer_state_is_finite(self.optimizer))
        if optimizer_state_resets:
            # A legacy/corrupt Adam moment can turn finite gradients and
            # parameters into NaNs on the first step. Keep the policy and reset
            # only optimizer history; the candidate still has to pass the
            # post-update finite check before publication.
            self.optimizer.state.clear()
        try:
            metrics = self._update(batch)
        except RuntimeError as error:
            if self.device.type != "mps" or not _is_mps_unsupported(error):
                raise
            self.device = torch.device("cpu")
            self.policy.to(self.device)
            _optimizer_to(self.optimizer, self.device)
            metrics = self._update(_batch_to(batch, self.device))
        metrics.quarantined_sequences = quarantined_sequences
        metrics.quarantined_samples = quarantined_samples
        metrics.optimizer_state_resets = optimizer_state_resets
        return metrics

    def _update(self, batch: SequenceBatch) -> UpdateMetrics:
        aggregate = torch.zeros(7, dtype=torch.float64)
        updates = 0
        optimizer_sample_exposures = 0
        max_kl = 0.0
        early_stopped = False
        skipped_optimizer_steps = 0
        for _ in range(self.config.optimization_epochs):
            for minibatch in _sample_aware_minibatches(
                batch, self.config.minibatch_samples
            ):
                metrics = self._minibatch(minibatch)
                if metrics is None:
                    skipped_optimizer_steps += 1
                    continue
                aggregate += torch.tensor(metrics, dtype=torch.float64)
                updates += 1
                optimizer_sample_exposures += int(metrics[6])
                max_kl = max(max_kl, float(metrics[3]))
                if self.config.target_kl > 0 and metrics[3] > self.config.target_kl:
                    early_stopped = True
                    break
            if early_stopped:
                break
        averaged = (aggregate / max(updates, 1)).tolist()
        if updates == 0:
            raise RuntimeError("no finite PPO optimizer step was available")
        rollout_valid_samples = int(batch.valid.sum().item())
        self._adapt_learning_rate(max_kl)
        return UpdateMetrics(
            policy_loss=averaged[0], value_loss=averaged[1], entropy=averaged[2],
            approximate_kl=averaged[3], clip_fraction=averaged[4], gradient_norm=averaged[5],
            valid_samples=int(averaged[6]),
            rollout_valid_samples=rollout_valid_samples,
            optimizer_sample_exposures=optimizer_sample_exposures,
            rollout_sample_coverage=min(
                1.0, optimizer_sample_exposures / max(1, rollout_valid_samples)
            ),
            max_kl=max_kl,
            optimizer_updates=updates, early_stopped=early_stopped,
            learning_rate=self.current_learning_rate,
            skipped_optimizer_steps=skipped_optimizer_steps,
        )

    @property
    def current_learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _adapt_learning_rate(self, measured_kl: float) -> None:
        # Recover learning speed whenever updates are consistently using less
        # than 80% of the configured KL budget.  The old fixed 0.002 trigger
        # left the optimizer pinned at its minimum even while live KL sat near
        # 0.006 against a 0.010 target.
        low_kl_threshold = max(
            self.config.adaptive_lr_low_kl,
            self.config.target_kl * 0.8,
        )
        if measured_kl > self.config.adaptive_lr_high_kl:
            self.low_kl_updates = 0
            desired = self.current_learning_rate * self.config.adaptive_lr_decrease
        elif measured_kl < low_kl_threshold:
            self.low_kl_updates += 1
            if self.low_kl_updates < self.config.adaptive_lr_low_updates:
                return
            self.low_kl_updates = 0
            desired = self.current_learning_rate * self.config.adaptive_lr_increase
        else:
            self.low_kl_updates = 0
            return
        desired = max(self.config.minimum_learning_rate, min(self.config.maximum_learning_rate, desired))
        for group in self.optimizer.param_groups:
            group["lr"] = desired

    def _minibatch(self, batch: SequenceBatch) -> tuple[float, ...] | None:
        total_valid = float(batch.valid.sum().item())
        if total_valid <= 0:
            raise ValueError("cannot optimize a minibatch with no valid samples")

        # Continuity boundaries can produce hundreds of short recurrent
        # fragments. Evaluate similarly sized fragments together so an
        # optimizer step is based on the configured number of *real* samples
        # without materializing every short sequence at the rollout's maximum
        # padded length. Loss weighting makes the microbatches equivalent to
        # one ragged minibatch, followed by one gradient clip/optimizer step.
        self.optimizer.zero_grad(set_to_none=True)
        aggregate = torch.zeros(5, dtype=torch.float64)
        for microbatch in _length_bucketed_microbatches(batch):
            terms = self._loss_terms(microbatch)
            microbatch_valid = float(terms[5])
            weight = microbatch_valid / total_valid
            loss = (
                terms[0]
                + self.config.value_coefficient * terms[1]
                - self.config.entropy_coefficient * terms[2]
            )
            if not torch.isfinite(loss) or not all(torch.isfinite(term) for term in terms[:5]):
                self.optimizer.zero_grad(set_to_none=True)
                return None
            (loss * weight).backward()
            aggregate += torch.tensor(
                [float(term.detach()) for term in terms[:5]], dtype=torch.float64
            ) * weight

        if not _gradients_are_finite(self.policy):
            self.optimizer.zero_grad(set_to_none=True)
            return None
        gradient_norm = nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.config.max_gradient_norm, error_if_nonfinite=True
        )
        self.optimizer.step()
        if not _module_is_finite(self.policy) or not _optimizer_state_is_finite(self.optimizer):
            raise RuntimeError("optimizer produced non-finite policy or Adam state")
        averaged = aggregate.tolist()
        return (
            averaged[0], averaged[1], averaged[2], averaged[3], averaged[4],
            float(gradient_norm), total_valid,
        )

    def _loss_terms(self, batch: SequenceBatch) -> tuple[torch.Tensor, ...]:
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
        # PPO ratios outside this range are already far beyond the clip region.
        # Bounding the exponent preserves the clipped objective while avoiding
        # inf * negative-advantage -> -inf gradients.
        log_ratio = (new_log_probability - batch.old_log_probability).clamp(-20.0, 20.0)
        ratio = torch.exp(log_ratio)
        unclipped = ratio * batch.advantage
        clipped = ratio.clamp(1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * batch.advantage
        policy_loss = -(torch.minimum(unclipped, clipped) * batch.valid).sum() / valid_count

        clipped_value = batch.old_value + (value - batch.old_value).clamp(
            -self.config.clip_ratio, self.config.clip_ratio
        )
        value_error = _clipped_huber_value_error(
            value, clipped_value, batch.returns, self.config.value_huber_delta,
        )
        value_loss = (value_error * batch.valid).sum() / valid_count
        entropy_mean = (entropy * batch.valid).sum() / valid_count
        approximate_kl = (
            ((torch.exp(log_ratio) - 1) - log_ratio) * batch.valid
        ).sum() / valid_count
        clip_fraction = (
            ((ratio - 1).abs() > self.config.clip_ratio).float() * batch.valid
        ).sum() / valid_count
        return (
            policy_loss, value_loss, entropy_mean, approximate_kl, clip_fraction,
            valid_count,
        )

    def auxiliary_imitation_update(self, records: list[dict], weight: float, batch_size: int = 512) -> float:
        if not records or weight <= 0:
            self.last_imitation_metrics = {
                "updates": 0, "samples": 0, "crystal_samples": 0,
                "crystal_fraction": 0.0, "elite_samples": 0,
                "elite_fraction": 0.0, "elite_events_sampled": 0,
            }
            return 0.0
        from .imitation import imitation_loss
        batches = _stratified_imitation_batches(records, batch_size)
        # Four crystal-rich minibatches provide enough signal to retain the
        # aim/select/place/detonate chain. Half weight per step caps the total
        # auxiliary influence at roughly twice the former one-step update.
        step_weight = weight * (0.5 if len(batches) > 1 else 1.0)
        losses: list[float] = []
        crystal_samples = 0
        elite_samples = 0
        elite_event_ids: set[str] = set()
        samples = 0
        self.policy.train()
        for batch in batches:
            loss = imitation_loss(self.policy, batch, self.device)
            if not torch.isfinite(loss):
                continue
            self.optimizer.zero_grad(set_to_none=True)
            (loss * step_weight).backward()
            if not _gradients_are_finite(self.policy):
                self.optimizer.zero_grad(set_to_none=True)
                continue
            nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.config.max_gradient_norm,
                error_if_nonfinite=True,
            )
            self.optimizer.step()
            if not _module_is_finite(self.policy) or not _optimizer_state_is_finite(self.optimizer):
                raise RuntimeError("imitation optimizer produced non-finite state")
            losses.append(float(loss.detach()))
            samples += len(batch)
            crystal_samples += sum(
                str(record.get("execution_source", "")) == "teacher_crystal"
                for record in batch
            )
            elite_samples += sum(
                str(record.get("execution_source", "")) == "elite_policy"
                for record in batch
            )
            elite_event_ids.update(
                _elite_event_key(record)
                for record in batch
                if str(record.get("execution_source", "")) == "elite_policy"
            )
        self.last_imitation_metrics = {
            "updates": len(losses),
            "samples": samples,
            "crystal_samples": crystal_samples,
            "crystal_fraction": crystal_samples / max(1, samples),
            "elite_samples": elite_samples,
            "elite_fraction": elite_samples / max(1, samples),
            "elite_events_sampled": len(elite_event_ids),
        }
        return sum(losses) / max(1, len(losses))


def _clipped_huber_value_error(
    value: torch.Tensor,
    clipped_value: torch.Tensor,
    returns: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """PPO-clipped robust critic error with bounded outlier gradients.

    The maximum preserves PPO's conservative value-update rule. Smooth-L1
    changes only the error geometry: once a target is more than ``delta``
    away, its gradient is capped instead of growing with a terminal reward.
    """
    if not delta > 0:
        raise ValueError("value_huber_delta must be positive")
    direct = F.smooth_l1_loss(value, returns, reduction="none", beta=delta)
    clipped = F.smooth_l1_loss(
        clipped_value, returns, reduction="none", beta=delta,
    )
    return torch.maximum(direct, clipped)


def _quarantine_nonfinite_sequences(
    batch: SequenceBatch,
) -> tuple[SequenceBatch, int, int]:
    """Remove only recurrent sequences containing unsafe numeric inputs."""
    count = batch.sequence_count
    finite = torch.ones(count, dtype=torch.bool, device=batch.valid.device)

    def include(value: torch.Tensor, sequence_dimension: int = 0) -> None:
        nonlocal finite
        moved = value.movedim(sequence_dimension, 0)
        finite &= torch.isfinite(moved.reshape(count, -1)).all(dim=1)

    for value in vars(batch.features).values():
        include(value)
    include(batch.hidden, 1)
    include(batch.actions.camera)
    for value in (
        batch.old_log_probability, batch.old_value, batch.advantage,
        batch.returns, batch.done, batch.valid,
    ):
        include(value)
    quarantined_sequences = int((~finite).sum().item())
    if quarantined_sequences == 0:
        return batch, 0, 0
    quarantined_samples = int(batch.valid[~finite].nan_to_num(0.0).sum().item())
    indices = finite.nonzero(as_tuple=False).flatten()
    if indices.numel() == 0:
        raise ValueError("all rollout sequences contain non-finite values")
    return batch.index(indices), quarantined_sequences, quarantined_samples


def _module_is_finite(module: nn.Module) -> bool:
    return all(torch.isfinite(parameter).all().item() for parameter in module.parameters())


def _gradients_are_finite(module: nn.Module) -> bool:
    return all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in module.parameters()
    )


def _optimizer_state_is_finite(optimizer: torch.optim.Optimizer) -> bool:
    return all(
        not isinstance(value, torch.Tensor) or torch.isfinite(value).all().item()
        for state in optimizer.state.values()
        for value in state.values()
    )


def _sample_aware_minibatches(
    batch: SequenceBatch, target_samples: int,
) -> list[SequenceBatch]:
    """Balance indivisible recurrent sequences by their valid sample count.

    ``minibatch_samples`` is a preferred optimizer sample count, not a count of
    padded tensor cells. Longest-processing-time assignment keeps the batches
    balanced to within one recurrent sequence while an initial random order
    breaks ties differently on every epoch.
    """
    if batch.sequence_count <= 0:
        raise ValueError("cannot pack an empty recurrent batch")
    lengths = _sequence_valid_lengths(batch)
    if any(length <= 0 for length in lengths):
        raise ValueError("every recurrent sequence must contain a valid sample")

    target = max(1, int(target_samples))
    total = sum(lengths)
    # Choose the nearest number of optimizer steps rather than always rounding
    # upward: a rollout that arrives a few samples over an exact multiple should
    # produce four ~512-sample steps, not five ~410-sample steps.
    minibatch_count = max(1, (total + target // 2) // target)
    minibatch_count = min(batch.sequence_count, minibatch_count)

    randomized = torch.randperm(batch.sequence_count).tolist()
    randomized.sort(key=lambda index: lengths[index], reverse=True)
    groups: list[list[int]] = [[] for _ in range(minibatch_count)]
    loads = [0] * minibatch_count
    for index in randomized:
        destination = min(
            range(minibatch_count),
            key=lambda group_index: (loads[group_index], len(groups[group_index])),
        )
        groups[destination].append(index)
        loads[destination] += lengths[index]

    # Do not let the balancing bins impose a fixed update order.
    group_order = torch.randperm(minibatch_count).tolist()
    device = batch.valid.device
    return [
        _trim_trailing_padding(batch.index(torch.tensor(groups[index], device=device)))
        for index in group_order
    ]


def _length_bucketed_microbatches(batch: SequenceBatch) -> list[SequenceBatch]:
    """Split one optimizer batch into bounded-padding gradient microbatches."""
    lengths = _sequence_valid_lengths(batch)
    buckets: dict[int, list[int]] = {}
    for index, length in enumerate(lengths):
        # Power-of-two buckets cap recurrent padding below 2x real samples,
        # while avoiding up to 32 separate sequential forward loops.
        upper_bound = 1 << (length - 1).bit_length()
        buckets.setdefault(upper_bound, []).append(index)
    device = batch.valid.device
    return [
        _trim_trailing_padding(batch.index(torch.tensor(indices, device=device)))
        for _, indices in sorted(buckets.items())
    ]


def _sequence_valid_lengths(batch: SequenceBatch) -> list[int]:
    return [
        int(value)
        for value in batch.valid.detach().sum(dim=1).to(device="cpu").tolist()
    ]


def _trim_trailing_padding(batch: SequenceBatch) -> SequenceBatch:
    active_columns = (batch.valid > 0).any(dim=0).nonzero(as_tuple=False)
    if active_columns.numel() == 0:
        raise ValueError("cannot trim a recurrent batch with no valid samples")
    length = int(active_columns[-1].item()) + 1
    if length == batch.sequence_length:
        return batch
    return SequenceBatch(
        features=type(batch.features)(**{
            name: value[:, :length] for name, value in vars(batch.features).items()
        }),
        hidden=batch.hidden,
        actions=type(batch.actions)(
            categorical={
                name: value[:, :length]
                for name, value in batch.actions.categorical.items()
            },
            camera=batch.actions.camera[:, :length],
        ),
        old_log_probability=batch.old_log_probability[:, :length],
        old_value=batch.old_value[:, :length],
        advantage=batch.advantage[:, :length],
        returns=batch.returns[:, :length],
        done=batch.done[:, :length],
        valid=batch.valid[:, :length],
    )


def _stratified_imitation_batches(records: list[dict], batch_size: int) -> list[list[dict]]:
    """Build source/phase-balanced batches, preserving episode/tick order.

    Verified autonomous successes occupy one event-balanced rehearsal batch per
    generation. Teacher crystal mechanics still receive four phase-balanced
    batches. A long successful episode therefore cannot contribute dozens of
    near-duplicate frames on every PPO generation.
    """
    import random
    from .imitation import classify_crystal_teacher_action

    maximum = max(1, int(batch_size))
    crystals: list[dict] = []
    crystal_record_ids: set[int] = set()
    for record in records:
        if str(record.get("execution_source", "")) != "teacher_crystal":
            continue
        phase = record.get("teacher_phase") or classify_crystal_teacher_action(record.get("action"))
        if phase not in {"aim", "select", "place", "detonate"}:
            continue
        crystal_record_ids.add(id(record))
        crystals.append(record if record.get("teacher_phase") == phase else {
            **record, "teacher_phase": phase,
        })
    elites = [
        record for record in records
        if str(record.get("execution_source", "")) == "elite_policy"
    ]
    elite_record_ids = {id(record) for record in elites}
    if not crystals and not elites:
        return [random.sample(records, min(maximum, len(records)))]
    others = [
        record for record in records
        if id(record) not in crystal_record_ids and id(record) not in elite_record_ids
    ]
    crystal_count = min(len(crystals), max(1, maximum // 2)) if crystals else 0
    if crystals:
        elite_limit = min(maximum - crystal_count, maximum // 4)
    else:
        elite_limit = maximum if not others else max(1, maximum // 2)
    elite_events = {_elite_event_key(record) for record in elites}
    elite_count = (
        min(len(elite_events), elite_limit)
        if elites and elite_limit > 0 else 0
    )
    batch_count = 4 if crystals else (2 if elites and others else 1)
    batches: list[list[dict]] = []
    for batch_index in range(batch_count):
        selected = _balanced_crystal_sample(crystals, crystal_count) if crystal_count else []
        if batch_index == 0 and elite_count:
            selected.extend(_balanced_elite_sample(elites, elite_count))
        other_count = min(len(others), max(0, maximum - len(selected)))
        if other_count:
            selected.extend(random.sample(others, other_count))
        selected.sort(key=_imitation_sequence_key)
        batches.append(selected)
    return batches


def _balanced_crystal_sample(records: list[dict], count: int) -> list[dict]:
    import random

    phases = ("aim", "select", "place", "detonate")
    groups = {
        phase: [record for record in records if record.get("teacher_phase") == phase]
        for phase in phases
    }
    for values in groups.values():
        random.shuffle(values)
    selected: list[dict] = []
    while len(selected) < count:
        progressed = False
        for phase in phases:
            values = groups[phase]
            if values and len(selected) < count:
                selected.append(values.pop())
                progressed = True
        if not progressed:
            break
    return selected


def _balanced_elite_sample(records: list[dict], count: int) -> list[dict]:
    """Choose at most one action per event, round-robin across strategy buckets."""
    import random

    grouped_events: dict[str, dict[str, list[dict]]] = {}
    for record in records:
        bucket = str(record.get("elite_bucket", "unknown"))
        event = _elite_event_key(record)
        grouped_events.setdefault(bucket, {}).setdefault(event, []).append(record)
    groups = {bucket: list(events.values()) for bucket, events in grouped_events.items()}
    for events in groups.values():
        random.shuffle(events)
    names = list(groups)
    random.shuffle(names)
    selected: list[dict] = []
    while len(selected) < count:
        progressed = False
        for name in names:
            events = groups[name]
            if events and len(selected) < count:
                selected.append(random.choice(events.pop()))
                progressed = True
        if not progressed:
            break
    return selected


def _elite_event_key(record: dict) -> str:
    explicit = str(record.get("elite_event_id", ""))
    if explicit:
        return explicit
    # Legacy/test records predate event IDs. Treat each match-agent pair as one
    # trajectory instead of allowing every action to masquerade as a new event.
    match_id = str(record.get("match_id", ""))
    agent_id = str(record.get("agent_id", ""))
    return f"legacy:{match_id}:{agent_id}"


def _imitation_sequence_key(record: dict) -> tuple[str, str, int]:
    observation = record.get("observation")
    match = observation.get("match") if isinstance(observation, dict) else {}
    match = match if isinstance(match, dict) else {}
    try:
        tick = int(record.get("tick", match.get("tick", 0)))
    except (TypeError, ValueError, OverflowError):
        tick = 0
    return str(record.get("match_id", "")), str(record.get("agent_id", "")), tick


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
