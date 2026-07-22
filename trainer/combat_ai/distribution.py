from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.distributions import Normal

from .features import (
    BLOCK_CRYSTAL_BASE_INDEX,
    BLOCK_CRYSTAL_TARGET_INDEX,
    BLOCK_RAYCASTABLE_INDEX,
    ENTITY_BODY_X_INDEX,
    ENTITY_BODY_Z_INDEX,
    ENTITY_CRYSTAL_TARGET_INDEX,
    HOTBAR_SEMANTIC_INDICES,
    LEGAL_TERRAIN_MODE_INDEX,
    OPPONENT_AIM_ALIGNMENT_INDEX,
    OPPONENT_BEARING_COS_INDEX,
    OPPONENT_BEARING_SIN_INDEX,
    OPPONENT_MELEE_RANGE_INDEX,
    OPPONENT_PITCH_ERROR_INDEX,
    PVP_ITEM_CRYSTAL,
    PVP_ITEM_OBSIDIAN,
    PVP_ITEM_PICKAXE,
    PVP_ITEM_SWORD,
    SELF_CRYSTAL_CAPABLE_INDEX,
    SELF_CRYSTAL_RETENTION_INDEX,
    TACTICAL_CRYSTAL_KIND_INDEX,
    TACTICAL_LEGAL_INDEX,
    TACTICAL_REACHABLE_INDEX,
    TACTICAL_VISIBLE_INDEX,
    FeatureBatch,
    PRIMARY_NAMES,
    categorical_masks,
)
from .model import INTENT_NAMES, PolicyOutput

# Actions arrive at 20 Hz and are deltas from the current view. The wire format
# permits full rotations for compatibility, but sampling 180/90-degree deltas
# each tick makes early exploration pin the camera at a pitch limit. These
# bounds still allow a fighter to reverse direction in under half a second.
CAMERA_SCALE = (0.45, 0.30)

# When an attack is legal the worker has already verified that a charged swing
# can hit the assigned fighter or an arena crystal under the crosshair. Bias
# exploration toward taking that opportunity. This changes no view angles and
# contains no target coordinates; PPO learns a residual and can override it.
ATTACK_READY_LOGIT_BONUS = 8.0
CRYSTAL_READY_LOGIT_BONUS = 8.0
BLOCK_BREAK_READY_LOGIT_BONUS = 6.0
TACTICAL_BUILD_READY_LOGIT_BONUS = 8.0
TACTICAL_BUILD_HOTBAR_BONUS = 8.0
CRYSTAL_ACQUIRE_HOTBAR_BONUS = 2.5
CRYSTAL_CAMERA_PRIOR_WEIGHT = 0.85
CRYSTAL_RETENTION_CAMERA_PRIOR_WEIGHT = 1.35
TACTICAL_BUILD_CAMERA_PRIOR_WEIGHT = 0.85
CRYSTAL_EYE_HEIGHT = 1.62
OPPONENT_CAMERA_PRIOR_WEIGHT = 0.65
OPPONENT_APPROACH_LOGIT_BONUS = 1.25
OPPONENT_SPRINT_LOGIT_BONUS = 1.0
HOTBAR_HOLD_LOGIT_BONUS = 8.0
INTENT_CRYSTAL_DETONATE_BONUS = 18.0
INTENT_CRYSTAL_PLACE_BONUS = 16.0
INTENT_SWORD_READY_BONUS = 14.0
INTENT_CRYSTAL_ACQUIRE_BONUS = 7.0
INTENT_TACTICAL_BONUS = 6.0


@dataclass
class ActionTensor:
    categorical: dict[str, torch.Tensor]
    camera: torch.Tensor

    def index(self, indices: torch.Tensor) -> "ActionTensor":
        return ActionTensor(
            categorical={name: value[indices] for name, value in self.categorical.items()},
            camera=self.camera[indices],
        )


def sample_actions(
    output: PolicyOutput,
    features: FeatureBatch,
    deterministic: bool = False,
    compute_entropy: bool = True,
) -> tuple[list[dict[str, Any]], ActionTensor, torch.Tensor, torch.Tensor]:
    masks = categorical_masks(features)
    categorical: dict[str, torch.Tensor] = {}
    total_log_probability = torch.zeros_like(output.value)
    total_entropy = torch.zeros_like(output.value)
    priors = _action_prior_context(features, output.value.dtype)
    # Intent and target are conditional, so sample those two first. All other
    # independent low-level heads share one padded softmax/multinomial call;
    # release_use follows primary because its legal mask is conditional.
    for name in ("intent", "target_index"):
        logits = output.logits[name]
        mask = _hierarchical_mask(name, logits, masks, categorical, features, priors)
        action, log_probability, entropy = _sample_categorical_group(
            [(name, action_logits(name, logits, features, priors), mask)],
            deterministic, compute_entropy,
        )[name]
        categorical[name] = action
        total_log_probability += log_probability
        total_entropy += entropy
    independent = []
    for name, logits in output.logits.items():
        if name in ("intent", "target_index", "release_use"):
            continue
        independent.append((
            name, action_logits(name, logits, features, priors),
            _hierarchical_mask(name, logits, masks, categorical, features, priors),
        ))
    sampled = _sample_categorical_group(independent, deterministic, compute_entropy)
    for name, (action, log_probability, entropy) in sampled.items():
        categorical[name] = action
        total_log_probability += log_probability
        total_entropy += entropy
    name = "release_use"
    logits = output.logits[name]
    action, log_probability, entropy = _sample_categorical_group(
        [(name, action_logits(name, logits, features, priors),
          _hierarchical_mask(name, logits, masks, categorical, features, priors))],
        deterministic, compute_entropy,
    )[name]
    categorical[name] = action
    total_log_probability += log_probability
    total_entropy += entropy
    camera, camera_log_probability, camera_entropy = _sample_camera(
        camera_action_mean(output.camera_mean, features, priors), output.camera_log_std, deterministic
    )
    total_log_probability += camera_log_probability
    if compute_entropy:
        total_entropy += camera_entropy
    tensor = ActionTensor(categorical=categorical, camera=camera)
    return _to_wire_actions(tensor), tensor, total_log_probability, total_entropy


def evaluate_actions(
    output: PolicyOutput,
    features: FeatureBatch,
    actions: ActionTensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    masks = categorical_masks(features)
    total_log_probability = torch.zeros_like(output.value)
    total_entropy = torch.zeros_like(output.value)
    priors = _action_prior_context(features, output.value.dtype)
    entries = []
    for name, logits in output.logits.items():
        entries.append((
            name, action_logits(name, logits, features, priors),
            _hierarchical_mask(name, logits, masks, actions.categorical, features, priors),
        ))
    for name, (log_probability, entropy) in _evaluate_categorical_group(
        entries, actions.categorical
    ).items():
        del name
        total_log_probability += log_probability
        total_entropy += entropy
    camera_log_probability, camera_entropy = _camera_log_probability(
        camera_action_mean(output.camera_mean, features, priors), output.camera_log_std, actions.camera
    )
    return total_log_probability + camera_log_probability, total_entropy + camera_entropy


def actions_from_wire(actions: list[dict[str, Any]], device: torch.device | str) -> ActionTensor:
    categorical = {
        "intent": torch.tensor([_intent_index(a) for a in actions], device=device),
        "target_index": torch.tensor([max(0, min(16, int(a.get("target_index", -1)) + 1)) for a in actions], device=device),
        "forward": torch.tensor([int(a["forward"]) + 1 for a in actions], device=device),
        "strafe": torch.tensor([int(a["strafe"]) + 1 for a in actions], device=device),
        "jump": torch.tensor([int(bool(a["jump"])) for a in actions], device=device),
        "sprint": torch.tensor([int(bool(a["sprint"])) for a in actions], device=device),
        "sneak": torch.tensor([int(bool(a["sneak"])) for a in actions], device=device),
        "primary": torch.tensor([PRIMARY_NAMES.index(a["primary"]) for a in actions], device=device),
        "release_use": torch.tensor([int(bool(a["release_use"])) for a in actions], device=device),
        "hotbar": torch.tensor([int(a["hotbar"]) + 1 for a in actions], device=device),
        "swap_offhand": torch.tensor([int(bool(a["swap_offhand"])) for a in actions], device=device),
    }
    camera = torch.tensor([[float(a["yaw_delta"]), float(a["pitch_delta"])] for a in actions], device=device)
    return ActionTensor(categorical=categorical, camera=camera)


def _sample_camera(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normal = Normal(mean, log_std.exp())
    latent = mean if deterministic else normal.rsample()
    # Match the inverse path used during PPO evaluation. Without this clamp an
    # extreme exploration sample can round to exactly +/-1 in float32, so its
    # stored action no longer reconstructs the sampled log probability.
    squashed = torch.tanh(latent).clamp(-0.999999, 0.999999)
    latent = torch.atanh(squashed)
    scale = torch.tensor(CAMERA_SCALE, dtype=mean.dtype, device=mean.device)
    action = squashed * scale
    log_probability = normal.log_prob(latent) - torch.log(scale * (1 - squashed.square()) + 1e-6)
    entropy = -log_probability
    return action, log_probability.sum(-1), entropy.sum(-1)


def _camera_log_probability(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = torch.tensor(CAMERA_SCALE, dtype=mean.dtype, device=mean.device)
    squashed = (action / scale).clamp(-0.999999, 0.999999)
    latent = torch.atanh(squashed)
    normal = Normal(mean, log_std.exp())
    log_probability = normal.log_prob(latent) - torch.log(scale * (1 - squashed.square()) + 1e-6)
    return log_probability.sum(-1), (-log_probability).sum(-1)


def _masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if torch.any(~mask.any(dim=-1)):
        raise ValueError("an action head has no legal choice")
    return logits.masked_fill(~mask, -1e9)


def _sample_categorical_group(
    entries: list[tuple[str, torch.Tensor, torch.Tensor]],
    deterministic: bool, compute_entropy: bool,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Sample differently-sized heads with one padded multinomial operation."""
    if not entries:
        return {}
    batch = entries[0][1].shape[0]
    width = max(logits.shape[1] for _, logits, _ in entries)
    rows = []
    sizes = []
    for _, logits, mask in entries:
        masked = _masked_logits(logits, mask)
        sizes.append(masked.shape[1])
        if masked.shape[1] < width:
            masked = torch.nn.functional.pad(
                masked, (0, width - masked.shape[1]), value=-torch.inf
            )
        rows.append(masked)
    packed = torch.cat(rows, dim=0)
    log_probabilities = torch.log_softmax(packed, dim=-1)
    probabilities = log_probabilities.exp()
    # Exponential-race sampling is exactly categorical but avoids the costly
    # CPU multinomial setup for these many tiny action-head batches.
    selected = (
        packed.argmax(dim=-1)
        if deterministic
        else (packed - torch.empty_like(packed).exponential_().log()).argmax(dim=-1)
    )
    selected_log_probability = log_probabilities.gather(1, selected.unsqueeze(1)).squeeze(1)
    # Packed heads use -inf padding. xlogy(0, 0) has the right forward value
    # but a NaN derivative in PyTorch, poisoning the shared trunk on PPO's
    # first entropy backward pass. Zero padded log-probabilities explicitly;
    # probabilities are already exactly zero there.
    finite_log_probabilities = log_probabilities.masked_fill(
        ~torch.isfinite(log_probabilities), 0.0
    )
    entropy = (
        -(probabilities * finite_log_probabilities).sum(dim=-1)
        if compute_entropy else torch.zeros_like(selected_log_probability)
    )
    result = {}
    for index, (name, _, _) in enumerate(entries):
        start, end = index * batch, (index + 1) * batch
        result[name] = (
            selected[start:end], selected_log_probability[start:end], entropy[start:end]
        )
    return result


def _evaluate_categorical_group(
    entries: list[tuple[str, torch.Tensor, torch.Tensor]],
    actions: dict[str, torch.Tensor],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    if not entries:
        return {}
    batch = entries[0][1].shape[0]
    width = max(logits.shape[1] for _, logits, _ in entries)
    rows = []
    selected = []
    for name, logits, mask in entries:
        masked = _masked_logits(logits, mask)
        if masked.shape[1] < width:
            masked = torch.nn.functional.pad(
                masked, (0, width - masked.shape[1]), value=-torch.inf
            )
        rows.append(masked)
        selected.append(actions[name])
    packed = torch.cat(rows, dim=0)
    chosen = torch.cat(selected, dim=0)
    log_probabilities = torch.log_softmax(packed, dim=-1)
    probabilities = log_probabilities.exp()
    selected_log_probability = log_probabilities.gather(1, chosen.unsqueeze(1)).squeeze(1)
    finite_log_probabilities = log_probabilities.masked_fill(
        ~torch.isfinite(log_probabilities), 0.0
    )
    entropy = -(probabilities * finite_log_probabilities).sum(dim=-1)
    return {
        name: (
            selected_log_probability[index * batch:(index + 1) * batch],
            entropy[index * batch:(index + 1) * batch],
        )
        for index, (name, _, _) in enumerate(entries)
    }


def action_logits(
    name: str, logits: torch.Tensor, features: FeatureBatch,
    priors: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    result = logits.clone()
    if name == "target_index":
        return result
    context = priors or _action_prior_context(features, result.dtype)
    attack_ready = context["attack_ready"]
    crystal_place_ready = context["crystal_place_ready"]
    crystal_attack_ready = context["crystal_attack_ready"]
    crystal_entity_target = context["crystal_entity_target"]
    crystal_base_target = context["crystal_base_target"]
    tactical_build = context["tactical_build"]
    tactical_build_ready = context["tactical_build_ready"]
    retention = context["retention"]
    sword_ready = context["sword_ready"]
    block_break_ready = context["block_break_ready"]
    approach = context["approach"]
    crystal_acquisition = context["crystal_acquisition"]
    combat_ready = context["combat_ready"]
    if name == "intent":
        legal_intents = context["legal_intents"]
        place_ready = crystal_place_ready * legal_intents[:, 2].to(result.dtype)
        detonate_ready = crystal_attack_ready * legal_intents[:, 3].to(result.dtype)
        immediate_crystal = torch.clamp(place_ready + detonate_ready, max=1.0)
        result[:, 3] += INTENT_CRYSTAL_DETONATE_BONUS * detonate_ready
        result[:, 2] += INTENT_CRYSTAL_PLACE_BONUS * place_ready * (1.0 - detonate_ready)
        result[:, 0] += INTENT_SWORD_READY_BONUS * attack_ready * (1.0 - immediate_crystal)
        result[:, 1] += INTENT_CRYSTAL_ACQUIRE_BONUS * crystal_acquisition * (1.0 - immediate_crystal)
        result[:, 4] += INTENT_TACTICAL_BONUS * tactical_build_ready * (1.0 - immediate_crystal)
        result[:, 5] += INTENT_TACTICAL_BONUS * block_break_ready * (1.0 - immediate_crystal)
        return result
    del crystal_entity_target, crystal_acquisition, combat_ready

    if name == "primary":
        result[:, 1] += ATTACK_READY_LOGIT_BONUS * sword_ready
        result[:, 1] += CRYSTAL_READY_LOGIT_BONUS * crystal_attack_ready
        result[:, 1] += BLOCK_BREAK_READY_LOGIT_BONUS * block_break_ready
        result[:, 2] += CRYSTAL_READY_LOGIT_BONUS * crystal_place_ready
        result[:, 2] += TACTICAL_BUILD_READY_LOGIT_BONUS * tactical_build_ready
    elif name == "hotbar":
        sword_slots = context["sword_slots"]
        crystal_slots = context["crystal_slots"]
        obsidian_slots = context["obsidian_slots"]
        pickaxe_slots = context["pickaxe_slots"]
        sword_switch = sword_ready * sword_slots.any(dim=1).to(result.dtype)
        crystal_switch = torch.clamp(
            crystal_place_ready + crystal_base_target, max=1.0
        ) * crystal_slots.any(dim=1).to(result.dtype)
        pickaxe_switch = block_break_ready * pickaxe_slots.any(dim=1).to(result.dtype)
        obsidian_switch = tactical_build * obsidian_slots.any(dim=1).to(result.dtype)
        result[:, 1:10] += sword_slots * (ATTACK_READY_LOGIT_BONUS * sword_ready).unsqueeze(-1)
        result[:, 1:10] += crystal_slots * (CRYSTAL_READY_LOGIT_BONUS * crystal_place_ready).unsqueeze(-1)
        result[:, 1:10] += crystal_slots * (
            CRYSTAL_ACQUIRE_HOTBAR_BONUS * (1.0 + retention) * crystal_base_target
        ).unsqueeze(-1)
        result[:, 1:10] += pickaxe_slots * (BLOCK_BREAK_READY_LOGIT_BONUS * block_break_ready).unsqueeze(-1)
        result[:, 1:10] += obsidian_slots * (TACTICAL_BUILD_HOTBAR_BONUS * tactical_build).unsqueeze(-1)
        tactical_switch = torch.clamp(
            sword_switch + crystal_switch + pickaxe_switch + obsidian_switch, max=1.0
        )
        result[:, 0] += HOTBAR_HOLD_LOGIT_BONUS * (1.0 - tactical_switch)
    elif name == "forward":
        result[:, 2] += OPPONENT_APPROACH_LOGIT_BONUS * approach
    elif name == "sprint":
        result[:, 1] += OPPONENT_SPRINT_LOGIT_BONUS * approach
    return result


def _action_prior_context(
    features: FeatureBatch, dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    sword_slots = _hotbar_category_mask(features, PVP_ITEM_SWORD, dtype)
    crystal_slots = _hotbar_category_mask(features, PVP_ITEM_CRYSTAL, dtype)
    obsidian_slots = _hotbar_category_mask(features, PVP_ITEM_OBSIDIAN, dtype)
    pickaxe_slots = _hotbar_category_mask(features, PVP_ITEM_PICKAXE, dtype)
    candidate_masks = _candidate_target_masks(features)
    legal_intents = _legal_intent_mask(
        features, candidate_masks, obsidian_slots, pickaxe_slots,
    )
    attack_ready = (features.legal[:, 17] > 0.5).to(dtype=dtype)
    crystal_place_ready = (features.legal[:, 18] > 0.5).to(dtype=dtype)
    crystal_attack_ready = (features.legal[:, 19] > 0.5).to(dtype=dtype)
    # Mining is useful only when there is no immediately executable combat
    # action. Without this gate the easier block objective steals both the
    # attack input and hotbar choice from sword/crystal opportunities.
    crystal_entity_target, crystal_base_target = _crystal_acquisition_masks(features)
    tactical_build, tactical_build_ready = _tactical_build_masks(
        features, crystal_entity_target, obsidian_slots,
    )
    # A one-per-episode worker-verified build support is the curriculum target
    # that teaches the missing setup skill. It may supersede acquisition of a
    # generated base, but never an immediately ready combat action or a live
    # crystal entity that should be detonated first.
    crystal_base_target = crystal_base_target * (1.0 - tactical_build)
    crystal_acquisition = torch.clamp(crystal_entity_target + crystal_base_target, max=1.0)
    retention = features.self_state[:, SELF_CRYSTAL_RETENTION_INDEX].to(dtype)
    # Dedicated crystal retention must get repetitions even at close spawns.
    # Remove only the hand-authored sword bonus while a crystal target exists;
    # the policy's own combat logits remain untouched and can still override.
    sword_ready = attack_ready * (1.0 - retention * crystal_acquisition)
    combat_ready = torch.clamp(
        attack_ready + crystal_place_ready + crystal_attack_ready + crystal_acquisition,
        max=1.0,
    )
    block_break_ready = (
        (features.legal[:, 20] > 0.5).to(dtype=dtype)
        * (1.0 - combat_ready)
        * (1.0 - tactical_build)
    )
    approach = _opponent_approach_mask(
        features, crystal_acquisition, tactical_build
    ).to(dtype)
    return {
        "attack_ready": attack_ready,
        "crystal_place_ready": crystal_place_ready,
        "crystal_attack_ready": crystal_attack_ready,
        "crystal_entity_target": crystal_entity_target,
        "crystal_base_target": crystal_base_target,
        "tactical_build": tactical_build,
        "tactical_build_ready": tactical_build_ready,
        "retention": retention,
        "sword_ready": sword_ready,
        "crystal_acquisition": crystal_acquisition,
        "combat_ready": combat_ready,
        "block_break_ready": block_break_ready,
        "approach": approach,
        "sword_slots": sword_slots,
        "crystal_slots": crystal_slots,
        "obsidian_slots": obsidian_slots,
        "pickaxe_slots": pickaxe_slots,
        "candidate_base_mask": candidate_masks[0],
        "candidate_crystal_mask": candidate_masks[1],
        "candidate_block_mask": candidate_masks[2],
        "legal_intents": legal_intents,
    }


def camera_action_mean(
    mean: torch.Tensor, features: FeatureBatch,
    priors: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Add bounded feature-only residuals toward assigned combat targets.

    Existing crystal priors only fired after the crosshair was already exact.
    The marked target features make acquisition dense while keeping the action
    sampled by the policy distribution (and therefore PPO-owned). When no
    crystal/block operation has priority, the assigned opponent's explicit
    bearing supplies the same kind of weak acquisition help. The worker never
    puts the spectator into the opponent field. Sampling, PPO evaluation and
    imitation all call this exact transform.
    """
    if priors is None:
        entity_target, crystal_block_target = _crystal_acquisition_masks(features)
        tactical_build, _ = _tactical_build_masks(features, entity_target)
        crystal_block_target = crystal_block_target * (1.0 - tactical_build)
    else:
        entity_target = priors["crystal_entity_target"]
        crystal_block_target = priors["crystal_base_target"]
        tactical_build = priors["tactical_build"]
    crystal_active = torch.clamp(entity_target + crystal_block_target, max=1.0)
    active = torch.clamp(crystal_active + tactical_build, max=1.0)

    entity_marker = (
        (features.entities[:, :, ENTITY_CRYSTAL_TARGET_INDEX] > 0.5)
        * features.entity_mask.bool()
    ).to(mean.dtype)
    marked_blocks = (
        (features.blocks[:, :, BLOCK_CRYSTAL_TARGET_INDEX] > 0.5)
        * features.block_mask.bool()
    )
    crystal_block_marker = (
        marked_blocks
        * (features.blocks[:, :, BLOCK_CRYSTAL_BASE_INDEX] > 0.5)
    ).to(mean.dtype)
    tactical_build_marker = (
        marked_blocks
        * (features.blocks[:, :, BLOCK_CRYSTAL_BASE_INDEX] <= 0.5)
    ).to(mean.dtype)

    # Entity positions are scaled by 12; block positions by 6. Aim near the
    # crystal's centre or the top-centre of its base from normal eye height.
    entity_legacy_position = (
        features.entities[:, :, 5:8] * 12.0 * entity_marker.unsqueeze(-1)
    ).sum(dim=1)
    entity_position = torch.stack((
        (features.entities[:, :, ENTITY_BODY_X_INDEX] * 12.0 * entity_marker).sum(dim=1),
        entity_legacy_position[:, 1],
        (features.entities[:, :, ENTITY_BODY_Z_INDEX] * 12.0 * entity_marker).sum(dim=1),
    ), dim=-1)
    entity_position[:, 1] += 1.0 - CRYSTAL_EYE_HEIGHT
    corrected_crystal_block = _marked_block_top_position(features, crystal_block_marker)
    corrected_tactical_support = _marked_block_top_position(features, tactical_build_marker)
    use_entity = entity_target.unsqueeze(-1)
    target = (
        entity_position * use_entity
        + corrected_crystal_block * crystal_block_target.unsqueeze(-1)
        + corrected_tactical_support * tactical_build.unsqueeze(-1)
    )
    horizontal = torch.sqrt(target[:, 0].square() + target[:, 2].square()).clamp_min(1e-6)
    yaw_error = torch.atan2(-target[:, 0], -target[:, 2])
    desired_pitch = torch.atan2(target[:, 1], horizontal)
    current_pitch = features.self_state[:, 8] * (torch.pi / 2)
    pitch_error = desired_pitch - current_pitch
    desired_delta = torch.stack((yaw_error, pitch_error), dim=-1)

    scale = torch.tensor(CAMERA_SCALE, dtype=mean.dtype, device=mean.device)
    squashed = (desired_delta / scale).clamp(-0.95, 0.95)
    target_latent = torch.atanh(squashed)
    retention = features.self_state[:, SELF_CRYSTAL_RETENTION_INDEX].to(mean.dtype)
    crystal_weight = (
        CRYSTAL_CAMERA_PRIOR_WEIGHT
        + retention * (CRYSTAL_RETENTION_CAMERA_PRIOR_WEIGHT - CRYSTAL_CAMERA_PRIOR_WEIGHT)
    ) * crystal_active
    operation_weight = crystal_weight + TACTICAL_BUILD_CAMERA_PRIOR_WEIGHT * tactical_build
    operation_residual = target_latent * operation_weight.unsqueeze(-1)

    verified_operation = torch.clamp(
        active
        + (features.legal[:, 18] > 0.5).to(mean.dtype)
        + (features.legal[:, 19] > 0.5).to(mean.dtype)
        + (features.legal[:, 20] > 0.5).to(mean.dtype),
        max=1.0,
    )
    opponent_active = features.opponent_mask[:, 0].to(mean.dtype) * (1.0 - verified_operation)
    opponent_yaw_error = torch.atan2(
        features.opponent[:, OPPONENT_BEARING_SIN_INDEX],
        features.opponent[:, OPPONENT_BEARING_COS_INDEX],
    )
    opponent_pitch_error = (
        features.opponent[:, OPPONENT_PITCH_ERROR_INDEX] * (torch.pi / 2)
    )
    opponent_delta = torch.stack((opponent_yaw_error, opponent_pitch_error), dim=-1)
    opponent_squashed = (opponent_delta / scale).clamp(-0.95, 0.95)
    opponent_latent = torch.atanh(opponent_squashed)
    opponent_weight = OPPONENT_CAMERA_PRIOR_WEIGHT * opponent_active
    # Blend in latent space rather than merely adding a turn residual. At zero
    # bearing/pitch error the target latent is zero, so this also attenuates a
    # legacy checkpoint's learned camera drift instead of letting it wander
    # off target and correcting one tick later. Verified crystal/block work
    # makes opponent_active zero, leaving the crystal path exactly unchanged.
    opponent_adjustment = (
        (opponent_latent - mean) * opponent_weight.unsqueeze(-1)
    )
    return mean + operation_residual + opponent_adjustment


def _marked_block_top_position(
    features: FeatureBatch, marker: torch.Tensor,
) -> torch.Tensor:
    """Return corrected body-relative top-centre coordinates for one marker."""

    position = (features.blocks[:, :, 0:3] * 6.0 * marker.unsqueeze(-1)).sum(dim=1)
    yaw_sin = features.self_state[:, 6].to(position.dtype)
    yaw_cos = features.self_state[:, 7].to(position.dtype)
    # Block samples store their integer world origin. Transform the horizontal
    # half-block centre offset into the worker's legacy egocentric frame before
    # applying the compatibility correction below.
    position[:, 0] += 0.5 * (yaw_cos + yaw_sin)
    position[:, 1] += 1.0 - CRYSTAL_EYE_HEIGHT
    position[:, 2] += 0.5 * (yaw_cos - yaw_sin)
    sin_two_yaw = 2.0 * yaw_sin * yaw_cos
    cos_two_yaw = yaw_cos.square() - yaw_sin.square()
    corrected_x = cos_two_yaw * position[:, 0] - sin_two_yaw * position[:, 2]
    corrected_z = sin_two_yaw * position[:, 0] + cos_two_yaw * position[:, 2]
    return torch.stack((corrected_x, position[:, 1], corrected_z), dim=-1)


def _crystal_acquisition_masks(features: FeatureBatch) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = features.self_state.dtype
    capable = (features.self_state[:, SELF_CRYSTAL_CAPABLE_INDEX] > 0.5).to(dtype)
    retention = (features.self_state[:, SELF_CRYSTAL_RETENTION_INDEX] > 0.5).to(dtype)
    melee_ready = (features.legal[:, 17] > 0.5).to(dtype) * (1.0 - retention)
    ready = torch.clamp(
        melee_ready
        + (features.legal[:, 18] > 0.5).to(dtype)
        + (features.legal[:, 19] > 0.5).to(dtype),
        max=1.0,
    )
    may_acquire = capable * (1.0 - ready)
    has_entity = torch.any(
        (features.entities[:, :, ENTITY_CRYSTAL_TARGET_INDEX] > 0.5)
        & features.entity_mask.bool(),
        dim=1,
    ).to(dtype)
    has_block = torch.any(
        (features.blocks[:, :, BLOCK_CRYSTAL_TARGET_INDEX] > 0.5)
        & (features.blocks[:, :, BLOCK_CRYSTAL_BASE_INDEX] > 0.5)
        & features.block_mask.bool(),
        dim=1,
    ).to(dtype)
    entity_target = may_acquire * has_entity
    block_target = may_acquire * (1.0 - has_entity) * has_block
    return entity_target, block_target


def _tactical_build_masks(
    features: FeatureBatch, crystal_entity_acquisition: torch.Tensor,
    obsidian_slots: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return policy-visible tactical build opportunity and click readiness."""

    dtype = features.self_state.dtype
    terrain = (features.legal[:, LEGAL_TERRAIN_MODE_INDEX] > 0.5).to(dtype)
    target_marker = (
        (features.blocks[:, :, BLOCK_CRYSTAL_TARGET_INDEX] > 0.5)
        & (features.blocks[:, :, BLOCK_CRYSTAL_BASE_INDEX] <= 0.5)
        & features.block_mask.bool()
    )
    has_target = torch.any(target_marker, dim=1).to(dtype)
    target_raycastable = torch.any(
        target_marker
        & (features.blocks[:, :, BLOCK_RAYCASTABLE_INDEX] > 0.5),
        dim=1,
    ).to(dtype)
    if obsidian_slots is None:
        obsidian_slots = _hotbar_category_mask(features, PVP_ITEM_OBSIDIAN, dtype)
    has_obsidian = obsidian_slots.any(dim=1).to(dtype)
    immediate_combat = torch.clamp(
        (features.legal[:, 17] > 0.5).to(dtype)
        + (features.legal[:, 18] > 0.5).to(dtype)
        + (features.legal[:, 19] > 0.5).to(dtype),
        max=1.0,
    )
    active = (
        terrain
        * has_target
        * has_obsidian
        * (1.0 - immediate_combat)
        * (1.0 - crystal_entity_acquisition)
    )
    ready = (
        active
        * target_raycastable
        * (features.legal[:, 2] > 0.5).to(dtype)
    )
    return active, ready


def _hotbar_category_mask(
    features: FeatureBatch, category: float, dtype: torch.dtype,
) -> torch.Tensor:
    # Hotbar semantics occupy a regular four-float stride. A view avoids
    # allocating an index tensor and dispatching index_select in every action
    # prior (historically nine times per sampled batch).
    categories = features.self_state[
        :, HOTBAR_SEMANTIC_INDICES[0]:HOTBAR_SEMANTIC_INDICES[-1] + 1:4
    ]
    occupied = features.legal[:, 8:17] > 0.5
    return ((categories - category).abs() < 1e-4).to(dtype) * occupied.to(dtype)


def _opponent_approach_mask(
    features: FeatureBatch, crystal_acquisition: torch.Tensor,
    tactical_build: torch.Tensor,
) -> torch.Tensor:
    dtype = features.self_state.dtype
    opponent = features.opponent_mask[:, 0].to(dtype)
    outside_melee = 1.0 - (features.opponent[:, OPPONENT_MELEE_RANGE_INDEX] > 0.5).to(dtype)
    roughly_aligned = (
        (features.opponent[:, OPPONENT_AIM_ALIGNMENT_INDEX] > 0.35)
        & (features.opponent[:, OPPONENT_BEARING_COS_INDEX] > 0.0)
    ).to(dtype)
    verified_operation = torch.clamp(
        crystal_acquisition
        + (features.legal[:, 18] > 0.5).to(dtype)
        + (features.legal[:, 19] > 0.5).to(dtype)
        + (features.legal[:, 20] > 0.5).to(dtype)
        + tactical_build,
        max=1.0,
    )
    return opponent * outside_melee * roughly_aligned * (1.0 - verified_operation)


def _conditional_mask(name: str, mask: torch.Tensor, primary: torch.Tensor | None) -> torch.Tensor:
    if name != "release_use" or primary is None:
        return mask
    result = mask.clone()
    result[:, 1] &= primary < 2
    return result


def _hierarchical_mask(
    name: str, logits: torch.Tensor, masks: dict[str, torch.Tensor],
    chosen: dict[str, torch.Tensor], features: FeatureBatch,
    priors: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    if name == "intent":
        return priors["legal_intents"] if priors is not None else _legal_intent_mask(features)
    if name == "target_index":
        result = torch.zeros_like(logits, dtype=torch.bool)
        intent = chosen.get("intent")
        if intent is None:
            result[:, 0] = True
            return result
        implicit = (intent == 0) | (intent == 1) | (intent >= 6)
        result[:, 0] = implicit
        if priors is None:
            base, crystal, block = _candidate_target_masks(features)
        else:
            base = priors["candidate_base_mask"]
            crystal = priors["candidate_crystal_mask"]
            block = priors["candidate_block_mask"]
        result[:, 1:] |= (intent == 2).unsqueeze(-1) & base
        result[:, 1:] |= (intent == 3).unsqueeze(-1) & crystal
        result[:, 1:] |= ((intent == 4) | (intent == 5)).unsqueeze(-1) & block
        return result
    return _conditional_mask(name, masks[name], chosen.get("primary"))


def _candidate_target_masks(
    features: FeatureBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    candidates = features.crystal_candidates
    present = features.crystal_candidate_mask.bool()
    kind = candidates[:, :, TACTICAL_CRYSTAL_KIND_INDEX]
    reachable = candidates[:, :, TACTICAL_REACHABLE_INDEX] > 0.5
    visible = candidates[:, :, TACTICAL_VISIBLE_INDEX] > 0.5
    placement_legal = candidates[:, :, TACTICAL_LEGAL_INDEX] > 0.5
    base = present & (kind > 0.5) & reachable & visible & placement_legal
    crystal = present & (kind < -0.5) & reachable & visible
    block = (
        features.tactical_block_mask.bool()
        & (features.tactical_blocks[:, :, TACTICAL_REACHABLE_INDEX] > 0.5)
        & (features.tactical_blocks[:, :, TACTICAL_VISIBLE_INDEX] > 0.5)
    )
    return base, crystal, block


def _legal_intent_mask(
    features: FeatureBatch,
    candidate_masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    obsidian_slots: torch.Tensor | None = None,
    pickaxe_slots: torch.Tensor | None = None,
) -> torch.Tensor:
    batch = features.self_state.shape[0]
    result = torch.zeros((batch, len(INTENT_NAMES)), dtype=torch.bool, device=features.self_state.device)
    # Implicit-target intents are always structurally legal. Acquire additionally
    # requires crystal capability so it cannot become a permanent empty select.
    result[:, 0] = True
    result[:, 1] = features.self_state[:, SELF_CRYSTAL_CAPABLE_INDEX] > 0.5
    result[:, 6:] = True
    base, crystal, block = candidate_masks or _candidate_target_masks(features)
    if obsidian_slots is None:
        obsidian_slots = _hotbar_category_mask(
            features, PVP_ITEM_OBSIDIAN, features.self_state.dtype,
        )
    if pickaxe_slots is None:
        pickaxe_slots = _hotbar_category_mask(
            features, PVP_ITEM_PICKAXE, features.self_state.dtype,
        )
    has_obsidian = obsidian_slots.any(dim=1)
    has_pickaxe = pickaxe_slots.any(dim=1)
    result[:, 2] = (features.legal[:, 18] > 0.5) & base.any(dim=1)
    result[:, 3] = (features.legal[:, 19] > 0.5) & crystal.any(dim=1)
    # V1 has no dedicated place bit for tactical blocks. Generic use legality
    # plus a verified reachable candidate is the exact capability the worker uses.
    result[:, 4] = (features.legal[:, 2] > 0.5) & has_obsidian & block.any(dim=1)
    result[:, 5] = (features.legal[:, 20] > 0.5) & has_pickaxe & block.any(dim=1)
    return result


def _intent_index(action: dict[str, Any]) -> int:
    value = action.get("intent")
    if value in INTENT_NAMES:
        return INTENT_NAMES.index(value)
    primary = action.get("primary")
    if primary == "attack":
        return INTENT_NAMES.index("sword_engage")
    if primary in ("use_main", "use_offhand"):
        return INTENT_NAMES.index("crystal_place")
    return INTENT_NAMES.index("reposition")


def _to_wire_actions(actions: ActionTensor) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    categorical = {name: value.detach().cpu().tolist() for name, value in actions.categorical.items()}
    camera = actions.camera.detach().cpu().tolist()
    for index in range(len(camera)):
        result.append({
            "schema_version": 2,
            "intent": INTENT_NAMES[int(categorical["intent"][index])],
            "target_index": int(categorical["target_index"][index]) - 1,
            "forward": int(categorical["forward"][index]) - 1,
            "strafe": int(categorical["strafe"][index]) - 1,
            "jump": bool(categorical["jump"][index]),
            "sprint": bool(categorical["sprint"][index]),
            "sneak": bool(categorical["sneak"][index]),
            "yaw_delta": float(camera[index][0]),
            "pitch_delta": float(camera[index][1]),
            "primary": PRIMARY_NAMES[int(categorical["primary"][index])],
            "release_use": bool(categorical["release_use"][index]),
            "hotbar": int(categorical["hotbar"][index]) - 1,
            "swap_offhand": bool(categorical["swap_offhand"][index]),
        })
    return result
