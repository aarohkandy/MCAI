import math

import torch
from torch.distributions import Categorical

from combat_ai.distribution import (
    CAMERA_SCALE,
    OPPONENT_CAMERA_PRIOR_WEIGHT,
    action_logits,
    camera_action_mean,
    evaluate_actions,
    sample_actions,
    _hierarchical_mask,
    _evaluate_categorical_group,
)
from combat_ai.features import batch_observations, categorical_masks
from combat_ai.model import CombatPolicy, INTENT_NAMES
from fixtures import observation


def test_grouped_categorical_matches_torch_log_probability_and_entropy():
    first_logits = torch.tensor([[1.0, -0.5, 0.25], [0.1, 0.2, 0.3]])
    second_logits = torch.tensor([[2.0, -1.0], [-0.5, 0.5]])
    first_mask = torch.tensor([[True, False, True], [True, True, True]])
    second_mask = torch.tensor([[True, True], [True, False]])
    actions = {"first": torch.tensor([2, 1]), "second": torch.tensor([0, 0])}
    grouped = _evaluate_categorical_group([
        ("first", first_logits, first_mask),
        ("second", second_logits, second_mask),
    ], actions)
    for name, logits, mask in (
        ("first", first_logits, first_mask), ("second", second_logits, second_mask),
    ):
        reference = Categorical(logits=logits.masked_fill(~mask, -1e9))
        assert torch.allclose(grouped[name][0], reference.log_prob(actions[name]), atol=1e-6)
        assert torch.allclose(grouped[name][1], reference.entropy(), atol=1e-6)


def test_policy_is_compact_and_outputs_legal_actions():
    policy = CombatPolicy()
    features = batch_observations([observation(), observation(tick=2)])
    output = policy(features, policy.initial_hidden(2, "cpu"))
    actions, _, log_probability, _ = sample_actions(output, features)
    assert 1_000_000 <= policy.parameter_count <= 2_000_000
    assert output.hidden.shape == (1, 2, 256)
    assert output.value.shape == (2,)
    assert torch.isfinite(log_probability).all()
    assert all(action["schema_version"] == 2 for action in actions)
    assert all("intent" in action and -1 <= action["target_index"] < 16 for action in actions)
    assert all(-math.pi <= action["yaw_delta"] <= math.pi for action in actions)
    assert all(-math.pi / 2 <= action["pitch_delta"] <= math.pi / 2 for action in actions)


def test_primary_mask_cannot_sample_attack():
    value = observation()
    value["action_mask"]["attack"] = False
    features = batch_observations([value])
    policy = CombatPolicy()
    with torch.no_grad():
        policy.categorical_heads["head_primary"].bias[:] = torch.tensor([0.0, 100.0, 0.0, 0.0])
    output = policy(features, policy.initial_hidden(1, "cpu"))
    actions, _, _, _ = sample_actions(output, features, deterministic=True)
    assert actions[0]["primary"] != "attack"


def test_ready_attack_has_an_exploration_prior_without_changing_aim():
    features = batch_observations([observation()])
    policy = CombatPolicy()
    with torch.no_grad():
        primary = policy.categorical_heads["head_primary"]
        primary.weight.zero_()
        primary.bias.zero_()
        policy.camera_mean.weight.zero_()
        policy.camera_mean.bias.zero_()
    output = policy(features, policy.initial_hidden(1, "cpu"))
    actions, _, _, _ = sample_actions(output, features, deterministic=True)
    assert actions[0]["primary"] == "attack"
    assert actions[0]["yaw_delta"] == 0.0
    assert actions[0]["pitch_delta"] == 0.0


def test_ready_melee_prior_selects_the_sword_slot():
    features = batch_observations([observation()])
    policy = CombatPolicy()
    with torch.no_grad():
        hotbar = policy.categorical_heads["head_hotbar"]
        hotbar.weight.zero_()
        hotbar.bias.zero_()
    output = policy(features, policy.initial_hidden(1, "cpu"))
    actions, _, _, _ = sample_actions(output, features, deterministic=True)
    assert actions[0]["hotbar"] == 0


def test_combat_hotbar_priors_follow_randomized_item_slots():
    cases = (
        ("sword", 6, {"combat_attack_ready": True}, "diamond_sword"),
        ("crystal", 7, {"combat_attack_ready": False, "crystal_place_ready": True}, "end_crystal"),
        ("pickaxe", 5, {
            "combat_attack_ready": False, "tactical_block_break_ready": True,
        }, "diamond_pickaxe"),
    )
    for _name, slot, mask, item_name in cases:
        value = observation()
        value["self"]["hotbar"] = [_combat_item("", 0) for _ in range(9)]
        maximum = 1561 if item_name.startswith("diamond_") else 0
        value["self"]["hotbar"][slot] = _combat_item(
            item_name, 1 if maximum else 64, durability=maximum, maximum=maximum
        )
        value["action_mask"].update({
            "combat_attack_ready": False,
            "crystal_place_ready": False,
            "crystal_attack_ready": False,
            "tactical_block_break_ready": False,
            **mask,
        })
        features = batch_observations([value])
        policy = CombatPolicy()
        with torch.no_grad():
            policy.categorical_heads["head_hotbar"].weight.zero_()
            policy.categorical_heads["head_hotbar"].bias.zero_()
        output = policy(features, policy.initial_hidden(1, "cpu"))
        actions, _, _, _ = sample_actions(output, features, deterministic=True)
        assert actions[0]["hotbar"] == slot, _name


def test_hotbar_defaults_to_no_change_without_a_tactical_switch():
    value = observation()
    value["action_mask"].update({
        "combat_attack_ready": False,
        "crystal_place_ready": False,
        "crystal_attack_ready": False,
        "tactical_block_break_ready": False,
    })
    features = batch_observations([value])
    policy = CombatPolicy()
    with torch.no_grad():
        policy.categorical_heads["head_hotbar"].weight.zero_()
        policy.categorical_heads["head_hotbar"].bias.zero_()
    output = policy(features, policy.initial_hidden(1, "cpu"))
    actions, _, _, _ = sample_actions(output, features, deterministic=True)
    assert actions[0]["hotbar"] == -1
    probabilities = torch.softmax(
        action_logits("hotbar", output.logits["hotbar"], features), dim=-1
    )
    assert probabilities[0, 0] > 0.995


def test_mining_uses_attack_input_without_receiving_combat_prior():
    value = observation()
    value["action_mask"]["combat_attack_ready"] = False
    features = batch_observations([value])
    policy = CombatPolicy()
    with torch.no_grad():
        primary = policy.categorical_heads["head_primary"]
        primary.weight.zero_()
        primary.bias.zero_()
    output = policy(features, policy.initial_hidden(1, "cpu"))
    actions, _, _, _ = sample_actions(output, features, deterministic=True)
    assert actions[0]["primary"] == "none"


def test_ready_crystal_base_prior_selects_crystals_and_uses_main_hand():
    value = observation()
    value["self"]["hotbar"][3] = _combat_item("end_crystal", 64)
    value["action_mask"].update({
        "combat_attack_ready": False,
        "crystal_place_ready": True,
        "crystal_attack_ready": False,
    })
    features = batch_observations([value])
    policy = CombatPolicy()
    with torch.no_grad():
        primary = policy.categorical_heads["head_primary"]
        primary.weight.zero_()
        primary.bias.zero_()
        hotbar = policy.categorical_heads["head_hotbar"]
        hotbar.weight.zero_()
        hotbar.bias.zero_()
    output = policy(features, policy.initial_hidden(1, "cpu"))
    actions, _, _, _ = sample_actions(output, features, deterministic=True)
    assert actions[0]["primary"] == "use_main"
    assert actions[0]["hotbar"] == 3


def test_ready_crystal_attack_prior_does_not_require_melee_readiness():
    value = observation()
    value["action_mask"].update({
        "combat_attack_ready": False,
        "crystal_place_ready": False,
        "crystal_attack_ready": True,
    })
    features = batch_observations([value])
    policy = CombatPolicy()
    with torch.no_grad():
        primary = policy.categorical_heads["head_primary"]
        primary.weight.zero_()
        primary.bias.zero_()
    output = policy(features, policy.initial_hidden(1, "cpu"))
    actions, _, _, _ = sample_actions(output, features, deterministic=True)
    assert actions[0]["primary"] == "attack"


def test_ready_block_break_prior_selects_pickaxe_and_attack():
    value = observation()
    value["self"]["hotbar"][1] = _combat_item(
        "diamond_pickaxe", 1, durability=1561, maximum=1561
    )
    value["action_mask"].update({
        "combat_attack_ready": False,
        "crystal_place_ready": False,
        "crystal_attack_ready": False,
        "tactical_block_break_ready": True,
    })
    features = batch_observations([value])
    policy = CombatPolicy()
    with torch.no_grad():
        primary = policy.categorical_heads["head_primary"]
        primary.weight.zero_()
        primary.bias.zero_()
        hotbar = policy.categorical_heads["head_hotbar"]
        hotbar.weight.zero_()
        hotbar.bias.zero_()
    output = policy(features, policy.initial_hidden(1, "cpu"))
    actions, _, _, _ = sample_actions(output, features, deterministic=True)
    assert actions[0]["primary"] == "attack"
    assert actions[0]["hotbar"] == 1


def test_block_break_prior_yields_to_ready_melee():
    value = observation()
    value["action_mask"].update({
        "combat_attack_ready": True,
        "crystal_place_ready": False,
        "crystal_attack_ready": False,
        "tactical_block_break_ready": True,
    })
    features = batch_observations([value])
    policy = CombatPolicy()
    with torch.no_grad():
        primary = policy.categorical_heads["head_primary"]
        primary.weight.zero_()
        primary.bias.zero_()
        hotbar = policy.categorical_heads["head_hotbar"]
        hotbar.weight.zero_()
        hotbar.bias.zero_()
    output = policy(features, policy.initial_hidden(1, "cpu"))
    actions, _, _, _ = sample_actions(output, features, deterministic=True)
    assert actions[0]["primary"] == "attack"
    assert actions[0]["hotbar"] == 0


def test_block_break_prior_yields_to_ready_crystal():
    value = observation()
    value["self"]["hotbar"][1] = _combat_item(
        "diamond_pickaxe", 1, durability=1561, maximum=1561
    )
    value["self"]["hotbar"][3] = _combat_item("end_crystal", 64)
    value["action_mask"].update({
        "combat_attack_ready": False,
        "crystal_place_ready": True,
        "crystal_attack_ready": False,
        "tactical_block_break_ready": True,
    })
    features = batch_observations([value])
    policy = CombatPolicy()
    with torch.no_grad():
        primary = policy.categorical_heads["head_primary"]
        primary.weight.zero_()
        primary.bias.zero_()
        hotbar = policy.categorical_heads["head_hotbar"]
        hotbar.weight.zero_()
        hotbar.bias.zero_()
    output = policy(features, policy.initial_hidden(1, "cpu"))
    actions, _, _, _ = sample_actions(output, features, deterministic=True)
    assert actions[0]["primary"] == "use_main"
    assert actions[0]["hotbar"] == 3


def test_crystal_retention_prior_acquires_reachable_base_before_clicking():
    value = _crystal_acquisition_observation()
    features = batch_observations([value])
    policy = CombatPolicy()
    with torch.no_grad():
        policy.camera_mean.weight.zero_()
        policy.camera_mean.bias.zero_()
        policy.categorical_heads["head_primary"].weight.zero_()
        policy.categorical_heads["head_primary"].bias.zero_()
        policy.categorical_heads["head_hotbar"].weight.zero_()
        policy.categorical_heads["head_hotbar"].bias.zero_()
    output = policy(features, policy.initial_hidden(1, "cpu"))
    actions, _, _, _ = sample_actions(output, features, deterministic=True)
    assert actions[0]["yaw_delta"] < -0.1
    assert actions[0]["primary"] == "none"
    assert actions[0]["hotbar"] == 3


def test_crystal_camera_prior_follows_randomized_pad_side():
    yaw_deltas = []
    for x in (2.0, -3.0):
        value = _crystal_acquisition_observation()
        value["blocks"][0]["relative_position"]["x"] = x
        features = batch_observations([value])
        policy = CombatPolicy()
        with torch.no_grad():
            policy.camera_mean.weight.zero_()
            policy.camera_mean.bias.zero_()
        output = policy(features, policy.initial_hidden(1, "cpu"))
        actions, _, _, _ = sample_actions(output, features, deterministic=True)
        yaw_deltas.append(actions[0]["yaw_delta"])

    assert yaw_deltas[0] < -0.1
    assert yaw_deltas[1] > 0.1


def test_crystal_retention_target_preempts_only_the_hand_authored_melee_prior():
    value = _crystal_acquisition_observation()
    value["action_mask"]["combat_attack_ready"] = True
    features = batch_observations([value])
    policy = CombatPolicy()
    with torch.no_grad():
        policy.camera_mean.weight.zero_()
        policy.camera_mean.bias.zero_()
        policy.categorical_heads["head_primary"].weight.zero_()
        policy.categorical_heads["head_primary"].bias.zero_()
        policy.categorical_heads["head_hotbar"].weight.zero_()
        policy.categorical_heads["head_hotbar"].bias.zero_()
    output = policy(features, policy.initial_hidden(1, "cpu"))
    actions, _, _, _ = sample_actions(output, features, deterministic=True)
    assert actions[0]["yaw_delta"] < -0.1
    assert actions[0]["primary"] == "none"
    assert actions[0]["hotbar"] == 3


def test_combined_lane_keeps_melee_priority_over_crystal_acquisition():
    value = _crystal_acquisition_observation()
    value["match"].update({"mode": "combined", "lane": "combined"})
    value["action_mask"]["combat_attack_ready"] = True
    features = batch_observations([value])
    policy = CombatPolicy()
    with torch.no_grad():
        policy.camera_mean.weight.zero_()
        policy.camera_mean.bias.zero_()
        policy.categorical_heads["head_primary"].weight.zero_()
        policy.categorical_heads["head_primary"].bias.zero_()
        policy.categorical_heads["head_hotbar"].weight.zero_()
        policy.categorical_heads["head_hotbar"].bias.zero_()
    output = policy(features, policy.initial_hidden(1, "cpu"))
    actions, _, _, _ = sample_actions(output, features, deterministic=True)
    assert actions[0]["yaw_delta"] == 0.0
    assert actions[0]["pitch_delta"] == 0.0
    assert actions[0]["primary"] == "attack"
    assert actions[0]["hotbar"] == 0


def test_crystal_camera_prior_yields_once_place_is_exactly_ready():
    value = _crystal_acquisition_observation()
    value["action_mask"]["crystal_place_ready"] = True
    features = batch_observations([value])
    policy = CombatPolicy()
    with torch.no_grad():
        policy.camera_mean.weight.zero_()
        policy.camera_mean.bias.zero_()
    output = policy(features, policy.initial_hidden(1, "cpu"))
    actions, _, _, _ = sample_actions(output, features, deterministic=True)
    assert actions[0]["yaw_delta"] == 0.0
    assert actions[0]["pitch_delta"] == 0.0


def test_crystal_camera_prior_corrects_legacy_egocentric_yaw_at_nonzero_headings():
    for yaw in (math.pi / 2, -math.pi / 2, math.pi / 4):
        value = _crystal_acquisition_observation()
        value["self"]["yaw"] = yaw
        value["blocks"] = []
        distance = 2.5
        value["entities"] = [{
            "kind": "end_crystal",
            # Worker legacy egocentric output for a world target directly in
            # front of this heading. The trainer compatibility rotation must
            # recover x=0,z=-distance before computing the camera delta.
            "relative_position": {
                "x": -math.sin(2 * yaw) * distance,
                "y": 0.62,
                "z": -math.cos(2 * yaw) * distance,
            },
            "relative_velocity": {"x": 0, "y": 0, "z": 0},
            "age_ticks": 1, "distance": distance, "raycastable": False,
        }]
        features = batch_observations([value])
        policy = CombatPolicy()
        with torch.no_grad():
            policy.camera_mean.weight.zero_()
            policy.camera_mean.bias.zero_()
        output = policy(features, policy.initial_hidden(1, "cpu"))
        actions, _, _, _ = sample_actions(output, features, deterministic=True)
        assert abs(actions[0]["yaw_delta"]) < 1e-5


def test_camera_sampling_is_incremental_and_log_probability_has_ppo_parity():
    features = batch_observations([observation()])
    policy = CombatPolicy()
    with torch.no_grad():
        policy.camera_mean.weight.zero_()
        policy.camera_mean.bias[:] = torch.tensor([100.0, -100.0])
    output = policy(features, policy.initial_hidden(1, "cpu"))
    deterministic, _, _, _ = sample_actions(output, features, deterministic=True)
    assert abs(deterministic[0]["yaw_delta"]) <= CAMERA_SCALE[0] + 1e-6
    assert abs(deterministic[0]["pitch_delta"]) <= CAMERA_SCALE[1] + 1e-6

    with torch.no_grad():
        policy.camera_mean.bias.zero_()
    output = policy(features, policy.initial_hidden(1, "cpu"))
    _, sampled, sampled_log_probability, _ = sample_actions(output, features)
    evaluated_log_probability, _ = evaluate_actions(output, features, sampled)
    assert torch.allclose(sampled_log_probability, evaluated_log_probability, atol=1e-5)


def test_crystal_action_and_camera_priors_have_ppo_log_probability_parity():
    features = batch_observations([_crystal_acquisition_observation()])
    policy = CombatPolicy()
    output = policy(features, policy.initial_hidden(1, "cpu"))

    _, sampled, sampled_log_probability, _ = sample_actions(output, features)
    evaluated_log_probability, _ = evaluate_actions(output, features, sampled)

    assert torch.allclose(sampled_log_probability, evaluated_log_probability, atol=1e-5)


def test_assigned_opponent_geometry_turns_and_approaches_outside_melee_range():
    value = observation()
    value["action_mask"].update({
        "combat_attack_ready": False,
        "crystal_place_ready": False,
        "crystal_attack_ready": False,
        "tactical_block_break_ready": False,
    })
    value["opponent"].update({
        "body_relative_position": {"x": 4.0, "y": 0.0, "z": -4.0},
        "body_relative_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
        "distance": math.sqrt(32), "horizontal_distance": math.sqrt(32),
        "bearing_error": -math.pi / 4, "pitch_error": 0.0,
        "closing_speed": 0.0, "within_melee_reach": False,
        "aim_alignment": math.sqrt(0.5), "facing_toward_self": 0.0,
    })
    features = batch_observations([value])
    policy = CombatPolicy()
    with torch.no_grad():
        policy.camera_mean.weight.zero_()
        policy.camera_mean.bias.zero_()
        policy.categorical_heads["head_forward"].weight.zero_()
        policy.categorical_heads["head_forward"].bias.zero_()
        policy.categorical_heads["head_sprint"].weight.zero_()
        policy.categorical_heads["head_sprint"].bias.zero_()
    output = policy(features, policy.initial_hidden(1, "cpu"))
    actions, _, _, _ = sample_actions(output, features, deterministic=True)
    assert actions[0]["yaw_delta"] < -0.1
    assert actions[0]["forward"] == 1
    assert actions[0]["sprint"] is True


def test_centered_opponent_blend_attenuates_learned_camera_drift():
    value = observation()
    value["opponent"].update({
        "body_relative_position": {"x": 0.0, "y": 0.0, "z": -5.0},
        "distance": 5.0, "horizontal_distance": 5.0,
        "bearing_error": 0.0, "pitch_error": 0.0,
        "closing_speed": 0.0, "within_melee_reach": False,
        "aim_alignment": 1.0, "facing_toward_self": 0.0,
    })
    features = batch_observations([value])
    learned_mean = torch.tensor([[0.6, -0.4]])

    blended = camera_action_mean(learned_mean, features)

    assert torch.allclose(
        blended, learned_mean * (1.0 - OPPONENT_CAMERA_PRIOR_WEIGHT), atol=1e-6
    )


def test_off_axis_opponent_camera_is_a_latent_space_blend():
    value = observation()
    bearing_error = -math.pi / 4
    pitch_error = 0.12
    value["opponent"].update({
        "body_relative_position": {"x": 4.0, "y": 0.7, "z": -4.0},
        "distance": 5.7, "horizontal_distance": math.sqrt(32),
        "bearing_error": bearing_error, "pitch_error": pitch_error,
        "closing_speed": 0.0, "within_melee_reach": False,
        "aim_alignment": 0.7, "facing_toward_self": 0.0,
    })
    features = batch_observations([value])
    learned_mean = torch.tensor([[0.25, -0.15]])
    scale = torch.tensor(CAMERA_SCALE)
    target = torch.atanh(torch.tensor([
        bearing_error / scale[0], pitch_error / scale[1],
    ]).clamp(-0.95, 0.95)).unsqueeze(0)

    blended = camera_action_mean(learned_mean, features)
    expected = (
        learned_mean * (1.0 - OPPONENT_CAMERA_PRIOR_WEIGHT)
        + target * OPPONENT_CAMERA_PRIOR_WEIGHT
    )

    assert torch.allclose(blended, expected, atol=1e-6)


def test_assigned_opponent_prior_has_sample_evaluate_log_probability_parity():
    value = observation()
    value["opponent"].update({
        "body_relative_position": {"x": -3.0, "y": 0.5, "z": -5.0},
        "distance": 5.85, "horizontal_distance": 5.83,
        "bearing_error": 0.54, "pitch_error": 0.08,
        "closing_speed": 0.2, "within_melee_reach": False,
        "aim_alignment": 0.85, "facing_toward_self": 0.7,
    })
    features = batch_observations([value])
    policy = CombatPolicy()
    output = policy(features, policy.initial_hidden(1, "cpu"))
    _, sampled, sampled_log_probability, _ = sample_actions(output, features)
    evaluated_log_probability, _ = evaluate_actions(output, features, sampled)
    assert torch.allclose(sampled_log_probability, evaluated_log_probability, atol=1e-5)


def test_tactical_build_prior_selects_obsidian_aims_support_and_uses_main():
    value = _tactical_build_observation(raycastable=True)
    features = batch_observations([value])
    policy = CombatPolicy()
    with torch.no_grad():
        for name in ("head_primary", "head_hotbar"):
            policy.categorical_heads[name].weight.zero_()
            policy.categorical_heads[name].bias.zero_()
        policy.camera_mean.weight.zero_()
        policy.camera_mean.bias.zero_()

    output = policy(features, policy.initial_hidden(1, "cpu"))
    actions, _, _, _ = sample_actions(output, features, deterministic=True)

    assert actions[0]["primary"] == "use_main"
    assert actions[0]["hotbar"] == 3
    assert actions[0]["yaw_delta"] < -0.1
    assert actions[0]["pitch_delta"] < -0.05


def test_tactical_build_acquisition_aims_and_selects_before_raycast_is_ready():
    value = _tactical_build_observation(raycastable=False)
    features = batch_observations([value])
    policy = CombatPolicy()
    with torch.no_grad():
        for name in ("head_primary", "head_hotbar"):
            policy.categorical_heads[name].weight.zero_()
            policy.categorical_heads[name].bias.zero_()
        policy.camera_mean.weight.zero_()
        policy.camera_mean.bias.zero_()

    output = policy(features, policy.initial_hidden(1, "cpu"))
    actions, _, _, _ = sample_actions(output, features, deterministic=True)

    assert actions[0]["primary"] == "none"
    assert actions[0]["hotbar"] == 3
    assert abs(actions[0]["yaw_delta"]) > 0.1


def test_worker_build_support_preempts_generated_base_acquisition():
    value = _tactical_build_observation(raycastable=True)
    value["self"]["hotbar"][2] = _combat_item("end_crystal", 64)
    value["blocks"].append({
        "name": "obsidian",
        "relative_position": {"x": -1.0, "y": -1.0, "z": -2.0},
        "collision": "solid", "hardness": 50, "replaceable": False,
        "break_progress": 0, "crystal_clearance": True, "exposed_faces": 5,
        "distance": 2.4, "within_reach": True, "raycastable": False,
        "sample_age_ticks": 0,
    })
    features = batch_observations([value])
    policy = CombatPolicy()
    with torch.no_grad():
        for name in ("head_primary", "head_hotbar"):
            policy.categorical_heads[name].weight.zero_()
            policy.categorical_heads[name].bias.zero_()
        policy.camera_mean.weight.zero_()
        policy.camera_mean.bias.zero_()

    output = policy(features, policy.initial_hidden(1, "cpu"))
    actions, _, _, _ = sample_actions(output, features, deterministic=True)

    assert actions[0]["primary"] == "use_main"
    assert actions[0]["hotbar"] == 3
    # The support is on the left in body coordinates; the generated base is on
    # the right. A negative turn proves the camera followed the support marker.
    assert actions[0]["yaw_delta"] < -0.1


def test_melee_and_crystal_readiness_preempt_tactical_build():
    melee = _tactical_build_observation(raycastable=True)
    melee["action_mask"]["combat_attack_ready"] = True
    crystal = _tactical_build_observation(raycastable=True)
    crystal["self"]["hotbar"][2] = _combat_item("end_crystal", 64)
    crystal["action_mask"]["crystal_place_ready"] = True
    crystal_attack = _tactical_build_observation(raycastable=True)
    crystal_attack["action_mask"]["crystal_attack_ready"] = True
    features = batch_observations([melee, crystal, crystal_attack])
    policy = CombatPolicy()
    with torch.no_grad():
        for name in ("head_primary", "head_hotbar"):
            policy.categorical_heads[name].weight.zero_()
            policy.categorical_heads[name].bias.zero_()

    output = policy(features, policy.initial_hidden(3, "cpu"))
    actions, _, _, _ = sample_actions(output, features, deterministic=True)

    assert actions[0]["primary"] == "attack"
    assert actions[0]["hotbar"] == 0
    assert actions[1]["primary"] == "use_main"
    assert actions[1]["hotbar"] == 2
    assert actions[2]["primary"] == "attack"
    assert actions[2]["hotbar"] != 3


def test_live_crystal_entity_acquisition_preempts_tactical_build_support():
    value = _tactical_build_observation(raycastable=True)
    value["entities"] = [{
        "kind": "end_crystal",
        "relative_position": {"x": -1.0, "y": 0.0, "z": -2.0},
        "relative_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
        "age_ticks": 2, "distance": 2.3, "raycastable": False,
    }]
    features = batch_observations([value])
    policy = CombatPolicy()
    with torch.no_grad():
        policy.categorical_heads["head_hotbar"].weight.zero_()
        policy.categorical_heads["head_hotbar"].bias.zero_()
        policy.camera_mean.weight.zero_()
        policy.camera_mean.bias.zero_()

    output = policy(features, policy.initial_hidden(1, "cpu"))
    actions, _, _, _ = sample_actions(output, features, deterministic=True)

    assert actions[0]["hotbar"] != 3
    assert actions[0]["yaw_delta"] > 0.1


def test_tactical_build_priors_have_sample_evaluate_log_probability_parity():
    features = batch_observations([_tactical_build_observation(raycastable=True)])
    policy = CombatPolicy()
    output = policy(features, policy.initial_hidden(1, "cpu"))

    _, sampled, sampled_log_probability, _ = sample_actions(output, features)
    evaluated_log_probability, _ = evaluate_actions(output, features, sampled)

    assert torch.allclose(sampled_log_probability, evaluated_log_probability, atol=1e-5)


def test_using_and_releasing_cannot_be_sampled_together():
    value = observation()
    value["action_mask"]["release_use"] = True
    features = batch_observations([value])
    policy = CombatPolicy()
    with torch.no_grad():
        policy.categorical_heads["head_primary"].bias[:] = torch.tensor([0.0, 0.0, 100.0, 0.0])
        policy.categorical_heads["head_release_use"].bias[:] = torch.tensor([0.0, 100.0])
    output = policy(features, policy.initial_hidden(1, "cpu"))
    actions, _, _, _ = sample_actions(output, features, deterministic=True)
    assert actions[0]["primary"] == "use_main"
    assert actions[0]["release_use"] is False


def _crystal_acquisition_observation() -> dict:
    value = observation()
    value["match"].update({"mode": "crystal", "lane": "crystal_retention"})
    value["self"]["hotbar"][3] = {
        "name": "end_crystal", "count": 64, "durability": 0,
        "max_durability": 0, "enchant_hash": 0,
    }
    value["action_mask"].update({
        "combat_attack_ready": False,
        "crystal_place_ready": False,
        "crystal_attack_ready": False,
    })
    value["blocks"] = [{
        "name": "obsidian", "relative_position": {"x": 2.0, "y": -1.0, "z": -1.0},
        "collision": "solid", "hardness": 50, "replaceable": False,
        "break_progress": 0, "crystal_clearance": True, "exposed_faces": 5,
        "distance": 2.3, "within_reach": True, "raycastable": False,
        "sample_age_ticks": 0,
    }]
    return value


def _tactical_build_observation(*, raycastable: bool) -> dict:
    value = observation()
    value["match"].update({"mode": "terrain", "lane": "terrain"})
    value["action_mask"].update({
        "combat_attack_ready": False,
        "crystal_place_ready": False,
        "crystal_attack_ready": False,
        "tactical_block_break_ready": False,
    })
    value["self"]["hotbar"][3] = _combat_item("obsidian", 64)
    value["blocks"] = [{
        "name": "stone",
        "relative_position": {"x": 1.0, "y": -1.0, "z": -2.0},
        "collision": "solid",
        "hardness": 1.5,
        "replaceable": False,
        "break_progress": 0,
        "crystal_clearance": False,
        "exposed_faces": 5,
        "distance": 2.4,
        "within_reach": True,
        "raycastable": raycastable,
        "sample_age_ticks": 0,
        "tactical_placement_target": True,
    }]
    return value


def _combat_item(
    name: str, count: int, *, durability: int = 0, maximum: int = 0,
) -> dict:
    return {
        "name": name, "count": count, "durability": durability,
        "max_durability": maximum, "enchant_hash": 0,
    }


def _v2_candidate_observation(*, place: bool = False, detonate: bool = False) -> dict:
    value = observation()
    value["schema_version"] = 2
    value["self"]["hotbar"][1] = _combat_item("end_crystal", 64)
    value["self"]["hotbar"][2] = _combat_item("obsidian", 64)
    value["self"]["hotbar"][3] = _combat_item("diamond_pickaxe", 1)
    value["action_mask"].update({
        "combat_attack_ready": True,
        "crystal_place_ready": place,
        "crystal_attack_ready": detonate,
        "tactical_block_break_ready": True,
    })
    value["tactical"] = {
        "crystal_candidates": [
            {
                "kind": "crystal", "source_index": 0, "distance": 2.2,
                "reachable": True, "visible": True, "placement_legal": False,
                "estimated_opponent_damage": 8, "estimated_self_damage": 1,
                "pop_potential": 0, "escape_direction": 1,
                "body_relative_position": {"x": 1, "y": 0, "z": -2},
            },
            {
                "kind": "base", "source_index": 1, "distance": 2.6,
                "reachable": True, "visible": True, "placement_legal": True,
                "estimated_opponent_damage": 9, "estimated_self_damage": 1,
                "pop_potential": 1, "escape_direction": -1,
                "body_relative_position": {"x": -1, "y": -1, "z": -2},
            },
        ],
        "block_candidates": [{
            "source_index": 0, "distance": 2, "purpose": "mine_path",
            "reachable": True, "visible": True, "cover_value": 0.2,
            "followup_crystal_viability": 0.5,
            "body_relative_position": {"x": 0, "y": 0, "z": -2},
        }],
        "recent_history": [],
        "survival": {"has_totem": True, "spare_totems": 1, "heal_available": False},
        "threat": {"score": 0.2},
    }
    return value


def _zero_hierarchy(policy: CombatPolicy) -> None:
    with torch.no_grad():
        for name in ("head_intent", "head_target_index"):
            policy.categorical_heads[name].weight.zero_()
            policy.categorical_heads[name].bias.zero_()


def test_immediate_crystal_intents_outrank_sword_and_select_only_matching_kind():
    features = batch_observations([
        _v2_candidate_observation(place=True),
        _v2_candidate_observation(place=True, detonate=True),
        _v2_candidate_observation(),
    ])
    policy = CombatPolicy()
    _zero_hierarchy(policy)
    with torch.no_grad():
        # Try to force no-target and candidate zero. Legal masks must still pick
        # the base at index one for placement.
        policy.categorical_heads["head_target_index"].bias[0] = 100
        policy.categorical_heads["head_target_index"].bias[1] = 90
    output = policy(features, policy.initial_hidden(3, "cpu"))
    actions, sampled, sampled_log_probability, _ = sample_actions(
        output, features, deterministic=True
    )
    assert (actions[0]["intent"], actions[0]["target_index"]) == ("crystal_place", 1)
    assert (actions[1]["intent"], actions[1]["target_index"]) == ("crystal_detonate", 0)
    assert (actions[2]["intent"], actions[2]["target_index"]) == ("sword_engage", -1)
    evaluated, _ = evaluate_actions(output, features, sampled)
    assert torch.allclose(sampled_log_probability, evaluated, atol=1e-5)


def test_acquire_and_all_other_implicit_intents_accept_only_minus_one_target():
    features = batch_observations([_v2_candidate_observation()])
    policy = CombatPolicy()
    _zero_hierarchy(policy)
    masks = categorical_masks(features)
    logits = torch.zeros((1, 17))
    for intent_name in (
        "sword_engage", "crystal_acquire", "heal_retotem", "disengage", "reposition",
    ):
        intent = torch.tensor([INTENT_NAMES.index(intent_name)])
        target_mask = _hierarchical_mask(
            "target_index", logits, masks, {"intent": intent}, features
        )
        assert target_mask[0, 0]
        assert not target_mask[0, 1:].any()


def test_conditional_intents_have_disjoint_candidate_target_masks():
    features = batch_observations([_v2_candidate_observation(place=True, detonate=True)])
    masks = categorical_masks(features)
    logits = torch.zeros((1, 17))
    expected = {
        "crystal_place": 2,      # internal 0=-1, base candidate one => 2
        "crystal_detonate": 1,   # crystal candidate zero => 1
        "build_pad": 1,
        "mine_path": 1,
    }
    for intent_name, only_index in expected.items():
        intent = torch.tensor([INTENT_NAMES.index(intent_name)])
        target_mask = _hierarchical_mask(
            "target_index", logits, masks, {"intent": intent}, features
        )
        assert target_mask[0].nonzero().flatten().tolist() == [only_index]
