from __future__ import annotations

import json
import math

import torch

from combat_ai import introspect as intro
from combat_ai.features import (
    BLOCK_SIZE,
    ENTITY_SIZE,
    LEGAL_SIZE,
    OPPONENT_SIZE,
    SELF_SIZE,
    batch_observations,
    encode_observation,
)
from combat_ai.model import CATEGORICAL_SIZES, CombatPolicy
from fixtures import observation, parity_observation


def _span(fields, name):
    for field, start, end in fields:
        if field == name:
            return start, end
    raise AssertionError(f"no span named {name}")


def _slice(vector, fields, name):
    start, end = _span(fields, name)
    return vector[start:end]


# --------------------------------------------------------------------------
# the span table must stay pinned to the hand-written encoders
# --------------------------------------------------------------------------

def test_field_spans_tile_each_feature_vector():
    sizes = {"self": SELF_SIZE, "opponent": OPPONENT_SIZE, "entities": ENTITY_SIZE,
             "blocks": BLOCK_SIZE, "legal": LEGAL_SIZE}
    for group, fields in intro.GROUP_FIELDS.items():
        cursor = 0
        for _name, start, end in fields:
            assert start == cursor, f"{group} spans are not contiguous at {start}"
            assert end > start
            cursor = end
        assert cursor == sizes[group], f"{group} spans cover {cursor}, expected {sizes[group]}"


def test_named_spans_match_the_real_encoders():
    value = intro.make_observation(
        health=10.0, opponent_distance=2.4, attack_legal=False,
        blocks=[intro._obsidian_block(1.5)],
        entities=[{"kind": "end_crystal", "relative_position": {"x": 0.0, "y": 0.0, "z": -2.0},
                   "relative_velocity": {"x": 0.0, "y": 0.0, "z": 0.0}, "age_ticks": 0,
                   "distance": 2.0, "raycastable": True}],
    )
    encoded = encode_observation(value)
    assert _slice(encoded["self_state"], intro.SELF_FIELDS, "health")[0] == 0.5
    assert _slice(encoded["self_state"], intro.SELF_FIELDS, "on_ground")[0] == 1.0
    assert list(_slice(encoded["self_state"], intro.SELF_FIELDS, "active_hand")) == [1.0, 0.0, 0.0]
    assert list(_slice(encoded["self_state"], intro.SELF_FIELDS, "raycast_kind")) == [0.0, 0.0, 1.0]
    # SELF_SIZE leaves headroom past the last written field.
    assert not _slice(encoded["self_state"], intro.SELF_FIELDS, "unused").any()

    opponent = encoded["opponent"]
    position = _slice(opponent, intro.OPPONENT_FIELDS, "relative_position")
    assert position[0] == 0.0 and position[1] == 0.0
    assert position[2] == -2.4 / 12  # forward is -z in the egocentric frame
    assert _slice(opponent, intro.OPPONENT_FIELDS, "line_of_sight")[0] == 1.0
    assert _slice(opponent, intro.OPPONENT_FIELDS, "health_known")[0] == 1.0

    entity = encoded["entities"][0]
    assert list(_slice(entity, intro.ENTITY_FIELDS, "kind_one_hot")) == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert _slice(entity, intro.ENTITY_FIELDS, "raycastable")[0] == 1.0

    block = encoded["blocks"][0]
    assert _slice(block, intro.BLOCK_FIELDS, "crystal_clearance")[0] == 1.0
    assert _slice(block, intro.BLOCK_FIELDS, "crystal_base_name")[0] == 1.0
    assert list(_slice(block, intro.BLOCK_FIELDS, "collision")) == [0.0, 1.0, 0.0, 0.0]

    legal = encoded["legal"]
    assert _slice(legal, intro.LEGAL_FIELDS, "attack_legal")[0] == 0.0
    assert _slice(legal, intro.LEGAL_FIELDS, "use_main_legal")[0] == 1.0
    assert list(_slice(legal, intro.LEGAL_FIELDS, "hotbar_legal")) == [1.0] * 9


def test_hotbar_choice_names_cover_the_head():
    for name, size in CATEGORICAL_SIZES.items():
        assert len(intro.CHOICE_NAMES[name]) == size
    assert intro.CHOICE_NAMES["hotbar"][4] == "slot3_crystal"


# --------------------------------------------------------------------------
# 1. activation health
# --------------------------------------------------------------------------

def test_activation_report_covers_every_encoder_layer():
    policy = CombatPolicy()
    features = batch_observations([observation(), parity_observation()])
    report = intro.activation_report(policy, features)
    expected = {f"{name}.{layer}"
                for name in ("self_encoder", "opponent_encoder", "entity_encoder",
                             "block_encoder", "legal_encoder", "fusion")
                for layer in ("hidden", "output")}
    assert set(report["layers"]) == expected
    assert report["total_units"] == 2 * (128 + 96 + 96 + 64 + 64 + 64 + 64 + 64 + 32 + 24 + 192 + 192) // 2
    assert report["saturated_unit_fraction"] < 0.05
    for entry in report["layers"].values():
        assert -1.0 <= entry["mean"] <= 1.0
        assert entry["absolute_mean"] <= 1.0


def test_activation_report_detects_saturation():
    policy = CombatPolicy()
    with torch.no_grad():
        for module in (policy.self_encoder, policy.opponent_encoder, policy.fusion):
            module.layers[0].weight.mul_(500.0)
            module.layers[0].bias.add_(50.0)
    features = batch_observations(intro.default_observations())
    report = intro.activation_report(policy, features)
    assert report["layers"]["self_encoder.hidden"]["saturated_fraction"] > 0.9
    assert report["layers"]["self_encoder.hidden"]["saturated_units"] > 100
    assert report["total_saturated_units"] > 0
    codes = {finding["code"] for finding in intro.diagnose(
        {**intro.introspect(None), "activations": report})}
    assert "activations_saturated" in codes or "activations_saturating" in codes


# --------------------------------------------------------------------------
# 2. GRU memory
# --------------------------------------------------------------------------

def test_memory_report_on_a_live_gru():
    report = intro.memory_report(CombatPolicy(), intro.approach_sequence(12))
    assert report["steps"] == 12
    assert report["hidden_size"] == 128
    assert report["temporal_std_mean"] > 0.0
    assert report["effective_rank"] > 1.0
    assert report["constant_unit_fraction"] < 1.0


def test_memory_report_flags_a_collapsed_gru():
    policy = CombatPolicy()
    with torch.no_grad():
        for parameter in policy.memory.parameters():
            parameter.zero_()
    report = intro.memory_report(policy, intro.approach_sequence(8))
    # A zeroed GRU decays a zero hidden state to zero forever: no variance,
    # no rank, and resetting the state changes nothing.
    assert report["constant_unit_fraction"] == 1.0
    assert report["effective_rank"] == 0.0
    assert report["memory_influence"] == 0.0


# --------------------------------------------------------------------------
# 3. saliency
# --------------------------------------------------------------------------

def test_saliency_shares_sum_to_one_and_name_real_fields():
    policy = CombatPolicy()
    features = batch_observations(intro.default_observations())
    report = intro.saliency_report(policy, features)
    assert set(report) == {*CATEGORICAL_SIZES, "primary:attack", "camera_yaw", "value"}
    for entry in report.values():
        assert math.isclose(sum(group["share"] for group in entry["groups"].values()), 1.0, abs_tol=1e-5)
        for group, fields in intro.GROUP_FIELDS.items():
            assert set(entry["groups"][group]["fields"]) == {name for name, _s, _e in fields}
        assert entry["top_fields"][0]["share"] >= entry["top_fields"][-1]["share"]
        best = entry["top_fields_per_dimension"][0]
        assert best["share_per_dimension"] >= entry["top_fields"][0]["share_per_dimension"]
        assert best["share_per_dimension"] * best["width"] <= 1.0 + 1e-6
    attack = report["primary:attack"]["groups"]
    assert attack["opponent"]["fields"]["relative_position"]["magnitude"] > 0.0


def test_saliency_isolates_the_group_that_still_has_a_gradient_path():
    policy = CombatPolicy()
    with torch.no_grad():
        for module in (policy.self_encoder, policy.entity_encoder,
                       policy.block_encoder, policy.legal_encoder):
            module.layers[0].weight.zero_()
    features = batch_observations(intro.default_observations())
    report = intro.saliency_report(policy, features)["primary:attack"]["groups"]
    assert report["opponent"]["share"] > 0.99
    for group in ("self", "entities", "blocks", "legal"):
        assert report[group]["share"] < 0.01


def test_saliency_ignores_padded_entity_slots():
    features = batch_observations([intro.make_observation(entities=[])])
    report = intro.saliency_report(CombatPolicy(), features)
    assert report["value"]["groups"]["entities"]["magnitude"] == 0.0


# --------------------------------------------------------------------------
# 4. head entropy
# --------------------------------------------------------------------------

def test_untrained_heads_sit_at_maximum_entropy():
    features = batch_observations([observation(), parity_observation()])
    report = intro.head_report(CombatPolicy(), features)
    for name, entry in report["heads"].items():
        if entry["legal_choices"] > 1.5:
            assert entry["entropy_ratio"] > 0.9, name
        assert math.isclose(sum(entry["probabilities"].values()), 1.0, abs_tol=1e-5)
    assert "forward" in report["uniform_heads"]
    assert report["collapsed_heads"] == []
    assert report["camera"]["gaussian_entropy"] < report["camera"]["squashed_entropy_estimate"]


def test_head_report_detects_a_collapsed_head():
    policy = CombatPolicy()
    with torch.no_grad():
        policy.categorical_heads["head_forward"].bias[:] = torch.tensor([0.0, 0.0, 60.0])
    features = batch_observations([observation()])
    report = intro.head_report(policy, features)
    assert report["heads"]["forward"]["entropy_ratio"] < 0.05
    assert report["heads"]["forward"]["top_choice"] == "forward"
    assert report["collapsed_heads"] == ["forward"]


def test_max_entropy_respects_the_action_mask():
    value = observation()
    value["action_mask"]["hotbar"] = [True, False, False, False, False, False, False, False, False]
    report = intro.head_report(CombatPolicy(), batch_observations([value]))
    # "keep" plus one legal slot.
    assert report["heads"]["hotbar"]["legal_choices"] == 2.0
    assert math.isclose(report["heads"]["hotbar"]["max_entropy"], math.log(2), abs_tol=1e-5)


# --------------------------------------------------------------------------
# 5. value calibration
# --------------------------------------------------------------------------

def test_value_report_flags_a_constant_head():
    policy = CombatPolicy()
    with torch.no_grad():
        policy.value_head.weight.zero_()
        policy.value_head.bias.fill_(0.25)
    report = intro.value_report(policy, batch_observations(intro.default_observations()))
    assert report["near_constant"] is True
    assert math.isclose(report["mean"], 0.25, abs_tol=1e-6)
    assert report["range"] == 0.0


def test_value_report_on_a_live_head():
    report = intro.value_report(CombatPolicy(), batch_observations(intro.default_observations()))
    assert report["samples"] == len(intro.default_observations())
    assert report["max"] >= report["mean"] >= report["min"]


# --------------------------------------------------------------------------
# 6. behavioural probes
# --------------------------------------------------------------------------

def test_probe_report_returns_probabilities_for_every_scenario():
    report = intro.probe_report(CombatPolicy())
    assert set(report["scenarios"]) == set(intro.probe_observations())
    for entry in report["scenarios"].values():
        for head, probabilities in entry["heads"].items():
            assert math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-5), head
        assert -180.0 <= entry["camera_yaw_degrees"] <= 180.0
    assert set(report["signals"]) >= {
        "attack_lift_in_reach", "forward_lift_when_far", "crystal_slot_lift_on_spot",
        "attack_probability_when_illegal",
    }


def test_probes_cannot_attack_when_the_mask_forbids_it():
    report = intro.probe_report(CombatPolicy())
    assert report["scenarios"]["attack_illegal_in_reach"]["heads"]["primary"]["attack"] == 0.0
    assert report["signals"]["attack_probability_when_illegal"] == 0.0


def test_probe_signals_move_when_the_policy_learns_to_attack_in_reach():
    """A policy wired to fire on close opponents must show a positive lift, so
    the signal tracks real behaviour rather than being decorative."""
    policy = CombatPolicy()
    start, _end = _span(intro.OPPONENT_FIELDS, "relative_position")
    with torch.no_grad():
        for module in (policy.self_encoder, policy.entity_encoder,
                       policy.block_encoder, policy.legal_encoder):
            module.layers[0].weight.zero_()
            module.layers[0].bias.zero_()
        # Opponent relative z is -distance/12, so tanh(1 + z) is a proximity
        # signal: ~0.66 in melee range, negative when the opponent is far.
        policy.opponent_encoder.layers[0].weight.zero_()
        policy.opponent_encoder.layers[0].bias.zero_()
        policy.opponent_encoder.layers[0].weight[0, start + 2] = 1.0
        policy.opponent_encoder.layers[0].bias[0] = 1.0
        policy.opponent_encoder.layers[2].weight.zero_()
        policy.opponent_encoder.layers[2].bias.zero_()
        policy.opponent_encoder.layers[2].weight[0, 0] = 2.0
        # Fusion input 96 is the first opponent-encoder output (self takes 0..95).
        policy.fusion.layers[0].weight.zero_()
        policy.fusion.layers[0].bias.zero_()
        policy.fusion.layers[0].weight[0, 96] = 2.0
        policy.fusion.layers[2].weight.zero_()
        policy.fusion.layers[2].bias.zero_()
        policy.fusion.layers[2].weight[0, 0] = 2.0
        for parameter in policy.memory.parameters():
            parameter.zero_()
        # Candidate-gate row 0 reads fused unit 0; the reset/update gates stay at
        # sigmoid(0)=0.5, so hidden unit 0 becomes 0.5*tanh(2*proximity).
        policy.memory.weight_ih_l0[2 * 128, 0] = 2.0
        policy.categorical_heads["head_primary"].weight.zero_()
        policy.categorical_heads["head_primary"].bias.zero_()
        policy.categorical_heads["head_primary"].weight[1, 0] = 6.0
    report = intro.probe_report(policy)
    assert report["signals"]["attack_lift_in_reach"] > 0.1
    assert report["scenarios"]["opponent_in_reach"]["heads"]["primary"]["attack"] > \
        report["scenarios"]["opponent_far"]["heads"]["primary"]["attack"]


# --------------------------------------------------------------------------
# 7. comparison
# --------------------------------------------------------------------------

def test_compare_identical_policies_reports_no_movement():
    policy = CombatPolicy()
    clone = CombatPolicy()
    clone.load_state_dict(policy.state_dict())
    features = batch_observations(intro.default_observations())
    report = intro.compare_policies(policy, clone, features)
    assert report["total_l2_delta"] == 0.0
    assert len(report["unchanged_tensors"]) == len(policy.state_dict())
    for entry in report["head_shift"].values():
        assert entry["entropy_delta"] == 0.0
        assert entry["total_variation"] == 0.0


def test_compare_ranks_the_tensor_that_actually_moved():
    baseline = CombatPolicy()
    candidate = CombatPolicy()
    candidate.load_state_dict(baseline.state_dict())
    with torch.no_grad():
        candidate.categorical_heads["head_primary"].bias.add_(torch.tensor([0.0, 3.0, 0.0, 0.0]))
    features = batch_observations(intro.default_observations())
    report = intro.compare_policies(baseline, candidate, features)
    assert report["most_changed_tensors"][0]["tensor"] == "categorical_heads.head_primary.bias"
    assert report["total_l2_delta"] > 0.0
    assert report["head_shift"]["primary"]["entropy_delta"] < 0.0
    assert report["head_shift"]["primary"]["probability_delta"]["attack"] > 0.3
    assert report["head_shift"]["forward"]["total_variation"] == 0.0


# --------------------------------------------------------------------------
# top-level report, checkpoint loading and CLI
# --------------------------------------------------------------------------

def test_introspect_runs_on_a_random_policy_and_is_json_serialisable():
    report = intro.introspect(None)
    assert report["checkpoint"]["source"] == "random-initialisation"
    assert report["parameter_count"] == CombatPolicy().parameter_count
    assert set(report) == {
        "checkpoint", "parameter_count", "batch_size", "activations", "memory",
        "saliency", "policy_heads", "value", "probes", "diagnosis",
    }
    encoded = json.dumps(report)
    assert json.loads(encoded)["batch_size"] == report["batch_size"]
    codes = {finding["code"] for finding in report["diagnosis"]}
    # An untrained policy is uniform everywhere but must not be broken.
    assert "heads_uniform" in codes
    assert "illegal_action_leak" not in codes
    assert "activations_saturated" not in codes


def test_introspect_reads_a_checkpoint_and_its_metadata(tmp_path):
    policy = CombatPolicy()
    destination = tmp_path / "latest.pt"
    torch.save({
        "format_version": 1, "policy": policy.state_dict(), "optimizer": {},
        "policy_version": 7, "total_agent_ticks": 123456,
        "metrics": {"reward": 1.5, "note": "smoke", "device": torch.device("cpu")},
    }, destination)
    report = intro.introspect(destination)
    assert report["checkpoint"]["policy_version"] == 7
    assert report["checkpoint"]["total_agent_ticks"] == 123456
    assert report["checkpoint"]["metrics"]["reward"] == 1.5
    assert report["checkpoint"]["metrics"]["note"] == "smoke"
    json.dumps(report)


def test_introspect_with_comparison_and_summary(tmp_path):
    baseline = CombatPolicy()
    trained = CombatPolicy()
    trained.load_state_dict(baseline.state_dict())
    with torch.no_grad():
        trained.value_head.weight.mul_(4.0)
    first = tmp_path / "old.pt"
    second = tmp_path / "new.pt"
    torch.save({"policy": baseline.state_dict()}, first)
    torch.save({"policy": trained.state_dict()}, second)
    report = intro.introspect(second, compare=first)
    assert report["comparison"]["most_changed_tensors"][0]["tensor"] == "value_head.weight"
    assert report["comparison"]["value_after"]["std"] > report["comparison"]["value_before"]["std"]
    text = intro.summarize(report)
    for heading in ("DIAGNOSIS", "1. LAYER ACTIVATION HEALTH", "2. GRU MEMORY",
                    "3. INPUT SALIENCY", "4. HEAD ENTROPY", "5. VALUE FUNCTION",
                    "6. BEHAVIOURAL PROBES", "7. COMPARISON"):
        assert heading in text
    assert "crystal_spot_ready" in text
    assert "value_head.weight" in text


def test_run_cli_emits_text_and_json():
    text = intro.run_cli(None, None, False)
    assert text.startswith("=")
    payload = json.loads(intro.run_cli(None, None, True))
    assert payload["policy_heads"]["heads"]["primary"]["top_choice"] in intro.CHOICE_NAMES["primary"]
