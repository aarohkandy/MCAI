import math

import torch

from combat_ai.distribution import sample_actions
from combat_ai.features import batch_observations
from combat_ai.model import CombatPolicy
from fixtures import observation


def test_policy_is_compact_and_outputs_legal_actions():
    policy = CombatPolicy()
    features = batch_observations([observation(), observation(tick=2)])
    output = policy(features, policy.initial_hidden(2, "cpu"))
    actions, _, log_probability, _ = sample_actions(output, features)
    assert policy.parameter_count < 1_000_000
    assert output.hidden.shape == (1, 2, 128)
    assert output.value.shape == (2,)
    assert torch.isfinite(log_probability).all()
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
