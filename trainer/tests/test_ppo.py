import numpy as np
import torch

from combat_ai.buffer import Transition, prepare_sequences
from combat_ai.config import PPOConfig
from combat_ai.distribution import actions_from_wire
from combat_ai.features import encode_observation
from combat_ai.model import CombatPolicy
from combat_ai.ppo import PPOTrainer
from fixtures import observation


def test_recurrent_ppo_update_runs_on_padded_sequences():
    policy = CombatPolicy()
    wire = {"schema_version": 1, "forward": 0, "strafe": 0, "jump": False, "sprint": False,
            "sneak": False, "yaw_delta": 0.0, "pitch_delta": 0.0, "primary": "none",
            "release_use": False, "hotbar": -1, "swap_offhand": False}
    action = actions_from_wire([wire], "cpu")
    transitions = []
    for tick in range(5):
        transitions.append(Transition(
            agent_id="a", episode_id="e", policy_version=0,
            features=encode_observation(observation(tick=tick)), hidden=np.zeros(128, dtype=np.float32),
            categorical_action={name: int(value[0]) for name, value in action.categorical.items()},
            camera_action=np.zeros(2, dtype=np.float32), old_log_probability=-5.0, old_value=0.0,
            reward=0.01, done=tick == 4, next_value=0.0,
        ))
    batch = prepare_sequences(transitions, 4, 0.995, 0.95, "cpu")
    trainer = PPOTrainer(policy, PPOConfig(optimization_epochs=1, minibatch_samples=8), torch.device("cpu"))
    metrics = trainer.update(batch)
    assert metrics.valid_samples > 0
    assert np.isfinite(metrics.policy_loss)
