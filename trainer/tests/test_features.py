import numpy as np

from combat_ai.features import BLOCK_SIZE, ENTITY_SIZE, LEGAL_SIZE, OPPONENT_SIZE, SELF_SIZE, encode_observation
from fixtures import observation


def test_feature_contract_has_fixed_shapes():
    encoded = encode_observation(observation())
    assert encoded["self_state"].shape == (SELF_SIZE,)
    assert encoded["opponent"].shape == (OPPONENT_SIZE,)
    assert encoded["entities"].shape == (16, ENTITY_SIZE)
    assert encoded["blocks"].shape == (48, BLOCK_SIZE)
    assert encoded["legal"].shape == (LEGAL_SIZE,)
    assert encoded["opponent_mask"].tolist() == [1.0]
    assert np.isfinite(encoded["self_state"]).all()


def test_missing_slots_are_explicitly_masked():
    value = observation()
    value["opponent"] = None
    encoded = encode_observation(value)
    assert encoded["opponent_mask"].sum() == 0
    assert encoded["entity_mask"].sum() == 0
    assert encoded["block_mask"].sum() == 0
