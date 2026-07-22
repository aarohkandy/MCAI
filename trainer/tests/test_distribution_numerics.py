from __future__ import annotations

import torch

from combat_ai.distribution import (
    _evaluate_categorical_group,
    _sample_categorical_group,
)


def test_masked_and_padded_categorical_entropy_has_finite_gradients():
    """Zero-probability actions must not poison PPO's entropy gradient.

    The first head contains a masked action and the differently sized heads
    force packed-group padding. Both paths create exact zero probabilities.
    A finite entropy value alone is insufficient: xlogy(0, 0), for example,
    has a NaN derivative which reaches every logit through softmax.
    """
    narrow_logits = torch.tensor([[0.2, -0.1]], requires_grad=True)
    wide_logits = torch.tensor([[0.3, -0.2, 0.1]], requires_grad=True)
    sampled = _sample_categorical_group(
        [
            (
                "narrow", narrow_logits,
                torch.tensor([[True, False]]),
            ),
            (
                "wide", wide_logits,
                torch.tensor([[True, True, True]]),
            ),
        ],
        deterministic=True,
        compute_entropy=True,
    )
    entropy = sampled["narrow"][2].sum() + sampled["wide"][2].sum()

    assert torch.isfinite(entropy)
    entropy.backward()

    assert narrow_logits.grad is not None
    assert wide_logits.grad is not None
    assert torch.isfinite(narrow_logits.grad).all()
    assert torch.isfinite(wide_logits.grad).all()


def test_masked_and_padded_evaluation_entropy_has_finite_gradients():
    narrow_logits = torch.tensor([[0.2, -0.1]], requires_grad=True)
    wide_logits = torch.tensor([[0.3, -0.2, 0.1]], requires_grad=True)
    evaluated = _evaluate_categorical_group(
        [
            ("narrow", narrow_logits, torch.tensor([[True, False]])),
            ("wide", wide_logits, torch.tensor([[True, True, True]])),
        ],
        {
            "narrow": torch.tensor([0]),
            "wide": torch.tensor([2]),
        },
    )
    entropy = evaluated["narrow"][1].sum() + evaluated["wide"][1].sum()

    assert torch.isfinite(entropy)
    entropy.backward()

    assert narrow_logits.grad is not None
    assert wide_logits.grad is not None
    assert torch.isfinite(narrow_logits.grad).all()
    assert torch.isfinite(wide_logits.grad).all()


def test_exponential_race_sampling_matches_categorical_probabilities():
    torch.manual_seed(19)
    expected = torch.tensor([0.1, 0.3, 0.6, 0.0])
    logits = expected.clamp_min(1e-12).log().expand(20_000, -1)
    mask = torch.tensor([[True, True, True, False]]).expand_as(logits)

    sampled = _sample_categorical_group(
        [("action", logits, mask)], deterministic=False, compute_entropy=False,
    )["action"][0]
    observed = torch.bincount(sampled, minlength=4).float() / sampled.numel()

    assert torch.allclose(observed, expected, atol=0.015)
