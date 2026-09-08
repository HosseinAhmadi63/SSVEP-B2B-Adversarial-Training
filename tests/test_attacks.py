import pytest
import torch
from torch import nn
from torch.nn import functional as F

from ssvep_b2b.attacks import (
    attack_defaults,
    bim,
    combine_attacks,
    cw_l2,
    fgsm,
    generate_attack,
    mim,
    pgd,
)


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.5, -0.5, -1.0],
                        [-1.0, 0.5, 1.0, -0.5],
                        [0.25, -1.0, 0.5, 1.0],
                    ]
                )
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs.flatten(start_dim=1))


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.tensor(
        [
            [[[0.3, 0.2], [-0.1, -0.2]]],
            [[[-0.2, 0.1], [0.3, -0.1]]],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1])
    return inputs, labels


def test_paper_attack_defaults() -> None:
    assert attack_defaults("FGSM") == {"epsilon": 0.01}
    assert attack_defaults("PGD") == {
        "epsilon": 0.01,
        "step_size": 0.001,
        "iterations": 50,
    }
    assert attack_defaults("MIM")["decay"] == 0.5
    assert attack_defaults("C&W")["confidence"] == 10.0
    assert attack_defaults("C&W")["iterations"] == 1000


def test_fgsm_is_untargeted_and_linf_bounded() -> None:
    model = TinyClassifier()
    inputs, labels = _batch()
    clean_loss = F.cross_entropy(model(inputs), labels)
    adversarial = fgsm(model, inputs, labels, epsilon=0.05)
    adversarial_loss = F.cross_entropy(model(adversarial), labels)
    assert adversarial_loss >= clean_loss
    assert (adversarial - inputs).abs().amax() <= 0.050001
    assert not adversarial.requires_grad


@pytest.mark.parametrize(
    ("attack", "parameters"),
    [
        (bim, {"iterations": 3, "step_size": 0.02}),
        (pgd, {"iterations": 3, "step_size": 0.02}),
        (mim, {"iterations": 3, "step_size": 0.02, "decay": 0.5}),
        (
            cw_l2,
            {
                "iterations": 4,
                "learning_rate": 0.02,
                "c": 1.0,
                "confidence": 0.0,
            },
        ),
    ],
)
def test_attacks_preserve_state_and_respect_budget(
    attack: object, parameters: dict[str, float | int]
) -> None:
    model = TinyClassifier()
    model.train()
    model.linear.weight.grad = torch.full_like(model.linear.weight, 2.0)
    original_gradient = model.linear.weight.grad.clone()
    inputs, labels = _batch()
    inputs.requires_grad_(True)
    adversarial = attack(model, inputs, labels, epsilon=0.05, **parameters)
    assert adversarial.shape == inputs.shape
    assert torch.isfinite(adversarial).all()
    assert (adversarial - inputs.detach()).abs().amax() <= 0.050001
    assert not adversarial.requires_grad
    assert inputs.grad is None
    assert model.training
    assert model.linear.weight.requires_grad
    assert torch.equal(model.linear.weight.grad, original_gradient)


def test_zero_start_pgd_matches_one_step_fgsm() -> None:
    model = TinyClassifier()
    inputs, labels = _batch()
    fgsm_result = fgsm(model, inputs, labels, epsilon=0.03)
    pgd_result = pgd(
        model,
        inputs,
        labels,
        epsilon=0.03,
        step_size=0.03,
        iterations=1,
    )
    assert torch.equal(fgsm_result, pgd_result)


def test_dispatcher_accepts_one_hot_labels_and_iteration_override() -> None:
    model = TinyClassifier()
    inputs, labels = _batch()
    one_hot = F.one_hot(labels, num_classes=3).float()
    dispatched = generate_attack(
        model,
        inputs,
        one_hot,
        "projected-gradient-descent",
        config={"eps": 0.04, "alpha": 0.01},
        iterations=2,
    )
    direct = pgd(
        model,
        inputs,
        labels,
        epsilon=0.04,
        step_size=0.01,
        iterations=2,
    )
    assert torch.equal(dispatched, direct)


def test_combination_adds_independent_clean_input_perturbations() -> None:
    model = TinyClassifier()
    inputs, labels = _batch()
    fgsm_result = fgsm(model, inputs, labels, epsilon=0.02)
    pgd_result = pgd(
        model,
        inputs,
        labels,
        epsilon=0.03,
        step_size=0.01,
        iterations=2,
    )
    combined = combine_attacks(
        model,
        inputs,
        labels,
        ["FGSM", "PGD"],
        configs={
            "FGSM": {"epsilon": 0.02},
            "PGD": {"epsilon": 0.03, "step_size": 0.01},
        },
        iteration_overrides={"PGD": 2},
    )
    expected = inputs + (fgsm_result - inputs) + (pgd_result - inputs)
    assert torch.allclose(combined, expected, atol=1.0e-7, rtol=0.0)


def test_attack_clipping_and_invalid_combination() -> None:
    model = TinyClassifier()
    inputs, labels = _batch()
    adversarial = fgsm(
        model,
        inputs,
        labels,
        epsilon=0.2,
        clip_min=-0.25,
        clip_max=0.25,
    )
    assert adversarial.min() >= -0.25
    assert adversarial.max() <= 0.25
    with pytest.raises(ValueError):
        combine_attacks(model, inputs, labels, ["FGSM", "fast gradient sign method"])
