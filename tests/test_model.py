import pytest
import torch

from ssvep_b2b.model import CNNTCN, CausalResidualBlock, build_model


def test_model_output_shape_and_input_gradient() -> None:
    model = CNNTCN(num_classes=12, seed=17)
    inputs = torch.randn(3, 1, 8, 64, requires_grad=True)
    logits = model(inputs)
    assert logits.shape == (3, 12)
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_model_initialization_is_seed_deterministic() -> None:
    torch.manual_seed(3)
    first = build_model(num_classes=4, seed=91)
    torch.manual_seed(700)
    second = build_model(num_classes=4, seed=91)
    third = build_model(num_classes=4, seed=92)
    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters(), strict=True
    ):
        assert torch.equal(first_parameter, second_parameter)
    assert any(
        not torch.equal(first_parameter, third_parameter)
        for first_parameter, third_parameter in zip(
            first.parameters(), third.parameters(), strict=True
        )
    )


def test_causal_block_does_not_use_future_samples() -> None:
    block = CausalResidualBlock(channels=1, kernel_size=3)
    with torch.no_grad():
        block.convolution.weight.fill_(1.0)
        block.convolution.bias.zero_()
    reference = torch.zeros(1, 1, 8)
    changed = reference.clone()
    changed[:, :, 5:] = 1.0
    reference_output = block(reference)
    changed_output = block(changed)
    assert torch.equal(reference_output[:, :, :5], changed_output[:, :, :5])
    assert not torch.equal(reference_output[:, :, 5:], changed_output[:, :, 5:])


@pytest.mark.parametrize(
    "shape",
    [
        (2, 8, 64),
        (2, 2, 8, 64),
        (2, 1, 1, 64),
        (2, 1, 8, 1),
    ],
)
def test_model_rejects_invalid_input_shape(shape: tuple[int, ...]) -> None:
    model = CNNTCN(num_classes=4)
    with pytest.raises(ValueError):
        model(torch.randn(*shape))
