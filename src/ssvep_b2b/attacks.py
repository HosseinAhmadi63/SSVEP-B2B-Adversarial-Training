from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

PAPER_ATTACK_DEFAULTS: dict[str, dict[str, float | int]] = {
    "fgsm": {"epsilon": 0.01},
    "bim": {"epsilon": 0.01, "step_size": 0.001, "iterations": 50},
    "pgd": {"epsilon": 0.01, "step_size": 0.001, "iterations": 50},
    "mim": {
        "epsilon": 0.01,
        "step_size": 0.001,
        "iterations": 50,
        "decay": 0.5,
    },
    "cw": {
        "epsilon": 0.01,
        "c": 0.001,
        "confidence": 10.0,
        "iterations": 1000,
        "learning_rate": 0.01,
    },
}


@contextmanager
def _frozen_model(model: nn.Module) -> Iterator[None]:
    modes = tuple((module, module.training) for module in model.modules())
    gradients = tuple((parameter, parameter.requires_grad) for parameter in model.parameters())
    try:
        model.eval()
        for parameter, _ in gradients:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, required in gradients:
            parameter.requires_grad_(required)
        for module, training in modes:
            module.training = training


def _validate_attack(
    inputs: Tensor,
    epsilon: float,
    clip_min: float | None,
    clip_max: float | None,
) -> None:
    if not inputs.is_floating_point():
        raise TypeError("inputs must use a floating-point dtype")
    if inputs.ndim < 2:
        raise ValueError("inputs must include batch and feature dimensions")
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    if clip_min is not None and clip_max is not None and clip_min > clip_max:
        raise ValueError("clip_min must not exceed clip_max")


def _targets(labels: Tensor, batch_size: int, device: torch.device) -> Tensor:
    if labels.ndim == 0 or labels.shape[0] != batch_size:
        raise ValueError("labels must have the same batch size as inputs")
    if labels.ndim == 1:
        targets = labels
    elif labels.shape[-1] == 1:
        targets = labels.reshape(batch_size)
    else:
        targets = labels.argmax(dim=-1)
    return targets.to(device=device, dtype=torch.long)


def _clip(inputs: Tensor, clip_min: float | None, clip_max: float | None) -> Tensor:
    clipped = inputs
    if clip_min is not None:
        clipped = clipped.clamp_min(clip_min)
    if clip_max is not None:
        clipped = clipped.clamp_max(clip_max)
    return clipped


def _project_linf(
    candidate: Tensor,
    clean: Tensor,
    epsilon: float,
    clip_min: float | None,
    clip_max: float | None,
) -> Tensor:
    perturbation = (candidate - clean).clamp(min=-epsilon, max=epsilon)
    return _clip(clean + perturbation, clip_min, clip_max)


def _loss_gradient(model: nn.Module, inputs: Tensor, targets: Tensor) -> Tensor:
    with torch.enable_grad():
        differentiable = inputs.detach().requires_grad_(True)
        logits = model(differentiable)
        if logits.ndim != 2 or logits.shape[0] != differentiable.shape[0]:
            raise ValueError("model must return logits with shape (batch, classes)")
        loss = F.cross_entropy(logits, targets, reduction="sum")
        gradient = torch.autograd.grad(loss, differentiable, only_inputs=True)[0]
    return gradient.detach()


def fgsm(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    epsilon: float = 0.01,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> Tensor:
    _validate_attack(inputs, epsilon, clip_min, clip_max)
    clean = inputs.detach()
    targets = _targets(labels, clean.shape[0], clean.device)
    with _frozen_model(model):
        gradient = _loss_gradient(model, clean, targets)
        adversarial = _project_linf(
            clean + epsilon * gradient.sign(), clean, epsilon, clip_min, clip_max
        )
    return adversarial.detach()


def _iterative_sign_attack(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    epsilon: float,
    step_size: float,
    iterations: int,
    clip_min: float | None,
    clip_max: float | None,
    decay: float | None,
) -> Tensor:
    _validate_attack(inputs, epsilon, clip_min, clip_max)
    if step_size < 0:
        raise ValueError("step_size must be nonnegative")
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    if decay is not None and decay < 0:
        raise ValueError("decay must be nonnegative")
    clean = inputs.detach()
    targets = _targets(labels, clean.shape[0], clean.device)
    adversarial = clean.clone()
    momentum = torch.zeros_like(clean)
    with _frozen_model(model):
        for _ in range(iterations):
            gradient = _loss_gradient(model, adversarial, targets)
            if decay is not None:
                dimensions = tuple(range(1, gradient.ndim))
                scale = gradient.abs().sum(dim=dimensions, keepdim=True)
                scale = scale.clamp_min(torch.finfo(gradient.dtype).eps)
                momentum = decay * momentum + gradient / scale
                direction = momentum.sign()
            else:
                direction = gradient.sign()
            adversarial = _project_linf(
                adversarial + step_size * direction,
                clean,
                epsilon,
                clip_min,
                clip_max,
            ).detach()
    return adversarial.detach()


def bim(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    epsilon: float = 0.01,
    step_size: float = 0.001,
    iterations: int = 50,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> Tensor:
    return _iterative_sign_attack(
        model,
        inputs,
        labels,
        epsilon,
        step_size,
        iterations,
        clip_min,
        clip_max,
        None,
    )


def pgd(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    epsilon: float = 0.01,
    step_size: float = 0.001,
    iterations: int = 50,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> Tensor:
    return _iterative_sign_attack(
        model,
        inputs,
        labels,
        epsilon,
        step_size,
        iterations,
        clip_min,
        clip_max,
        None,
    )


def mim(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    epsilon: float = 0.01,
    step_size: float = 0.001,
    iterations: int = 50,
    decay: float = 0.5,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> Tensor:
    return _iterative_sign_attack(
        model,
        inputs,
        labels,
        epsilon,
        step_size,
        iterations,
        clip_min,
        clip_max,
        decay,
    )


def _cw_margin(logits: Tensor, targets: Tensor, confidence: float) -> Tensor:
    if logits.shape[1] < 2:
        raise ValueError("C&W requires at least two output classes")
    true_logits = logits.gather(1, targets[:, None]).squeeze(1)
    target_mask = F.one_hot(targets, num_classes=logits.shape[1]).bool()
    other_logits = logits.masked_fill(target_mask, -torch.inf).max(dim=1).values
    return (true_logits - other_logits + confidence).clamp_min(0.0)


def cw_l2(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    epsilon: float = 0.01,
    c: float = 0.001,
    confidence: float = 10.0,
    iterations: int = 1000,
    learning_rate: float = 0.01,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> Tensor:
    _validate_attack(inputs, epsilon, clip_min, clip_max)
    if c < 0:
        raise ValueError("c must be nonnegative")
    if confidence < 0:
        raise ValueError("confidence must be nonnegative")
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    if learning_rate < 0:
        raise ValueError("learning_rate must be nonnegative")
    clean = inputs.detach()
    targets = _targets(labels, clean.shape[0], clean.device)
    perturbation = torch.zeros_like(clean, requires_grad=True)
    optimizer = torch.optim.Adam([perturbation], lr=learning_rate)
    best = clean.clone()
    best_norm = torch.full(
        (clean.shape[0],), torch.inf, dtype=clean.dtype, device=clean.device
    )
    best_found = torch.zeros(clean.shape[0], dtype=torch.bool, device=clean.device)
    with _frozen_model(model), torch.enable_grad():
        for _ in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            adversarial = _project_linf(
                clean + perturbation, clean, epsilon, clip_min, clip_max
            )
            logits = model(adversarial)
            if logits.ndim != 2 or logits.shape[0] != clean.shape[0]:
                raise ValueError("model must return logits with shape (batch, classes)")
            squared_l2 = (adversarial - clean).flatten(start_dim=1).square().sum(dim=1)
            margin = _cw_margin(logits, targets, confidence)
            loss = (squared_l2 + c * margin).mean()
            with torch.no_grad():
                successful = logits.argmax(dim=1).ne(targets)
                better = successful & squared_l2.lt(best_norm)
                selector = better.reshape((-1,) + (1,) * (clean.ndim - 1))
                best = torch.where(selector, adversarial.detach(), best)
                best_norm = torch.where(better, squared_l2.detach(), best_norm)
                best_found |= successful
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                projected = _project_linf(
                    clean + perturbation, clean, epsilon, clip_min, clip_max
                )
                perturbation.copy_(projected - clean)
        with torch.no_grad():
            final = _project_linf(
                clean + perturbation, clean, epsilon, clip_min, clip_max
            )
            final_logits = model(final)
            final_norm = (final - clean).flatten(start_dim=1).square().sum(dim=1)
            final_success = final_logits.argmax(dim=1).ne(targets)
            final_better = final_success & final_norm.lt(best_norm)
            selector = final_better.reshape((-1,) + (1,) * (clean.ndim - 1))
            best = torch.where(selector, final, best)
            best_found |= final_success
            selector = best_found.reshape((-1,) + (1,) * (clean.ndim - 1))
            result = torch.where(selector, best, final)
    return result.detach()


def _canonical_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "fgsm": "fgsm",
        "fast_gradient_sign_method": "fgsm",
        "bim": "bim",
        "basic_iterative_method": "bim",
        "pgd": "pgd",
        "projected_gradient_descent": "pgd",
        "mim": "mim",
        "momentum_iterative_method": "mim",
        "cw": "cw",
        "c&w": "cw",
        "c_w": "cw",
        "cw_l2": "cw",
        "carlini_wagner": "cw",
    }
    if normalized not in aliases:
        raise ValueError(f"unknown attack: {name}")
    return aliases[normalized]


def _normalized_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    aliases = {
        "eps": "epsilon",
        "alpha": "step_size",
        "steps": "iterations",
        "t": "iterations",
        "mu": "decay",
        "momentum": "decay",
        "kappa": "confidence",
        "delta": "epsilon",
        "delta_limit": "epsilon",
        "lr": "learning_rate",
    }
    normalized = {aliases.get(key, key): value for key, value in parameters.items()}
    random_start = normalized.pop("random_start", False)
    if random_start:
        raise ValueError("this implementation uses a zero-start iterative attack")
    norm = str(normalized.pop("norm", "l2")).lower().replace("_", "")
    if norm not in {"l2", "2"}:
        raise ValueError("C&W is implemented with the L2 objective")
    return normalized


def attack_defaults(name: str) -> dict[str, float | int]:
    return dict(PAPER_ATTACK_DEFAULTS[_canonical_name(name)])


def generate_attack(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    attack: str,
    *,
    config: Mapping[str, Any] | None = None,
    iterations: int | None = None,
    **overrides: Any,
) -> Tensor:
    canonical = _canonical_name(attack)
    parameters: dict[str, Any] = attack_defaults(canonical)
    if config is not None:
        parameters.update(_normalized_parameters(config))
    parameters.update(_normalized_parameters(overrides))
    if iterations is not None and canonical != "fgsm":
        parameters["iterations"] = iterations
    functions = {
        "fgsm": fgsm,
        "bim": bim,
        "pgd": pgd,
        "mim": mim,
        "cw": cw_l2,
    }
    return functions[canonical](model, inputs, labels, **parameters)


def _configuration_for(
    configs: Mapping[str, Mapping[str, Any]], name: str
) -> Mapping[str, Any]:
    canonical = _canonical_name(name)
    for key, value in configs.items():
        if _canonical_name(key) == canonical:
            return value
    return {}


def combine_attacks(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    attacks: Sequence[str],
    *,
    configs: Mapping[str, Mapping[str, Any]] | None = None,
    attack_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    iteration_overrides: Mapping[str, int] | None = None,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> Tensor:
    if not attacks:
        raise ValueError("at least one attack is required")
    canonical = tuple(_canonical_name(name) for name in attacks)
    if len(set(canonical)) != len(canonical):
        raise ValueError("attack combinations must not contain duplicates")
    if configs is not None and attack_kwargs is not None:
        raise ValueError("provide either configs or attack_kwargs")
    clean = inputs.detach()
    combined = clean.clone()
    configurations = configs or attack_kwargs or {}
    iteration_values = iteration_overrides or {}
    normalized_iterations = {
        _canonical_name(name): value for name, value in iteration_values.items()
    }
    for name, normalized in zip(attacks, canonical, strict=True):
        parameters = dict(_configuration_for(configurations, name))
        if clip_min is not None:
            parameters.setdefault("clip_min", clip_min)
        if clip_max is not None:
            parameters.setdefault("clip_max", clip_max)
        adversarial = generate_attack(
            model,
            clean,
            labels,
            name,
            config=parameters,
            iterations=normalized_iterations.get(normalized),
        )
        combined = combined + adversarial - clean
    return _clip(combined, clip_min, clip_max).detach()


additive_attack_combination = combine_attacks
