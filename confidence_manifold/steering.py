"""Activation steering for causal validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from confidence_manifold.model import find_layers


@dataclass
class SteeringResult:
    """Result from steering intervention."""

    alpha: float
    generated_text: str
    baseline_text: str
    changed: bool


@dataclass
class SweepResult:
    """Results from alpha sweep."""

    alphas: Float[np.ndarray, "n"]
    error_rates: Float[np.ndarray, "n"]
    texts: list[str]


class Steerer:
    """Intervene on residual stream by adding direction vector."""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.hooks: list = []
        self._layer: int | None = None
        self._direction: Tensor | None = None
        self._alpha: float = 0.0

    def _make_hook(self, layer_idx: int) -> Callable:
        """Create forward hook for intervention."""

        def hook(module, input, output):
            if layer_idx != self._layer or self._direction is None:
                return output

            if isinstance(output, tuple):
                hidden = output[0].clone()
                hidden[:, :, :] += self._alpha * self._direction
                return (hidden,) + output[1:]
            else:
                return output + self._alpha * self._direction

        return hook

    def _register_hooks(self):
        """Register hooks on all layers."""
        self._remove_hooks()
        for idx, layer in enumerate(find_layers(self.model)):
            hook = layer.register_forward_hook(self._make_hook(idx))
            self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        layer: int,
        direction: Float[np.ndarray, "dim"],
        alpha: float,
        max_new_tokens: int = 64,
    ) -> str:
        """Generate with steering intervention."""
        self._layer = layer
        self._direction = torch.tensor(direction, device=self.device, dtype=self.model.dtype)
        self._direction = self._direction / self._direction.norm()  # Normalize
        self._alpha = alpha

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        self._register_hooks()
        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=False,  # Critical: cache bypasses hooks
            )
        finally:
            self._remove_hooks()

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def steer(
        self,
        prompt: str,
        layer: int,
        direction: Float[np.ndarray, "dim"],
        alpha: float,
        max_new_tokens: int = 64,
    ) -> SteeringResult:
        """Generate with and without steering, compare."""
        baseline = self.generate(prompt, layer, direction, alpha=0.0, max_new_tokens=max_new_tokens)
        steered = self.generate(prompt, layer, direction, alpha=alpha, max_new_tokens=max_new_tokens)

        return SteeringResult(
            alpha=alpha,
            generated_text=steered,
            baseline_text=baseline,
            changed=steered != baseline,
        )

    def sweep_alpha(
        self,
        prompts: list[str],
        correct_answers: list[str],
        layer: int,
        direction: Float[np.ndarray, "dim"],
        alphas: Float[np.ndarray, "n"] | None = None,
        max_new_tokens: int = 64,
    ) -> SweepResult:
        """
        Sweep alpha values and measure error rate.

        Args:
            prompts: Input prompts
            correct_answers: Ground truth answers for each prompt
            layer: Layer to intervene on
            direction: Steering direction
            alphas: Alpha values to sweep (default: -5 to 5)
            max_new_tokens: Max generation length

        Returns:
            SweepResult with error rates per alpha
        """
        if alphas is None:
            alphas = np.linspace(-5, 5, 21)

        error_rates = []
        all_texts = []

        for alpha in tqdm(alphas, desc="Sweeping alpha"):
            errors = 0
            texts = []
            for prompt, correct in zip(prompts, correct_answers):
                generated = self.generate(prompt, layer, direction, alpha, max_new_tokens)
                texts.append(generated)
                # Simple overlap check for correctness
                if correct.lower() not in generated.lower():
                    errors += 1
            error_rates.append(errors / len(prompts))
            all_texts.extend(texts)

        return SweepResult(
            alphas=np.array(alphas),
            error_rates=np.array(error_rates),
            texts=all_texts,
        )


def compare_directions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    correct_answers: list[str],
    layer: int,
    learned_direction: Float[np.ndarray, "dim"],
    n_random: int = 5,
    alphas: Float[np.ndarray, "n"] | None = None,
) -> dict[str, SweepResult]:
    """
    Compare learned direction vs random controls.

    Returns dict with 'learned', 'random_mean', 'orthogonal' sweep results.
    """
    steerer = Steerer(model, tokenizer)

    results = {}

    # Learned direction
    results["learned"] = steerer.sweep_alpha(
        prompts, correct_answers, layer, learned_direction, alphas
    )

    # Random directions
    dim = len(learned_direction)
    random_errors = []
    for _ in range(n_random):
        random_dir = np.random.randn(dim)
        random_dir /= np.linalg.norm(random_dir)
        sweep = steerer.sweep_alpha(prompts, correct_answers, layer, random_dir, alphas)
        random_errors.append(sweep.error_rates)

    results["random"] = SweepResult(
        alphas=results["learned"].alphas,
        error_rates=np.mean(random_errors, axis=0),
        texts=[],
    )

    # Orthogonal direction
    ortho_dir = np.random.randn(dim)
    ortho_dir -= (ortho_dir @ learned_direction) * learned_direction
    ortho_dir /= np.linalg.norm(ortho_dir)
    results["orthogonal"] = steerer.sweep_alpha(
        prompts, correct_answers, layer, ortho_dir, alphas
    )

    return results
