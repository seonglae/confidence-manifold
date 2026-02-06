#!/usr/bin/env python
"""
Steering experiment: Validate learned direction causally affects outputs.

Usage:
    python scripts/core/steering_experiment.py --model qwen2-7b --direction probe_direction.npy
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from confidence_manifold import Extractor, load_dataset, load_model
from confidence_manifold.data import to_arrays
from confidence_manifold.probe import train_probe
from confidence_manifold.steering import Steerer, compare_directions

MODELS = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
    "qwen2-1.5b": "Qwen/Qwen2-1.5B-Instruct",
    "qwen2-7b": "Qwen/Qwen2-7B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
}


def run_steering_experiment(
    model_name: str,
    layer: int,
    direction: np.ndarray | None = None,
    max_samples: int = 200,
    output_dir: str = "experiments",
):
    """Run steering experiment."""
    print(f"\n{'='*60}")
    print(f"Steering Experiment")
    print(f"Model: {model_name}")
    print(f"Layer: {layer}")
    print(f"{'='*60}\n")

    # Load model
    model_path = MODELS.get(model_name, model_name)
    print(f"Loading {model_path}...")
    model, tokenizer = load_model(model_path)
    extractor = Extractor(model, tokenizer)

    # Load data
    print("Loading TruthfulQA...")
    samples = load_dataset("truthfulqa", max_samples=max_samples * 2)
    questions, answers, labels, groups = to_arrays(samples)

    # Split: first 200 for probe, rest for steering eval
    train_idx = list(range(min(200, len(questions) // 2)))
    test_idx = list(range(len(train_idx), min(len(train_idx) + max_samples, len(questions))))

    # Train probe to get direction (if not provided)
    if direction is None:
        print("\nTraining probe to get direction...")
        train_q = [questions[i] for i in train_idx]
        train_a = [answers[i] for i in train_idx]
        train_y = np.array([labels[i] for i in train_idx], dtype=int)

        embeddings = []
        for q, a in tqdm(zip(train_q, train_a), total=len(train_q), desc="Extracting"):
            result = extractor.extract(q, a)
            embeddings.append(result.hidden_states[layer].cpu().float().numpy())
        X_train = np.stack(embeddings)

        probe_result = train_probe(X_train, train_y, pls_dim=8)
        direction = probe_result.direction
        print(f"Probe AUC: {probe_result.auc:.3f}")

    # Prepare steering prompts
    test_q = [questions[i] for i in test_idx]
    test_correct = [answers[i] for i in test_idx if labels[i]]  # Only correct answers as ground truth

    # Match lengths
    min_len = min(len(test_q), len(test_correct))
    test_q = test_q[:min_len]
    test_correct = test_correct[:min_len]

    print(f"\nSteering evaluation on {len(test_q)} prompts...")

    # Run comparison
    alphas = np.linspace(-5, 5, 21)
    results = compare_directions(
        model, tokenizer, test_q[:50], test_correct[:50], layer, direction, n_random=3, alphas=alphas
    )

    # Print summary
    print("\n" + "="*40)
    print("Results Summary")
    print("="*40)

    learned = results["learned"]
    random_ctrl = results["random"]
    ortho = results["orthogonal"]

    effect_learned = learned.error_rates[0] - learned.error_rates[-1]
    effect_random = random_ctrl.error_rates[0] - random_ctrl.error_rates[-1]
    effect_ortho = ortho.error_rates[0] - ortho.error_rates[-1]

    print(f"Learned direction: {effect_learned*100:+.1f}pp effect")
    print(f"Random direction:  {effect_random*100:+.1f}pp effect")
    print(f"Orthogonal:        {effect_ortho*100:+.1f}pp effect")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{model_name}_steering"
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "config": {
            "model": model_name,
            "layer": layer,
            "n_prompts": len(test_q[:50]),
            "alphas": alphas.tolist(),
        },
        "learned": {
            "error_rates": learned.error_rates.tolist(),
            "effect_pp": effect_learned * 100,
        },
        "random": {
            "error_rates": random_ctrl.error_rates.tolist(),
            "effect_pp": effect_random * 100,
        },
        "orthogonal": {
            "error_rates": ortho.error_rates.tolist(),
            "effect_pp": effect_ortho * 100,
        },
    }

    with open(run_dir / "results.json", "w") as f:
        json.dump(output, f, indent=2)

    np.save(run_dir / "direction.npy", direction)

    print(f"\nResults saved to {run_dir}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Run steering experiment")
    parser.add_argument("--model", type=str, default="gpt2", choices=list(MODELS.keys()))
    parser.add_argument("--layer", type=int, default=None, help="Layer to steer (default: 75% depth)")
    parser.add_argument("--direction", type=str, help="Path to .npy direction file")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--output", type=str, default="experiments")
    args = parser.parse_args()

    direction = np.load(args.direction) if args.direction else None

    # Default layer to 75% depth
    if args.layer is None:
        model_path = MODELS.get(args.model, args.model)
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        args.layer = int(config.num_hidden_layers * 0.75)

    run_steering_experiment(args.model, args.layer, direction, args.samples, args.output)


if __name__ == "__main__":
    main()
