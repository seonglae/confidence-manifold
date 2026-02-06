#!/usr/bin/env python
"""
Core experiment: Train and evaluate correctness detection probes.

Usage:
    python scripts/core/probe_experiment.py --model qwen2-7b --samples 1000
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
from confidence_manifold.probe import cross_validate, nested_cross_validate, dimension_sweep, train_probe

# Model name mapping
MODELS = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
    "qwen2-1.5b": "Qwen/Qwen2-1.5B-Instruct",
    "qwen2-7b": "Qwen/Qwen2-7B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "gemma-2b": "google/gemma-2-2b-it",
}


def extract_embeddings(
    extractor: Extractor,
    questions: list[str],
    answers: list[str],
    layer: int,
) -> np.ndarray:
    """Extract embeddings for all samples at specified layer."""
    embeddings = []
    for q, a in tqdm(zip(questions, answers), total=len(questions), desc=f"Extracting L{layer}"):
        result = extractor.extract(q, a)
        embeddings.append(result.hidden_states[layer].cpu().float().numpy())
    return np.stack(embeddings)


def run_experiment(
    model_name: str,
    dataset_name: str = "truthfulqa",
    max_samples: int = 1000,
    output_dir: str = "experiments",
):
    """Run probe experiment."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Max samples: {max_samples}")
    print(f"{'='*60}\n")

    # Load model
    model_path = MODELS.get(model_name, model_name)
    print(f"Loading {model_path}...")
    model, tokenizer = load_model(model_path)
    extractor = Extractor(model, tokenizer)

    # Load data
    print(f"Loading {dataset_name}...")
    samples = load_dataset(dataset_name, max_samples=max_samples)
    questions, answers, labels, groups = to_arrays(samples)
    y = np.array(labels, dtype=int)
    groups = np.array(groups)

    print(f"Samples: {len(samples)} ({sum(labels)} correct, {len(labels)-sum(labels)} incorrect)")

    # Find optimal layer
    print("\nFinding optimal layer...")
    layer_results = {}
    for layer in range(0, extractor.num_layers, max(1, extractor.num_layers // 8)):
        X = extract_embeddings(extractor, questions, answers, layer)
        cv = cross_validate(X, y, groups, pls_dim=8)
        layer_results[layer] = cv.mean_auc
        print(f"  Layer {layer}: AUC = {cv.mean_auc:.3f} ± {cv.std_auc:.3f}")

    best_layer = max(layer_results, key=layer_results.get)
    print(f"\nBest layer: {best_layer} (AUC = {layer_results[best_layer]:.3f})")

    # Extract at best layer
    print(f"\nExtracting embeddings at layer {best_layer}...")
    X = extract_embeddings(extractor, questions, answers, best_layer)

    # Dimension sweep
    print("\nDimension sweep...")
    dim_results = dimension_sweep(X, y, groups, dims=[1, 2, 3, 4, 5, 6, 8, 12, 16, 32])
    for dim, cv in sorted(dim_results.items()):
        print(f"  {dim}D: AUC = {cv.mean_auc:.3f} ± {cv.std_auc:.3f}")

    best_dim = max(dim_results, key=lambda d: dim_results[d].mean_auc)
    print(f"\nBest dimension: {best_dim}D (AUC = {dim_results[best_dim].mean_auc:.3f})")

    # Final CV with best settings
    print(f"\nFinal cross-validation (layer={best_layer}, dim={best_dim})...")
    final_cv = cross_validate(X, y, groups, pls_dim=best_dim)
    print(f"Final AUC: {final_cv.mean_auc:.3f} ± {final_cv.std_auc:.3f}")

    # Nested CV for unbiased evaluation (hyperparameter selection in inner loop)
    print("\nNested CV (unbiased estimate)...")
    nested_cv = nested_cross_validate(X, y, groups)
    print(f"Nested AUC: {nested_cv.mean_auc:.3f} ± {nested_cv.std_auc:.3f}")
    print(f"Bias: {final_cv.mean_auc - nested_cv.mean_auc:+.3f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{model_name}_{dataset_name}"
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "config": {
            "model": model_name,
            "model_path": model_path,
            "dataset": dataset_name,
            "max_samples": max_samples,
            "num_layers": extractor.num_layers,
            "hidden_dim": extractor.hidden_dim,
        },
        "layer_sweep": {str(k): v for k, v in layer_results.items()},
        "best_layer": best_layer,
        "dimension_sweep": {str(d): {"mean_auc": cv.mean_auc, "std_auc": cv.std_auc} for d, cv in dim_results.items()},
        "best_dim": best_dim,
        "final": {
            "mean_auc": final_cv.mean_auc,
            "std_auc": final_cv.std_auc,
            "fold_aucs": final_cv.fold_aucs,
        },
        "nested_cv": {
            "mean_auc": nested_cv.mean_auc,
            "std_auc": nested_cv.std_auc,
            "fold_aucs": nested_cv.fold_aucs,
            "bias": final_cv.mean_auc - nested_cv.mean_auc,
        },
    }

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {run_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run probe experiment")
    parser.add_argument("--model", type=str, default="gpt2", choices=list(MODELS.keys()) + ["custom"])
    parser.add_argument("--model-path", type=str, help="Custom model path (if --model=custom)")
    parser.add_argument("--dataset", type=str, default="truthfulqa")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--output", type=str, default="experiments")
    args = parser.parse_args()

    model_name = args.model_path if args.model == "custom" else args.model
    run_experiment(model_name, args.dataset, args.samples, args.output)


if __name__ == "__main__":
    main()
