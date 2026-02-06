#!/usr/bin/env python
"""
Geometry experiment: Layer evolution, intrinsic dimension, classifier comparison.

Usage:
    python scripts/core/geometry_experiment.py --model qwen2-7b
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
from confidence_manifold.geometry import estimate_dimension, pca_dimension
from confidence_manifold.probe import cross_validate, compare_classifiers, dimension_sweep
from confidence_manifold.baselines import evaluate_baselines

MODELS = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
    "qwen2-1.5b": "Qwen/Qwen2-1.5B-Instruct",
    "qwen2-7b": "Qwen/Qwen2-7B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
}


def extract_all_layers(extractor, questions, answers):
    """Extract embeddings from all layers."""
    n_samples = len(questions)
    n_layers = extractor.num_layers

    # Initialize
    all_embeddings = {l: [] for l in range(n_layers)}

    for q, a in tqdm(zip(questions, answers), total=n_samples, desc="Extracting"):
        result = extractor.extract(q, a)
        for l in range(n_layers):
            all_embeddings[l].append(result.hidden_states[l].cpu().float().numpy())

    return {l: np.stack(embs) for l, embs in all_embeddings.items()}


def run_experiment(
    model_name: str,
    max_samples: int = 800,
    output_dir: str = "experiments",
):
    """Run geometry analysis experiment."""
    print(f"\n{'='*60}")
    print(f"Geometry Analysis Experiment")
    print(f"Model: {model_name}")
    print(f"{'='*60}\n")

    # Load model
    model_path = MODELS.get(model_name, model_name)
    model, tokenizer = load_model(model_path)
    extractor = Extractor(model, tokenizer)

    # Load data
    print("Loading TruthfulQA...")
    samples = load_dataset("truthfulqa", max_samples=max_samples)
    questions, answers, labels, groups = to_arrays(samples)
    y = np.array(labels, dtype=int)
    groups = np.array(groups)

    # Extract all layers
    print("\nExtracting embeddings from all layers...")
    layer_embeddings = extract_all_layers(extractor, questions, answers)

    # 1. Layer evolution: AUC and intrinsic dimension
    print("\n" + "="*40)
    print("1. Layer Evolution")
    print("="*40)

    layer_results = {}
    for layer in range(0, extractor.num_layers, max(1, extractor.num_layers // 10)):
        X = layer_embeddings[layer]

        # AUC
        cv = cross_validate(X, y, groups, pls_dim=8)

        # Intrinsic dimension
        mle_dim = estimate_dimension(X, k=10)
        pca_dim, _ = pca_dimension(X, threshold=0.95)

        layer_results[layer] = {
            "auc": cv.mean_auc,
            "auc_std": cv.std_auc,
            "mle_dim": mle_dim,
            "pca_dim": pca_dim,
        }
        print(f"  L{layer:2d}: AUC={cv.mean_auc:.3f}, MLE_dim={mle_dim:.1f}, PCA_dim={pca_dim}")

    # Find best layer
    best_layer = max(layer_results, key=lambda l: layer_results[l]["auc"])
    print(f"\nBest layer: {best_layer} (AUC={layer_results[best_layer]['auc']:.3f})")

    # 2. Dimension sweep at best layer
    print("\n" + "="*40)
    print("2. Dimension Sweep")
    print("="*40)

    X_best = layer_embeddings[best_layer]
    dim_results = dimension_sweep(X_best, y, groups, dims=[1, 2, 3, 4, 5, 6, 8, 12, 16, 32])
    for dim, cv in sorted(dim_results.items()):
        print(f"  {dim:2d}D: AUC={cv.mean_auc:.3f} ± {cv.std_auc:.3f}")

    best_dim = max(dim_results, key=lambda d: dim_results[d].mean_auc)
    print(f"\nBest dimension: {best_dim}D (AUC={dim_results[best_dim].mean_auc:.3f})")

    # 3. Classifier comparison at best layer
    print("\n" + "="*40)
    print("3. Classifier Comparison (8D PLS)")
    print("="*40)

    clf_results = compare_classifiers(X_best, y, groups, pls_dim=8)
    for name, cv in sorted(clf_results.items(), key=lambda x: -x[1].mean_auc):
        print(f"  {name:15s}: AUC={cv.mean_auc:.3f} ± {cv.std_auc:.3f}")

    # 4. Unsupervised baselines
    print("\n" + "="*40)
    print("4. Unsupervised Baselines")
    print("="*40)

    baseline_results = evaluate_baselines(X_best, y)
    for name, result in sorted(baseline_results.items(), key=lambda x: -x[1].auc):
        print(f"  {name:20s}: AUC={result.auc:.3f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{model_name}_geometry"
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "config": {
            "model": model_name,
            "num_layers": extractor.num_layers,
            "hidden_dim": extractor.hidden_dim,
            "max_samples": max_samples,
        },
        "layer_evolution": {str(k): v for k, v in layer_results.items()},
        "best_layer": best_layer,
        "dimension_sweep": {str(d): {"auc": cv.mean_auc, "std": cv.std_auc} for d, cv in dim_results.items()},
        "best_dim": best_dim,
        "classifiers": {name: {"auc": cv.mean_auc, "std": cv.std_auc} for name, cv in clf_results.items()},
        "unsupervised": {name: {"auc": r.auc} for name, r in baseline_results.items()},
    }

    with open(run_dir / "results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {run_dir}")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", choices=list(MODELS.keys()))
    parser.add_argument("--samples", type=int, default=800)
    parser.add_argument("--output", type=str, default="experiments")
    args = parser.parse_args()

    run_experiment(args.model, args.samples, args.output)


if __name__ == "__main__":
    main()
