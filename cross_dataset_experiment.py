#!/usr/bin/env python
"""
Cross-dataset transfer experiment.

Train on TruthfulQA, evaluate on SciQ, CSQA, FEVER, HaluEval.

Usage:
    python scripts/core/cross_dataset_experiment.py --model qwen2-7b
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
from confidence_manifold.probe import train_probe, cross_validate

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

TRANSFER_DATASETS = ["sciq", "commonsenseqa", "fever", "halueval"]


def extract_embeddings(extractor, questions, answers, layer):
    """Extract embeddings at specified layer."""
    embeddings = []
    for q, a in tqdm(zip(questions, answers), total=len(questions), desc=f"L{layer}"):
        result = extractor.extract(q, a)
        embeddings.append(result.hidden_states[layer].cpu().float().numpy())
    return np.stack(embeddings)


def run_experiment(
    model_name: str,
    layer: int | None = None,
    train_samples: int = 800,
    test_samples: int = 500,
    output_dir: str = "experiments",
):
    """Run cross-dataset transfer experiment."""
    print(f"\n{'='*60}")
    print(f"Cross-Dataset Transfer Experiment")
    print(f"Model: {model_name}")
    print(f"{'='*60}\n")

    # Load model
    model_path = MODELS.get(model_name, model_name)
    model, tokenizer = load_model(model_path)
    extractor = Extractor(model, tokenizer)

    # Auto-select layer if not specified
    if layer is None:
        layer = int(extractor.num_layers * 0.75)
    print(f"Using layer {layer}/{extractor.num_layers}")

    # Load TruthfulQA for training
    print("\nLoading TruthfulQA...")
    train_samples_data = load_dataset("truthfulqa", max_samples=train_samples)
    train_q, train_a, train_labels, train_groups = to_arrays(train_samples_data)
    train_y = np.array(train_labels, dtype=int)
    train_groups = np.array(train_groups)

    # Extract training embeddings
    print("Extracting training embeddings...")
    X_train = extract_embeddings(extractor, train_q, train_a, layer)

    # In-domain CV
    print("\nIn-domain evaluation (TruthfulQA)...")
    indomain_cv = cross_validate(X_train, train_y, train_groups, pls_dim=8)
    print(f"  TruthfulQA AUC: {indomain_cv.mean_auc:.3f} ± {indomain_cv.std_auc:.3f}")

    # Train probe on full TruthfulQA
    probe_result = train_probe(X_train, train_y, pls_dim=8)

    # Cross-domain transfer
    results = {
        "truthfulqa": {
            "in_domain": True,
            "auc": indomain_cv.mean_auc,
            "std": indomain_cv.std_auc,
        }
    }

    print("\nCross-domain evaluation...")
    for ds_name in TRANSFER_DATASETS:
        print(f"\n  Loading {ds_name}...")
        try:
            test_samples_data = load_dataset(ds_name, max_samples=test_samples)
            test_q, test_a, test_labels, _ = to_arrays(test_samples_data)
            test_y = np.array(test_labels, dtype=int)

            # Extract test embeddings
            X_test = extract_embeddings(extractor, test_q, test_a, layer)

            # Apply probe's preprocessing
            X_test_scaled = probe_result.scaler.transform(X_test)
            if probe_result.pls is not None:
                X_test_scaled = probe_result.pls.transform(X_test_scaled)

            # Predict
            probs = probe_result.probe.predict_proba(X_test_scaled)[:, 1]
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(test_y, probs)

            results[ds_name] = {"in_domain": False, "auc": auc, "n_samples": len(test_y)}
            print(f"  {ds_name}: AUC = {auc:.3f} (n={len(test_y)})")

        except Exception as e:
            print(f"  {ds_name}: FAILED - {e}")
            results[ds_name] = {"in_domain": False, "auc": None, "error": str(e)}

    # Summary
    print("\n" + "="*40)
    print("Summary")
    print("="*40)
    cross_aucs = [r["auc"] for r in results.values() if not r.get("in_domain") and r.get("auc")]
    if cross_aucs:
        print(f"Cross-domain mean AUC: {np.mean(cross_aucs):.3f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{model_name}_cross_dataset"
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "config": {
            "model": model_name,
            "layer": layer,
            "train_samples": train_samples,
            "test_samples": test_samples,
        },
        "results": results,
    }

    with open(run_dir / "results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {run_dir}")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", choices=list(MODELS.keys()))
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--train-samples", type=int, default=800)
    parser.add_argument("--test-samples", type=int, default=500)
    parser.add_argument("--output", type=str, default="experiments")
    args = parser.parse_args()

    run_experiment(args.model, args.layer, args.train_samples, args.test_samples, args.output)


if __name__ == "__main__":
    main()
