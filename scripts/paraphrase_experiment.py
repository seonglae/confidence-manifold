#!/usr/bin/env python
"""
Paraphrase control experiment.

Verify that probes detect correctness, not answer style.
Generates paraphrase variants and analyzes variance decomposition.

Usage:
    python paraphrase_experiment.py --model qwen2-7b
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
from confidence_manifold.probe import train_probe, evaluate_probe, cross_validate

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

PARAPHRASE_TEMPLATES = [
    "{answer}",  # Original
    "The answer is: {answer}",
    "To be precise, {answer}",
    "In other words, {answer}",
    "Simply put, {answer}",
]


def generate_paraphrases(answer: str) -> list[str]:
    """Generate paraphrase variants of an answer."""
    return [template.format(answer=answer) for template in PARAPHRASE_TEMPLATES]


def run_experiment(
    model_name: str,
    layer: int | None = None,
    max_samples: int = 400,
    output_dir: str = "experiments",
):
    """Run paraphrase control experiment."""
    print(f"\n{'='*60}")
    print(f"Paraphrase Control Experiment")
    print(f"Model: {model_name}")
    print(f"{'='*60}\n")

    # Load model
    model_path = MODELS.get(model_name, model_name)
    model, tokenizer = load_model(model_path)
    extractor = Extractor(model, tokenizer)

    # Auto-select layer
    if layer is None:
        layer = int(extractor.num_layers * 0.75)
    print(f"Using layer {layer}/{extractor.num_layers}")

    # Load data
    print("Loading TruthfulQA...")
    samples = load_dataset("truthfulqa", max_samples=max_samples)
    questions, answers, labels, groups = to_arrays(samples)

    # Generate paraphrased versions
    print("\nGenerating paraphrases...")
    all_questions = []
    all_answers = []
    all_labels = []
    all_groups = []
    all_paraphrase_ids = []
    answer_ids = []  # Track which original answer each belongs to

    answer_id = 0
    for q, a, label, group in zip(questions, answers, labels, groups):
        paraphrases = generate_paraphrases(a)
        for p_idx, para in enumerate(paraphrases):
            all_questions.append(q)
            all_answers.append(para)
            all_labels.append(label)
            all_groups.append(group)
            all_paraphrase_ids.append(p_idx)
            answer_ids.append(answer_id)
        answer_id += 1

    # Extract embeddings
    print(f"\nExtracting embeddings ({len(all_questions)} samples)...")
    embeddings = []
    for q, a in tqdm(zip(all_questions, all_answers), total=len(all_questions)):
        result = extractor.extract(q, a)
        embeddings.append(result.hidden_states[layer].cpu().float().numpy())

    X = np.stack(embeddings)
    y = np.array(all_labels, dtype=int)
    answer_ids = np.array(answer_ids)
    paraphrase_ids = np.array(all_paraphrase_ids)

    # Variance decomposition (using raw embeddings to avoid PLS leakage)
    print("\nComputing variance decomposition (raw space)...")

    # Use standardized embeddings (no PLS to avoid fitting on labels)
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Compute within-answer variance (same answer, different paraphrases)
    unique_answers = np.unique(answer_ids)
    within_variances = []
    for aid in unique_answers:
        mask = answer_ids == aid
        if mask.sum() > 1:
            X_answer = X_scaled[mask]
            within_variances.append(np.var(X_answer, axis=0).sum())

    within_var = np.mean(within_variances)

    # Compute between-answer variance (different correctness labels)
    correct_mask = y == 1
    incorrect_mask = y == 0
    mu_correct = X_scaled[correct_mask].mean(axis=0)
    mu_incorrect = X_scaled[incorrect_mask].mean(axis=0)
    between_var = np.sum((mu_correct - mu_incorrect) ** 2)

    f_ratio = between_var / (within_var + 1e-10)

    print(f"  Within-answer variance: {within_var:.2f}")
    print(f"  Between-answer variance: {between_var:.2f}")
    print(f"  F-ratio: {f_ratio:.2f}")

    # Train on original, test on paraphrased (different questions)
    print("\nCross-validation: train original → test paraphrased...")

    # Split by question groups
    original_mask = paraphrase_ids == 0
    X_original = X[original_mask]
    y_original = y[original_mask]
    groups_original = np.array(all_groups)[original_mask]

    # Use original for training, paraphrased for testing
    from sklearn.model_selection import GroupKFold

    kfold = GroupKFold(n_splits=5)
    fold_aucs_original = []
    fold_aucs_paraphrase = []

    for train_idx, test_idx in kfold.split(X_original, y_original, groups_original):
        # Get question IDs for test
        test_questions = set(groups_original[test_idx])

        # Train on original answers
        probe_result = train_probe(X_original[train_idx], y_original[train_idx], pls_dim=8)

        # Test on original
        auc_orig, _ = evaluate_probe(probe_result, X_original[test_idx], y_original[test_idx])
        fold_aucs_original.append(auc_orig)

        # Test on paraphrased (same questions as test set)
        para_mask = np.array([g in test_questions and p > 0 for g, p in zip(all_groups, paraphrase_ids)])
        if para_mask.sum() > 0:
            X_para_test = X[para_mask]
            y_para_test = y[para_mask]
            auc_para, _ = evaluate_probe(probe_result, X_para_test, y_para_test)
            fold_aucs_paraphrase.append(auc_para)

    print(f"  Original test AUC: {np.mean(fold_aucs_original):.3f} ± {np.std(fold_aucs_original):.3f}")
    print(f"  Paraphrase test AUC: {np.mean(fold_aucs_paraphrase):.3f} ± {np.std(fold_aucs_paraphrase):.3f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{model_name}_paraphrase"
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "config": {
            "model": model_name,
            "layer": layer,
            "max_samples": max_samples,
            "n_paraphrases": len(PARAPHRASE_TEMPLATES),
        },
        "variance_decomposition": {
            "within_answer_variance": float(within_var),
            "between_answer_variance": float(between_var),
            "f_ratio": float(f_ratio),
        },
        "generalization": {
            "original_auc": float(np.mean(fold_aucs_original)),
            "original_std": float(np.std(fold_aucs_original)),
            "paraphrase_auc": float(np.mean(fold_aucs_paraphrase)),
            "paraphrase_std": float(np.std(fold_aucs_paraphrase)),
        },
    }

    with open(run_dir / "results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {run_dir}")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", choices=list(MODELS.keys()))
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--output", type=str, default="experiments")
    args = parser.parse_args()

    run_experiment(args.model, args.layer, args.samples, args.output)


if __name__ == "__main__":
    main()
