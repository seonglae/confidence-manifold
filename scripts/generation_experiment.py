#!/usr/bin/env python
"""
Generation geometry experiment.

Test if geometric properties persist in model-generated outputs
(vs teacher-forced evaluation).

Usage:
    python generation_experiment.py --model qwen2-7b
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from confidence_manifold import Extractor, load_dataset, load_model
from confidence_manifold.data import to_arrays
from confidence_manifold.probe import (
    cross_validate,
    dimension_sweep,
    compare_classifiers,
)

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


def generate_answers(
    model,
    tokenizer,
    questions: list[str],
    correct_answers: list[str],
    max_new_tokens: int = 50,
) -> tuple[list[str], list[int]]:
    """
    Generate model answers and label by semantic match to correct answer.

    Returns:
        (generated_answers, labels) where label=1 if matches correct answer
    """
    device = next(model.parameters()).device
    generated = []
    labels = []

    for q, correct in tqdm(zip(questions, correct_answers), total=len(questions), desc="Generating"):
        # Format prompt
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": f"Answer briefly: {q}"}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"Question: {q}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy for reproducibility
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_text = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        gen_text = gen_text.strip()
        generated.append(gen_text)

        # Simple semantic match: check if correct answer appears in generation
        correct_lower = correct.lower().strip()
        gen_lower = gen_text.lower()

        # Match if correct answer substring appears or high word overlap
        correct_words = set(correct_lower.split())
        gen_words = set(gen_lower.split())
        overlap = len(correct_words & gen_words) / max(len(correct_words), 1)

        is_correct = correct_lower in gen_lower or overlap > 0.5
        labels.append(1 if is_correct else 0)

    return generated, labels


def run_experiment(
    model_name: str,
    layer: int | None = None,
    max_samples: int = 200,
    output_dir: str = "experiments",
):
    """Run generation geometry experiment."""
    print(f"\n{'='*60}")
    print(f"Generation Geometry Experiment")
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

    # Get unique questions with correct answers
    unique_questions = []
    correct_answers = []
    seen = set()
    for q, a, label in zip(questions, answers, labels):
        if q not in seen and label == 1:
            unique_questions.append(q)
            correct_answers.append(a)
            seen.add(q)

    print(f"Unique questions: {len(unique_questions)}")

    # === Teacher-forced evaluation ===
    print("\n" + "="*40)
    print("1. Teacher-Forced Evaluation")
    print("="*40)

    tf_embeddings = []
    for q, a in tqdm(zip(questions, answers), total=len(questions), desc="Teacher-forced"):
        result = extractor.extract(q, a)
        tf_embeddings.append(result.hidden_states[layer].cpu().float().numpy())

    X_tf = np.stack(tf_embeddings)
    y_tf = np.array(labels, dtype=int)
    groups_tf = np.array(groups)

    tf_cv = cross_validate(X_tf, y_tf, groups_tf, pls_dim=8)
    print(f"  AUC: {tf_cv.mean_auc:.3f} ± {tf_cv.std_auc:.3f}")

    # Dimension sweep
    tf_dims = dimension_sweep(X_tf, y_tf, groups_tf)
    tf_best_dim = max(tf_dims, key=lambda d: tf_dims[d].mean_auc)
    print(f"  Best dimension: {tf_best_dim}D (AUC={tf_dims[tf_best_dim].mean_auc:.3f})")

    # Classifier comparison
    tf_classifiers = compare_classifiers(X_tf, y_tf, groups_tf, pls_dim=8)

    # === Generation evaluation ===
    print("\n" + "="*40)
    print("2. Generation Evaluation")
    print("="*40)

    # Generate answers
    gen_answers, gen_labels = generate_answers(
        model, tokenizer, unique_questions, correct_answers
    )

    print(f"  Generated {len(gen_answers)} answers")
    print(f"  Correct: {sum(gen_labels)} ({100*sum(gen_labels)/len(gen_labels):.1f}%)")

    # Extract embeddings for generated answers
    gen_embeddings = []
    for q, a in tqdm(zip(unique_questions, gen_answers), total=len(unique_questions), desc="Gen embeddings"):
        result = extractor.extract(q, a)
        gen_embeddings.append(result.hidden_states[layer].cpu().float().numpy())

    X_gen = np.stack(gen_embeddings)
    y_gen = np.array(gen_labels, dtype=int)

    # Check class balance
    if y_gen.sum() < 5 or (1 - y_gen).sum() < 5:
        print("  WARNING: Insufficient class balance for evaluation")
        gen_auc = 0.5
        gen_best_dim = None
    else:
        gen_cv = cross_validate(X_gen, y_gen, pls_dim=8)
        gen_auc = gen_cv.mean_auc
        print(f"  AUC: {gen_cv.mean_auc:.3f} ± {gen_cv.std_auc:.3f}")

        # Dimension sweep
        gen_dims = dimension_sweep(X_gen, y_gen)
        gen_best_dim = max(gen_dims, key=lambda d: gen_dims[d].mean_auc)
        print(f"  Best dimension: {gen_best_dim}D (AUC={gen_dims[gen_best_dim].mean_auc:.3f})")

        # Classifier comparison
        gen_classifiers = compare_classifiers(X_gen, y_gen, pls_dim=8)

    # === Summary ===
    print("\n" + "="*40)
    print("3. Summary")
    print("="*40)

    gap = tf_cv.mean_auc - gen_auc
    print(f"  Teacher-forced AUC: {tf_cv.mean_auc:.3f}")
    print(f"  Generation AUC: {gen_auc:.3f}")
    print(f"  Gap: {gap:.3f} ({100*gap/tf_cv.mean_auc:.1f}%)")
    print(f"  Dimension shift: {tf_best_dim}D → {gen_best_dim}D")

    # Check if geometric properties preserved
    tf_centroid = tf_classifiers.get("centroid", tf_classifiers.get("linear"))
    tf_linear = tf_classifiers.get("linear")
    centroid_matches_probe = abs(tf_centroid.mean_auc - tf_linear.mean_auc) < 0.02

    print(f"  Centroid ≈ Probe (TF): {centroid_matches_probe} ({tf_centroid.mean_auc:.3f} vs {tf_linear.mean_auc:.3f})")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{model_name}_generation"
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "config": {
            "model": model_name,
            "layer": layer,
            "max_samples": max_samples,
        },
        "teacher_forced": {
            "auc": tf_cv.mean_auc,
            "std": tf_cv.std_auc,
            "best_dim": tf_best_dim,
            "classifiers": {name: {"auc": cv.mean_auc, "std": cv.std_auc} for name, cv in tf_classifiers.items()},
        },
        "generation": {
            "auc": gen_auc,
            "best_dim": gen_best_dim,
            "n_correct": int(sum(gen_labels)),
            "n_total": len(gen_labels),
        },
        "gap": {
            "absolute": float(gap),
            "relative_pct": float(100 * gap / tf_cv.mean_auc) if tf_cv.mean_auc > 0 else 0,
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
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--output", type=str, default="experiments")
    args = parser.parse_args()

    run_experiment(args.model, args.layer, args.samples, args.output)


if __name__ == "__main__":
    main()
