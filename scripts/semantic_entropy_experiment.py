#!/usr/bin/env python
"""
Semantic entropy experiment (NLI clustering).

Usage:
    python semantic_entropy_experiment.py --model gpt2 --dataset truthfulqa --samples 50
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from confidence_manifold import load_model
from confidence_manifold.data import load_dataset, to_arrays
from confidence_manifold.semantic_entropy import SemanticEntropyProbe


def parse_args():
    parser = argparse.ArgumentParser(description="Semantic entropy experiment")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--dataset", type=str, default="truthfulqa")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=10, help="Responses per question")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--strict-entailment", action="store_true")
    parser.add_argument("--output", type=str, default="experiments")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.samples}")

    model, tokenizer = load_model(args.model)
    samples = load_dataset(args.dataset, max_samples=args.samples)
    questions, _, labels, _ = to_arrays(samples)

    y = np.array(labels, dtype=int)  # 1=correct, 0=incorrect

    probe = SemanticEntropyProbe(
        n_samples=args.n_samples,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        strict_entailment=args.strict_entailment,
    )

    results = probe.compute_batch(model, tokenizer, questions, show_progress=True)
    semantic_entropies = np.array([r.semantic_entropy for r in results])
    predictive_entropies = np.array([r.predictive_entropy for r in results])

    # Lower entropy => more correct
    se_scores = -semantic_entropies
    pe_scores = -predictive_entropies

    se_auc = roc_auc_score(y, se_scores)
    pe_auc = roc_auc_score(y, pe_scores)

    print(f"Semantic entropy AUC: {se_auc:.3f}")
    print(f"Predictive entropy AUC: {pe_auc:.3f}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"semantic_entropy_{timestamp}_{args.model.replace('/', '_')}.json"

    payload = {
        "config": {
            "model": args.model,
            "dataset": args.dataset,
            "samples": args.samples,
            "n_samples": args.n_samples,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "strict_entailment": args.strict_entailment,
        },
        "n_processed": int(len(semantic_entropies)),
        "semantic_entropy_auc": float(se_auc),
        "predictive_entropy_auc": float(pe_auc),
        "semantic_entropy_mean": float(semantic_entropies.mean()),
        "predictive_entropy_mean": float(predictive_entropies.mean()),
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
