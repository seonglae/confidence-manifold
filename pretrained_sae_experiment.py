#!/usr/bin/env python
"""
Pretrained SAE experiment.

Usage:
    python pretrained_sae_experiment.py --model gpt2 --sae-model gpt2-small --layer 6 --samples 200
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from confidence_manifold import Extractor, load_model
from confidence_manifold.data import load_dataset, to_arrays
from confidence_manifold.probe import cross_validate
from confidence_manifold.sae import (
    load_pretrained_sae,
    encode_with_pretrained_sae,
    get_sparse_features,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrained SAE experiment")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--sae-model", type=str, default="gpt2-small")
    parser.add_argument("--dataset", type=str, default="truthfulqa")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--density-threshold", type=float, default=0.01)
    parser.add_argument("--pls-dim", type=int, default=8)
    parser.add_argument("--output", type=str, default="experiments")
    return parser.parse_args()


def extract_layer_embeddings(extractor: Extractor, questions: list[str], answers: list[str], layer: int):
    embeddings = []
    for q, a in zip(questions, answers):
        h = extractor.extract_layer(q, a, layer).detach().cpu().float().numpy()
        embeddings.append(h)
    return np.stack(embeddings)


def main():
    args = parse_args()

    print(f"Model: {args.model}")
    print(f"SAE model: {args.sae_model}")
    print(f"Layer: {args.layer}")

    model, tokenizer = load_model(args.model)
    extractor = Extractor(model, tokenizer)

    samples = load_dataset(args.dataset, max_samples=args.samples)
    questions, answers, labels, groups = to_arrays(samples)
    y = np.array(labels, dtype=int)
    groups = np.array(groups)

    X = extract_layer_embeddings(extractor, questions, answers, args.layer)

    pls_dim = args.pls_dim if args.pls_dim > 0 else None
    raw_cv = cross_validate(X, y, groups, pls_dim=pls_dim)

    try:
        sae_result = load_pretrained_sae(args.sae_model, args.layer, device=str(extractor.device))
    except Exception as e:
        print(f"Failed to load SAE: {e}")
        return

    device = next(sae_result.sae.parameters()).device
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    features, _ = encode_with_pretrained_sae(sae_result, X_tensor)
    X_sae = features.detach().cpu().numpy()

    sae_cv = cross_validate(X_sae, y, groups, pls_dim=None)

    sparse_idx = get_sparse_features(sae_result, threshold=args.density_threshold)
    sparse_cv = None
    if len(sparse_idx) > 0:
        X_sparse = X_sae[:, sparse_idx]
        sparse_cv = cross_validate(X_sparse, y, groups, pls_dim=None)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"pretrained_sae_{timestamp}_{args.model.replace('/', '_')}.json"

    payload = {
        "config": {
            "model": args.model,
            "sae_model": args.sae_model,
            "dataset": args.dataset,
            "samples": args.samples,
            "layer": args.layer,
            "density_threshold": args.density_threshold,
            "pls_dim": args.pls_dim,
        },
        "raw_probe": {
            "mean_auc": float(raw_cv.mean_auc),
            "std_auc": float(raw_cv.std_auc),
        },
        "sae_probe": {
            "mean_auc": float(sae_cv.mean_auc),
            "std_auc": float(sae_cv.std_auc),
        },
        "sparse_sae_probe": (
            {
                "mean_auc": float(sparse_cv.mean_auc),
                "std_auc": float(sparse_cv.std_auc),
                "n_sparse_features": int(len(sparse_idx)),
            }
            if sparse_cv is not None
            else {"n_sparse_features": int(len(sparse_idx))}
        ),
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
