"""Baseline methods for comparison: output-based and unsupervised."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from jaxtyping import Float
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class BaselineResult:
    """Result from baseline method."""

    name: str
    auc: float
    scores: Float[np.ndarray, "n"]


# =============================================================================
# Output-based methods
# =============================================================================


@torch.no_grad()
def p_true_baseline(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    questions: list[str],
    answers: list[str],
) -> Float[np.ndarray, "n"]:
    """
    P(True) baseline: Ask model if answer is correct.

    Returns P("Yes") for each sample.
    """
    device = next(model.parameters()).device
    scores = []

    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]

    for q, a in tqdm(zip(questions, answers), total=len(questions), desc="P(True)"):
        prompt = f"Question: {q}\nAnswer: {a}\nIs this answer correct? "
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]

        p_yes = torch.softmax(logits[[yes_id, no_id]], dim=0)[0].item()
        scores.append(p_yes)

    return np.array(scores)


@torch.no_grad()
def token_entropy_baseline(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    questions: list[str],
    answers: list[str],
) -> Float[np.ndarray, "n"]:
    """
    Token entropy baseline: Average entropy over answer tokens.

    Lower entropy = more confident (so we return negative for AUC direction).
    """
    device = next(model.parameters()).device
    scores = []

    for q, a in tqdm(zip(questions, answers), total=len(questions), desc="Entropy"):
        prompt = f"Question: {q}\nAnswer: {a}"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model(**inputs)
        logits = outputs.logits[0]  # (seq, vocab)

        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        mean_entropy = entropy.mean().item()

        # Negative because lower entropy = more confident = more likely correct
        scores.append(-mean_entropy)

    return np.array(scores)


@torch.no_grad()
def semantic_entropy_baseline(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    questions: list[str],
    n_samples: int = 5,
    temperature: float = 0.7,
    max_new_tokens: int = 50,
    mode: Literal["simple", "nli"] = "simple",
    strict_entailment: bool = False,
) -> Float[np.ndarray, "n"]:
    """
    Semantic entropy baseline (Farquhar et al. 2024).

    mode="simple": fast string-based clustering (approximation)
    mode="nli": NLI-based bidirectional entailment clustering (paper setting)
    """
    device = next(model.parameters()).device
    scores = []

    if mode == "nli":
        from confidence_manifold.semantic_entropy import SemanticEntropyProbe

        probe = SemanticEntropyProbe(
            n_samples=n_samples,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            strict_entailment=strict_entailment,
        )
        results = probe.compute_batch(model, tokenizer, questions, device=str(device))
        return np.array([-r.semantic_entropy for r in results])

    for q in tqdm(questions, desc="Semantic Entropy"):
        prompt = f"Question: {q}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        generations = []
        for _ in range(n_samples):
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            gen_text = tokenizer.decode(
                output[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )
            generations.append(gen_text.strip())

        clusters = {}
        for gen in generations:
            key = gen.lower().strip()[:50]
            clusters[key] = clusters.get(key, 0) + 1

        probs = np.array(list(clusters.values())) / n_samples
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        scores.append(-entropy)

    return np.array(scores)


def nll_baseline(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    questions: list[str],
    answers: list[str],
) -> Float[np.ndarray, "n"]:
    """
    Negative log-likelihood baseline.

    Lower NLL = higher probability = more likely correct.
    """
    device = next(model.parameters()).device
    scores = []

    for q, a in tqdm(zip(questions, answers), total=len(questions), desc="NLL"):
        prompt = f"Question: {q}\nAnswer: {a}"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model(**inputs, labels=inputs.input_ids)
        nll = outputs.loss.item()
        scores.append(-nll)  # Negative so higher = more confident

    return np.array(scores)


# =============================================================================
# Unsupervised methods (on embeddings)
# =============================================================================


def l2_norm_baseline(X: Float[np.ndarray, "n dim"]) -> Float[np.ndarray, "n"]:
    """L2 norm of embeddings as confidence proxy."""
    return np.linalg.norm(X, axis=1)


def reconstruction_error_baseline(
    X: Float[np.ndarray, "n dim"],
    n_components: int = 32,
) -> Float[np.ndarray, "n"]:
    """PCA reconstruction error as anomaly/uncertainty proxy."""
    pca = PCA(n_components=min(n_components, X.shape[1], X.shape[0]))
    X_reduced = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_reduced)
    errors = np.linalg.norm(X - X_reconstructed, axis=1)
    return -errors  # Negative: lower error = more typical = more confident


def lof_baseline(
    X: Float[np.ndarray, "n dim"],
    n_neighbors: int = 20,
) -> Float[np.ndarray, "n"]:
    """Local Outlier Factor as uncertainty proxy."""
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False)
    lof.fit(X)
    scores = -lof.negative_outlier_factor_  # Higher = more outlier
    return -scores  # Negative: less outlier = more confident


def cluster_uncertainty_baseline(
    X: Float[np.ndarray, "n dim"],
    n_clusters: int = 8,
) -> Float[np.ndarray, "n"]:
    """Distance to nearest cluster centroid as uncertainty."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    distances = kmeans.transform(X).min(axis=1)
    return -distances  # Negative: closer to cluster = more confident


def local_dimension_baseline(
    X: Float[np.ndarray, "n dim"],
    k: int = 10,
) -> Float[np.ndarray, "n"]:
    """Local intrinsic dimension as confidence proxy."""
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, _ = nbrs.kneighbors(X)
    distances = distances[:, 1:]  # Remove self
    distances = np.maximum(distances, 1e-10)

    T_k = distances[:, -1:]
    log_ratios = np.log(T_k / distances[:, :-1])
    dim_estimates = (k - 1) / np.sum(log_ratios, axis=1)
    dim_estimates = np.clip(dim_estimates, 0, X.shape[1])

    return -dim_estimates  # Negative: lower dimension = more structured = confident


def evaluate_baselines(
    X: Float[np.ndarray, "n dim"],
    y: np.ndarray,
) -> dict[str, BaselineResult]:
    """
    Evaluate all unsupervised baselines.

    Returns dict mapping name to BaselineResult.
    """
    baselines = {
        "l2_norm": l2_norm_baseline,
        "reconstruction_error": reconstruction_error_baseline,
        "lof": lof_baseline,
        "cluster_uncertainty": cluster_uncertainty_baseline,
        "local_dimension": local_dimension_baseline,
    }

    results = {}
    for name, fn in baselines.items():
        try:
            scores = fn(X)
            auc = roc_auc_score(y, scores)
        except Exception:
            scores = np.zeros(len(y))
            auc = 0.5

        results[name] = BaselineResult(name=name, auc=auc, scores=scores)

    return results


def evaluate_output_baselines(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    questions: list[str],
    answers: list[str],
    y: np.ndarray,
    include_semantic_entropy: bool = False,
    semantic_entropy_mode: Literal["simple", "nli"] = "simple",
    strict_entailment: bool = False,
) -> dict[str, BaselineResult]:
    """
    Evaluate output-based baselines.

    Returns dict mapping name to BaselineResult.
    """
    results = {}

    # P(True)
    try:
        scores = p_true_baseline(model, tokenizer, questions, answers)
        auc = roc_auc_score(y, scores)
        results["p_true"] = BaselineResult(name="p_true", auc=auc, scores=scores)
    except Exception as e:
        print(f"P(True) failed: {e}")

    # Token entropy
    try:
        scores = token_entropy_baseline(model, tokenizer, questions, answers)
        auc = roc_auc_score(y, scores)
        results["entropy"] = BaselineResult(name="entropy", auc=auc, scores=scores)
    except Exception as e:
        print(f"Entropy failed: {e}")

    # NLL
    try:
        scores = nll_baseline(model, tokenizer, questions, answers)
        auc = roc_auc_score(y, scores)
        results["nll"] = BaselineResult(name="nll", auc=auc, scores=scores)
    except Exception as e:
        print(f"NLL failed: {e}")

    # Semantic entropy (optional)
    if include_semantic_entropy:
        try:
            scores = semantic_entropy_baseline(
                model,
                tokenizer,
                questions,
                mode=semantic_entropy_mode,
                strict_entailment=strict_entailment,
            )
            auc = roc_auc_score(y, scores)
            results["semantic_entropy"] = BaselineResult(
                name="semantic_entropy",
                auc=auc,
                scores=scores,
            )
        except Exception as e:
            print(f"Semantic entropy failed: {e}")

    return results
