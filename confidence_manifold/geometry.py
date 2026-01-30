"""Geometric analysis of confidence manifolds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


@dataclass
class DimensionResult:
    """Result of dimension estimation."""

    dimension: float
    method: str
    details: dict | None = None


def estimate_dimension(
    X: Float[np.ndarray, "n dim"],
    k: int = 10,
    aggregate: str = "mean",
) -> float:
    """
    Levina-Bickel MLE estimator for intrinsic dimension.

    Reference: "Maximum Likelihood Estimation of Intrinsic Dimension" (NIPS 2004)

    Args:
        X: Data matrix (n_samples, n_features)
        k: Number of nearest neighbors
        aggregate: "mean" or "median" over per-point estimates

    Returns:
        Estimated intrinsic dimension
    """
    n_samples, n_features = X.shape

    if k >= n_samples:
        k = n_samples - 1

    # Find k+1 nearest neighbors (includes self)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X)
    distances, _ = nbrs.kneighbors(X)
    distances = distances[:, 1:]  # Remove self-distance
    distances = np.maximum(distances, 1e-10)

    # MLE estimate: d = (k-1) / sum(log(T_k / T_j))
    T_k = distances[:, -1:]
    log_ratios = np.log(T_k / distances[:, :-1])
    sum_log_ratios = np.sum(log_ratios, axis=1)
    sum_log_ratios = np.where(np.abs(sum_log_ratios) < 1e-10, 1e-10, sum_log_ratios)
    dim_estimates = (k - 1) / sum_log_ratios

    # Filter invalid
    valid = np.isfinite(dim_estimates) & (dim_estimates > 0) & (dim_estimates < n_features)
    dim_estimates = dim_estimates[valid]

    if len(dim_estimates) == 0:
        return float("nan")

    return float(np.mean(dim_estimates) if aggregate == "mean" else np.median(dim_estimates))


def pca_dimension(
    X: Float[np.ndarray, "n dim"],
    threshold: float = 0.95,
) -> tuple[int, Float[np.ndarray, "k"]]:
    """
    Estimate dimension as number of PCs explaining threshold variance.

    Returns:
        Tuple of (n_components, explained_variance_ratio)
    """
    n_components = min(X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(X)

    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_dim = int(np.searchsorted(cumsum, threshold) + 1)

    return n_dim, pca.explained_variance_ratio_


def correlation_dimension(
    X: Float[np.ndarray, "n dim"],
    n_radii: int = 20,
) -> float:
    """
    Estimate correlation dimension via box-counting.

    Computes C(r) = fraction of pairs within distance r,
    then estimates d from log(C(r)) ~ d * log(r).
    """
    distances = pdist(X)

    r_min = np.percentile(distances, 1)
    r_max = np.percentile(distances, 99)
    radii = np.logspace(np.log10(max(r_min, 1e-10)), np.log10(r_max), n_radii)

    n_pairs = len(distances)
    correlation = np.array([np.sum(distances < r) / n_pairs for r in radii])

    valid = (correlation > 0) & (correlation < 1)
    if np.sum(valid) < 3:
        return float("nan")

    log_r = np.log(radii[valid])
    log_c = np.log(correlation[valid])
    slope, _ = np.polyfit(log_r, log_c, 1)

    return float(slope)


def layer_dimension_profile(
    hidden_states: dict[int, Float[np.ndarray, "n dim"]],
    method: str = "mle",
    **kwargs,
) -> dict[int, float]:
    """
    Estimate intrinsic dimension at each layer.

    Args:
        hidden_states: Dict mapping layer index to activations
        method: "mle", "pca", or "correlation"

    Returns:
        Dict mapping layer index to estimated dimension
    """
    results = {}
    for layer_idx, X in hidden_states.items():
        if method == "mle":
            dim = estimate_dimension(X, **kwargs)
        elif method == "pca":
            dim, _ = pca_dimension(X, **kwargs)
        elif method == "correlation":
            dim = correlation_dimension(X, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        results[layer_idx] = dim
    return results


def grassmannian_distance(
    U: Float[np.ndarray, "dim k1"],
    V: Float[np.ndarray, "dim k2"],
) -> float:
    """
    Compute Grassmannian distance between two subspaces.

    Based on principal angles between subspaces.
    """
    # Orthonormalize
    U, _ = np.linalg.qr(U)
    V, _ = np.linalg.qr(V)

    # Singular values of U^T V give cos of principal angles
    M = U.T @ V
    _, s, _ = np.linalg.svd(M)
    s = np.clip(s, -1, 1)

    # Grassmannian distance = sqrt(sum of squared angles)
    angles = np.arccos(s)
    return float(np.sqrt(np.sum(angles**2)))


def probe_alignment(
    direction1: Float[np.ndarray, "dim"],
    direction2: Float[np.ndarray, "dim"],
) -> float:
    """
    Compute cosine similarity between two probe directions.

    Returns absolute value (directions are antipodal-invariant).
    """
    d1 = direction1 / np.linalg.norm(direction1)
    d2 = direction2 / np.linalg.norm(direction2)
    return float(np.abs(d1 @ d2))


def cross_layer_similarity(
    directions: dict[int, Float[np.ndarray, "dim"]],
) -> Float[np.ndarray, "n n"]:
    """
    Compute pairwise cosine similarity matrix between layer directions.

    Returns:
        Similarity matrix (n_layers, n_layers)
    """
    layers = sorted(directions.keys())
    n = len(layers)
    sim = np.zeros((n, n))

    for i, l1 in enumerate(layers):
        for j, l2 in enumerate(layers):
            sim[i, j] = probe_alignment(directions[l1], directions[l2])

    return sim
