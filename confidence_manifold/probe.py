"""Linear probes for correctness detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from jaxtyping import Float
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeResult:
    """Results from probe training/evaluation."""

    auc: float
    accuracy: float
    direction: Float[np.ndarray, "dim"]  # Normalized probe direction
    scaler: StandardScaler
    probe: LogisticRegression
    pls: PLSRegression | None = None


@dataclass
class CVResult:
    """Cross-validation results."""

    mean_auc: float
    std_auc: float
    fold_aucs: list[float]
    best_layer: int | None = None
    best_dim: int | None = None


def train_probe(
    X: Float[np.ndarray, "n dim"],
    y: np.ndarray,
    pls_dim: int | None = 8,
    regularization: Literal["l1", "l2"] = "l2",
    C: float = 0.1,
) -> ProbeResult:
    """
    Train a linear probe on hidden states.

    Args:
        X: Hidden state activations (n_samples, hidden_dim)
        y: Binary labels (0=incorrect, 1=correct)
        pls_dim: PLS dimension reduction (None to skip)
        regularization: L1 or L2 regularization
        C: Regularization strength (smaller = stronger)

    Returns:
        ProbeResult with trained probe and direction
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pls = None
    if pls_dim is not None and pls_dim < X.shape[1]:
        pls = PLSRegression(n_components=pls_dim)
        pls.fit(X_scaled, y)
        X_scaled = pls.transform(X_scaled)

    probe = LogisticRegression(
        penalty=regularization,
        C=C,
        solver="saga" if regularization == "l1" else "lbfgs",
        max_iter=1000,
        class_weight="balanced",
    )
    probe.fit(X_scaled, y)

    # Get probe direction (in original space if PLS used)
    direction = probe.coef_[0]
    if pls is not None:
        direction = pls.inverse_transform(direction.reshape(1, -1)).flatten()
    direction = direction / np.linalg.norm(direction)

    # Evaluate on training data
    probs = probe.predict_proba(X_scaled)[:, 1]
    preds = probe.predict(X_scaled)

    return ProbeResult(
        auc=roc_auc_score(y, probs),
        accuracy=accuracy_score(y, preds),
        direction=direction,
        scaler=scaler,
        probe=probe,
        pls=pls,
    )


def evaluate_probe(
    probe_result: ProbeResult,
    X: Float[np.ndarray, "n dim"],
    y: np.ndarray,
) -> tuple[float, float]:
    """
    Evaluate trained probe on test data.

    Returns:
        Tuple of (AUC, accuracy)
    """
    X_scaled = probe_result.scaler.transform(X)
    if probe_result.pls is not None:
        X_scaled = probe_result.pls.transform(X_scaled)

    probs = probe_result.probe.predict_proba(X_scaled)[:, 1]
    preds = probe_result.probe.predict(X_scaled)

    return roc_auc_score(y, probs), accuracy_score(y, preds)


def cross_validate(
    X: Float[np.ndarray, "n dim"],
    y: np.ndarray,
    groups: np.ndarray | None = None,
    n_folds: int = 5,
    pls_dim: int | None = 8,
    C: float = 0.1,
) -> CVResult:
    """
    Cross-validate probe with GroupKFold (prevents question leakage).

    Args:
        X: Hidden states (n_samples, hidden_dim)
        y: Labels
        groups: Group IDs for GroupKFold (e.g., question IDs)
        n_folds: Number of folds
        pls_dim: PLS dimensions
        C: Regularization strength

    Returns:
        CVResult with mean/std AUC
    """
    if groups is not None:
        kfold = GroupKFold(n_splits=n_folds)
        splits = kfold.split(X, y, groups)
    else:
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = kfold.split(X, y)

    fold_aucs = []
    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        result = train_probe(X_train, y_train, pls_dim=pls_dim, C=C)
        auc, _ = evaluate_probe(result, X_test, y_test)
        fold_aucs.append(auc)

    return CVResult(
        mean_auc=np.mean(fold_aucs),
        std_auc=np.std(fold_aucs),
        fold_aucs=fold_aucs,
    )


def dimension_sweep(
    X: Float[np.ndarray, "n dim"],
    y: np.ndarray,
    groups: np.ndarray | None = None,
    dims: list[int] | None = None,
    n_folds: int = 5,
) -> dict[int, CVResult]:
    """
    Sweep PLS dimensions to find optimal.

    Returns:
        Dict mapping dimension to CVResult
    """
    if dims is None:
        dims = [1, 2, 3, 4, 5, 6, 8, 12, 16, 32]

    results = {}
    for dim in dims:
        if dim >= X.shape[1]:
            continue
        results[dim] = cross_validate(X, y, groups, n_folds, pls_dim=dim)
    return results


def centroid_classifier(
    X_train: Float[np.ndarray, "n dim"],
    y_train: np.ndarray,
    X_test: Float[np.ndarray, "m dim"],
) -> Float[np.ndarray, "m"]:
    """
    Simple centroid-based classifier (matches probe performance).

    Returns probability of correct (y=1).
    """
    mu_correct = X_train[y_train == 1].mean(axis=0)
    mu_incorrect = X_train[y_train == 0].mean(axis=0)

    dist_correct = np.linalg.norm(X_test - mu_correct, axis=1)
    dist_incorrect = np.linalg.norm(X_test - mu_incorrect, axis=1)

    # Softmax over negative distances
    exp_correct = np.exp(-dist_correct)
    exp_incorrect = np.exp(-dist_incorrect)

    return exp_correct / (exp_correct + exp_incorrect)


def mahalanobis_classifier(
    X_train: Float[np.ndarray, "n dim"],
    y_train: np.ndarray,
    X_test: Float[np.ndarray, "m dim"],
) -> Float[np.ndarray, "m"]:
    """Mahalanobis distance classifier with class-specific covariance."""
    from scipy.spatial.distance import mahalanobis

    X_correct = X_train[y_train == 1]
    X_incorrect = X_train[y_train == 0]

    mu_c = X_correct.mean(axis=0)
    mu_i = X_incorrect.mean(axis=0)

    # Shared covariance (pooled)
    cov = np.cov(X_train.T) + 1e-6 * np.eye(X_train.shape[1])
    cov_inv = np.linalg.inv(cov)

    probs = []
    for x in X_test:
        d_c = mahalanobis(x, mu_c, cov_inv)
        d_i = mahalanobis(x, mu_i, cov_inv)
        p = np.exp(-d_c) / (np.exp(-d_c) + np.exp(-d_i))
        probs.append(p)

    return np.array(probs)


def knn_classifier(
    X_train: Float[np.ndarray, "n dim"],
    y_train: np.ndarray,
    X_test: Float[np.ndarray, "m dim"],
    k: int = 10,
) -> Float[np.ndarray, "m"]:
    """K-nearest neighbors classifier."""
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn.predict_proba(X_test)[:, 1]


def svm_classifier(
    X_train: Float[np.ndarray, "n dim"],
    y_train: np.ndarray,
    X_test: Float[np.ndarray, "m dim"],
) -> Float[np.ndarray, "m"]:
    """RBF kernel SVM classifier."""
    from sklearn.svm import SVC

    svm = SVC(kernel="rbf", probability=True, class_weight="balanced")
    svm.fit(X_train, y_train)
    return svm.predict_proba(X_test)[:, 1]


def compare_classifiers(
    X: Float[np.ndarray, "n dim"],
    y: np.ndarray,
    groups: np.ndarray | None = None,
    pls_dim: int = 8,
    n_folds: int = 5,
) -> dict[str, CVResult]:
    """
    Compare all geometric classifiers via cross-validation.

    PLS is fitted per-fold on training data only to prevent leakage.

    Returns dict mapping classifier name to CVResult.
    """
    classifiers = {
        "linear": lambda Xtr, ytr, Xte: train_probe(Xtr, ytr, pls_dim=None).probe.predict_proba(
            StandardScaler().fit(Xtr).transform(Xte)
        )[:, 1],
        "centroid": centroid_classifier,
        "mahalanobis": mahalanobis_classifier,
        "knn": knn_classifier,
        "svm": svm_classifier,
    }

    if groups is not None:
        kfold = GroupKFold(n_splits=n_folds)
        splits = list(kfold.split(X, y, groups))
    else:
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = list(kfold.split(X, y))

    results = {}
    for name, clf_fn in classifiers.items():
        fold_aucs = []
        for train_idx, test_idx in splits:
            X_train_raw, X_test_raw = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Fit scaler and PLS on training data only
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_raw)
            X_test_scaled = scaler.transform(X_test_raw)

            if pls_dim < X.shape[1]:
                pls = PLSRegression(n_components=pls_dim)
                pls.fit(X_train_scaled, y_train)
                X_train = pls.transform(X_train_scaled)
                X_test = pls.transform(X_test_scaled)
            else:
                X_train, X_test = X_train_scaled, X_test_scaled

            try:
                probs = clf_fn(X_train, y_train, X_test)
                auc = roc_auc_score(y_test, probs)
            except Exception:
                auc = 0.5
            fold_aucs.append(auc)

        results[name] = CVResult(
            mean_auc=np.mean(fold_aucs),
            std_auc=np.std(fold_aucs),
            fold_aucs=fold_aucs,
        )

    return results


def nested_cross_validate(
    X: Float[np.ndarray, "n dim"],
    y: np.ndarray,
    groups: np.ndarray | None = None,
    outer_folds: int = 5,
    inner_folds: int = 3,
    dims: list[int] | None = None,
) -> CVResult:
    """
    Nested cross-validation for unbiased hyperparameter selection.

    Outer loop: evaluation (GroupKFold)
    Inner loop: dimension selection (StratifiedKFold)

    Returns:
        CVResult with unbiased AUC estimates
    """
    if dims is None:
        dims = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16]

    if groups is not None:
        outer_kfold = GroupKFold(n_splits=outer_folds)
        outer_splits = list(outer_kfold.split(X, y, groups))
    else:
        outer_kfold = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)
        outer_splits = list(outer_kfold.split(X, y))

    fold_aucs = []
    selected_dims = []

    for train_idx, test_idx in outer_splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Inner CV for dimension selection
        inner_kfold = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=42)
        dim_scores = {d: [] for d in dims if d < X_train.shape[1]}

        for inner_train, inner_val in inner_kfold.split(X_train, y_train):
            X_inner_train = X_train[inner_train]
            X_inner_val = X_train[inner_val]
            y_inner_train = y_train[inner_train]
            y_inner_val = y_train[inner_val]

            for dim in dim_scores:
                try:
                    result = train_probe(X_inner_train, y_inner_train, pls_dim=dim)
                    auc, _ = evaluate_probe(result, X_inner_val, y_inner_val)
                    dim_scores[dim].append(auc)
                except Exception:
                    dim_scores[dim].append(0.5)

        # Select best dimension
        best_dim = max(dim_scores, key=lambda d: np.mean(dim_scores[d]))
        selected_dims.append(best_dim)

        # Train final model with best dim, evaluate on outer test
        result = train_probe(X_train, y_train, pls_dim=best_dim)
        auc, _ = evaluate_probe(result, X_test, y_test)
        fold_aucs.append(auc)

    return CVResult(
        mean_auc=np.mean(fold_aucs),
        std_auc=np.std(fold_aucs),
        fold_aucs=fold_aucs,
        best_dim=int(np.median(selected_dims)),
    )


def fewshot_evaluation(
    X: Float[np.ndarray, "n dim"],
    y: np.ndarray,
    n_shots: list[int] = [5, 25, 100, 200],
    n_trials: int = 10,
    pls_dim: int = 8,
) -> dict[int, dict[str, float]]:
    """
    Evaluate few-shot label efficiency.

    Returns dict mapping n_shots to {mean_auc, std_auc}.
    """
    results = {}

    for n in n_shots:
        if 2 * n > len(y):
            continue

        aucs = []
        for seed in range(n_trials):
            rng = np.random.RandomState(seed)

            # Sample n per class
            idx_pos = np.where(y == 1)[0]
            idx_neg = np.where(y == 0)[0]
            train_idx = np.concatenate([
                rng.choice(idx_pos, min(n, len(idx_pos)), replace=False),
                rng.choice(idx_neg, min(n, len(idx_neg)), replace=False),
            ])
            test_idx = np.array([i for i in range(len(y)) if i not in train_idx])

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Centroid classifier
            probs = centroid_classifier(X_train, y_train, X_test)
            aucs.append(roc_auc_score(y_test, probs))

        results[n] = {"mean_auc": np.mean(aucs), "std_auc": np.std(aucs)}

    return results
