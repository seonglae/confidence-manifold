"""Confidence Manifold: Geometric analysis of confidence representations in LLMs."""

from confidence_manifold.model import load_model, Extractor
from confidence_manifold.probe import (
    train_probe,
    evaluate_probe,
    ProbeResult,
    cross_validate,
    nested_cross_validate,
    dimension_sweep,
    compare_classifiers,
    centroid_classifier,
    fewshot_evaluation,
)
from confidence_manifold.steering import Steerer, SteeringResult
from confidence_manifold.geometry import estimate_dimension, pca_dimension
from confidence_manifold.data import load_dataset, Sample
from confidence_manifold.baselines import evaluate_baselines, evaluate_output_baselines

from confidence_manifold.semantic_entropy import (
    SemanticEntropyProbe,
    SemanticEntropyResult,
    semantic_entropy_scores,
)
from confidence_manifold.sae import (
    load_pretrained_sae,
    get_available_models,
    get_sparse_features,
    encode_with_pretrained_sae,
    PretrainedSAEResult,
)

__version__ = "0.1.0"
__all__ = [
    "load_model",
    "Extractor",
    "train_probe",
    "evaluate_probe",
    "ProbeResult",
    "cross_validate",
    "nested_cross_validate",
    "dimension_sweep",
    "compare_classifiers",
    "centroid_classifier",
    "fewshot_evaluation",
    "Steerer",
    "SteeringResult",
    "estimate_dimension",
    "pca_dimension",
    "load_dataset",
    "Sample",
    "evaluate_baselines",
    "evaluate_output_baselines",
    "SemanticEntropyProbe",
    "SemanticEntropyResult",
    "semantic_entropy_scores",
    "load_pretrained_sae",
    "get_available_models",
    "get_sparse_features",
    "encode_with_pretrained_sae",
    "PretrainedSAEResult",
]
