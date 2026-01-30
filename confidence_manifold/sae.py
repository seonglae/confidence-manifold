"""Pre-trained SAE loader using SAELens from Neuronpedia."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

try:
    from sae_lens import SAE
    SAE_LENS_AVAILABLE = True
except ImportError:
    SAE_LENS_AVAILABLE = False
    SAE = None


@dataclass
class PretrainedSAEConfig:
    """Configuration for pre-trained SAE from Neuronpedia."""

    release: str
    id_template: str
    sae_type: str
    n_layers: int


PRETRAINED_SAES: Dict[str, PretrainedSAEConfig] = {
    "gemma-2b": PretrainedSAEConfig(
        release="gemma-scope-2b-pt-res-canonical",
        id_template="layer_{}/width_16k/canonical",
        sae_type="residual",
        n_layers=26,
    ),
    "llama-3.1-8b": PretrainedSAEConfig(
        release="llama_scope_lxr_8x",
        id_template="l{}r_8x",
        sae_type="residual",
        n_layers=32,
    ),
    "gpt2-small": PretrainedSAEConfig(
        release="gpt2-small-res-jb",
        id_template="blocks.{}.hook_resid_pre",
        sae_type="residual",
        n_layers=12,
    ),
}


@dataclass
class PretrainedSAEResult:
    """Result from loading a pre-trained SAE."""

    sae: Any
    config: Dict[str, Any]
    log_sparsity: Optional[torch.Tensor]
    feature_density: Optional[np.ndarray]
    model_name: str
    layer: int


def load_pretrained_sae(
    model_name: str,
    layer: int,
    device: str | None = None,
) -> PretrainedSAEResult:
    """Load a pre-trained SAE from Neuronpedia via SAELens."""
    if not SAE_LENS_AVAILABLE:
        raise ImportError(
            "sae-lens is required for pre-trained SAEs. Install with: pip install sae-lens"
        )

    if model_name not in PRETRAINED_SAES:
        available = ", ".join(PRETRAINED_SAES.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

    config = PRETRAINED_SAES[model_name]
    if layer >= config.n_layers:
        raise ValueError(
            f"Layer {layer} exceeds max layer {config.n_layers - 1} for {model_name}"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sae_id = config.id_template.format(layer)
    sae, cfg_dict, log_sparsity = SAE.from_pretrained(
        release=config.release,
        sae_id=sae_id,
        device=device,
    )

    feature_density = None
    if log_sparsity is not None:
        feature_density = torch.exp(log_sparsity).cpu().numpy()

    return PretrainedSAEResult(
        sae=sae,
        config=cfg_dict,
        log_sparsity=log_sparsity,
        feature_density=feature_density,
        model_name=model_name,
        layer=layer,
    )


def get_available_models() -> Dict[str, PretrainedSAEConfig]:
    """Get all available pre-trained SAE models."""
    return PRETRAINED_SAES.copy()


def encode_with_pretrained_sae(
    sae_result: PretrainedSAEResult,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode hidden states using a pre-trained SAE."""
    sae = sae_result.sae

    with torch.no_grad():
        features = sae.encode(hidden_states)
        reconstructed = sae.decode(features)

    return features, reconstructed


def get_sparse_features(
    sae_result: PretrainedSAEResult,
    threshold: float = 0.01,
) -> np.ndarray:
    """Get indices of sparse features (low density)."""
    if sae_result.feature_density is None:
        raise ValueError("Feature density not available for this SAE")

    return np.where(sae_result.feature_density < threshold)[0]


def get_dense_features(
    sae_result: PretrainedSAEResult,
    threshold: float = 0.1,
) -> np.ndarray:
    """Get indices of dense features (high density)."""
    if sae_result.feature_density is None:
        raise ValueError("Feature density not available for this SAE")

    return np.where(sae_result.feature_density >= threshold)[0]
