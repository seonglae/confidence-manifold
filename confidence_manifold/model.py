"""Model loading and hidden state extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


def get_device() -> str:
    """Detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(device: str) -> torch.dtype:
    """Get optimal dtype for device."""
    if device == "cuda":
        return torch.bfloat16
    return torch.float32  # MPS doesn't support bfloat16


def load_model(
    model_name: str,
    device: str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load model and tokenizer with optimal settings.

    Args:
        model_name: HuggingFace model name or path
        device: Device to load to (auto-detected if None)
        dtype: Tensor dtype (auto-selected if None)

    Returns:
        Tuple of (model, tokenizer)
    """
    if device is None:
        device = get_device()
    if dtype is None:
        dtype = get_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            output_hidden_states=True,
            trust_remote_code=True,
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device if device == "cuda" else None,
            torch_dtype=dtype,
            output_hidden_states=True,
            trust_remote_code=True,
        )
    model.eval()
    return model, tokenizer


def find_layers(model: PreTrainedModel) -> list:
    """Find transformer layers in model (architecture-agnostic)."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)  # Llama, Qwen, Mistral
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)  # GPT-2
    if hasattr(model, "model") and hasattr(model.model, "decoder"):
        return list(model.model.decoder.layers)  # Some encoder-decoder
    raise ValueError("Cannot find transformer layers")


@dataclass
class ExtractionResult:
    """Result of hidden state extraction."""

    hidden_states: Float[Tensor, "layers dim"]  # Per-layer activations
    logits: Float[Tensor, "vocab"]  # Output logits
    tokens: list[int]  # Token IDs


class Extractor:
    """Extract hidden states from transformer models."""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.num_layers = model.config.num_hidden_layers
        self.hidden_dim = model.config.hidden_size

    def format_prompt(self, question: str, answer: str) -> str:
        """Format as chat or simple prompt."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                pass
        return f"Question: {question}\nAnswer: {answer}"

    @torch.no_grad()
    def extract(
        self,
        question: str,
        answer: str,
        position: Literal["last", "mean"] = "last",
    ) -> ExtractionResult:
        """
        Extract hidden states for a question-answer pair.

        Args:
            question: Question text
            answer: Answer text
            position: "last" for last token, "mean" for mean pooling

        Returns:
            ExtractionResult with hidden states from all layers
        """
        text = self.format_prompt(question, answer)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        # Stack layer outputs: (num_layers, seq, hidden)
        all_hidden = torch.stack(outputs.hidden_states[1:], dim=0).squeeze(1)

        if position == "last":
            hidden = all_hidden[:, -1, :]  # (layers, hidden)
        else:
            hidden = all_hidden.mean(dim=1)  # (layers, hidden)

        return ExtractionResult(
            hidden_states=hidden,
            logits=outputs.logits[0, -1, :],
            tokens=inputs["input_ids"].squeeze(0).tolist(),
        )

    @torch.no_grad()
    def extract_batch(
        self,
        questions: list[str],
        answers: list[str],
        position: Literal["last", "mean"] = "last",
    ) -> list[ExtractionResult]:
        """Extract hidden states for batch of samples."""
        return [self.extract(q, a, position) for q, a in zip(questions, answers)]

    @torch.no_grad()
    def extract_layer(
        self,
        question: str,
        answer: str,
        layer: int,
        position: Literal["last", "mean"] = "last",
    ) -> Float[Tensor, "dim"]:
        """Extract hidden state from specific layer."""
        result = self.extract(question, answer, position)
        return result.hidden_states[layer]
