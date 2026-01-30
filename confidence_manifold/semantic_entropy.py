"""Semantic entropy using NLI clustering."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class SemanticCluster:
    """A cluster of semantically equivalent responses."""

    responses: list[str]
    log_probs: list[float]
    representative: str
    cluster_prob: float


@dataclass
class SemanticEntropyResult:
    """Result of semantic entropy computation."""

    question: str
    responses: list[str]
    log_probs: list[float]
    clusters: list[SemanticCluster]
    semantic_entropy: float
    predictive_entropy: float
    n_clusters: int


def generate_multiple_responses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    question: str,
    n_samples: int = 10,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    device: str | None = None,
) -> tuple[list[str], list[float]]:
    """Generate multiple responses and their log probabilities."""
    if device is None:
        device = str(next(model.parameters()).device)

    prompt = f"Q: {question}
A:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    responses: list[str] = []
    log_probs: list[float] = []

    for _ in range(n_samples):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        responses.append(response)

        if hasattr(outputs, "scores") and outputs.scores:
            log_prob = 0.0
            for i, score in enumerate(outputs.scores):
                if i < len(generated_ids):
                    token_id = generated_ids[i]
                    log_probs_step = torch.log_softmax(score[0], dim=-1)
                    log_prob += log_probs_step[token_id].item()
            log_probs.append(log_prob)
        else:
            log_probs.append(float(np.log(1 / max(n_samples, 1))))

    return responses, log_probs


def _get_nli_model():
    """Lazy-load DeBERTa NLI model for entailment checking."""
    if not hasattr(_get_nli_model, "_model"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        _get_nli_model._tokenizer = AutoTokenizer.from_pretrained(model_name)
        _get_nli_model._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _get_nli_model._model.eval()
        if torch.cuda.is_available():
            _get_nli_model._model = _get_nli_model._model.cuda()
    return _get_nli_model._model, _get_nli_model._tokenizer


def check_entailment_nli(
    premise: str,
    hypothesis: str,
    model=None,
    tokenizer=None,
) -> int:
    """Check entailment using NLI model.

    Returns:
        0: contradiction
        1: neutral
        2: entailment
    """
    if model is None or tokenizer is None:
        model, tokenizer = _get_nli_model()

    device = next(model.parameters()).device
    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.logits.argmax(dim=-1).item()


def compute_bidirectional_entailment(
    response1: str,
    response2: str,
    entailment_model=None,
    entailment_tokenizer=None,
    strict: bool = False,
) -> bool:
    """Check semantic equivalence using bidirectional entailment."""
    forward = check_entailment_nli(response1, response2, entailment_model, entailment_tokenizer)
    backward = check_entailment_nli(response2, response1, entailment_model, entailment_tokenizer)

    if strict:
        return forward == 2 and backward == 2

    implications = [forward, backward]
    no_contradiction = 0 not in implications
    not_both_neutral = implications != [1, 1]
    return no_contradiction and not_both_neutral


def cluster_by_semantic_equivalence(
    responses: list[str],
    log_probs: list[float],
    entailment_model=None,
    entailment_tokenizer=None,
    strict: bool = False,
) -> list[SemanticCluster]:
    """Cluster responses by semantic equivalence."""
    n = len(responses)
    cluster_ids = list(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if compute_bidirectional_entailment(
                responses[i],
                responses[j],
                entailment_model,
                entailment_tokenizer,
                strict=strict,
            ):
                old_id, new_id = max(cluster_ids[i], cluster_ids[j]), min(
                    cluster_ids[i], cluster_ids[j]
                )
                for k in range(n):
                    if cluster_ids[k] == old_id:
                        cluster_ids[k] = new_id

    cluster_map: dict[int, list[int]] = defaultdict(list)
    for i, cid in enumerate(cluster_ids):
        cluster_map[cid].append(i)

    clusters: list[SemanticCluster] = []
    for indices in cluster_map.values():
        cluster_responses = [responses[i] for i in indices]
        cluster_log_probs = [log_probs[i] for i in indices]

        cluster_log_prob = np.logaddexp.reduce(cluster_log_probs)
        clusters.append(
            SemanticCluster(
                responses=cluster_responses,
                log_probs=cluster_log_probs,
                representative=cluster_responses[0],
                cluster_prob=float(np.exp(cluster_log_prob)),
            )
        )

    total_prob = sum(c.cluster_prob for c in clusters)
    if total_prob > 0:
        for c in clusters:
            c.cluster_prob /= total_prob

    return clusters


def compute_semantic_entropy(clusters: list[SemanticCluster]) -> float:
    """Compute entropy over semantic clusters."""
    probs = np.array([c.cluster_prob for c in clusters])
    probs = probs[probs > 0]

    if len(probs) == 0:
        return 0.0

    return float(-np.sum(probs * np.log(probs + 1e-10)))


def compute_predictive_entropy(log_probs: list[float]) -> float:
    """Compute predictive entropy over sequence log-probabilities."""
    probs = np.exp(np.array(log_probs))
    probs = probs / (probs.sum() + 1e-10)
    probs = probs[probs > 0]

    return float(-np.sum(probs * np.log(probs + 1e-10)))


class SemanticEntropyProbe:
    """Compute semantic entropy using NLI clustering."""

    def __init__(
        self,
        n_samples: int = 10,
        temperature: float = 0.7,
        max_new_tokens: int = 50,
        strict_entailment: bool = False,
    ) -> None:
        self.n_samples = n_samples
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.strict_entailment = strict_entailment

    def compute_for_question(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        question: str,
        device: str | None = None,
    ) -> SemanticEntropyResult:
        """Compute semantic entropy for a single question."""
        responses, log_probs = generate_multiple_responses(
            model,
            tokenizer,
            question,
            n_samples=self.n_samples,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            device=device,
        )

        clusters = cluster_by_semantic_equivalence(
            responses,
            log_probs,
            strict=self.strict_entailment,
        )

        semantic_entropy = compute_semantic_entropy(clusters)
        predictive_entropy = compute_predictive_entropy(log_probs)

        return SemanticEntropyResult(
            question=question,
            responses=responses,
            log_probs=log_probs,
            clusters=clusters,
            semantic_entropy=semantic_entropy,
            predictive_entropy=predictive_entropy,
            n_clusters=len(clusters),
        )

    def compute_batch(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        questions: list[str],
        device: str | None = None,
        show_progress: bool = True,
    ) -> list[SemanticEntropyResult]:
        """Compute semantic entropy for multiple questions."""
        from tqdm import tqdm

        iterator = tqdm(questions, desc="Semantic entropy") if show_progress else questions
        results: list[SemanticEntropyResult] = []

        for question in iterator:
            results.append(self.compute_for_question(model, tokenizer, question, device))

        return results


def semantic_entropy_scores(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    questions: list[str],
    n_samples: int = 10,
    temperature: float = 0.7,
    max_new_tokens: int = 50,
    strict_entailment: bool = False,
) -> np.ndarray:
    """Convenience wrapper to compute semantic entropy scores for questions."""
    probe = SemanticEntropyProbe(
        n_samples=n_samples,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        strict_entailment=strict_entailment,
    )
    results = probe.compute_batch(model, tokenizer, questions, show_progress=True)
    return np.array([r.semantic_entropy for r in results])
