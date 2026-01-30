"""Dataset loaders for correctness detection experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterator, Literal

from datasets import load_dataset as hf_load_dataset


@dataclass
class Sample:
    """A single sample for correctness detection."""

    question: str
    answer: str
    correct: bool  # True = correct/factual, False = incorrect/hallucinated
    question_id: int | str | None = None  # For GroupKFold
    metadata: dict = field(default_factory=dict)


DatasetName = Literal[
    "truthfulqa", "fever", "sciq", "commonsenseqa", "halueval", "triviaqa", "gsm8k"
]


def load_dataset(
    name: DatasetName,
    max_samples: int | None = None,
    seed: int = 42,
) -> list[Sample]:
    """
    Load dataset by name.

    Args:
        name: Dataset name
        max_samples: Maximum samples to load
        seed: Random seed for shuffling negatives

    Returns:
        List of Sample objects with correct/incorrect pairs
    """
    random.seed(seed)
    loaders = {
        "truthfulqa": _load_truthfulqa,
        "fever": _load_fever,
        "sciq": _load_sciq,
        "commonsenseqa": _load_commonsenseqa,
        "halueval": _load_halueval,
        "triviaqa": _load_triviaqa,
        "gsm8k": _load_gsm8k,
    }

    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(loaders.keys())}")

    samples = loaders[name](max_samples)
    return samples


def _load_truthfulqa(max_samples: int | None) -> list[Sample]:
    """Load TruthfulQA with paired correct/incorrect answers."""
    ds = hf_load_dataset("truthfulqa/truthful_qa", "generation", split="validation")

    samples = []
    for idx, item in enumerate(ds):
        q = item["question"]

        # Correct answers
        for ans in item["correct_answers"]:
            samples.append(Sample(question=q, answer=ans, correct=True, question_id=idx))

        # Incorrect answers
        for ans in item["incorrect_answers"]:
            samples.append(Sample(question=q, answer=ans, correct=False, question_id=idx))

        if max_samples and len(samples) >= max_samples:
            break

    return samples[:max_samples] if max_samples else samples


def _load_fever(max_samples: int | None) -> list[Sample]:
    """Load FEVER fact verification."""
    try:
        ds = hf_load_dataset("copenlu/fever_gold_evidence", split="validation")
    except Exception:
        ds = hf_load_dataset("pierreguillou/FEVER", split="validation")

    samples = []
    for idx, item in enumerate(ds):
        if item["label"] == "NOT ENOUGH INFO":
            continue

        samples.append(
            Sample(
                question="Is this statement factually accurate?",
                answer=item["claim"],
                correct=(item["label"] == "SUPPORTS"),
                question_id=idx,
            )
        )

        if max_samples and len(samples) >= max_samples:
            break

    return samples[:max_samples] if max_samples else samples


def _load_sciq(max_samples: int | None) -> list[Sample]:
    """Load SciQ science questions."""
    ds = hf_load_dataset("allenai/sciq", split="validation")

    samples = []
    for idx, item in enumerate(ds):
        q = item["question"]
        correct = item["correct_answer"]
        wrong = random.choice([item["distractor1"], item["distractor2"], item["distractor3"]])

        samples.append(Sample(question=q, answer=correct, correct=True, question_id=idx))
        samples.append(Sample(question=q, answer=wrong, correct=False, question_id=idx))

        if max_samples and len(samples) >= max_samples:
            break

    return samples[:max_samples] if max_samples else samples


def _load_commonsenseqa(max_samples: int | None) -> list[Sample]:
    """Load CommonsenseQA."""
    ds = hf_load_dataset("tau/commonsense_qa", split="validation")

    samples = []
    for idx, item in enumerate(ds):
        q = item["question"]
        labels = item["choices"]["label"]
        texts = item["choices"]["text"]
        correct_idx = labels.index(item["answerKey"])

        correct = texts[correct_idx]
        wrong = random.choice([t for i, t in enumerate(texts) if i != correct_idx])

        samples.append(Sample(question=q, answer=correct, correct=True, question_id=idx))
        samples.append(Sample(question=q, answer=wrong, correct=False, question_id=idx))

        if max_samples and len(samples) >= max_samples:
            break

    return samples[:max_samples] if max_samples else samples


def _load_halueval(max_samples: int | None) -> list[Sample]:
    """Load HaluEval QA samples."""
    ds = hf_load_dataset("pminervini/HaluEval", "qa_samples", split="data")

    samples = []
    for idx, item in enumerate(ds):
        is_correct = item.get("hallucination", "").lower() != "yes"
        samples.append(
            Sample(
                question=item["question"],
                answer=item["answer"],
                correct=is_correct,
                question_id=idx,
            )
        )

        if max_samples and len(samples) >= max_samples:
            break

    return samples[:max_samples] if max_samples else samples


def _load_triviaqa(max_samples: int | None) -> list[Sample]:
    """Load TriviaQA with shuffled negatives."""
    ds = hf_load_dataset("trivia_qa", "rc.nocontext", split="validation")

    correct_items = []
    all_answers = []

    for item in ds:
        if not item["answer"] or not item["answer"].get("value"):
            continue
        correct_items.append({"q": item["question"], "a": item["answer"]["value"]})
        all_answers.append(item["answer"]["value"])
        if max_samples and len(correct_items) >= max_samples // 2:
            break

    samples = []
    for idx, item in enumerate(correct_items):
        samples.append(Sample(question=item["q"], answer=item["a"], correct=True, question_id=idx))
        wrong = random.choice([a for a in all_answers if a != item["a"]])
        samples.append(Sample(question=item["q"], answer=wrong, correct=False, question_id=idx))

    return samples[:max_samples] if max_samples else samples


def _load_gsm8k(max_samples: int | None) -> list[Sample]:
    """Load GSM8K math with perturbed negatives."""
    ds = hf_load_dataset("openai/gsm8k", "main", split="test")

    samples = []
    for idx, item in enumerate(ds):
        if "####" not in item["answer"]:
            continue

        final = item["answer"].split("####")[-1].strip()
        try:
            num = float(final.replace(",", ""))
        except ValueError:
            continue

        samples.append(
            Sample(question=item["question"], answer=f"The answer is {final}", correct=True, question_id=idx)
        )

        # Perturb for incorrect
        wrong_num = num * random.choice([0.5, 0.9, 1.1, 1.5, 2.0])
        wrong_num = int(wrong_num) if num == int(num) else wrong_num
        samples.append(
            Sample(question=item["question"], answer=f"The answer is {wrong_num}", correct=False, question_id=idx)
        )

        if max_samples and len(samples) >= max_samples:
            break

    return samples[:max_samples] if max_samples else samples


def iter_batches(samples: list[Sample], batch_size: int) -> Iterator[list[Sample]]:
    """Yield batches of samples."""
    for i in range(0, len(samples), batch_size):
        yield samples[i : i + batch_size]


def to_arrays(samples: list[Sample]) -> tuple[list[str], list[str], list[bool], list]:
    """Convert samples to parallel arrays."""
    questions = [s.question for s in samples]
    answers = [s.answer for s in samples]
    labels = [s.correct for s in samples]
    groups = [s.question_id for s in samples]
    return questions, answers, labels, groups
