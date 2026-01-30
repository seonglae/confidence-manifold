# The Confidence Manifold

Geometric Structure of Correctness Representations in Language Models

## Key Findings

When a language model asserts that "the capital of Australia is Sydney," does it *know* this is wrong?

We characterize the geometry of correctness representations across 9 models from 5 architecture families:

| Finding | Details |
|---------|---------|
| **3–8D discriminative subspace** | Performance *degrades* with additional dimensions |
| **Linear separability** | No nonlinear classifier improves over linear |
| **Centroid ≈ Probe** | Simple centroid distance matches trained probe (0.90 AUC) |
| **Few-shot detection** | 25 labeled examples achieve 89% of full-data accuracy |
| **Causal validation** | Steering produces 10.9pp error rate changes; random directions show no effect |
| **Internal >> Output** | Probes: 0.80–0.97 AUC; P(True)/entropy: 0.44–0.64 AUC |

The correctness signal exists internally but is not expressed in outputs. Centroid matching probe performance indicates class separation is a **mean shift**, making detection geometric rather than learned.

## Installation

```bash
uv sync        # recommended
pip install -e .  # alternative
```

## Quick Start

```python
from confidence_manifold import load_model, Extractor, load_dataset, cross_validate
from confidence_manifold.data import to_arrays
import numpy as np

# Load model and data
model, tokenizer = load_model("gpt2")
extractor = Extractor(model, tokenizer)
samples = load_dataset("truthfulqa", max_samples=500)
questions, answers, labels, groups = to_arrays(samples)

# Extract hidden states at optimal layer (L11 for GPT-2)
embeddings = []
for q, a in zip(questions, answers):
    result = extractor.extract(q, a)
    embeddings.append(result.hidden_states[11].cpu().numpy())
X = np.stack(embeddings)
y = np.array(labels, dtype=int)

# Cross-validate with PLS dimension reduction
cv_result = cross_validate(X, y, np.array(groups), pls_dim=8)
print(f"AUC: {cv_result.mean_auc:.3f} ± {cv_result.std_auc:.3f}")
```

## Experiments

```bash
# Probe experiment: layer sweep, dimension sweep, classifier comparison
python probe_experiment.py --model gpt2 --samples 1000

# Geometry experiment: intrinsic dimension, classifier comparison, baselines
python geometry_experiment.py --model qwen2-7b

# Cross-dataset transfer: train TruthfulQA → test SciQ, CSQA, FEVER
python cross_dataset_experiment.py --model qwen2-7b

# Steering experiment: causal validation via activation intervention
python steering_experiment.py --model qwen2-7b --layer 20

# Paraphrase control: verify detection of correctness vs answer style
python paraphrase_experiment.py --model qwen2-7b

# Generation geometry: test if properties persist in model-generated outputs
python generation_experiment.py --model qwen2-7b
```

## Results Summary

**Model Comparison (TruthfulQA, GroupKFold AUC)**

| Model | Size | Optimal Layer | AUC |
|-------|------|---------------|-----|
| Llama-3B | 3B | L12/28 (43%) | 0.97 |
| Qwen2-7B | 7B | L20/28 (75%) | 0.94 |
| Gemma-2B | 2B | L15/26 (62%) | 0.93 |
| Mistral-7B | 7B | L23/32 (75%) | 0.92 |
| GPT-2-Large | 774M | L35/36 (100%) | 0.84 |
| GPT-2 | 124M | L11/12 (100%) | 0.80 |

**Key insight**: Instruction-tuned models peak at mid-layers (43–75%); base models peak at final layers (100%).

## Supported Models

- GPT-2 family (gpt2, gpt2-medium, gpt2-large)
- Qwen2 (1.5B, 7B Instruct)
- Mistral-7B-Instruct
- Llama-3.2 (1B, 3B Instruct)
- Gemma-2-2B-it

## Supported Datasets

- TruthfulQA (primary benchmark)
- FEVER (fact verification)
- SciQ (science QA)
- CommonsenseQA
- HaluEval
- TriviaQA
- GSM8K

**Stack**: PyTorch · Transformers · scikit-learn · jaxtyping

## Citation

```bibtex
@article{cho2026confidence,
  title={The Confidence Manifold: Geometric Structure of Correctness Representations in Language Models},
  author={Cho, Seonglae and Wu, Zekun and Da Costa, Kleyton and Koshiyama, Adriano},
  year={2026},
  note={Preprint}
}
```

## License

MIT
