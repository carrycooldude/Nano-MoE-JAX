# NanoMoE — Mixture-of-Experts in JAX

A lightweight, educational **Mixture-of-Experts (MoE)** GPT-style language model built from scratch in **JAX / Flax**.

Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT), NanoMoE replaces the standard FFN in each transformer block with a sparse MoE layer — only **top-k experts** activate per token, giving increased model capacity with reduced compute per forward pass.

## Architecture

```
Input Tokens
    ↓
Token Embedding + Positional Embedding
    ↓
┌─────────────────────────────────────┐
│         Transformer Block ×N        │
│                                     │
│  LayerNorm → Causal Multi-Head Attn │
│      ↓ + Residual                   │
│  LayerNorm → MoE Layer              │
│      ↓ + Residual                   │
│                                     │
│  ┌─── MoE Layer ─────────────────┐  │
│  │ Router (Top-K Gating)         │  │
│  │   ├─ Expert 1 (FFN)           │  │
│  │   ├─ Expert 2 (FFN)           │  │
│  │   ├─ ...                      │  │
│  │   └─ Expert N (FFN)           │  │
│  │ → Weighted Sum of Top-K       │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
    ↓
LayerNorm → Linear Head → Logits
```
<img width="942" height="1930" alt="image" src="https://github.com/user-attachments/assets/61c93464-e2c3-4fac-a089-5ce327fdec21" />


### Key Features

- **Sparse MoE Routing** — Top-K gating with softmax; only a subset of experts runs per token
- **Load-Balancing Loss** — Switch Transformer-style auxiliary loss for uniform expert utilisation
- **Pure JAX/Flax** — No custom CUDA kernels; portable across CPU, GPU, and TPU
- **Autoregressive Generation** — Temperature + top-k sampling for text generation
- **Fully JIT-compiled** training and evaluation steps

## Quick Start

### Install

```bash
git clone https://github.com/carrycooldude/MoE-JAX.git
cd MoE-JAX
pip install -r requirements.txt
```

> **Note:** For GPU support, install the appropriate `jaxlib` CUDA wheel — see [JAX installation](https://github.com/google/jax#installation).

### Train on Tiny Shakespeare

```bash
python examples/train_shakespeare.py
```

This downloads Tiny Shakespeare (~1 MB), trains a character-level NanoMoE, and generates sample text.

### Run Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
MoE-JAX/
├── nano_moe/
│   ├── __init__.py        # Public API
│   ├── config.py          # Hyperparameter dataclass
│   ├── layers.py          # ExpertFFN, Router, MoELayer, Attention, TransformerBlock
│   ├── model.py           # NanoMoE model + generate()
│   ├── train.py           # Training loop, JIT-compiled steps
│   └── utils.py           # Param counting, batching, data loading
├── examples/
│   └── train_shakespeare.py
├── tests/
│   ├── test_layers.py
│   └── test_model.py
├── requirements.txt
└── README.md
```

## Default Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `d_model` | 128 | Hidden dimension |
| `n_layers` | 4 | Transformer blocks |
| `n_heads` | 4 | Attention heads |
| `d_ff` | 512 | Expert FFN inner dim |
| `n_experts` | 4 | Experts per MoE layer |
| `top_k` | 2 | Active experts per token |
| `block_size` | 128 | Max context length |
| `aux_loss_coeff` | 0.01 | Load-balancing loss weight |

## How It Works

1. **Router** projects each token to `n_experts` logits and selects the top-k experts
2. **Experts** are independent 2-layer FFNs (d_model → d_ff → d_model, GELU activation)
3. **Weighted Sum** combines the top-k expert outputs using normalised softmax gates
4. **Auxiliary Loss** penalises uneven routing to prevent expert collapse

## License

MIT
