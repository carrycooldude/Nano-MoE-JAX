---
sidebar_position: 99
title: Quickstart
---

# Quickstart

Get NanoMoE running in under 2 minutes.

## Install

### From PyPI

```bash
pip install nano-moe-jax
```

### From Source

```bash
git clone https://github.com/carrycooldude/MoE-JAX.git
cd MoE-JAX
pip install -e .
```

## Train on Shakespeare

```bash
python examples/train_shakespeare.py
```

This will:
1. Download Tiny Shakespeare (~1.1MB)
2. Train a 2.4M parameter NanoMoE for 5000 steps
3. Print loss metrics every 250 steps
4. Generate a 500-character sample at the end

**Expected output:**

```
============================================================
  NanoMoE — Tiny Shakespeare
  Params: 2,409,025
============================================================

[step     1/5000]  train loss: 4.2320  |  val loss: 4.0883 ★
[step   250/5000]  train loss: 2.5591  |  val loss: 2.5524 ★
...
[step  5000/5000]  train loss: 1.5395  |  val loss: 1.6584 ★

============================================================
  Generating sample text …
============================================================
```

## Use as a Library

```python
import jax
import jax.numpy as jnp
from nano_moe import NanoMoEConfig, NanoMoE

# 1. Configure
config = NanoMoEConfig(
    vocab_size=65,
    n_experts=4,
    top_k=2,
)

# 2. Initialize
model = NanoMoE(config=config)
rng = jax.random.PRNGKey(42)
dummy = jnp.ones((1, 128), dtype=jnp.int32)
params = model.init(rng, dummy)["params"]

# 3. Forward pass
logits, aux_loss = model.apply({"params": params}, dummy)
print(f"Logits shape: {logits.shape}")   # (1, 128, 65)
print(f"Aux loss: {aux_loss:.4f}")       # ~4.0 (balanced)
```

## Run Tests

```bash
python -m pytest tests/ -v
```

Expected: **13/13 tests pass** ✅

## Project Structure

```
MoE-JAX/
├── nano_moe/
│   ├── __init__.py        # Public API
│   ├── config.py          # NanoMoEConfig
│   ├── layers.py          # ExpertFFN, Router, MoELayer, etc.
│   ├── model.py           # NanoMoE model
│   ├── train.py           # Training loop
│   └── utils.py           # Data loading, helpers
├── examples/
│   └── train_shakespeare.py
├── tests/
│   ├── test_layers.py
│   └── test_model.py
└── pyproject.toml
```
