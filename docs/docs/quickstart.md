---
sidebar_position: 99
title: Quickstart
---

# Quickstart

Get NanoMoE running in under 2 minutes.

## Install

```bash
pip install nano-moe-jax
```

That's it — no need to clone any repository. Everything below works with just the pip package.

## Option 1: Train on Shakespeare (Pip-Only, No Clone Needed)

Copy-paste this into a file called `train.py` and run it. **No GitHub clone required.**

```python
"""Train NanoMoE on Tiny Shakespeare — works with just `pip install nano-moe-jax`."""

import jax
import jax.numpy as jnp

from nano_moe import NanoMoEConfig, NanoMoE
from nano_moe.train import create_train_state, train_loop
from nano_moe.utils import load_shakespeare

# 1. Load data (auto-downloads Tiny Shakespeare ~1.1MB)
train_data, val_data, encode, decode, vocab_size = load_shakespeare()

# 2. Configure
config = NanoMoEConfig(
    vocab_size=vocab_size,
    block_size=128,
    d_model=128,
    n_heads=4,
    n_layers=4,
    n_experts=4,
    top_k=2,
    batch_size=32,
    learning_rate=3e-4,
    max_steps=5000,
)

# 3. Create model
model = NanoMoE(config=config)

# 4. Train
rng = jax.random.PRNGKey(42)
state = train_loop(model, config, train_data, val_data, rng)

# 5. Generate text
prompt = jnp.array([encode("\n")])
generated = model.generate(state.params, jax.random.PRNGKey(0), prompt, max_new_tokens=500)
print(decode(generated[0].tolist()))
```

```bash
python train.py
```

:::tip No GitHub Clone Needed
The `load_shakespeare()` function automatically downloads the dataset. The entire training script above is self-contained — you only need `pip install nano-moe-jax`.
:::

## Option 2: Clone the Repository

If you want the full project with example scripts, tests, and docs:

```bash
git clone https://github.com/carrycooldude/Nano-MoE-JAX.git
cd Nano-MoE-JAX
pip install -e .

# Run the included example
python examples/train_shakespeare.py
```

## Use as a Library

```python
import jax
import jax.numpy as jnp
from nano_moe import NanoMoEConfig, NanoMoE

# Create a custom model
config = NanoMoEConfig(
    vocab_size=65,
    n_experts=8,     # more experts
    top_k=2,         # activate 2 per token
    d_model=256,     # wider
    n_layers=6,      # deeper
)

model = NanoMoE(config=config)
rng = jax.random.PRNGKey(42)
dummy = jnp.ones((1, 128), dtype=jnp.int32)
params = model.init(rng, dummy)["params"]

# Forward pass
logits, aux_loss = model.apply({"params": params}, dummy)
print(f"Logits: {logits.shape}")   # (1, 128, 65)
print(f"Aux loss: {aux_loss:.4f}") # ~4.0 (balanced)
```

## Run Tests

```bash
# Only available if you cloned the repo
python -m pytest tests/ -v
```

Expected: **13/13 tests pass** ✅

## Project Structure

```
nano-moe-jax (pip package)
├── nano_moe/
│   ├── config.py    ← NanoMoEConfig
│   ├── layers.py    ← ExpertFFN, Router, MoELayer, etc.
│   ├── model.py     ← NanoMoE model + generate()
│   ├── train.py     ← Training loop (create_train_state, train_loop)
│   └── utils.py     ← load_shakespeare(), get_batch(), count_params()

GitHub repo (git clone)
├── examples/
│   └── train_shakespeare.py  ← Ready-to-run demo script
├── tests/
│   ├── test_layers.py
│   └── test_model.py
└── docs/                     ← This documentation site
```
