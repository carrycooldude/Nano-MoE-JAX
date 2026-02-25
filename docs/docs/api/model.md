---
sidebar_position: 3
title: Model
---

# Model API

`nano_moe.model`

## NanoMoE

```python
class NanoMoE(nn.Module):
    config: NanoMoEConfig
```

The complete NanoMoE language model: embeddings → transformer blocks → LM head.

### `__call__(x, deterministic=True)`

Forward pass through the model.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `jnp.ndarray` of shape `(B, T)` | Integer token indices |
| `deterministic` | `bool` | If `True`, disable dropout (for evaluation) |

**Returns:** `Tuple[logits, aux_loss]`

| Return | Shape | Description |
|--------|-------|-------------|
| `logits` | `(B, T, vocab_size)` | Raw logits for next-token prediction |
| `aux_loss` | scalar | Mean auxiliary loss across all transformer blocks |

**Example:**

```python
import jax
import jax.numpy as jnp
from nano_moe import NanoMoEConfig, NanoMoE

config = NanoMoEConfig(vocab_size=65)
model = NanoMoE(config=config)

rng = jax.random.PRNGKey(0)
tokens = jnp.ones((1, 32), dtype=jnp.int32)
params = model.init(rng, tokens)["params"]

logits, aux_loss = model.apply({"params": params}, tokens)
# logits.shape: (1, 32, 65)
# aux_loss: scalar
```

### `generate(params, rng, prompt, max_new_tokens, temperature=0.8, top_k=40)`

Autoregressive text generation with temperature scaling and top-k filtering.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | `dict` | — | Model parameters |
| `rng` | `PRNGKey` | — | Random key for sampling |
| `prompt` | `jnp.ndarray` `(1, T)` | — | Starting token sequence |
| `max_new_tokens` | `int` | — | Number of tokens to generate |
| `temperature` | `float` | `0.8` | Sampling temperature (lower = more deterministic) |
| `top_k` | `int` | `40` | Number of top tokens to sample from |

**Returns:** `jnp.ndarray` of shape `(1, T + max_new_tokens)` — the prompt concatenated with generated tokens.

**Example:**

```python
prompt = jnp.array([[0]])  # newline character
generated = model.generate(params, rng, prompt, max_new_tokens=200)
text = decode(generated[0].tolist())
print(text)
```
