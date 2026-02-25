---
sidebar_position: 4
title: Training
---

# Training API

`nano_moe.train`

## create_train_state

```python
def create_train_state(rng, model, config) -> TrainState
```

Initialize model parameters and optimizer state.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `rng` | `PRNGKey` | Random key for parameter initialization |
| `model` | `NanoMoE` | Model instance |
| `config` | `NanoMoEConfig` | Configuration |

**Returns:** `flax.training.train_state.TrainState` with AdamW optimizer.

---

## train_step

```python
@jax.jit
def train_step(state, x, y, rng) -> Tuple[TrainState, loss, ce_loss, aux_loss]
```

Single JIT-compiled training step.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | `TrainState` | Current parameters + optimizer state |
| `x` | `jnp.ndarray (B, T)` | Input token batch |
| `y` | `jnp.ndarray (B, T)` | Target token batch (shifted by 1) |
| `rng` | `PRNGKey` | Random key for dropout |

**Returns:** `(updated_state, total_loss, ce_loss, aux_loss)`

---

## eval_step

```python
@jax.jit
def eval_step(state, x, y) -> Tuple[loss, ce_loss, aux_loss]
```

Single JIT-compiled evaluation step (no dropout, no gradient update).

---

## train_loop

```python
def train_loop(model, config, train_data, val_data, rng) -> TrainState
```

Full training loop with periodic evaluation and logging.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `NanoMoE` | Model instance |
| `config` | `NanoMoEConfig` | Configuration |
| `train_data` | `jnp.ndarray` | Training token sequence |
| `val_data` | `jnp.ndarray` | Validation token sequence |
| `rng` | `PRNGKey` | Random key |

**Returns:** Final `TrainState` with trained parameters.
