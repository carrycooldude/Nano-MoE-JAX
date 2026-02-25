---
sidebar_position: 5
title: Utilities
---

# Utilities API

`nano_moe.utils`

## count_params

```python
def count_params(params) -> int
```

Count total number of trainable parameters in a Flax parameter tree.

```python
from nano_moe.utils import count_params
n = count_params(params)
print(f"Parameters: {n:,}")  # Parameters: 2,409,025
```

---

## get_batch

```python
def get_batch(data, batch_size, block_size, rng) -> Tuple[x, y]
```

Sample a random batch of sequences from the data.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `jnp.ndarray` | Full token sequence |
| `batch_size` | `int` | Number of sequences |
| `block_size` | `int` | Sequence length |
| `rng` | `PRNGKey` | Random key |

**Returns:** `(x, y)` where `y` is `x` shifted by 1 position (next-token targets).

---

## load_shakespeare

```python
def load_shakespeare(data_dir="data") -> Tuple[train_data, val_data, encode, decode, vocab_size]
```

Download Tiny Shakespeare, split into train/val, and create encode/decode functions.

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `train_data` | `jnp.ndarray` | ~1M training tokens |
| `val_data` | `jnp.ndarray` | ~111K validation tokens |
| `encode` | `Callable` | `str → List[int]` |
| `decode` | `Callable` | `List[int] → str` |
| `vocab_size` | `int` | 65 unique characters |
