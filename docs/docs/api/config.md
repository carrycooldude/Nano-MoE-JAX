---
sidebar_position: 1
title: NanoMoEConfig
---

# NanoMoEConfig

`nano_moe.config.NanoMoEConfig`

A frozen dataclass holding all hyperparameters for the model, training, and data.

## Usage

```python
from nano_moe import NanoMoEConfig

# Default configuration
config = NanoMoEConfig()

# Custom configuration
config = NanoMoEConfig(
    vocab_size=256,
    n_layers=6,
    n_experts=8,
    top_k=2,
    d_model=256,
)
```

## Parameters

### Model Architecture

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | `int` | `65` | Size of the token vocabulary |
| `block_size` | `int` | `128` | Maximum sequence length (context window) |
| `d_model` | `int` | `128` | Hidden dimension (embedding size) |
| `n_heads` | `int` | `4` | Number of attention heads |
| `n_layers` | `int` | `4` | Number of transformer blocks |
| `d_ff` | `int` | `512` | Inner dimension of each expert FFN |

### MoE Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_experts` | `int` | `4` | Number of expert FFNs per MoE layer |
| `top_k` | `int` | `2` | Experts activated per token |
| `aux_loss_weight` | `float` | `0.01` | Weight of load-balancing auxiliary loss |

### Training

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | `int` | `32` | Sequences per training batch |
| `learning_rate` | `float` | `3e-4` | AdamW learning rate |
| `max_steps` | `int` | `5000` | Total training steps |
| `dropout` | `float` | `0.1` | Dropout probability |
| `weight_decay` | `float` | `0.1` | AdamW weight decay |

### Evaluation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eval_interval` | `int` | `250` | Steps between evaluations |
| `eval_iters` | `int` | `200` | Batches per evaluation |

:::info Frozen Dataclass
`NanoMoEConfig` is a `@dataclass(frozen=True)`, meaning instances are immutable after creation. To change a value, create a new config. This is intentional — it prevents accidental mutation during training.
:::
