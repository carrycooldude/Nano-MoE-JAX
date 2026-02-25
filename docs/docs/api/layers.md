---
sidebar_position: 2
title: Layers
---

# Layers API

`nano_moe.layers`

All core neural network layers used by NanoMoE.

## ExpertFFN

```python
class ExpertFFN(nn.Module):
    config: NanoMoEConfig
```

A single expert feed-forward network: `Dense(d_ff) → GELU → Dense(d_model)`.

**Input:** `(*, d_model)` — any shape with last dimension `d_model`

**Output:** `(*, d_model)` — same shape as input

---

## Router

```python
class Router(nn.Module):
    config: NanoMoEConfig
```

Top-K gating network with load-balancing auxiliary loss.

**Input:** `(batch, seq_len, d_model)`

**Returns:** `Tuple[gates, indices, aux_loss]`

| Return | Shape | Description |
|--------|-------|-------------|
| `gates` | `(B, T, top_k)` | Softmax weights for selected experts |
| `indices` | `(B, T, top_k)` | Indices of selected experts |
| `aux_loss` | scalar | Load-balancing auxiliary loss |

---

## MoELayer

```python
class MoELayer(nn.Module):
    config: NanoMoEConfig
```

Full Mixture-of-Experts layer: Router + N expert FFNs + weighted combination.

**Input:** `(batch, seq_len, d_model)`, `deterministic: bool`

**Returns:** `Tuple[output, aux_loss]`

| Return | Shape | Description |
|--------|-------|-------------|
| `output` | `(B, T, d_model)` | Weighted combination of expert outputs |
| `aux_loss` | scalar | Load-balancing loss from router |

---

## MultiHeadAttention

```python
class MultiHeadAttention(nn.Module):
    config: NanoMoEConfig
```

Multi-head causal self-attention with learned Q, K, V projections.

**Input:** `(batch, seq_len, d_model)`, `deterministic: bool`

**Output:** `(batch, seq_len, d_model)`

---

## TransformerBlock

```python
class TransformerBlock(nn.Module):
    config: NanoMoEConfig
```

Pre-norm transformer block: `LayerNorm → Attention → Residual → LayerNorm → MoE → Residual`.

**Input:** `(batch, seq_len, d_model)`, `deterministic: bool`

**Returns:** `Tuple[output, aux_loss]`

| Return | Shape | Description |
|--------|-------|-------------|
| `output` | `(B, T, d_model)` | Block output |
| `aux_loss` | scalar | Auxiliary loss from the MoE sub-layer |
