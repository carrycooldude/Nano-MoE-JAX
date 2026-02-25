---
sidebar_position: 6
title: Transformer Block
---

# Transformer Block

The transformer block is the main repeating unit. It combines **self-attention** (for inter-token communication) with the **MoE layer** (for per-token processing), connected through residual pathways.

## Structure

```mermaid
graph TB
    Input["Input x<br/>(B, T, D)"]

    Input --> LN1["LayerNorm"]
    LN1 --> ATTN["Multi-Head<br/>Causal Attention"]
    ATTN --> Drop1["Dropout"]

    Input --> Res1(("+"))
    Drop1 --> Res1

    Res1 --> LN2["LayerNorm"]
    LN2 --> MOE["MoE Layer<br/>(Router + 4 Experts)"]
    MOE --> Drop2["Dropout"]

    Res1 --> Res2(("+"))
    Drop2 --> Res2

    Res2 --> Output["Output<br/>(B, T, D) + aux_loss"]
```

## Two Sub-layers

### Sub-layer 1: Self-Attention

```python
# Pre-norm → Attention → Residual
h = x + dropout(attention(layer_norm(x)))
```

Tokens communicate with each other. Position 5 can read from positions 1–5 to understand context.

### Sub-layer 2: MoE

```python
# Pre-norm → MoE → Residual
output = h + dropout(moe_layer(layer_norm(h)))
```

Each token is processed independently by its selected experts. **This is where MoE happens!**

## Residual Connections

```mermaid
graph LR
    X["x"] --> Add1("+")
    X --> SA["Self-Attention(LayerNorm(x))"]
    SA --> Add1
    Add1 --> H["h"]

    H --> Add2("+")
    H --> MoE["MoE(LayerNorm(h))"]
    MoE --> Add2
    Add2 --> Out["output"]
```

:::tip Why Residual Connections?
Without residual connections, deep networks suffer from **vanishing gradients** — the gradient signal becomes too weak to update early layers. Residual connections create a "gradient highway" that allows gradients to flow directly from the loss to any layer.

In math: if `output = x + f(x)`, then `∂output/∂x = 1 + ∂f/∂x`. The `1` term ensures the gradient is always at least 1, preventing vanishing.
:::

## Code

```python
class TransformerBlock(nn.Module):
    config: NanoMoEConfig

    @nn.compact
    def __call__(self, x, deterministic=False):
        cfg = self.config

        # Sub-layer 1: Attention
        h = nn.LayerNorm()(x)
        h = MultiHeadAttention(config=cfg)(h, deterministic)
        h = nn.Dropout(cfg.dropout)(h, deterministic=deterministic)
        x = x + h  # residual

        # Sub-layer 2: MoE
        h = nn.LayerNorm()(x)
        h, aux_loss = MoELayer(config=cfg)(h, deterministic)
        h = nn.Dropout(cfg.dropout)(h, deterministic=deterministic)
        x = x + h  # residual

        return x, aux_loss
```

## Stacking Blocks

NanoMoE uses 4 blocks by default. Each block refines the representation:

```mermaid
graph LR
    Emb["Embeddings"] --> B1["Block 1<br/>Basic patterns"]
    B1 --> B2["Block 2<br/>Composition"]
    B2 --> B3["Block 3<br/>Higher-order"]
    B3 --> B4["Block 4<br/>Prediction-ready"]
    B4 --> Head["LM Head"]
```

Earlier blocks tend to learn **local patterns** (character combinations, common words), while later blocks learn **longer-range dependencies** (sentence structure, style).
