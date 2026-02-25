---
sidebar_position: 7
title: Full Model
---

# Full NanoMoE Model

The `NanoMoE` class puts everything together: embeddings → stacked transformer blocks → language model head.

## Complete Data Flow

```mermaid
graph TB
    subgraph Input["1. Input Processing"]
        Tokens["Integer Tokens<br/>[15, 42, 7, ...]"] --> TokEmb["Token Embedding<br/>(65 × 128 table)"]
        TokEmb --> Add["+ Position Embedding<br/>(128 × 128 table)"]
        Add --> DropI["Dropout (p=0.1)"]
    end

    subgraph Blocks["2. Transformer Blocks (×4)"]
        direction TB
        B1["Block 1"] -->|"aux_loss₁"| B2["Block 2"]
        B2 -->|"aux_loss₂"| B3["Block 3"]
        B3 -->|"aux_loss₃"| B4["Block 4"]
        B4 -->|"aux_loss₄"| Out["Output"]
    end

    subgraph Head["3. Output Head"]
        FLN["LayerNorm"] --> LM["Linear (128 → 65)"]
        LM --> Logits["Logits<br/>(B, T, 65)"]
    end

    DropI --> B1
    Out --> FLN

    Logits --> AUX["Total aux_loss =<br/>mean(aux₁ + aux₂ + aux₃ + aux₄)"]
```

## Two Outputs

The model's `__call__` returns **two** things:

```python
logits, aux_loss = model.apply({"params": params}, input_tokens)
```

| Output | Shape | Description |
|--------|-------|-------------|
| `logits` | `(batch, seq_len, vocab_size)` | Raw predictions for next token |
| `aux_loss` | scalar | Mean auxiliary load-balancing loss across all layers |

## Autoregressive Generation

After training, the model generates text one token at a time:

```mermaid
graph LR
    S1["Seed: '\\n'"] --> M1["Model"]
    M1 --> L1["Logits → Temperature → Top-K → Sample"]
    L1 --> T1["'T'"]
    T1 --> M2["Model"]
    M2 --> L2["..."]
    L2 --> T2["'h'"]
    T2 --> M3["..."]
    M3 --> Final["...→ 'The king...'"]
```

The `generate()` method:

```python
def generate(self, params, rng, prompt, max_new_tokens, temperature=0.8, top_k=40):
    tokens = prompt
    for _ in range(max_new_tokens):
        # Crop to context window
        context = tokens[:, -block_size:]

        # Get predictions
        logits, _ = self.apply({"params": params}, context)

        # Sample next token
        logits = logits[:, -1, :] / temperature
        top_vals, _ = jax.lax.top_k(logits, k=top_k)
        logits = jnp.where(logits < top_vals[:, -1:], -1e9, logits)
        next_token = jax.random.categorical(rng, logits)

        tokens = jnp.concatenate([tokens, next_token[:, None]], axis=1)
    return tokens
```

### Generation Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `temperature` | 0.8 | Lower = more deterministic, higher = more creative |
| `top_k` | 40 | Only sample from the top K most likely tokens |
| `max_new_tokens` | 500 | Maximum tokens to generate |
