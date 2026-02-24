"""Core layers: Expert FFN, Router, MoE, Multi-Head Attention, Transformer Block."""

from typing import Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from nano_moe.config import NanoMoEConfig


# ---------------------------------------------------------------------------
# Expert Feed-Forward Network
# ---------------------------------------------------------------------------

class ExpertFFN(nn.Module):
    """Two-layer FFN with GELU activation (a single expert).

    Architecture: d_model → d_ff → d_model
    """

    d_ff: int
    d_model: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.d_ff, kernel_init=nn.initializers.he_normal())(x)
        x = nn.gelu(x)
        x = nn.Dense(self.d_model, kernel_init=nn.initializers.he_normal())(x)
        return x


# ---------------------------------------------------------------------------
# Router / Gating Network
# ---------------------------------------------------------------------------

class Router(nn.Module):
    """Top-k gating network that routes tokens to experts.

    Produces per-token expert weights and computes an auxiliary
    load-balancing loss to encourage uniform expert utilisation.
    """

    n_experts: int
    top_k: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Route tokens to experts.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            gates:     (batch, seq_len, top_k) — softmax weights for selected experts.
            indices:   (batch, seq_len, top_k) — indices of selected experts.
            aux_loss:  Scalar load-balancing loss.
        """
        # Compute raw logits → (batch, seq_len, n_experts)
        logits = nn.Dense(self.n_experts, use_bias=False,
                          kernel_init=nn.initializers.xavier_uniform())(x)

        # Full softmax over experts for load-balance computation
        probs = jax.nn.softmax(logits, axis=-1)  # (B, T, E)

        # Top-k selection
        top_k_values, top_k_indices = jax.lax.top_k(logits, self.top_k)  # (B, T, K)

        # Normalised gates only over the selected experts
        gates = jax.nn.softmax(top_k_values, axis=-1)  # (B, T, K)

        # ---- Auxiliary load-balancing loss (Switch Transformer style) ----
        # f_i = fraction of tokens routed to expert i
        # P_i = mean routing probability for expert i
        # aux_loss = n_experts * sum_i(f_i * P_i)
        # One-hot of top-1 choice → (B, T, E)
        top1_idx = top_k_indices[..., 0]
        dispatch_mask = jax.nn.one_hot(top1_idx, self.n_experts)  # (B, T, E)
        f = jnp.mean(dispatch_mask, axis=(0, 1))  # (E,)
        P = jnp.mean(probs, axis=(0, 1))           # (E,)
        aux_loss = self.n_experts * jnp.sum(f * P)

        return gates, top_k_indices, aux_loss


# ---------------------------------------------------------------------------
# Mixture-of-Experts Layer
# ---------------------------------------------------------------------------

class MoELayer(nn.Module):
    """Mixture-of-Experts layer: router + N expert FFNs.

    Each token is routed to top_k experts; their outputs are
    combined via the gating weights.
    """

    config: NanoMoEConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass.

        Args:
            x: (batch, seq_len, d_model)
            deterministic: If True, disable dropout.

        Returns:
            output:   (batch, seq_len, d_model) — weighted expert outputs.
            aux_loss: Scalar load-balancing loss.
        """
        cfg = self.config
        B, T, D = x.shape

        # Route tokens → gates (B,T,K), indices (B,T,K), aux_loss scalar
        gates, indices, aux_loss = Router(
            n_experts=cfg.n_experts, top_k=cfg.top_k
        )(x)

        # Initialise all experts as a list of modules
        experts = [
            ExpertFFN(d_ff=cfg.d_ff, d_model=cfg.d_model, name=f"expert_{i}")
            for i in range(cfg.n_experts)
        ]

        # Compute ALL expert outputs → stack to (n_experts, B, T, D)
        # Using a simple loop — for nano-scale this is efficient and
        # avoids the dynamic-shape issues of scatter/gather in JAX.
        expert_outputs = jnp.stack(
            [expert(x) for expert in experts], axis=0
        )  # (E, B, T, D)

        # Gather the top-k expert outputs for each token
        # indices shape: (B, T, K)
        # We need to pick from expert_outputs along axis 0
        # expert_outputs[indices[b, t, k], b, t, :] for all b, t, k
        batch_idx = jnp.arange(B)[:, None, None]   # (B, 1, 1)
        seq_idx = jnp.arange(T)[None, :, None]     # (1, T, 1)
        selected = expert_outputs[indices, batch_idx, seq_idx, :]  # (B, T, K, D)

        # Weighted combination: gates (B, T, K, 1) * selected (B, T, K, D)
        output = jnp.sum(gates[..., None] * selected, axis=2)  # (B, T, D)

        # Optional dropout on the combined output
        output = nn.Dropout(rate=cfg.dropout_rate)(output, deterministic=deterministic)

        return output, aux_loss


# ---------------------------------------------------------------------------
# Multi-Head Causal Self-Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Standard multi-head causal self-attention with dropout."""

    config: NanoMoEConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        cfg = self.config
        B, T, D = x.shape
        head_dim = D // cfg.n_heads

        # QKV projection
        qkv = nn.Dense(3 * D, kernel_init=nn.initializers.xavier_uniform())(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)  # each (B, T, D)

        # Reshape to (B, n_heads, T, head_dim)
        q = q.reshape(B, T, cfg.n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, cfg.n_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, cfg.n_heads, head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention with causal mask
        scale = jnp.sqrt(jnp.float32(head_dim))
        attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / scale  # (B, H, T, T)

        # Causal mask: upper-triangular → −∞
        causal_mask = jnp.triu(jnp.ones((T, T), dtype=jnp.bool_), k=1)
        attn_weights = jnp.where(causal_mask[None, None, :, :], -1e9, attn_weights)

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_weights = nn.Dropout(rate=cfg.dropout_rate)(attn_weights, deterministic=deterministic)

        # Weighted sum → (B, H, T, head_dim) → (B, T, D)
        attn_out = jnp.matmul(attn_weights, v)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, D)

        # Output projection
        out = nn.Dense(D, kernel_init=nn.initializers.xavier_uniform())(attn_out)
        out = nn.Dropout(rate=cfg.dropout_rate)(out, deterministic=deterministic)
        return out


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN → Attention → residual, LN → MoE → residual."""

    config: NanoMoEConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass.

        Returns:
            output:   (B, T, D) — block output.
            aux_loss: Scalar MoE load-balancing loss from this block.
        """
        # Self-attention sub-layer
        residual = x
        x = nn.LayerNorm()(x)
        x = MultiHeadAttention(config=self.config)(x, deterministic=deterministic)
        x = x + residual

        # MoE sub-layer
        residual = x
        x_norm = nn.LayerNorm()(x)
        moe_out, aux_loss = MoELayer(config=self.config)(x_norm, deterministic=deterministic)
        x = moe_out + residual

        return x, aux_loss
