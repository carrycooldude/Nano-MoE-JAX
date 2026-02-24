"""Hyperparameter configuration for NanoMoE."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NanoMoEConfig:
    """All hyperparameters for the NanoMoE model.

    Attributes:
        vocab_size: Size of the token vocabulary (character-level by default).
        n_layers: Number of transformer blocks.
        n_heads: Number of attention heads.
        d_model: Hidden / embedding dimension.
        d_ff: Inner dimension of expert feed-forward networks.
        n_experts: Total number of expert FFNs in each MoE layer.
        top_k: Number of experts activated per token.
        block_size: Maximum sequence length (context window).
        dropout_rate: Dropout probability (used during training).
        aux_loss_coeff: Weight of the load-balancing auxiliary loss.
        learning_rate: Peak learning rate for AdamW.
        weight_decay: AdamW weight decay coefficient.
        batch_size: Training batch size.
        max_iters: Maximum training iterations.
        eval_interval: Iterations between evaluation runs.
        eval_iters: Number of batches used for evaluation.
    """

    # --- Model architecture ---
    vocab_size: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 128
    d_ff: int = 512
    n_experts: int = 4
    top_k: int = 2
    block_size: int = 128

    # --- Regularization ---
    dropout_rate: float = 0.1
    aux_loss_coeff: float = 0.01

    # --- Optimiser ---
    learning_rate: float = 3e-4
    weight_decay: float = 0.1

    # --- Training schedule ---
    batch_size: int = 32
    max_iters: int = 5000
    eval_interval: int = 250
    eval_iters: int = 50
