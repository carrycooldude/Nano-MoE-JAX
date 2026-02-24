"""Helper utilities: parameter counting, batching, data loading."""

import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def count_params(params) -> int:
    """Total number of trainable parameters in a parameter tree.

    Args:
        params: A pytree of parameter arrays.

    Returns:
        Total scalar count of parameters.
    """
    return sum(p.size for p in jax.tree_util.tree_leaves(params))


def get_batch(
    data: jnp.ndarray,
    batch_size: int,
    block_size: int,
    rng: jax.Array,
):
    """Sample a random batch of (input, target) pairs from sequential data.

    Args:
        data: 1-D integer array of token ids.
        batch_size: Number of sequences in the batch.
        block_size: Sequence length.
        rng: PRNG key.

    Returns:
        x: (batch_size, block_size) input tokens.
        y: (batch_size, block_size) target tokens (shifted by one).
    """
    max_start = len(data) - block_size - 1
    starts = jax.random.randint(rng, (batch_size,), 0, max_start)
    # Use numpy for indexing (avoids tracing issues)
    starts_np = np.array(starts)
    x = np.stack([data[s : s + block_size] for s in starts_np])
    y = np.stack([data[s + 1 : s + 1 + block_size] for s in starts_np])
    return jnp.array(x), jnp.array(y)


def load_text_data(
    path: str | None = None,
    url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    val_fraction: float = 0.1,
):
    """Load a text file (or download tiny-shakespeare) and split into train/val.

    Args:
        path: Optional local path to a text file. If None, downloads tiny-shakespeare.
        url: URL to download if path is None.
        val_fraction: Fraction of data reserved for validation.

    Returns:
        train_data: 1-D numpy int array of token ids.
        val_data:   1-D numpy int array of token ids.
        vocab_size: Number of unique characters.
        encode:     Function str → list[int].
        decode:     Function list[int] → str.
    """
    if path is not None:
        text = Path(path).read_text(encoding="utf-8")
    else:
        cache_path = Path("data/tinyshakespeare.txt")
        if cache_path.exists():
            text = cache_path.read_text(encoding="utf-8")
        else:
            print(f"Downloading tiny-shakespeare from {url} ...")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, cache_path)
            text = cache_path.read_text(encoding="utf-8")

    # Character-level tokenisation
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    def encode(s: str) -> list:
        return [stoi[c] for c in s]

    def decode(ids) -> str:
        return "".join(itos[int(i)] for i in ids)

    data = np.array(encode(text), dtype=np.int32)
    split = int(len(data) * (1 - val_fraction))
    train_data = data[:split]
    val_data = data[split:]

    print(f"Loaded {len(data):,} characters  |  vocab size: {vocab_size}  |  "
          f"train: {len(train_data):,}  val: {len(val_data):,}")

    return train_data, val_data, vocab_size, encode, decode
