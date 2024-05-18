import bisect
import functools
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional, Tuple
import jax
import jax.experimental.pjit as pjit
import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec
from jax.typing import ArrayLike


def prepare_input(input_ids, max_length):
    pad_width = max_length - input_ids.shape[1]
    if pad_width > 0:
        return (
            jnp.pad(input_ids, [(0, 0), (pad_width, 0)]),
            jnp.ones((input_ids.shape[0], max_length)).at[:, :pad_width].set(0)
        )
    else:
        return input_ids[:, abs(pad_width):, :, :], jnp.ones((input_ids.shape[0], max_length))


def sample_next_token(
        logits,
        rng: jax.random.PRNGKey,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        do_sample: bool = True
):
    """
    Applies temperature, top-p, and top-k filtering to logits and samples the next token.

    Args:
      logits: Logits predicted by the model, shape (batch, seq, vocab_size).
      rng: jax.random.PRNGKey
      temperature: Temperature for scaling the logits.
      top_p: Top-p probability threshold for filtering logits.
      top_k: Top-k number of logits to keep after filtering by probability.
      do_sample: boolean indicating whether to sample a new token or not

    Returns:
      Sampled tokens, shape (batch,).
    """

    # Apply temperature scaling.
    logits = logits / temperature

    batch, seq, vocab_size = logits.shape

    # Apply top-p filtering.
    if top_p < 1.0 and do_sample:
        sorted_logits = jnp.sort(logits, axis=-1)[..., ::-1]
        sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
        cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)

        # Create a mask for logits exceeding the top-p threshold.
        cutoff_index = jnp.argmax(cumulative_probs >= top_p, axis=-1)
        cutoff_index = jnp.expand_dims(cutoff_index, axis=-1)
        cutoff_mask = jnp.arange(vocab_size) < cutoff_index

        # Mask logits exceeding the top-p threshold.
        logits = jnp.where(cutoff_mask, logits, jnp.full_like(logits, float('-inf')))

    # Apply top-k filtering.
    if top_k > 0 and do_sample:
        top_k_logits, top_k_indices = jax.lax.top_k(logits, k=top_k)
        top_k_mask = jnp.zeros_like(logits, dtype=bool)
        top_k_mask = top_k_mask.at[jnp.arange(batch)[:, None], jnp.arange(seq)[:, None], top_k_indices].set(True)

        # Mask logits outside the top-k.
        logits = jnp.where(top_k_mask, logits, jnp.full_like(logits, float('-inf')))

    # Sample from the filtered logits.
    probs = jax.nn.softmax(logits, axis=-1)
    if do_sample:
        return jnp.atleast_2d(jax.random.categorical(rng, probs)[:, -1])
    return jnp.atleast_2d(jnp.argmax(probs, -1)[:, -1])
