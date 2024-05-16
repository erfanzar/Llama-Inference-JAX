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


class SampleSettings(NamedTuple):
    temperature: ArrayLike
    nucleus_p: ArrayLike
    mask: ArrayLike
    active: ArrayLike


class SampleOutput(NamedTuple):
    token_id: ArrayLike
    prob: ArrayLike
    top_k_token_ids: ArrayLike
    top_k_probs: ArrayLike


def prepare_input(input_ids, max_length):
    pad_width = max_length - input_ids.shape[1]
    if pad_width > 0:
        return (
            jnp.pad(input_ids, [(0, 0), (0, pad_width)]),
            jnp.ones((input_ids.shape[0], max_length)).at[:, :pad_width].set(0)
        )
    else:
        return input_ids[:, abs(pad_width):, :, :], jnp.ones((input_ids.shape[0], max_length))


def top_p_filter(logits: jax.Array, top_p: jax.Array) -> jax.Array:
    """Performs nucleus filtering on logits."""
    assert logits.ndim == top_p.ndim, f"Expected {logits.ndim} equal {top_p.ndim}"
    sorted_logits = jax.lax.sort(logits, is_stable=False)
    sorted_probs = jax.nn.softmax(sorted_logits)
    threshold_idx = jnp.argmax(jnp.cumsum(sorted_probs, -1) >= 1 - top_p, axis=-1)
    threshold_largest_logits = jnp.take_along_axis(
        sorted_logits, threshold_idx[..., jnp.newaxis], axis=-1
    )
    assert threshold_largest_logits.shape == logits.shape[:-1] + (1,)
    mask = logits >= threshold_largest_logits
    logits = jnp.where(mask, logits, -1e10)
    return logits


def sample_token(
        rngs: jax.random.PRNGKey,
        logits,
        settings: SampleSettings,
) -> SampleOutput:
    # Expand the settings shape to match the logit shape.
    settings = SampleSettings(
        temperature=jnp.expand_dims(settings.temperature, (1, 2)),  # Input [B], output [B, 1, 1].
        nucleus_p=jnp.expand_dims(settings.nucleus_p, (1, 2)),  # Input [B], output [B, 1, 1].
        mask=jnp.expand_dims(settings.mask, -1),
        active=settings.active,  # [B].
    )
    logits = logits / settings.temperature.astype(logits.dtype)
    # Mask out all disallowed tokens by assigning them a near-zero probability.
    logits = jnp.where(settings.mask, logits, -1e10)
    # Mask out all tokens that don't fall into the p-th percentile.
    logits = top_p_filter(logits, settings.nucleus_p.astype(logits.dtype))

    new_token = jax.random.categorical(rngs, logits, axis=-1)
    probabilities = jax.nn.softmax(logits)
    token_prob = jnp.take_along_axis(probabilities, jnp.expand_dims(new_token, 1), axis=2)
    token_prob = jnp.squeeze(token_prob, 1)

    # Gather the top-k tokens and probabilities.
    top_k_probs, top_k_token_ids = jax.lax.top_k(probabilities, 10)
    top_k_probs = jnp.squeeze(top_k_probs, 1)
    top_k_token_ids = jnp.squeeze(top_k_token_ids, 1)
    return SampleOutput(
        new_token,
        token_prob,
        top_k_token_ids,
        top_k_probs,
    )
