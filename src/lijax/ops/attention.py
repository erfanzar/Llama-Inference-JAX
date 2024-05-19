import math

import jax.nn
from jax import numpy as jnp, Array, lax
from typing import Optional


def dot_product_attention(
        query: Array,
        key: Array,
        value: Array,
        bias: Optional[Array] = None,
        precision: lax.PrecisionLike = lax.Precision.DEFAULT
):
    """
    (softmax(q@k.T / sqrt(hd))) @ v
    :param query: Array : Query State
    :param key: Array : key State
    :param value: Array : value State
    :param bias: Optional[Array] : Bias
    :param precision: lax.PrecisionLike: precision used for computing attention,
    :return: attention
    """
    depth = query.shape[-1]
    query = query / math.sqrt(depth)
    attention_weights = jnp.einsum(
        "bqhd,bkhd->bhqk", query, key, precision=precision
    )
    if bias is not None:
        attention_weights = attention_weights + bias

    attention_weights = jax.nn.softmax(attention_weights.astype("float32"))
    attention = jnp.einsum(
        "bhqk,bkhd->bqhd", attention_weights, value, precision=precision
    )
    return attention
