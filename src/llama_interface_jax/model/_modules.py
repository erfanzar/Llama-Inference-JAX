from typing import NamedTuple, Optional, List, Union, Dict, Tuple, Literal

import jax
from jax import numpy as jnp, Array
from ..ops import matmul, un_quantize_array, pt2jax, quantize_array
from functools import partial


class FreqsCis(NamedTuple):
    sin: Array
    cos: Array


class LiJAXLinear(NamedTuple):
    weight: Array
    bias: Optional[Array] = None
    weight_scale: Optional[Array] = None

    @partial(jax.jit, static_argnames=["kernel", "interpret", "block_size", "dtype"])
    def __call__(
            self,
            x,
            kernel: Literal["pallas", "normal"] = "pallas",
            interpret: bool = True,
            block_size: Optional[int] = None,
            dtype: jnp.dtype = jnp.float16,
    ) -> Array:
        weight = self.weight
        if self.weight_scale is not None:
            weight = un_quantize_array(
                weight,
                self.weight_scale,
                dtype
            )
        res = matmul(
            lhs=x,
            rhs=weight,
            operator=kernel,
            interpret=interpret,
            block_size=block_size
        )
        return res

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"weight={self.weight.shape},"
            f" quantized={self.weight_scale is not None},"
            f" bias={self.bias.shape if self.bias is not None else None}"
            f")"
        )

    @classmethod
    def from_torch(cls, linear: "torch.nn.Linear", quantize=False) -> "LiJAXLinear":  # type:ignore
        bias = linear.bias
        weight = pt2jax(linear.weight).T
        weight_scale = None
        if quantize:
            weight, weight_scale = quantize_array(weight)
        return cls(
            weight=weight,
            weight_scale=weight_scale,
            bias=None if bias is None else pt2jax(bias)
        )


class LiJAXEmbed(NamedTuple):
    embedding: Array
    embedding_scale: Optional[Array] = None

    @partial(jax.jit, static_argnames=["dtype"])
    def __call__(
            self,
            ids,
            dtype: jnp.dtype = jnp.float16,
    ) -> Array:
        embedding = self.embedding
        if self.embedding_scale is not None:
            embedding = un_quantize_array(
                embedding,
                self.embedding_scale,
                dtype
            )

        return embedding[ids]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"embedding={self.embedding.shape}, "
            f"quantized={self.embedding_scale is not None}"
            f")"
        )

    @classmethod
    def from_torch(cls, embed: "torch.nn.Embedding", quantize=False) -> "LiJAXEmbed":  # type:ignore
        embedding = pt2jax(embed.weight)
        embedding_scale = None
        if quantize:
            embedding, embedding_scale = quantize_array(embedding)
        return cls(
            embedding=embedding,
            embedding_scale=embedding_scale,
        )


class KVMemory(NamedTuple):
    key: Array
    value: Array
    step: Optional[Array]

    @classmethod
    def init_memory(
            cls,
            batch_size: int,
            sequence_length: int,
            head_dims: int,
            num_key_value_heads: int,
            dtype=jnp.bfloat16,
            step: Optional[Array] = None
    ) -> "KVMemory":  # type:ignore
        return cls(
            key=jnp.zeros((batch_size, sequence_length, num_key_value_heads, head_dims), dtype=dtype),
            value=jnp.zeros((batch_size, sequence_length, num_key_value_heads, head_dims), dtype=dtype),
            step=step
        )

    @classmethod
    def init_layer_memories(
            cls,
            batch_size: int,
            sequence_length: int,
            head_dims: int,
            num_key_value_heads: int,
            num_hidden_layers: int,
            dtype=jnp.bfloat16,
            step: Optional[Array] = None
    ) -> "List[KVMemory]":  # type:ignore
        return [
            cls(
                key=jnp.zeros((batch_size, sequence_length, num_key_value_heads, head_dims), dtype=dtype),
                value=jnp.zeros((batch_size, sequence_length, num_key_value_heads, head_dims), dtype=dtype),
                step=step
            ) for _ in range(num_hidden_layers)
        ]


def _rotate_half(array):
    x1 = array[..., : array.shape[-1] // 2]
    x2 = array[..., array.shape[-1] // 2:]
    return jax.numpy.concatenate((-x2, x1), axis=-1)


def _apply_rotary_pos_embedding(array, sin, cos):
    batch, head, sequence, dim = array.shape
    return (array * cos[:, :, :sequence, :]) + (_rotate_half(array) * sin[:, :, :sequence, :])


@partial(jax.jit, static_argnames=["runtime_dtype", "freq_cis"])
def rotary_embedding(
        query: Array,
        key: Array,
        freq_cis: FreqsCis,
        position_ids: Array,
        runtime_dtype: jnp.dtype = jnp.float16
):
    sin = freq_cis.sin[position_ids][:, None, :, :]
    cos = freq_cis.cos[position_ids][:, None, :, :]
    key = _apply_rotary_pos_embedding(key, sin=sin, cos=cos)
    query = _apply_rotary_pos_embedding(query, sin=sin, cos=cos)
    return query.astype(runtime_dtype), key.astype(runtime_dtype)
