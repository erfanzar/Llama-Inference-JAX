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
            kernel: Literal["pallas", "norm"] = "pallas",
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
