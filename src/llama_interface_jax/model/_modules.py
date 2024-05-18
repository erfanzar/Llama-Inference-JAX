import functools
from typing import NamedTuple, Optional, List, Literal
import math
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
    bias_scale: Optional[Array] = None

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

        if x.ndim != weight.ndim:
            assert x.ndim == weight.ndim + 1, (
                f"Expected weights and x to have same number of dimensions, "
                f"got x:{x.ndim}, got weight:{weight.ndim}"
            )
            weight = jnp.expand_dims(weight, 0)

        res = matmul(
            lhs=x.astype(dtype),
            rhs=weight.astype(dtype),
            operator=kernel,
            interpret=interpret,
            block_size=block_size
        )
        if self.bias is not None:
            bias = self.bias
            if self.bias_scale is not None:
                bias = un_quantize_array(
                    weight,
                    self.bias_scale,
                    dtype
                )
            res = res + bias
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
    ) -> "KVMemory":  # type:ignore
        return cls(
            key=jnp.zeros((batch_size, sequence_length, num_key_value_heads, head_dims), dtype=dtype),
            value=jnp.zeros((batch_size, sequence_length, num_key_value_heads, head_dims), dtype=dtype),
            step=jnp.array(0, dtype="i4")
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
    ) -> "List[KVMemory]":  # type:ignore
        return [
            cls(
                key=jnp.zeros((batch_size, sequence_length, num_key_value_heads, head_dims), dtype=dtype),
                value=jnp.zeros((batch_size, sequence_length, num_key_value_heads, head_dims), dtype=dtype),
                step=jnp.array(0, dtype="i4")
            ) for _ in range(num_hidden_layers)
        ]

    def __repr__(self):
        return f"{self.__class__.__name__}(key={self.key.shape}, value={self.value.shape}, step={self.step})"


def _rotate_half(array):
    x1 = array[..., : array.shape[-1] // 2]
    x2 = array[..., array.shape[-1] // 2:]
    return jax.numpy.concatenate((-x2, x1), axis=-1)


def _apply_rotary_pos_embedding(array, sin, cos):
    batch, head, sequence, dim = array.shape
    return (array * cos[:, :, :sequence, :]) + (_rotate_half(array) * sin[:, :, :sequence, :])


@partial(
    jax.jit,  # Let XLA cache everything ...
    static_argnames=[
        "dim",
        "max_position_embeddings",
        "base",
        "scaling_factor",
        "rope_type",
        "t_dtype",
        "original_max_position_embeddings",
        "long_factor",
        "short_factor"
    ]
)
@functools.lru_cache()
def precompute_freqs_cis(
        dim,
        max_position_embeddings=2048,
        base=10000,
        scaling_factor=1.0,
        rope_type: Optional[Literal["none", "linear", "dynamic", "yarn", "su",]] = None,
        t_dtype: jnp.dtype = jnp.int32,
        original_max_position_embeddings: Optional[int] = None,
        long_factor: Optional[List[float]] = None,
        short_factor: Optional[List[float]] = None
) -> FreqsCis:
    def _calc_yarn_scaling_factor(scale):
        if scale <= 1.0:
            return 1.0
        return math.sqrt(1 + math.log(scale) / math.log(original_max_position_embeddings))

    def _calc_su_scaling_factor(scale):
        if scale <= 1.0:
            return 1.0
        return math.sqrt(1 + math.log(scale) / math.log(original_max_position_embeddings))

    if t_dtype == jnp.int64:
        jax.config.update("jax_enable_x64", True)

    if rope_type is None or rope_type == "none":
        t = jax.numpy.arange(max_position_embeddings, dtype=t_dtype)
        inv_freq = 1.0 / (base ** (jax.numpy.arange(0, dim, 2, dtype=jax.numpy.float32) / dim))
        freq = jax.numpy.einsum("i , j -> i j", t, inv_freq).astype("float32")
        embed = jax.numpy.concatenate((freq, freq), axis=-1)
        return FreqsCis(sin=jnp.sin(embed)[:, :], cos=jnp.cos(embed)[:, :])
    elif rope_type == "linear":
        t = jax.numpy.arange(max_position_embeddings, dtype=t_dtype)
        t = t / scaling_factor
        inv_freq = 1.0 / (base ** (jax.numpy.arange(0, dim, 2, dtype=jax.numpy.float32) / dim))
        freq = jax.numpy.einsum("i , j -> i j", t, inv_freq).astype("float32")
        embed = jax.numpy.concatenate((freq, freq), axis=-1)
        return FreqsCis(sin=jnp.sin(embed)[:, :], cos=jnp.cos(embed)[:, :])
    elif rope_type == "dynamic":
        t = jax.numpy.arange(max_position_embeddings, dtype=t_dtype)
        base = base * (scaling_factor - (scaling_factor - 1)) ** (dim / (dim - 2))
        inv_freq = 1.0 / (base ** (jax.numpy.arange(0, dim, 2, dtype=jax.numpy.float32) / dim))
        freq = jax.numpy.einsum("i , j -> i j", t, inv_freq).astype("float32")
        embed = jax.numpy.concatenate((freq, freq), axis=-1)
        return FreqsCis(sin=jnp.sin(embed)[:, :], cos=jnp.cos(embed)[:, :])
    elif rope_type == "su":
        assert original_max_position_embeddings is not None, "No original max position embeddings is provided"
        if max_position_embeddings > original_max_position_embeddings:
            ext_factors = jnp.array(long_factor, dtype=jnp.float32)
        else:
            ext_factors = jnp.array(short_factor, dtype=jnp.float32)
        t_row = (jnp.arange(0, dim, 2, dtype=t_dtype).astype(jnp.float32) / dim)
        inv_freq = 1.0 / (ext_factors * base ** t_row)[None, :, None]
        position_ids = jnp.arange(0, max_position_embeddings, dtype="i4").reshape(1, -1)[:, None, :].astype("float32")
        freqs = (inv_freq @ position_ids).transpose(0, 2, 1)
        scaling_factor = _calc_su_scaling_factor(max_position_embeddings / original_max_position_embeddings)
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        cos = jnp.cos(emb) * scaling_factor
        sin = jnp.sin(emb) * scaling_factor
        return FreqsCis(sin=sin[0], cos=cos[0])
    elif rope_type == "yarn":
        assert original_max_position_embeddings is not None, "No original max position embeddings is provided"
        if max_position_embeddings > original_max_position_embeddings:
            ext_factors = jnp.array(long_factor, dtype=jnp.float32)
        else:
            ext_factors = jnp.array(short_factor, dtype=jnp.float32)
        t_row = (jnp.arange(0, dim, 2, dtype=t_dtype).astype(jnp.float32) / dim)
        inv_freq = 1.0 / (ext_factors * base ** t_row)[None, :, None]
        position_ids = jnp.arange(0, max_position_embeddings, dtype="i4").reshape(1, -1)[:, None, :].astype("float32")
        freqs = (inv_freq @ position_ids).transpose(0, 2, 1)
        scaling_factor = _calc_yarn_scaling_factor(max_position_embeddings / original_max_position_embeddings)
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        cos = jnp.cos(emb) * scaling_factor
        sin = jnp.sin(emb) * scaling_factor
        return FreqsCis(sin=sin[0], cos=cos[0])

    raise "wrong rope type has been given"


@partial(jax.jit, static_argnames=["runtime_dtype", "freqs_cis"])
def rotary_embedding(
        query: Array,
        key: Array,
        freqs_cis: FreqsCis,
        position_ids: Array,
        runtime_dtype: jnp.dtype = jnp.float16
):
    def _gather_embeddings(idx):
        return freqs_cis.sin[idx], freqs_cis.cos[idx]

    # sin, cos = freqs_cis.sin[position_ids][None, None, :, :], freqs_cis.cos[position_ids][None, None, :, :]
    sin, cos = jax.vmap(_gather_embeddings, in_axes=0, out_axes=0)(position_ids)
    sin, cos = jnp.expand_dims(sin, 1), jnp.expand_dims(cos, 1)
    key = _apply_rotary_pos_embedding(key, sin=sin, cos=cos)
    query = _apply_rotary_pos_embedding(query, sin=sin, cos=cos)
    return query.astype(runtime_dtype), key.astype(runtime_dtype)
