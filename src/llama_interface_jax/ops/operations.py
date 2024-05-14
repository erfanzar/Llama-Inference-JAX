import jax.lax
from functools import partial
from .pallas_oprations import matmul as matmul_pallas
from typing import Literal, Tuple
from jax import Array, numpy as jnp


@partial(jax.jit, static_argnames=["operator", "block_sizes"])
def matmul(
        lhs: Array,
        rhs: Array,
        *,
        block_sizes: int | None = None,
        operator: Literal["pallas", "norm"] = "pallas",
        interpret: bool = False
):
    if operator == "pallas":
        assert lhs.ndim == rhs.ndim
        return matmul_pallas(
            lhs=lhs,
            rhs=rhs,
            block_sizes=block_sizes,
            interpret=interpret
        )
    elif operator == "norm":
        return jax.lax.batch_matmul(lhs, rhs)
    else:
        raise ValueError("unknown operator")


def un_quantize_array(
        quantized: jnp.ndarray,
        scale: jnp.ndarray,
        float_dtype: jnp.dtype = jnp.float16,
) -> Array:
    max_scale = (jnp.iinfo(quantized.dtype).max + abs(jnp.iinfo(quantized.dtype).min)) / 2
    return (jax.lax.convert_element_type(quantized, float_dtype) * scale) / max_scale


def quantize_array(array: Array) -> Tuple[Array, Array]:
    scale = jnp.max(jnp.abs(array), axis=-1, keepdims=True)
    return jax.lax.convert_element_type(
        jnp.rint(array * ((jnp.iinfo(jnp.int8).max + abs(jnp.iinfo(jnp.int8).min)) / 2 / scale)), jnp.int8
    ), scale
