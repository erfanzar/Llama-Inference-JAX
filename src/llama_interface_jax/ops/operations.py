import jax.lax
from functools import partial
from .pallas_oprations import matmul_2d_u_ref
from typing import Literal
from jax import Array


@partial(jax.jit, static_argnames=["operator", "block_sizes"])
def matmul(
        lhs: Array,
        rhs: Array,
        *,
        block_sizes: tuple[int, ...] | None = None,
        operator: Literal["pallas", "norm"] = "pallas"
):
    if operator == "pallas":
        assert lhs.ndim == rhs.ndim
        # if lhs.ndim == 3:
        #     return matmul_3d(
        #         lhs=lhs,
        #         rhs=rhs,
        #         block_sizes=block_sizes
        #     )
        
        if lhs.ndim == 2:
            return matmul_2d_u_ref(
                lhs=lhs,
                rhs=rhs,
                block_sizes=block_sizes
            )
        else:
            raise NotImplemented(f"pallas matmul kernel not implemented for given array with {lhs.ndim} dims.")
    elif operator == "norm":
        return jax.lax.batch_matmul(lhs, rhs)
    else:
        raise ValueError("unknown operator")
