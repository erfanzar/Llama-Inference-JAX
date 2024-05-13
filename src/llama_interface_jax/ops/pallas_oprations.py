# A simple fused matmul operation for pallas

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
from functools import partial

from jax.experimental import pallas as pl
from jax import numpy as jnp, ShapeDtypeStruct, Array, jit

DEFAULT_SUB_LINE = 4


def _matmul_2d_kernel_u_ref(lhs_ref, rhs_ref, o_ref, *, block_size):
    for k in range(lhs_ref.shape[1] // block_size):
        o_ref[:, :] += lhs_ref[:, k * block_size:(k + 1) * block_size] @ rhs_ref[k * block_size:(k + 1) * block_size, :]


@partial(jit, static_argnames=["block_sizes"])
def matmul_2d_u_ref(lhs: Array, rhs: Array, block_sizes: tuple[int, ...] | None = None) -> Array:
    assert lhs.ndim == rhs.ndim, "lhs and rhs must have same ndim"

    if block_sizes is None:
        assert (lhs.shape[0] / DEFAULT_SUB_LINE).is_integer(), "couldn't create default subline (sizes wont match)."
        subline = lhs.shape[0] // DEFAULT_SUB_LINE
        block_sizes = (subline, lhs.shape[1], subline)

    block_f, block_s, block_k = block_sizes
    return pl.pallas_call(
        f=partial(_matmul_2d_kernel_u_ref, block_size=block_k),
        out_shape=ShapeDtypeStruct(
            shape=lhs.shape[:-1] + (rhs.shape[-1],),
            dtype=lhs.dtype
        ),
        in_specs=[
            pl.BlockSpec(index_map=lambda i, j: (i, 0), block_shape=(block_f, lhs.shape[1])),
            pl.BlockSpec(index_map=lambda i, j: (0, j), block_shape=(rhs.shape[0], block_s))
        ],
        out_specs=pl.BlockSpec(index_map=lambda i, j: (i, j), block_shape=(block_f, block_s)),
        grid=(DEFAULT_SUB_LINE, DEFAULT_SUB_LINE),
    )(lhs, rhs)


if __name__ == "__main__":
    batch_size = 8
    seq_len = 256
    num_heads = 32
    head_dims = 128
    # x = jax.random.normal(jax.random.key(0), (seq_len, head_dims), dtype=jnp.float32)
    # y = jax.random.normal(jax.random.key(1), (head_dims, seq_len), dtype=jnp.float32)
    x = jnp.ones((512, 256), dtype=jnp.float32)
    y = jnp.ones((256, 1024), dtype=jnp.float32)
    res = matmul_2d_u_ref(x, y)
    org = x @ y

    org_sum = jnp.mean(jnp.sum(org, axis=0))
    res_sum = jnp.mean(jnp.sum(res, axis=0))

    print(org_sum)
    print(res_sum)

    print(jnp.allclose(res, res))
