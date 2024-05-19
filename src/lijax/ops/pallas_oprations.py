# A simple fused matmul operation for pallas
import jax
from functools import partial
from jax.experimental import pallas as pl
from jax import numpy as jnp, ShapeDtypeStruct, Array, jit, lax

DEFAULT_SUB_LINE = 4
FIXED_MATMUL_DIM = 2


def _matmul_2d_kernel_u_ref(lhs_ref, rhs_ref, o_ref, *, block_size):
    for idx in range(lhs_ref.shape[1] // block_size):
        o_ref[:, :] += lhs_ref[
                       :, idx * block_size:(idx + 1) * block_size
                       ] @ rhs_ref[idx * block_size:(idx + 1) * block_size, :
                           ]


@partial(jit, static_argnames=["block_sizes"])
def _matmul_2d_u_ref(lhs: Array, rhs: Array, block_sizes: tuple[int, ...] | None = None) -> Array:
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


def _matmul_kernel(x_ref, y_ref, z_ref, *, precision):
    z_ref[...] = jnp.dot(x_ref[...], y_ref[...], precision=precision)


@partial(jit, static_argnames=["interpret", "precision", "block_size"])
def _matmul_2d(
        lhs: Array,
        rhs: Array,
        block_size: int | None = None,
        precision: lax.PrecisionLike = None,
        interpret: bool = False
) -> Array:
    assert lhs.ndim == rhs.ndim, "lhs and rhs must have same ndim"
    if block_size is None:
        block_rhs = pl.cdiv(rhs.shape[1], 16)
        block_lhs = pl.cdiv(lhs.shape[0], 16)
    else:
        block_rhs = block_size
        block_lhs = block_size
    return pl.pallas_call(
        partial(_matmul_kernel, precision=precision),
        out_shape=jax.ShapeDtypeStruct((lhs.shape[0], rhs.shape[1]), lhs.dtype),
        grid=(block_lhs, block_rhs),
        in_specs=[
            pl.BlockSpec(lambda i, j: (i, 0), (lhs.shape[0] // block_lhs, lhs.shape[1])),
            pl.BlockSpec(lambda i, j: (0, j), (rhs.shape[0], rhs.shape[1] // block_rhs))
        ],
        out_specs=pl.BlockSpec(
            lambda i, j: (i, j), (lhs.shape[0] // block_lhs, rhs.shape[1] // block_rhs)
        ),
        interpret=interpret,
    )(lhs, rhs)


@partial(jit, static_argnames=["interpret", "precision", "block_size"])
def matmul(
        lhs: Array,
        rhs: Array,
        block_size: int | None = None,
        precision: lax.PrecisionLike = None,
        interpret: bool = False
):
    func = partial(_matmul_2d, block_size=block_size, precision=precision, interpret=interpret)
    assert lhs.ndim == rhs.ndim, "lhs and rhs must have same ndim"
    assert lhs.ndim >= 2, "Only 1d arrays are not supported."
    tdim = lhs.ndim
    if tdim > FIXED_MATMUL_DIM:
        for _ in range(tdim - FIXED_MATMUL_DIM):
            func = jax.vmap(func)
    return func(lhs, rhs)


if __name__ == "__main__":
    batch_size = 8
    seq_len = 64
    num_heads = 32
    head_dims = 128
    k1, k2, k3 = jax.random.split(jax.random.key(0), num=3)
    q = jax.random.normal(k1, (batch_size, seq_len, num_heads, head_dims))
    k = jax.random.normal(k2, (batch_size, seq_len, num_heads, head_dims))

    res = matmul(q, k, block_size=64)
    org = q @ k

    org_sum = jnp.mean(jnp.sum(org, axis=0))
    res_sum = jnp.mean(jnp.sum(res, axis=0))

    print(org_sum)
    print(res_sum)

    print(jnp.allclose(res, org))
