import jax
from jax.interpreters import pxla
from jax.sharding import Mesh, PartitionSpec
from jax.experimental import mesh_utils
from functools import lru_cache
from jax.lax import with_sharding_constraint as _with_sharding_constraint

AXIS_SHARDING_NAMES = ("data", "tensor")


@lru_cache()
def get_mesh(dims: tuple[int, int] = (1, -1)):
    mesh_array = jax.numpy.arange(0, len(jax.devices())).reshape(dims)
    return Mesh(mesh_utils.create_device_mesh(mesh_array.shape), AXIS_SHARDING_NAMES)


def check_sharding(mesh: Mesh, array: jax.Array):
    assert array.ndim <= 2

    o_partition = "tensor" if (array.shape[-1] / mesh.shape["tensor"]).is_integer() else None
    f_partition = "data" if (array.shape[0] / mesh.shape["data"]).is_integer() else (
        "tensor" if (array.shape[0] / mesh.shape["tensor"]).is_integer() else None
    )

    if array.ndim == 2:
        return PartitionSpec(f_partition, o_partition)
    elif array.ndim == 1:
        if o_partition is not None:
            return PartitionSpec(o_partition)
        return PartitionSpec(f_partition)
    raise ValueError()


def names_in_current_mesh(*names):
    mesh_axis_names = pxla.thread_resources.env.physical_mesh.axis_names
    return set(names) <= set(mesh_axis_names)


def get_names_from_partition_spec(partition_specs):
    names = set()
    if isinstance(partition_specs, dict):
        partition_specs = partition_specs.values()
    for item in partition_specs:
        if item is None:
            continue
        elif isinstance(item, str):
            names.add(item)
        else:
            names.update(get_names_from_partition_spec(item))

    return list(names)


def with_sharding_constraint(x, partition_specs):
    axis_names = get_names_from_partition_spec(partition_specs)
    if names_in_current_mesh(*axis_names):
        return _with_sharding_constraint(x, partition_specs)
    return x
