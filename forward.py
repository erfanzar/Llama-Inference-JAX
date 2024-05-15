import pickle

from src.llama_interface_jax.model import forward_llama_lm_head
from jax import numpy as jnp


def main():
    block = pickle.load(open("lijax_llama_model.pkl", "rb"))
    input_ids = jnp.array([1, 2, 3, 4, 5, 6, ], dtype="i4").reshape(1, -1)
    res, new_past_key_value = forward_llama_lm_head(
        block=block,
        input_ids=input_ids,
        runtime_kernel="normal",
        init_cache=True
    )
    """
[[[ -7.68   -10.63     1.446  ...  -3.635   -6.56    -3.758 ]
[ -0.2146  19.52     1.4375 ...   0.583   -2.098    3.172 ]
[ -5.26    -5.54     5.875  ...  -4.49    -6.445   -0.971 ]
[ -5.5     -5.484    5.316  ...  -4.184   -5.68    -1.433 ]
[ -4.93    -5.293    5.58   ...  -4.188   -4.293   -0.604 ]
[ -4.527   -5.223    4.477  ...  -2.432   -4.96    -1.414 ]]]

[[[ -7.695  -10.66     1.449  ...  -3.643   -6.57    -3.764 ]
  [ -0.208   19.53     1.429  ...   0.588   -2.084    3.176 ]
  [ -5.266   -5.54     5.875  ...  -4.49    -6.445   -0.9673]
  [ -5.496   -5.48     5.316  ...  -4.184   -5.676   -1.433 ]
  [ -4.934   -5.29     5.582  ...  -4.188   -4.29    -0.604 ]
  [ -4.527   -5.223    4.48   ...  -2.434   -4.965   -1.412 ]]]
    """
    print(res)
    print(new_past_key_value)


if __name__ == '__main__':
    main()
