# Llama-inference-jax

Accelerated inference with Llama Models in JAX for high-speed, pure JAX implementation.

> [!NOTE]
>
> This project will only support Llama Models (at least for now), and focuses on local machines
> so if you are more likely trying to use this project for any other purposes I suggest you check
> out [EasyDeL](https://github.com/erfanzar/EasyDeL).

## Overview

Llama-inference-jax is a library designed to perform accelerated inference using Llama Models in JAX, providing
high-speed and pure JAX implementation. Llama Models are known for their efficiency and accuracy in various machine
learning tasks, and integrating them with JAX allows for seamless deployment on accelerators like GPUs and TPUs.

## Features

- Accelerated inference with Llama Models.
- Pure JAX implementation for high-speed execution.
- Seamless deployment on GPUs and TPUs.
- Custom Pallas Kernels.
- Parameter Quantization.
- Standalone weights.
- Flash Attention Support on CPU/GPU/TPU.

## Usage

##### Converting Your Own Llama Model to LiJAX as easy as possible

```python
from lijax.covertors import convert_llama_model
import pickle as pkl

lijax_model = convert_llama_model(
    pre_trained_model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
    extra_loading_options_for_model=dict(),  # Kwargs to hf model loading
    quantize_mlp=True,
    quantize_embed=True,
    quantize_lm_head=True,
    quantize_self_attn=True
)

print(lijax_model)

# Saving Model 

pkl.dump(lijax_model, open("lijax_llama_3_8b", "wb"))

# Loading Saved Model 

_new_lijax_model = pkl.load(open("lijax_llama_3_8b", "rb"))
```

#### Generation Process



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.