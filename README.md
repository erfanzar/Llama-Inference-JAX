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
- PyTrees and JAX compatible Blocks for Model.

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
lijax_model.shard()

print(lijax_model)

# Saving Model 

pkl.dump(lijax_model, open("lijax_llama_3_8b", "wb"))

# Loading Saved Model 

_new_lijax_model = pkl.load(open("lijax_llama_3_8b", "rb"))
_new_lijax_model.shard()  # sharding model is optional across available GPUs,TPUs
```

#### Generation Process

```python
import jax.numpy
from transformers import AutoTokenizer
from lijax.model import llama_generate
from lijax.covertors import convert_llama_model

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
lijax_model = convert_llama_model("meta-llama/Meta-Llama-3-8B-Instruct")
lijax_model.shard()
generated_ids = None
printed_length = 0
for token in llama_generate(
        block=lijax_model,
        input_ids=tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "hi"}
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="np"
        ),
        use_flash_attention=False,
        # runtime_kernel="pallas",
        runtime_kernel="normal",
        max_length=2048,
        max_new_tokens=32,
        eos_token_id=tokenizer.eos_token_id,
        temperature=1.6,
        # do_sample=True,
        top_k=20,
        top_p=0.95,
):
    generated_ids = jax.numpy.concatenate([generated_ids, token], -1) if generated_ids is not None else token
    stream = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=False)
    print(stream[printed_length:], end="")
    printed_length = len(stream)

```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.