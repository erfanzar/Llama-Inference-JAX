import jax.random

from src.llama_interface_jax.generation import prepare_input, sample_next_token
from src.llama_interface_jax.model import LlamaForCausalLMWeight, forward_llama_lm_head, KVMemory, llama_generate
from src.llama_interface_jax.ops import pt2jax
from src.llama_interface_jax.covertors import convert_llama_model_weights_to_lijax
from transformers import LlamaConfig, LlamaForCausalLM
from torch import nn
import torch
from jax import numpy as jnp, lax, random

torch.manual_seed(0)


def main():
    hf_config = LlamaConfig(
        vocab_size=32000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2
    )
    hf_model = LlamaForCausalLM(config=hf_config)
    lijax_model = convert_llama_model_weights_to_lijax(
        hf_model,
        None,
        True,
        True,
        True,
        True
    )
    input_ids_torch = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long).reshape(1, -1)
    input_ids_jax = pt2jax(input_ids_torch)

    res_torch = hf_model.generate(
        input_ids_torch,
        max_new_tokens=2
    )
    print(res_torch)
    llama_generate(
        block=lijax_model,
        input_ids=input_ids_jax,
        max_new_tokens=512,
        max_length=512 + 32
    )
    # print(res_torch)
    # res_lijax, new_key_values = forward_llama_lm_head(
    #     block=lijax_model,
    #     input_ids=input_ids_jax,
    #     attention_mask=attention_mask_jax,
    #     runtime_kernel="pallas",
    #     use_flash_attention=False,
    #     past_key_values=KVMemory.init_layer_memories(
    #         1,
    #         32,
    #         lijax_model.config.hidden_size // lijax_model.config.num_attention_heads,
    #         lijax_model.config.num_key_value_heads,
    #         lijax_model.config.num_hidden_layers,
    #         dtype=jnp.float16
    #     )
    # )
    # next_token = sample_next_token(res_lijax, jax.random.PRNGKey(0), do_sample=False).reshape(1, -1)
    #
    # print("next Token : ", next_token)


if __name__ == '__main__':
    main()
