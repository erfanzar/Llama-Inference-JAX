import jax.random

from src.lijax.generation import prepare_input, sample_next_token
from src.lijax.model import LlamaForCausalLMWeight, forward_llama_lm_head, KVMemory, llama_generate
from src.lijax.ops import pt2jax
from src.lijax.covertors import convert_llama_model_weights_to_lijax
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
        num_hidden_layers=1,
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
    input_ids_jax = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 648, 6541]], dtype="i4")
    max_new_tokens = 32
    max_seq = max_new_tokens + 8
    input_ids_jax, attention_mask_jax = prepare_input(input_ids=input_ids_jax, max_length=5)
    input_ids_torch, attention_mask_torch = torch.tensor(input_ids_jax.tolist()), torch.tensor(
        attention_mask_jax.tolist()
    )
    res_torch = hf_model.generate(
        input_ids_torch,
        attention_mask=attention_mask_torch,
        max_new_tokens=max_new_tokens
    )
    print(res_torch)
    for nt in llama_generate(
            block=lijax_model,
            input_ids=input_ids_jax,
            attention_mask=attention_mask_jax.astype("i4"),
            max_new_tokens=max_new_tokens,
            max_length=max_new_tokens + 32,
            use_flash_attention=False,
            runtime_kernel="normal",
            do_padding=False,
    ):
        print(nt)
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
