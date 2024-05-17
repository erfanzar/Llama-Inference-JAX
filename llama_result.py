from src.llama_interface_jax.model import LlamaForCausalLMWeight, forward_llama_lm_head, KVMemory
from src.llama_interface_jax.ops import pt2jax
from src.llama_interface_jax.covertors import convert_llama_model_weights_to_lijax
from transformers import LlamaConfig, LlamaForCausalLM
from torch import nn
import torch
from jax import numpy as jnp, lax, random


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
    res_torch = hf_model(input_ids_torch)
    res_lijax, new_key_values = forward_llama_lm_head(
        block=lijax_model,
        input_ids=input_ids_jax,
        runtime_kernel="pallas",
        use_flash_attention=False,
        past_key_values=KVMemory.init_layer_memories(
            1,
            2048,
            lijax_model.config.hidden_size // lijax_model.config.num_attention_heads,
            lijax_model.config.num_key_value_heads,
            lijax_model.config.num_hidden_layers,
            dtype=jnp.float16
        )
    )
    print(res_torch["past_key_values"][0][0][0, -1, 0, -1])
    print(new_key_values[0].key[0, -1, 0, -1])
    print(res_torch.logits.sum().mean())
    print(res_lijax.sum().mean())


if __name__ == '__main__':
    main()
