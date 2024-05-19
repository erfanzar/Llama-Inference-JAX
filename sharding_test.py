from jax._src.partition_spec import PartitionSpec

from src.lijax.sharding import get_mesh, check_sharding
from src.lijax.generation import prepare_input, sample_next_token
from src.lijax.model import LlamaForCausalLMWeight, forward_llama_lm_head, KVMemory, llama_generate
from src.lijax.ops import pt2jax
from src.lijax.covertors import convert_llama_model_weights_to_lijax
from transformers import LlamaConfig, LlamaForCausalLM
from torch import nn
import torch
from jax import numpy as jnp, lax, random

torch.manual_seed(0)
from src.lijax.sharding import get_mesh


def main():
    # hf_config = LlamaConfig(
    #     hidden_size=256,
    #     intermediate_size=512,
    #     num_hidden_layers=1,
    #     num_attention_heads=8,
    #     num_key_value_heads=2
    # )
    # hf_model = LlamaForCausalLM(config=hf_config)
    # lijax_model = convert_llama_model_weights_to_lijax(
    #     hf_model,
    #     None,
    #     True,
    #     True,
    #     True,
    #     True
    # )
    mseh = get_mesh()


if __name__ == "__main__":
    main()
