import pickle

from src.llama_interface_jax.model import (
    forward_llama_lm_head,
    KVMemory,
    LlamaForCausalLMWeight,
)
from src.llama_interface_jax.generation import prepare_input, sample_next_token

from src.llama_interface_jax.utils import GenerateRNG
from jax import numpy as jnp

rng = GenerateRNG()


def main():
    batch_size = 1
    max_sequence_length = 32
    block: LlamaForCausalLMWeight = pickle.load(open("lijax_llama_model.pkl", "rb"))
    input_ids = jnp.array([1, 2, 3, 4, 5, 6, ], dtype="i4").reshape(1, -1)
    input_ids, attention_mask = prepare_input(input_ids, max_sequence_length)
    memory = KVMemory.init_layer_memories(
        batch_size=batch_size,
        dtype=jnp.float16,
        head_dims=block.config.hidden_size // block.config.num_attention_heads,
        num_key_value_heads=block.config.num_attention_heads,
        sequence_length=max_sequence_length,
        num_hidden_layers=block.config.num_hidden_layers
    )
    res, memory = forward_llama_lm_head(
        block=block,
        input_ids=input_ids,
        attention_mask=attention_mask,
        runtime_kernel="normal",
        use_flash_attention=False,
        past_key_values=memory
    )
    out = sample_next_token(
        res, rng.rng, 0.8, 0.95, 20
    )
    print(res)
    print(out)


if __name__ == '__main__':
    main()
