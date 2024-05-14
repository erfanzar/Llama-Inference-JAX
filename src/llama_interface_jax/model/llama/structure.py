from typing import NamedTuple, Optional, List, Union, Dict, Tuple, Literal
from jax import numpy as jnp, Array
from dataclasses import dataclass
from ...ops import un_quantize_array, flash_attention, matmul, dot_product_attention


class AttentionWeights(NamedTuple):
    q_proj: Array
    k_proj: Array
    v_proj: Array
    o_proj: Array

    q_proj_scale: Optional[Array] = None
    k_proj_scale: Optional[Array] = None
    v_proj_scale: Optional[Array] = None
    o_proj_scale: Optional[Array] = None


class MLPWeights(NamedTuple):
    gate_proj: Array
    down_proj: Array
    up_proj: Array

    gate_proj_scale: Optional[Array] = None
    down_proj_scale: Optional[Array] = None
    up_proj_scale: Optional[Array] = None


class LlamaBlockWeight(NamedTuple):
    mlp: MLPWeights
    self_attn: AttentionWeights

    input_layer_norm: Array
    post_attention_layer_norm: Array

    input_layer_norm_scale: Optional[Array] = None
    post_attention_layer_norm_scale: Optional[Array] = None


class LlamaModelWeight(NamedTuple):
    embed_tokens: Array
    layer: List[LlamaBlockWeight]
    norm: Array

    embed_tokens_scale: Optional[Array] = None
    norm_scale: Optional[Array] = None


class LlamaForCausalLMWeight(NamedTuple):
    model: LlamaModelWeight
    lm_head: Array
    lm_head_scale: Optional[Array] = None


@dataclass(frozen=False)
class LlamaConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    number_rep_kv: int = 1
    num_key_value_heads: Optional[int] = None
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    use_cache: bool = True
    bos_token_id: int = 0
    eos_token_id: int = 1
    attention_dropout: float = 0.0
    rope_theta: float = 10000.
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    rope_scaling: Optional[Dict[str, Union[str, float]]] = None
    hidden_act: str = "silu"
    chat_template: Optional[str] = None


def forward_llama_attention(
        config: LlamaConfig,
        block: AttentionWeights,
        hidden_states: Array,
        attention_mask: Array,
        causal_mask: Array,
        freqs_cis: Tuple[Array, Array],
        past_key_values: Optional[Tuple[Array, Array]] = None,
        init_cache: bool = False,
        runtime_dtype: jnp.dtype = jnp.float16,
        runtime_kernel: Literal["pallas", "normal"] = "pallas",
        use_flash_attention: bool = True,
        block_key: int = 128,
        block_query: int = 128
):
    query_weight = block.q_proj if block.q_proj_scale is None else un_quantize_array(block.q_proj, block.q_proj_scale)
    key_weight = block.k_proj if block.k_proj_scale is None else un_quantize_array(block.k_proj, block.k_proj_scale)
    value_weight = block.v_proj if block.v_proj_scale is None else un_quantize_array(block.v_proj, block.v_proj_scale)

    query = matmul(hidden_states, query_weight)  # B,S,D @ D,HD
    key = matmul(hidden_states, key_weight)  # B,S,D @ D,HD
    value = matmul(hidden_states, value_weight)  # B,S,D @ D,HD


def forward_llama_mlp(
        config: LlamaConfig,
        block: MLPWeights,
        hidden_states: Array,
        runtime_dtype: jnp.dtype = jnp.float16,
): ...


def forward_llama_block(
        config: LlamaConfig,
        block: LlamaBlockWeight,
        hidden_states: Array,
        attention_mask: Array,
        causal_mask: Array,
        freqs_cis: Tuple[Array, Array],
        past_key_values: Optional[Tuple[Array, Array]] = None,
        init_cache: bool = False,
        runtime_dtype: jnp.dtype = jnp.float16,
        runtime_kernel: Literal["pallas", "normal"] = "pallas",
        use_flash_attention: bool = True,
        block_key: int = 128,
        block_query: int = 128
): ...


def forward_llama_model(
        config: LlamaConfig,
        block: LlamaBlockWeight,
        hidden_states: Array,
        attention_mask: Array,
        causal_mask: Array,
        freqs_cis: Tuple[Array, Array],
        past_key_values: Optional[Tuple[Array, Array]] = None,
        init_cache: bool = False,
        runtime_dtype: jnp.dtype = jnp.float16,
        runtime_kernel: Literal["pallas", "normal"] = "pallas",
        use_flash_attention: bool = True,
        block_key: int = 128,
        block_query: int = 128
): ...


def forward_llama_lm_head(
        config: LlamaConfig,
        block: LlamaForCausalLMWeight,
        hidden_states: Array,
        attention_mask: Array,
        causal_mask: Array,
        freqs_cis: Tuple[Array, Array],
        past_key_values: Optional[Tuple[Array, Array]] = None,
        init_cache: bool = False,
        runtime_dtype: jnp.dtype = jnp.float16,
        runtime_kernel: Literal["pallas", "normal"] = "pallas",
        use_flash_attention: bool = True,
        block_key: int = 128,
        block_query: int = 128
): ...
