from typing import NamedTuple, Optional, List, Union, Dict, Tuple, Literal
from jax import numpy as jnp, Array
from dataclasses import dataclass
from ...ops import un_quantize_array, flash_attention, matmul, dot_product_attention
from .._modules import LiJAXLinear, LiJAXEmbed


@dataclass(frozen=False)
class LiJAXLlamaConfig:
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


class LlamaAttentionWeights(NamedTuple):
    config: LiJAXLlamaConfig
    q_proj: LiJAXLinear
    k_proj: LiJAXLinear
    v_proj: LiJAXLinear
    o_proj: LiJAXLinear


class LlamaMLPWeights(NamedTuple):
    config: LiJAXLlamaConfig
    # Llama MLP don't use Bias
    gate_proj: LiJAXLinear
    down_proj: LiJAXLinear
    up_proj: LiJAXLinear


class LlamaRMSNorm(NamedTuple):
    weight: Array
    weight_scale: Optional[Array] = None


class LlamaBlockWeight(NamedTuple):
    config: LiJAXLlamaConfig
    mlp: LlamaMLPWeights
    self_attn: LlamaAttentionWeights
    input_layer_norm: LlamaRMSNorm
    post_attention_layer_norm: LlamaRMSNorm


class LlamaModelWeight(NamedTuple):
    config: LiJAXLlamaConfig
    embed_tokens: LiJAXEmbed
    layers: List[LlamaBlockWeight]
    norm: LlamaRMSNorm


class LlamaForCausalLMWeight(NamedTuple):
    config: LiJAXLlamaConfig
    model: LlamaModelWeight
    lm_head: LiJAXLinear


def forward_llama_attention(
        config: LiJAXLlamaConfig,
        block: LlamaAttentionWeights,
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
    # query_weight = block.q_proj if block.q_proj_scale is None else un_quantize_array(block.q_proj, block.q_proj_scale)
    # key_weight = block.k_proj if block.k_proj_scale is None else un_quantize_array(block.k_proj, block.k_proj_scale)
    # value_weight = block.v_proj if block.v_proj_scale is None else un_quantize_array(block.v_proj, block.v_proj_scale)

    # query = matmul(hidden_states, query_weight)  # B,S,D @ D,HD
    # key = matmul(hidden_states, key_weight)  # B,S,D @ D,HD
    # value = matmul(hidden_states, value_weight)  # B,S,D @ D,HD
    ...


def forward_llama_mlp(
        config: LiJAXLlamaConfig,
        block: LlamaMLPWeights,
        hidden_states: Array,
        runtime_dtype: jnp.dtype = jnp.float16,
): ...


def forward_llama_block(
        config: LiJAXLlamaConfig,
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
        config: LiJAXLlamaConfig,
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
        config: LiJAXLlamaConfig,
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
