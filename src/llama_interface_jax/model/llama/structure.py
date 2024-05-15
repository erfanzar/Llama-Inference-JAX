import math
from typing import NamedTuple, Optional, List, Union, Dict, Tuple, Literal
from functools import partial
import jax.lax
from jax import numpy as jnp, Array
from dataclasses import dataclass
from ...ops import un_quantize_array, flash_attention, matmul, dot_product_attention, repeat_key_value
from .._modules import LiJAXLinear, LiJAXEmbed, FreqsCis, rotary_embedding, KVMemory


def _pad_repr(repr_str: str) -> str:
    return repr_str.replace("\n", "\n\t")


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

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"\tq_proj={_pad_repr(self.q_proj.__repr__())}\n"
            f"\tk_proj={_pad_repr(self.k_proj.__repr__())}\n"
            f"\tv_proj={_pad_repr(self.v_proj.__repr__())}\n"
            f"\to_proj={_pad_repr(self.o_proj.__repr__())}\n"
            f")"
        )


class LlamaMLPWeights(NamedTuple):
    config: LiJAXLlamaConfig
    # Llama MLP don't use Bias
    gate_proj: LiJAXLinear
    down_proj: LiJAXLinear
    up_proj: LiJAXLinear

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"\tgate_proj={_pad_repr(self.gate_proj.__repr__())}\n"
            f"\tdown_proj={_pad_repr(self.down_proj.__repr__())}\n"
            f"\tup_proj={_pad_repr(self.up_proj.__repr__())}\n"
            f")"
        )


class LlamaRMSNorm(NamedTuple):
    weight: Array
    weight_scale: Optional[Array] = None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(shape = ({self.weight.shape},), quantized = {self.weight_scale is None})"
        )


class LlamaBlockWeight(NamedTuple):
    config: LiJAXLlamaConfig
    mlp: LlamaMLPWeights
    self_attn: LlamaAttentionWeights
    input_layer_norm: LlamaRMSNorm
    post_attention_layer_norm: LlamaRMSNorm

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"\tmlp={_pad_repr(self.mlp.__repr__())}\n"
            f"\tself_attn={_pad_repr(self.self_attn.__repr__())}\n"
            f"\tinput_layer_norm={_pad_repr(self.input_layer_norm.__repr__())}\n"
            f"\tpost_attention_layer_norm={_pad_repr(self.post_attention_layer_norm.__repr__())}\n"
            f")"
        )


class LlamaModelWeight(NamedTuple):
    config: LiJAXLlamaConfig
    embed_tokens: LiJAXEmbed
    layers: List[LlamaBlockWeight]
    norm: LlamaRMSNorm

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"\tembed_tokens={self.embed_tokens}\n"
            f"\tlayers 0x{len(self.layers)} {_pad_repr(self.layers[0].__repr__())}\n"
            f"\tnorm={_pad_repr(self.norm.__repr__())}\n"
            f")"
        )


class LlamaForCausalLMWeight(NamedTuple):
    config: LiJAXLlamaConfig
    model: LlamaModelWeight
    lm_head: LiJAXLinear

    def __repr__(self) -> str:
        model_repr = self.model.__repr__().replace("\n", "\n\t")
        lm_head_repr = self.lm_head.__repr__().replace("\n", "\n\t")
        return (
            f"{self.__class__.__name__}(\n"
            f"\tconfig={self.config}\n"
            f"\tmodel={model_repr}\n"
            f"\tlm_head={lm_head_repr}\n"
            f")"
        )


_c_axes = lambda x: x.transpose(0, 2, 1, 3)


@partial(
    jax.jit,
    static_argnames=[
        "block",
        "freqs_cis",
        "runtime_dtype",
        "runtime_kernel",
        "interpret",
        "use_flash_attention",
        "block_key",
        "block_query"
    ]
)
def forward_llama_attention(
        block: LlamaAttentionWeights,
        hidden_states: Array,
        attention_mask: Array,
        causal_mask: Array,
        freqs_cis: FreqsCis,
        past_key_values: Optional[KVMemory] = None,
        init_cache: bool = False,
        runtime_dtype: jnp.dtype = jnp.float16,
        runtime_kernel: Literal["pallas", "normal"] = "pallas",
        interpret: bool = True,
        use_flash_attention: bool = True,
        block_key: int = 128,
        block_query: int = 128
) -> Tuple[Array, Optional[KVMemory]]:
    config = block.config
    batch_size, sequence_length, hidden_size = hidden_states.shape
    head_dim = config.hidden_size // config.num_attention_heads
    num_rep_heads = config.num_attention_heads // config.num_key_value_heads

    query = block.q_proj(hidden_states, kernel=runtime_kernel, interpret=interpret, dtype=runtime_dtype)
    key = block.k_proj(hidden_states, kernel=runtime_kernel, interpret=interpret, dtype=runtime_dtype)
    value = block.v_proj(hidden_states, kernel=runtime_kernel, interpret=interpret, dtype=runtime_dtype)

    query = query.reshape(batch_size, sequence_length, config.num_attention_heads, head_dim)
    key = key.reshape(batch_size, sequence_length, config.num_key_value_heads, head_dim)
    value = value.reshape(batch_size, sequence_length, config.num_key_value_heads, head_dim)
    query, key = rotary_embedding(
        query=_c_axes(query),
        key=_c_axes(key),
        freqs_cis=freqs_cis,
        position_ids=past_key_values.step if past_key_values else 0,
        runtime_dtype=runtime_dtype
    )
    key = repeat_key_value(key, num_rep_heads)
    value = repeat_key_value(_c_axes(value), num_rep_heads)

    query = _c_axes(query)
    key = _c_axes(key)
    value = _c_axes(value)
    new_past_key_values = None
    if past_key_values is not None:
        @partial(jax.vmap)
        def dynamic_update_slice_in_dim(memory, start, update):
            return jax.lax.dynamic_update_slice_in_dim(memory, update, start, axis=0)

        key = dynamic_update_slice_in_dim(past_key_values.key, past_key_values.step, key)
        value = dynamic_update_slice_in_dim(past_key_values.value, past_key_values.step, value)
        new_step = past_key_values.step + sequence_length
        attention_mask = jnp.arange(past_key_values.key.shape[1]) < new_step[:, None]
        attention_mask = attention_mask[:, None, None, :]  # [B, H, T, T]
        new_past_key_values = KVMemory(
            key=key,
            value=value,
            step=new_step,
        )
    if init_cache and past_key_values is None:
        new_past_key_values = KVMemory.init_memory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            head_dims=head_dim,
            num_key_value_heads=config.num_key_value_heads,
            dtype=runtime_dtype
        )

    query_length, key_length = query.shape[1], key.shape[1]
    causal_mask = causal_mask[:, :, :query_length, :key_length]

    batch_size = hidden_states.shape[0]
    causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])
    attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
    attention_mask = jnp.logical_and(attention_mask, causal_mask)

    attention_bias = jax.lax.select(
        attention_mask > 0,
        jnp.full(attention_mask.shape, 0.).astype(runtime_dtype),
        jnp.full(attention_mask.shape, jnp.finfo(runtime_dtype).min).astype(runtime_dtype),
    )

    if use_flash_attention:
        attention_output = flash_attention(
            query=query,
            key=key,
            value=value,
            bias=attention_bias,
            interpret=interpret,
            block_q=block_query,
            block_k=block_key,
            sm_scale=1 / math.sqrt(head_dim),
        )
    else:
        attention_output = dot_product_attention(query=query, key=key, value=value, bias=attention_bias)
    attention_output = attention_output.reshape(batch_size, sequence_length, hidden_size)
    output = block.o_proj(attention_output, kernel=runtime_kernel, interpret=interpret, dtype=runtime_dtype)
    return output, new_past_key_values


@partial(
    jax.jit,
    static_argnames=[
        "block", "runtime_dtype", "runtime_kernel", "interpret"
    ]
)
def forward_llama_mlp(
        block: LlamaMLPWeights,
        hidden_states: Array,
        runtime_dtype: jnp.dtype = jnp.float16,
        runtime_kernel: Literal["pallas", "normal"] = "pallas",
        interpret: bool = True,
):
    return block.down_proj(
        jax.nn.silu(block.gate_proj(
            hidden_states,
            kernel=runtime_kernel,
            interpret=interpret,
            dtype=runtime_dtype
        )) * block.up_proj(
            hidden_states,
            kernel=runtime_kernel,
            interpret=interpret,
            dtype=runtime_dtype
        ),
        kernel=runtime_kernel,
        interpret=interpret,
        dtype=runtime_dtype
    )


def forward_llama_block(
        block: LlamaBlockWeight,
        hidden_states: Array,
        attention_mask: Array,
        causal_mask: Array,
        freqs_cis: FreqsCis,
        past_key_values: Optional[KVMemory] = None,
        init_cache: bool = False,
        runtime_dtype: jnp.dtype = jnp.float16,
        runtime_kernel: Literal["pallas", "normal"] = "pallas",
        interpret: bool = True,
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
