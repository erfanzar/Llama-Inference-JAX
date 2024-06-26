import functools
import math
import os
import pickle
from typing import Optional, List, Union, Dict, Tuple, Literal, Generator, NamedTuple
from functools import partial
import jax.lax
from jax import numpy as jnp, Array
from dataclasses import dataclass
from ...ops import (
    un_quantize_array,
    flash_attention,
    dot_product_attention,
    repeat_key_value,
    quantize_array,
    pt2jax
)
from ...utils import GenerateRNG
from .._modules import (
    LiJAXLinear,
    LiJAXEmbed,
    LiJAXRMSNorm,
    FreqsCis,
    rotary_embedding,
    KVMemory,
    precompute_freqs_cis
)
from ...sharding import get_mesh, Mesh
from ...generation import sample_next_token, prepare_input


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


@jax.tree_util.register_pytree_node_class
@dataclass
class LlamaAttentionWeights:
    q_proj: LiJAXLinear
    k_proj: LiJAXLinear
    v_proj: LiJAXLinear
    o_proj: LiJAXLinear

    def tree_flatten(self):
        return (self.q_proj, self.k_proj, self.v_proj, self.o_proj), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"\tq_proj={_pad_repr(self.q_proj.__repr__())}\n"
            f"\tk_proj={_pad_repr(self.k_proj.__repr__())}\n"
            f"\tv_proj={_pad_repr(self.v_proj.__repr__())}\n"
            f"\to_proj={_pad_repr(self.o_proj.__repr__())}\n"
            f")"
        )

    def shard(self, mesh: Mesh = get_mesh((1, -1))):
        self.q_proj.shard(mesh=mesh)
        self.k_proj.shard(mesh=mesh)
        self.v_proj.shard(mesh=mesh)
        self.o_proj.shard(mesh=mesh)


@jax.tree_util.register_pytree_node_class
@dataclass
class LlamaMLPWeights:
    gate_proj: LiJAXLinear
    down_proj: LiJAXLinear
    up_proj: LiJAXLinear

    def tree_flatten(self):
        return (self.gate_proj, self.down_proj, self.up_proj), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"\tgate_proj={_pad_repr(self.gate_proj.__repr__())}\n"
            f"\tdown_proj={_pad_repr(self.down_proj.__repr__())}\n"
            f"\tup_proj={_pad_repr(self.up_proj.__repr__())}\n"
            f")"
        )

    def shard(self, mesh: Mesh = get_mesh((1, -1))):
        self.gate_proj.shard(mesh=mesh)
        self.down_proj.shard(mesh=mesh)
        self.up_proj.shard(mesh=mesh)


@jax.tree_util.register_pytree_node_class
@dataclass
class LlamaBlockWeight:
    mlp: LlamaMLPWeights
    self_attn: LlamaAttentionWeights
    input_layer_norm: LiJAXRMSNorm
    post_attention_layer_norm: LiJAXRMSNorm

    def tree_flatten(self):
        return (self.mlp, self.self_attn, self.input_layer_norm, self.post_attention_layer_norm), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"\tmlp={_pad_repr(self.mlp.__repr__())}\n"
            f"\tself_attn={_pad_repr(self.self_attn.__repr__())}\n"
            f"\tinput_layer_norm={_pad_repr(self.input_layer_norm.__repr__())}\n"
            f"\tpost_attention_layer_norm={_pad_repr(self.post_attention_layer_norm.__repr__())}\n"
            f")"
        )

    def shard(self, mesh: Mesh = get_mesh((1, -1))):
        self.mlp.shard(mesh=mesh)
        self.self_attn.shard(mesh=mesh)
        self.input_layer_norm.shard(mesh=mesh)
        self.post_attention_layer_norm.shard(mesh=mesh)


@jax.tree_util.register_pytree_node_class
@dataclass
class LlamaModelWeight:
    embed_tokens: LiJAXEmbed
    layers: List[LlamaBlockWeight]
    norm: LiJAXRMSNorm

    def tree_flatten(self):
        return (self.embed_tokens, self.layers, self.norm), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"\tembed_tokens={self.embed_tokens}\n"
            f"\tlayers 0x{len(self.layers)} {_pad_repr(self.layers[0].__repr__())}\n"
            f"\tnorm={_pad_repr(self.norm.__repr__())}\n"
            f")"
        )

    def shard(self, mesh: Mesh = get_mesh((1, -1))):
        self.embed_tokens.shard(mesh=mesh)
        for layer in self.layers:
            layer.shard(mesh=mesh)
        self.norm.shard(mesh=mesh)


@jax.tree_util.register_pytree_node_class
@dataclass
class LlamaForCausalLMWeight:
    config: LiJAXLlamaConfig
    model: LlamaModelWeight
    lm_head: LiJAXLinear

    def tree_flatten(self):
        return (self.config, self.model, self.lm_head), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

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

    def shard(self, mesh: Mesh = get_mesh((1, -1))):
        self.model.shard(mesh=mesh)
        self.lm_head.shard(mesh=mesh)

    def save(self, path: os.PathLike | str):
        pickle.dump(self, open(path, "wb"))

    @classmethod
    def load(cls, path: os.PathLike | str):
        return pickle.load(open(path, "rb"))


_c_axes = lambda x: x.transpose(0, 2, 1, 3)


@partial(
    jax.jit,
    static_argnames=[
        "runtime_dtype",
        "runtime_kernel",
        "interpret",
        "use_flash_attention",
        "block_key",
        "block_query",
        "hidden_size",
        "num_key_value_heads",
        "num_attention_heads",
        "max_sequence_length",
    ]
)
def forward_llama_attention(
        block: LlamaAttentionWeights,
        hidden_states: Array,
        attention_mask: Array,
        position_ids: Array,
        causal_mask: Array,
        freqs_cis: FreqsCis,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        max_sequence_length: int,
        past_key_values: Optional[KVMemory] = None,
        runtime_dtype: jnp.dtype = jnp.float16,
        runtime_kernel: Literal["pallas", "normal"] = "pallas",
        interpret: bool = True,
        use_flash_attention: bool = True,
        block_key: int = 128,
        block_query: int = 128,
) -> Tuple[Array, Optional[KVMemory]]:
    batch_size, sequence_length, _ = hidden_states.shape
    head_dim = hidden_size // num_attention_heads
    num_rep_heads = num_attention_heads // num_key_value_heads
    query = block.q_proj(hidden_states, kernel=runtime_kernel, interpret=interpret, dtype=runtime_dtype)
    key = block.k_proj(hidden_states, kernel=runtime_kernel, interpret=interpret, dtype=runtime_dtype)
    value = block.v_proj(hidden_states, kernel=runtime_kernel, interpret=interpret, dtype=runtime_dtype)

    query = query.reshape(batch_size, sequence_length, num_attention_heads, head_dim)
    key = key.reshape(batch_size, sequence_length, num_key_value_heads, head_dim)
    value = value.reshape(batch_size, sequence_length, num_key_value_heads, head_dim)

    query_length, key_length = query.shape[1], key.shape[1]

    query, key = rotary_embedding(
        query=_c_axes(query),  # B S H D -> B H S D
        key=_c_axes(key),  # B S H D -> B H S D
        freqs_cis=freqs_cis,
        position_ids=position_ids,
        runtime_dtype=runtime_dtype
    )
    assert attention_mask.ndim == 2
    new_past_key_values = None
    if past_key_values is not None:
        cur_index = past_key_values.step
        *batch_dims, max_length, num_heads, depth_per_head = past_key_values.value.shape

        causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])
        causal_mask = jax.lax.dynamic_slice(
            causal_mask,
            (0, 0, cur_index, 0),
            (1, 1, sequence_length, past_key_values.key.shape[1])
        )
        attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
        attention_mask = jnp.logical_and(attention_mask, causal_mask)

        query = _c_axes(query)  # B H S D -> B S H D
        key = _c_axes(key)  # B H S D -> B S H D
        indices = (0,) * len(batch_dims) + (cur_index, 0, 0)  # type:ignore
        value = jax.lax.dynamic_update_slice(past_key_values.value, value, indices)
        key = jax.lax.dynamic_update_slice(past_key_values.key, key, indices)
        pad_mask = jnp.broadcast_to(
            jnp.arange(max_sequence_length) < cur_index + sequence_length,
            (batch_size, 1, sequence_length, max_sequence_length)
        )
        attention_mask = jnp.logical_and(pad_mask, attention_mask)

        new_past_key_values = KVMemory(
            key=key,
            value=value,
            step=cur_index + sequence_length,
        )
        query = _c_axes(query)  # B S H D -> B H S D
        key = _c_axes(key)  # B S H D -> B H S D
    else:

        causal_mask = causal_mask[:, :, :query_length, :key_length]
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])
        attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
        attention_mask = jnp.logical_and(attention_mask, causal_mask)

    key = repeat_key_value(key, num_rep_heads)
    value = repeat_key_value(_c_axes(value), num_rep_heads)  # B S H D -> B H S D

    query = _c_axes(query)  # B H S D -> B S H D
    key = _c_axes(key)  # B H S D -> B S H D
    value = _c_axes(value)  # B H S D -> B S H D
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
        "runtime_dtype",
        "runtime_kernel",
        "interpret"
    ]
)
def forward_llama_mlp(
        block: LlamaMLPWeights,
        hidden_states: Array,
        runtime_dtype: jnp.dtype = jnp.float16,
        runtime_kernel: Literal["pallas", "normal"] = "pallas",
        interpret: bool = True,
) -> Array:
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


# @partial(
#     jax.jit,
#     static_argnames=[
#         "rms_norm_eps",
#         "num_attention_heads",
#         "causal_mask",
#         "hidden_size",
#         "num_key_value_heads",
#         "max_sequence_length",
#         "runtime_dtype",
#         "runtime_kernel",
#         "use_flash_attention",
#         "interpret",
#         "block_key",
#         "block_query",
#     ]
# )
def forward_llama_block(
        block: LlamaBlockWeight,
        hidden_states: Array,
        position_ids: Array,
        attention_mask: Array,
        causal_mask: Array,
        freqs_cis: FreqsCis,
        rms_norm_eps: float,
        num_attention_heads: int,
        hidden_size: int,
        num_key_value_heads: int,
        max_sequence_length: int,
        past_key_values: Optional[KVMemory] = None,
        runtime_dtype: jnp.dtype = jnp.float16,
        runtime_kernel: Literal["pallas", "normal"] = "pallas",
        interpret: bool = True,
        use_flash_attention: bool = True,
        block_key: int = 128,
        block_query: int = 128,
) -> Tuple[Array, Optional[KVMemory]]:
    attention_output, new_key_values = forward_llama_attention(
        block=block.self_attn,
        hidden_states=block.input_layer_norm(
            x=hidden_states,
            eps=rms_norm_eps,
            dtype=runtime_dtype
        ),
        interpret=interpret,
        position_ids=position_ids,
        attention_mask=attention_mask,
        block_key=block_key,
        block_query=block_query,
        causal_mask=causal_mask,
        freqs_cis=freqs_cis,
        past_key_values=past_key_values,
        runtime_dtype=runtime_dtype,
        runtime_kernel=runtime_kernel,
        use_flash_attention=use_flash_attention,
        num_key_value_heads=num_key_value_heads,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        max_sequence_length=max_sequence_length
    )
    hidden_states = hidden_states + attention_output

    ffn_output = forward_llama_mlp(
        block=block.mlp,
        hidden_states=block.post_attention_layer_norm(hidden_states, rms_norm_eps, runtime_dtype),
        runtime_dtype=runtime_dtype,
        runtime_kernel=runtime_kernel,
        interpret=interpret,
    )
    hidden_states = hidden_states + ffn_output
    return hidden_states, new_key_values


@functools.lru_cache()
def get_c_mask(max_position_embeddings):
    return jnp.tril(
        jnp.ones(
            (1, 1, max_position_embeddings, max_position_embeddings),
            dtype="bool"
        )
    )


# @partial(
#     jax.jit,
#     static_argnames=[
#         "rms_norm_eps",
#         "num_attention_heads",
#         "hidden_size",
#         "num_key_value_heads",
#         "max_sequence_length",
#         "num_hidden_layers",
#         "max_position_embeddings",
#         "rope_theta",
#         "rope_type",
#         "runtime_dtype",
#         "runtime_kernel",
#         "use_flash_attention",
#         "interpret",
#         "block_key",
#         "block_query"
#     ]
# )
def forward_llama_model(
        block: LlamaModelWeight,
        input_ids: Array,
        position_ids: Array,
        attention_mask: Array,
        rms_norm_eps: float,
        num_attention_heads: int,
        hidden_size: int,
        num_key_value_heads: int,
        max_sequence_length: int,
        num_hidden_layers: int,
        max_position_embeddings: int,
        rope_theta: float | int,
        rope_type: Optional[str] = None,
        scaling_factor: Optional[float] = None,
        past_key_values: Optional[List[KVMemory]] = None,
        input_embeds: Optional[Array] = None,
        runtime_dtype: jnp.dtype = jnp.float16,
        runtime_kernel: Literal["pallas", "normal"] = "pallas",
        use_flash_attention: bool = True,
        interpret: bool = True,
        block_key: int = 128,
        block_query: int = 128,
) -> Tuple[Array, Optional[List[KVMemory]]]:
    new_key_values = []
    if past_key_values is None:
        past_key_values = [None] * num_hidden_layers
    if input_embeds is None:
        input_embeds = block.embed_tokens(input_ids, runtime_dtype)
    hidden_states = input_embeds
    causal_mask = get_c_mask(max_position_embeddings)
    freqs_cis = precompute_freqs_cis(
        dim=hidden_size // num_attention_heads,
        max_position_embeddings=max_position_embeddings,
        base=rope_theta,
        rope_type=rope_type,
        scaling_factor=scaling_factor
    )

    for layer_idx in range(num_hidden_layers):
        hidden_states, new_key_value = forward_llama_block(
            block=block.layers[layer_idx],
            hidden_states=hidden_states,
            position_ids=position_ids,
            block_key=block_key,
            past_key_values=past_key_values[layer_idx],
            runtime_kernel=runtime_kernel,
            freqs_cis=freqs_cis,
            block_query=block_query,
            causal_mask=causal_mask,
            runtime_dtype=runtime_dtype,
            attention_mask=attention_mask,
            use_flash_attention=use_flash_attention,
            interpret=interpret,
            num_key_value_heads=num_key_value_heads,
            num_attention_heads=num_attention_heads,
            max_sequence_length=max_sequence_length,
            hidden_size=hidden_size,
            rms_norm_eps=rms_norm_eps,
        )
        new_key_values.append(new_key_value)
    return block.norm(hidden_states, rms_norm_eps, runtime_dtype), new_key_values


def forward_llama_lm_head(
        block: LlamaForCausalLMWeight,
        input_ids: Array,
        attention_mask: Optional[Array] = None,
        position_ids: Optional[Array] = None,
        past_key_values: Optional[List[KVMemory]] = None,
        input_embeds: Optional[Array] = None,
        runtime_dtype: jnp.dtype = jnp.float16,
        runtime_kernel: Literal["pallas", "normal"] = "pallas",
        use_flash_attention: bool = True,
        interpret: bool = True,
        block_key: int = 128,
        block_query: int = 128,
        max_sequence_length: Optional[int] = None
) -> Tuple[Array, Optional[List[KVMemory]]]:
    batch, seq_len = input_ids.shape
    config = block.config
    max_sequence_length = max_sequence_length or block.config.max_position_embeddings
    if attention_mask is None:
        attention_mask = jnp.ones((batch, seq_len), dtype="i4")
    if position_ids is None:
        position_ids = jnp.arange(0, seq_len, dtype="i4").reshape(1, -1).repeat(batch, 0)
    hidden_states, new_key_values = forward_llama_model(
        block=block.model,
        input_ids=input_ids,
        position_ids=position_ids,
        past_key_values=past_key_values,
        runtime_kernel=runtime_kernel,
        block_query=block_query,
        runtime_dtype=runtime_dtype,
        attention_mask=attention_mask,
        interpret=interpret,
        use_flash_attention=use_flash_attention,
        block_key=block_key,
        input_embeds=input_embeds,
        rms_norm_eps=config.rms_norm_eps,
        hidden_size=config.hidden_size,
        num_key_value_heads=config.num_key_value_heads,
        max_sequence_length=max_sequence_length,
        num_attention_heads=config.num_attention_heads,
        scaling_factor=getattr(config.rope_scaling, "scaling_factor", None),
        rope_type=getattr(config.rope_scaling, "rope_type", None),
        max_position_embeddings=config.max_position_embeddings,
        num_hidden_layers=config.num_hidden_layers,
        rope_theta=config.rope_theta,
    )
    lm_head_out = block.lm_head(hidden_states, interpret=interpret, dtype=runtime_dtype, kernel=runtime_kernel)
    return lm_head_out, new_key_values


def llama_generate(
        block: LlamaForCausalLMWeight,
        input_ids: Array,
        attention_mask: Optional[Array] = None,
        position_ids: Optional[Array] = None,
        past_key_values: Optional[List[KVMemory]] = None,
        runtime_dtype: jnp.dtype = jnp.float16,
        runtime_kernel: Literal["pallas", "normal"] = "pallas",
        use_flash_attention: bool = True,
        interpret: bool = True,
        block_key: int = 128,
        block_query: int = 128,
        max_new_tokens: int = 512,
        max_length: Optional[int] = None,
        temperature: float = 0,
        top_p: float = 0.95,
        do_padding: bool = False,
        rng_generator: GenerateRNG = GenerateRNG(seed=48),
        eos_token_id: Optional[int] = None,
) -> Generator[Array, None, None] | None:
    batch_size, seq_length = input_ids.shape
    if max_length is None:
        max_length = block.config.max_position_embeddings
    if do_padding:
        input_ids, attention_mask = prepare_input(input_ids, max_length - max_new_tokens)
        position_ids = attention_mask.cumsum(axis=-1) - 1
        input_ids, position_ids = map(
            lambda x: x.astype("i4"),
            [input_ids, position_ids]
        )
        attention_mask = attention_mask.astype("bool")
    else:
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids, dtype="bool")
        if position_ids is None:
            position_ids = jnp.atleast_2d(attention_mask.cumsum(axis=-1) - 1).astype("i4")

    extended_attention_mask = jnp.ones((batch_size, max_length), dtype="bool")
    attention_mask = jax.lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
    if past_key_values is None:
        past_key_values = KVMemory.init_layer_memories(
            batch_size=batch_size,
            dtype=runtime_dtype,
            num_key_value_heads=block.config.num_key_value_heads,
            num_hidden_layers=block.config.num_hidden_layers,
            head_dims=block.config.hidden_size // block.config.num_attention_heads,
            sequence_length=max_length,
            step=jnp.array(input_ids.shape[-1], dtype="i4")
        )
    for _ in range(max_new_tokens):
        logits, past_key_values = forward_llama_lm_head(
            block=block,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            runtime_kernel=runtime_kernel,
            runtime_dtype=runtime_dtype,
            block_query=block_query,
            block_key=block_key,
            interpret=interpret,
            use_flash_attention=use_flash_attention,
            max_sequence_length=max_length
        )

        next_token = sample_next_token(
            logits=logits,
            rng=rng_generator.rng,
            temperature=temperature,
            top_p=top_p,
        )

        if next_token == eos_token_id:
            break
        input_ids = next_token
        attention_mask = jnp.ones_like(next_token, dtype="bool")
        position_ids = position_ids[:, -1:] + 1
        input_ids, position_ids = map(
            lambda x: x.astype("i4"),
            [input_ids, position_ids]
        )
        attention_mask = attention_mask.astype("bool")
        yield next_token
