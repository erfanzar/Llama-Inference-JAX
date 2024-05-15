import jax.numpy
import torch
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer
from typing import Optional

from ..model import (
    LlamaModelWeight,
    LlamaBlockWeight,
    LlamaMLPWeights,
    LlamaForCausalLMWeight,
    LlamaAttentionWeights,
    LlamaRMSNorm,
    LiJAXLlamaConfig as LiJAXLlamaConfig,
    LiJAXLinear as LiJAXLinear,
    LiJAXEmbed as LiJAXEmbed
)


def pt2jax(tensor: torch.Tensor) -> jax.Array:
    return jax.numpy.asarray(tensor.detach().cpu().numpy())


def convert_llama_model_weights_to_lijax(
        torch_model: LlamaForCausalLM,
        chat_template: Optional[str] = None,
        quantize_self_attn: bool = True,
        quantize_mlp: bool = True,
        quantize_embed: bool = True,
        quantize_lm_head: bool = True
) -> LlamaForCausalLMWeight:
    config: LlamaConfig = torch_model.config  # type:ignore
    lijax_config = LiJAXLlamaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        number_rep_kv=config.number_rep_kv,
        num_key_value_heads=config.num_key_value_heads,
        max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps,
        initializer_range=config.initializer_range,
        use_cache=config.use_cache,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        attention_dropout=config.attention_dropout,
        rope_theta=config.rope_theta,
        attention_bias=config.attention_bias,
        tie_word_embeddings=config.tie_word_embeddings,
        rope_scaling=config.rope_scaling,
        hidden_act=config.hidden_act,
        chat_template=chat_template,
    )
    layers = []

    for layer_idx in config.num_hidden_layers:
        self_attn = LlamaAttentionWeights(
            config=lijax_config,
            q_proj=LiJAXLinear.from_torch(torch_model.model.layers[layer_idx].self_attn.q_proj,
                                          quantize=quantize_self_attn),
            k_proj=LiJAXLinear.from_torch(torch_model.model.layers[layer_idx].self_attn.k_proj,
                                          quantize=quantize_self_attn),
            v_proj=LiJAXLinear.from_torch(torch_model.model.layers[layer_idx].self_attn.v_proj,
                                          quantize=quantize_self_attn),
            o_proj=LiJAXLinear.from_torch(torch_model.model.layers[layer_idx].self_attn.o_proj,
                                          quantize=quantize_self_attn),
        )

        mlp = LlamaMLPWeights(
            config=lijax_config,
            gate_proj=LiJAXLinear.from_torch(torch_model.model.layers[layer_idx].mlp.gate_proj, quantize=quantize_mlp),
            up_proj=LiJAXLinear.from_torch(torch_model.model.layers[layer_idx].mlp.up_proj, quantize=quantize_mlp),
            down_proj=LiJAXLinear.from_torch(torch_model.model.layers[layer_idx].mlp.down_proj, quantize=quantize_mlp),
        )
        input_layer_norm = LlamaRMSNorm(weight=pt2jax(torch_model.model.layers[layer_idx].input_layer_norm.weight))
        post_attention_layer_norm = LlamaRMSNorm(
            weight=pt2jax(torch_model.model.layers[layer_idx].post_attention_layer_norm.weight))
        layers.append(
            LlamaBlockWeight(
                config=lijax_config,
                self_attn=self_attn,
                mlp=mlp,
                input_layer_norm=input_layer_norm,
                post_attention_layer_norm=post_attention_layer_norm
            )
        )
    embed = LiJAXEmbed.from_torch(torch_model.model.embed_tokens, quantize=quantize_embed)
    norm = LlamaRMSNorm(weight=pt2jax(torch_model.model.norm.weight))
    model = LlamaModelWeight(config=lijax_config, layers=layers, embed_tokens=embed, norm=norm)
    lm_head = LiJAXLinear.from_torch(torch_model.lm_head, quantize=quantize_lm_head)
    causal_language_model = LlamaForCausalLMWeight(config=lijax_config, model=model, lm_head=lm_head)

    return causal_language_model


def convert_llama_model(
        pre_trained_model_name_or_path: str,
        extra_loading_options_for_model: Optional[dict] = None,
        quantize_self_attn: bool = True,
        quantize_mlp: bool = True,
        quantize_embed: bool = True,
        quantize_lm_head: bool = True
):
    if extra_loading_options_for_model is None:
        extra_loading_options_for_model = {}

    model: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(
        pre_trained_model_name_or_path,
        **extra_loading_options_for_model
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_name_or_path, trust_remote_code=True)
        chat_template = tokenizer.chat_template
    except Exception as e:  # noqa
        chat_template = None

    lijax_model = convert_llama_model_weights_to_lijax(
        torch_model=model,
        chat_template=chat_template,
        quantize_mlp=quantize_mlp,
        quantize_embed=quantize_embed,
        quantize_lm_head=quantize_lm_head,
        quantize_self_attn=quantize_self_attn
    )

    print(lijax_model)
