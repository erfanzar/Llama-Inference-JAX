import time

import os

os.environ["JAX_TRACEBACK_FILTERING"] = "off"

import jax.numpy
from transformers import AutoTokenizer
from src.lijax.model import llama_generate, LlamaForCausalLMWeight
from src.lijax.covertors import convert_llama_model


def prompt_model(
        instruction: str
) -> str:
    return f"### Instruction:\n{instruction}\n### Response:\n"


def main():
    tokenizer = AutoTokenizer.from_pretrained("jan-hq/LlamaCorn-1.1B-Chat")
    lijax_model = convert_llama_model("jan-hq/LlamaCorn-1.1B-Chat")
    lijax_model.save("LlamaCorn.lijax")
    lijax_model = LlamaForCausalLMWeight.load("LlamaCorn.lijax")
    lijax_model.shard()
    tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if (loop.last and add_generation_prompt) or not loop.last %}{{ '<|im_end|>' + '\n'}}{% endif %}{% endfor %}{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    history = []
    while True:
        history.append({"role": "user", "content": input("> ")})
        generated_ids = None
        printed_length = 0
        for token in llama_generate(
                block=lijax_model,
                input_ids=tokenizer.apply_chat_template(
                    history,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="np"
                ),
                use_flash_attention=False,
                # runtime_kernel="pallas",
                runtime_kernel="normal",
                max_length=2048,
                max_new_tokens=512,
                eos_token_id=tokenizer.eos_token_id,
                temperature=1.6,
                # do_sample=True,
                top_k=20,
                top_p=0.95,
        ):
            generated_ids = jax.numpy.concatenate([generated_ids, token], -1) if generated_ids is not None else token
            stream = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=False)
            print(stream[printed_length:], end="")

            printed_length = len(stream)
            if "<|im_end|>" in stream:
                stream = stream.split("<|im_end|>")[0]
                break
        history.append({"role": "assistant", "content": stream})
        print()


if __name__ == '__main__':
    main()
