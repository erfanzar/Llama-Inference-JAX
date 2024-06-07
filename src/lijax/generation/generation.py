import jax
import jax.numpy as jnp
from queue import Queue
from typing import Optional


def prepare_input(input_ids, max_length):
    pad_width = max_length - input_ids.shape[1]
    if pad_width > 0:
        return (
            jnp.pad(input_ids, [(0, 0), (pad_width, 0)]),
            jnp.ones((input_ids.shape[0], max_length)).at[:, :pad_width].set(0)
        )
    else:
        return input_ids[:, abs(pad_width):], jnp.ones((input_ids.shape[0], max_length))


def apply_top_p_sampling(logits, top_p, prng_key):
    """Applies top-p (nucleus) sampling to the logits."""
    assert 0 <= top_p <= 1

    probs_sort, probs_idx = jax.lax.sort_key_val(logits, -jnp.ones_like(logits))
    probs_sum = jnp.cumsum(probs_sort, axis=-1)
    mask = probs_sum - probs_sort > top_p
    probs_sort = jnp.where(mask, 0.0, probs_sort)
    probs_sort = probs_sort / jnp.sum(probs_sort, axis=-1, keepdims=True)
    next_token = jax.random.categorical(prng_key, probs_sort, axis=-1, shape=probs_sort.shape[:-1] + (1,))
    return jnp.take_along_axis(probs_idx, next_token, axis=-1)


def sample_next_token(
        logits,
        rng: jax.random.PRNGKey,
        temperature=1.0,
        top_p=1.0,
):
    """
    Applies temperature, top-p, and top-k filtering to logits and samples the next token.

    Args:
      logits: Logits predicted by the model, shape (batch, seq, vocab_size).
      rng: jax.random.PRNGKey
      temperature: Temperature for scaling the logits.
      top_p: Top-p probability threshold for filtering logits.

    Returns:
      Sampled tokens, shape (batch,).
    """

    # Apply temperature scaling.
    if temperature > 0:
        logits = jax.nn.softmax(logits / temperature, axis=-1)
        return apply_top_p_sampling(logits, top_p, rng)
    else:
        return jnp.argmax(jax.nn.softmax(logits, axis=-1), -1)[:, -1:]


class BaseStreamer:
    def put(self, value):
        raise NotImplementedError()

    def end(self):
        raise NotImplementedError()


class TextStreamer(BaseStreamer):

    def __init__(self, tokenizer: "AutoTokenizer", skip_prompt: bool = False, **decode_kwargs):  # type:ignore
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs

        self.token_cache = []
        self.print_len = 0
        self.next_tokens_are_prompt = True

    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return
        self.token_cache.extend(value.tolist())
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
        if text.endswith("\n"):
            printable_text = text[self.print_len:]
            self.token_cache = []
            self.print_len = 0
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len:]
            self.print_len += len(printable_text)
        else:
            printable_text = text[self.print_len: text.rfind(" ") + 1]
            self.print_len += len(printable_text)

        self.on_finalized_text(printable_text)

    def end(self):
        if len(self.token_cache) > 0:
            text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
            printable_text = text[self.print_len:]
            self.token_cache = []
            self.print_len = 0
        else:
            printable_text = ""

        self.next_tokens_are_prompt = True
        self.on_finalized_text(printable_text, stream_end=True)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        print(text, flush=True, end="" if not stream_end else None)

    @staticmethod
    def _is_chinese_char(cp):
        if (
                (0x4E00 <= cp <= 0x9FFF)
                or (0x3400 <= cp <= 0x4DBF)  #
                or (0x20000 <= cp <= 0x2A6DF)  #
                or (0x2A700 <= cp <= 0x2B73F)  #
                or (0x2B740 <= cp <= 0x2B81F)  #
                or (0x2B820 <= cp <= 0x2CEAF)  #
                or (0xF900 <= cp <= 0xFAFF)
                or (0x2F800 <= cp <= 0x2FA1F)  #
        ):  #
            return True

        return False


class TextIteratorStreamer(TextStreamer):

    def __init__(
            self,
            tokenizer: "AutoTokenizer",  # type:ignore
            skip_prompt: bool = False,
            timeout: Optional[float] = None,
            **decode_kwargs
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.text_queue = Queue()
        self.stop_signal = None
        self.timeout = timeout

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
        self.text_queue.put(text, timeout=self.timeout)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value
