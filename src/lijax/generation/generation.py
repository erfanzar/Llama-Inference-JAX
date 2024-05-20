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


def sample_next_token(
        logits,
        rng: jax.random.PRNGKey,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        do_sample: bool = True
):
    """
    Applies temperature, top-p, and top-k filtering to logits and samples the next token.

    Args:
      logits: Logits predicted by the model, shape (batch, seq, vocab_size).
      rng: jax.random.PRNGKey
      temperature: Temperature for scaling the logits.
      top_p: Top-p probability threshold for filtering logits.
      top_k: Top-k number of logits to keep after filtering by probability.
      do_sample: boolean indicating whether to sample a new token or not

    Returns:
      Sampled tokens, shape (batch,).
    """

    # Apply temperature scaling.
    if do_sample:
        logits = logits / temperature

    batch, seq, vocab_size = logits.shape

    # Apply top-p filtering.
    if top_p < 1.0 and do_sample:
        sorted_logits = jnp.sort(logits, axis=-1)[..., ::-1]
        sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
        cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)

        # Create a mask for logits exceeding the top-p threshold.
        cutoff_index = jnp.argmax(cumulative_probs >= top_p, axis=-1)
        cutoff_index = jnp.expand_dims(cutoff_index, axis=-1)
        cutoff_mask = jnp.arange(vocab_size) < cutoff_index

        # Mask logits exceeding the top-p threshold.
        logits = jnp.where(cutoff_mask, logits, jnp.full_like(logits, float('-inf')))

    # Apply top-k filtering.
    if top_k > 0 and do_sample:
        top_k_logits, top_k_indices = jax.lax.top_k(logits, k=top_k)
        top_k_mask = jnp.zeros_like(logits, dtype=bool)
        top_k_mask = top_k_mask.at[jnp.arange(batch)[:, None], jnp.arange(seq)[:, None], top_k_indices].set(True)

        # Mask logits outside the top-k.
        logits = jnp.where(top_k_mask, logits, jnp.full_like(logits, float('-inf')))

    # Sample from the filtered logits.
    probs = jax.nn.softmax(logits, axis=-1)
    if do_sample:
        return jnp.atleast_2d(jax.random.categorical(rng, probs)[:, -1])
    return jnp.atleast_2d(jnp.argmax(probs, -1)[:, -1])


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
