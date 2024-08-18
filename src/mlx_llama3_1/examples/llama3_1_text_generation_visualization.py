import hashlib
from functools import lru_cache
from typing import Dict, List, Literal, Optional, Tuple, TypedDict, cast

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
from mlx_lm.models.llama import Model
from rich.console import Console, Group
from rich.layout import Layout
from rich.padding import Padding, PaddingDimensions
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from transformers import PreTrainedTokenizer
from transformers.utils.generic import TensorType

from src.mlx_llama3_1.llama3_1 import load_llama3_1
from src.mlx_llama3_1.rich import rich_print, with_spinner

epsilon = 1e-5  # float16の場合にlog(0)を防ぐための微小値 TODO: 今回fp16だが別の重みの場合は要検討


def system_prompt_template(content: str) -> dict:
    return {"role": "system", "content": content}


def user_prompt_template(content: str) -> dict:
    return {"role": "user", "content": content}


def get_color(v: float, cmap=plt.get_cmap(), scale=1.0, min_=0.4, max_=0.8):
    """print(_v, ":" ,token)で確認できるがEntropyは暗いほど良い（値が小さい）.
    Probabilityは明るいほど確信度が高い（値が大きい）.
    """
    v_scaled = np.clip(v / scale, min_, max_)
    c = cmap(v_scaled)
    c = np.array(c)[:3] * 255
    c = "rgb(" + ",".join([str(v) for v in c.astype(int)]) + ")"
    return c


def get_next_token_logits(logits: mx.array) -> mx.array:
    """バッチの最初の要素から次のトークン（語彙全体）に対する予測を取得"""
    return logits[0, -1, :]


def get_context_logits(logits: mx.array) -> mx.array:
    """バッチの最初の要素からこれまでに与えたトークン（語彙全体）に対する予測を取得"""
    return logits[0, :-1, :]


# @lru_cache(maxsize=None)
def calculate_loglikelihood(model: Model, tokenizer: PreTrainedTokenizer, prompt: str, completion: str) -> float:
    text = prompt + completion

    start_idx = len(tokenizer.encode(prompt, add_special_tokens=False))
    end_idx = len(tokenizer.encode(text, add_special_tokens=False))

    X = mx.array(tokenizer.encode(text, add_special_tokens=False))
    logits = model(X[None, :-1])[0, start_idx - 1 :, :]
    target = X[start_idx:end_idx]

    probs = mx.softmax(logits, axis=-1)
    log_probs = mx.log(probs + epsilon)

    sequence_length = log_probs.shape[0]
    indices = mx.arange(0, sequence_length, 1)

    token_log_probs = log_probs[indices, target]

    return cast(float, mx.sum(token_log_probs).item())


class CompletionsResult:

    def __init__(self):
        self._logprobs: List[float] = []
        self._entropies: List[float] = []
        self._probabilities: List[float] = []
        self._tokens: List[str] = []
        self._prompt = ""
        self._loglikelihood = 0.0

    def set_perplexity(self, perplexity: float):
        self._perplexity = perplexity

    def append_entropy(self, entropy: float):
        self._entropies.append(entropy)

    def append_probability(self, probability: float):
        self._probabilities.append(probability)

    def append_token(self, token: str):
        self._tokens.append(token)

    def append_logprobs(self, logprobs: float):
        self._logprobs.append(logprobs)

    def set_prompt(self, prompt: str):
        self._prompt = prompt

    def get_perplexity(self):
        total_logprobs = sum(self._logprobs)
        return np.exp(-total_logprobs / len(self._logprobs))

    def get_totallogprobs(self):
        return sum(self._logprobs)

    def set_loglikelihood(self, loglikelihood: float):
        self._loglikelihood = loglikelihood

    def get_loglikelihood(self):
        return self._loglikelihood

    # 可視化として後から保存したの追加できると良い
    def rich_print(self):
        console = Console()
        width = console.width - 100

        # プロンプトセクション
        prompt_panel = Panel(
            Text(self._prompt, overflow="fold"), title="[bold cyan]Prompt", border_style="cyan", width=width
        )

        # トークンとエントロピーの可視化
        entropy_vis = Text(overflow="fold")
        for token, entropy in zip(self._tokens, self._entropies):
            entropy_vis.append(token, style=f"on {get_color(entropy, scale=max(self._entropies))}")

        # トークンと確率の可視化
        prob_vis = Text(overflow="fold")
        for token, prob in zip(self._tokens, self._probabilities):
            prob_vis.append(token, style=f"on {get_color(prob, scale=1.0)}")

        # エントロピーと確率のパネルを作成
        entropy_panel = Panel(
            entropy_vis, title="[bold green]Entropy-based coloring", border_style="green", width=width
        )
        prob_panel = Panel(
            prob_vis, title="[bold yellow]Probability-based coloring", border_style="yellow", width=width
        )

        # メトリクステーブル
        table = Table(
            title="Language Model Metrics",
            title_style="bold",
            show_header=True,
            header_style="bold magenta",
            width=width,
        )
        table.add_column("Token", style="cyan", no_wrap=True)
        table.add_column("Entropy", style="green")
        table.add_column("Probability", style="yellow")
        table.add_column("Log Probability", style="blue")

        for token, entropy, prob, logprob in zip(self._tokens, self._entropies, self._probabilities, self._logprobs):
            table.add_row(token, f"{entropy:.4f}", f"{prob:.4f}", f"{logprob:.4f}")
        table.add_section()
        table.add_row("Total", "---", "---", f"{self.get_totallogprobs():.4f}", style="bold")
        table.add_row(
            "Average",
            f"{np.mean(self._entropies):.4f}",
            f"{np.exp(np.mean(self._logprobs)):.4f}",
            f"{np.mean(self._logprobs):.4f}",
            style="bold",
        )
        table.add_row("Perplexity", "---", "---", f"{self.get_perplexity():.4f}", style="bold")
        table.add_row("Log-Likelihood", "---", "---", f"{self.get_loglikelihood():.4f}", style="bold")

        # layout["table"].update(table)
        output = Group(prompt_panel, entropy_panel, prob_panel, table)

        # 出力
        console.print(Padding(output, (1, 1, 1, 1)))


result = CompletionsResult()


class MultiByteBuffer:
    def __init__(self):
        self._buffer: mx.array = mx.array([])

    @staticmethod
    def is_multibyte_token(token: str):
        multi_byte_tokens = {"�", "", " "}
        return set(token) <= multi_byte_tokens

    def append(self, value: float):
        self._buffer = mx.concatenate([self._buffer, mx.array([value])])

    def mean(self):
        _mean = mx.mean(self._buffer)
        return cast(float, _mean.item())

    def clear(self):
        self._buffer = mx.array([])

    def len(self):
        return len(self._buffer)

    def is_empty(self) -> bool:
        return self.len() == 0


@with_spinner("Generating text...")
def generate_text_(
    model: Model,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_tokens: int = 100,
):
    input_ids = mx.array(tokenizer.encode(prompt, return_tensors=TensorType("mlx")))

    input_ids_shape: tuple[int, int] = input_ids.shape
    _, inputs_seq_len = input_ids_shape

    state = input_ids

    result.set_prompt(prompt)
    _decoded_prev = ""
    _multibyte_buffer_logprobs = MultiByteBuffer()
    _multibyte_buffer_entropy = MultiByteBuffer()
    _multibyte_buffer_probability = MultiByteBuffer()

    for i in range(max_tokens):
        logits = model(state)

        probs = mx.softmax(get_next_token_logits(logits), axis=-1)
        next_token_id = mx.argmax(probs, axis=-1)

        state = mx.concatenate([state, next_token_id.reshape(1, 1)], axis=1)

        _v_logprobs = cast(float, mx.log(probs[next_token_id] + epsilon).item())
        _v_entropy = -mx.sum(probs * mx.log(probs + epsilon), axis=-1)
        _v_entropy = cast(float, _v_entropy.item())
        _v_probability = cast(float, probs[next_token_id].item())

        completion = state[0, inputs_seq_len:]
        _decoded = tokenizer.decode(completion.tolist())
        token = _decoded[len(_decoded_prev) :]

        if MultiByteBuffer.is_multibyte_token(token):
            _multibyte_buffer_logprobs.append(cast(float, _v_logprobs))
            _multibyte_buffer_entropy.append(cast(float, _v_entropy))
            _multibyte_buffer_probability.append(cast(float, _v_probability))

        else:
            if not _multibyte_buffer_logprobs.is_empty():
                _multibyte_buffer_logprobs.append(cast(float, _v_logprobs))
                _v_logprobs = _multibyte_buffer_logprobs.mean()
            if not _multibyte_buffer_entropy.is_empty():
                _multibyte_buffer_entropy.append(cast(float, _v_entropy))
                _v_entropy = _multibyte_buffer_entropy.mean()
            if not _multibyte_buffer_probability.is_empty():
                _multibyte_buffer_probability.append(cast(float, _v_probability))
                _v_probability = _multibyte_buffer_probability.mean()
            _decoded_prev = _decoded
            _multibyte_buffer_logprobs.clear()
            _multibyte_buffer_entropy.clear()
            _multibyte_buffer_probability.clear()
            result.append_logprobs(_v_logprobs)
            result.append_token(token)
            result.append_entropy(_v_entropy)
            result.append_probability(_v_probability)

        if next_token_id.item() == tokenizer.eos_token_id:
            break
    loglikelihood = calculate_loglikelihood(model, tokenizer, prompt, "".join(result._tokens).strip())
    result.set_loglikelihood(loglikelihood)

    return state


# 愚直に計算しているのでかなり遅いがgenerated_textを返すことは検証済み
def generate_text(model: Model, tokenizer: PreTrainedTokenizer, prompt: str, max_length=1000):
    input_ids = tokenizer.encode(prompt, return_tensors="np")
    input_ids = mx.array(input_ids)

    eos_token_id = tokenizer.eos_token_id

    for _ in range(max_length):
        logits = model(input_ids)
        next_token_logits = get_next_token_logits(logits)
        probs = mx.softmax(next_token_logits, axis=-1)
        next_token = mx.argmax(probs, keepdims=True)
        input_ids = mx.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)

        if next_token.item() == eos_token_id:
            break

    generated_text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)

    return generated_text


def main():
    model, tokenizer = load_llama3_1()

    messages = [
        system_prompt_template("You are a man."),
        user_prompt_template("Do you usually eat rice or bread for breakfast?"),
    ]

    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, max_length=1000)

    generate_text_(model, tokenizer, str(inputs))

    result.rich_print()


if __name__ == "__main__":
    main()
