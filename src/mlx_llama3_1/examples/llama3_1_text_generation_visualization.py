from typing import cast, Literal, Optional, List
from src.mlx_llama3_1.llama3_1 import load_llama3_1
import matplotlib.pyplot as plt
import numpy as np
from mlx_lm.models.llama import Model
from transformers import PreTrainedTokenizer

from rich.console import Console
from rich.padding import Padding, PaddingDimensions
from rich.style import Style

import mlx.core as mx

epsilon = 1e-5  # float16の場合にlog(0)を防ぐための微小値 TODO: 今回fp16だが別の重みの場合は要検討

console = Console()


def system_prompt_template(content: str) -> dict:
    return {"role": "system", "content": content}


def user_prompt_template(content: str) -> dict:
    return {"role": "user", "content": content}


def rich_print(txt: str, padding: PaddingDimensions = (1, 3), style: Optional[Style] = None, **kwargs):
    console.print(
        Padding("[bold]" + txt.strip() + "[/]", padding),
        style=style,
        justify="left",
        crop=False,
        **kwargs,
    )


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


type VISUALIZATION_MODE = Literal["entropy", "probability"]


class CompletionsResult:

    def __init__(self):
        self._perplexity = 0.0
        self._logprobs = 0.0
        self._entropies: List[float] = []
        self._probabilities: List[float] = []
        self._tokens: List[str] = []
        self._prompt = ""

    def set_perplexity(self, perplexity: float):
        self._perplexity = perplexity

    def append_entropy(self, entropy: float):
        self._entropies.append(entropy)

    def append_probability(self, probability: float):
        self._probabilities.append(probability)

    def append_token(self, token: str):
        self._tokens.append(token)

    def set_prompt(self, prompt: str):
        self._prompt = prompt

    def rich_print(self):
        print("[prompt]")
        rich_print(f"[white]{self._prompt}[/]")

        print("[completion / entropy]")
        rich_print("", (1, 3, 0, 3))
        token_buffer: List[str] = []
        for token, entropy in zip(self._tokens, self._entropies):
            c_entropy = get_color(entropy, scale=1.0)
            token_buffer.append(f"[{c_entropy}]{token}[/{c_entropy}]")
            if token.endswith("\n"):
                rich_print("".join(token_buffer).strip(), (0, 3))
                token_buffer = []
        rich_print("".join(token_buffer).strip(), (0, 3, 1, 3))

        print("[completion / probability]")
        token_buffer: List[str] = []
        for token, probability in zip(self._tokens, self._probabilities):
            c_probability = get_color(probability, scale=1.0)
            token_buffer.append(f"[{c_probability}]{token}[/{c_probability}]")
            if token.endswith("\n"):
                rich_print("".join(token_buffer).strip(), (0, 3))
                token_buffer = []
        rich_print("".join(token_buffer).strip(), (0, 3, 1, 3))


result = CompletionsResult()


def generate_text_(
    model: Model,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    metric: VISUALIZATION_MODE,
    scale: float = 1.0,
    max_tokens: int = 100,
    compute_for_prompt: bool = True,
):
    # if metric:
    #     print(f"=== visualized {metric=} ===")

    input_ids = mx.array(tokenizer.encode(prompt, return_tensors="np"))

    input_ids_shape: tuple[int, int] = input_ids.shape
    _, inputs_seq_len = input_ids_shape

    state = input_ids

    for i in range(max_tokens):
        logits = model(state)

        if i == 0:
            # print("[prompt]")
            # if compute_for_prompt:
            #     probs_context = mx.softmax(logits[0, :-1, :], axis=-1)

            #     if metric == "entropy":
            #         _v_context = -mx.sum(probs_context * mx.log(probs_context + epsilon), axis=-1)
            #     elif metric == "probability":
            #         seq_len: int = cast(int, probs_context.shape[0])
            #         _v_context = probs_context[mx.arange(0, seq_len, 1), state[0, 1:]]

            #     _v_context_list = cast(List[float], _v_context.tolist())

            #     _decoded_prev = ""

            #     token_buffer: List[str] = []
            #     _multibyte_buffer = mx.array([])
            #     for i_context, _v in enumerate(_v_context_list):
            #         _decoded = tokenizer.decode(state[0, 1 : 1 + (i_context + 1)].tolist())
            #         token = _decoded[len(_decoded_prev) :]
            #         if set(token) <= {"�", "", " "}:
            #             _multibyte_buffer = mx.concatenate([_multibyte_buffer, mx.array([_v])])
            #         else:
            #             if len(_multibyte_buffer) > 0:
            #                 _v = cast(float, mx.mean(mx.concatenate([_multibyte_buffer, mx.array([_v])])).item())
            #             c = get_color(_v, scale=scale)
            #             token_buffer.append(f"[{c}]{token}[/{c}]")
            #             _decoded_prev = _decoded

            #     rich_print("".join(token_buffer).strip())
            # else:
            #     rich_print(f"[white]{prompt}[/]")

            # print("[completion]")
            # rich_print("", (1, 3, 0, 3))
            result.set_prompt(prompt)

            _decoded_prev = ""
            token_buffer: List[str] = []
            _multibyte_buffer_entropy = mx.array([])
            _multibyte_buffer_probability = mx.array([])

        probs = mx.softmax(get_next_token_logits(logits), axis=-1)
        next_token_id = mx.argmax(probs, axis=-1)

        state = mx.concatenate([state, next_token_id.reshape(1, 1)], axis=1)

        # if metric == "entropy":
        _v_entropy = -mx.sum(probs * mx.log(probs + epsilon), axis=-1)
        _v_entropy = _v_entropy.item()
        # elif metric == "probability":
        _v_probability = probs[next_token_id].item()

        completion = state[0, inputs_seq_len:]
        _decoded = tokenizer.decode(completion.tolist())
        token = _decoded[len(_decoded_prev) :]

        if set(token) <= {"�", "", " "}:
            _multibyte_buffer_entropy = mx.concatenate([_multibyte_buffer_entropy, mx.array([_v_entropy])])
            _multibyte_buffer_probability = mx.concatenate([_multibyte_buffer_probability, mx.array([_v_probability])])

        else:
            if len(_multibyte_buffer_entropy) > 0:
                _v_entropy = mx.mean(mx.concatenate([_multibyte_buffer_entropy, mx.array([_v_entropy])])).item()
            if len(_multibyte_buffer_probability) > 0:
                _v_probability = mx.mean(
                    mx.concatenate([_multibyte_buffer_probability, mx.array([_v_probability])])
                ).item()
            _v_entropy = cast(float, _v_entropy)
            _v_probability = cast(float, _v_probability)
            # c = get_color(_v, scale=scale)
            # token_buffer.append(f"[{c}]{token}[/{c}]")
            _decoded_prev = _decoded
            _multibyte_buffer_entropy = mx.array([])
            _multibyte_buffer_probability = mx.array([])
            # if token.endswith("\n"):
            # rich_print("".join(token_buffer).strip(), (0, 3))
            # token_buffer = []
            result.append_token(token)
            result.append_entropy(_v_entropy)
            result.append_probability(_v_probability)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

    # rich_print("".join(token_buffer).strip(), (0, 3, 1, 3))
    # print()
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
        # system_prompt_template("あなたはフレンドリーなチャットボットです"),
        user_prompt_template("ポケモンは全部で何匹いますか？"),
    ]

    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, max_length=1000)

    # print(generate_text(model, tokenizer, str(inputs)))

    generate_text_(model, tokenizer, str(inputs), "entropy")

    # generate_text_(model, tokenizer, str(inputs), "probability")

    result.rich_print()


if __name__ == "__main__":
    main()
