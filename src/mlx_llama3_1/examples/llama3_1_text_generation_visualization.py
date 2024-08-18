from typing import cast

import mlx.core as mx
from transformers import PreTrainedTokenizer
from transformers.utils.generic import TensorType

from src.mlx_llama3_1.caching_model import CachingModel
from src.mlx_llama3_1.llama3_1 import load_llama3_1
from src.mlx_llama3_1.llm_metrics import LLMMetrics
from src.mlx_llama3_1.utils import (
    epsilon,
    get_next_token_logits,
    system_prompt_template,
    user_prompt_template,
)
from src.mlx_llama3_1.visualizer import visualize_metrics, with_spinner


class MultiByteBuffer:
    """MultiByte文字をバッファリングするクラス. 平均を取って格納する"""

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


# TODO かなり冗長なのでリファクタリングする
@with_spinner("Generating text...")
def text_to_llm_metrics(
    model: CachingModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
) -> LLMMetrics:
    """CachingModelに蓄えたlogitsを用いてLLMMetricsを生成する関数"""
    input_ids = mx.array(tokenizer.encode(prompt, return_tensors=TensorType("mlx")))
    result = LLMMetrics()

    input_ids_shape: tuple[int, int] = input_ids.shape
    _, inputs_seq_len = input_ids_shape

    state = input_ids

    result.set_prompt(prompt)
    decoded_prev = ""
    multibyte_buffer_logprobs = MultiByteBuffer()
    multibyte_buffer_entropy = MultiByteBuffer()
    multibyte_buffer_probability = MultiByteBuffer()

    for _, logits in model._cache.items():
        probs = mx.softmax(get_next_token_logits(logits), axis=-1)
        next_token_id = mx.argmax(probs, axis=-1)

        state = mx.concatenate([state, next_token_id.reshape(1, 1)], axis=1)

        logprobs = cast(float, mx.log(probs[next_token_id] + epsilon).item())
        entropy = -mx.sum(probs * mx.log(probs + epsilon), axis=-1)
        entropy = cast(float, entropy.item())
        probability = cast(float, probs[next_token_id].item())

        completion = state[0, inputs_seq_len:]
        decoded = tokenizer.decode(completion.tolist())
        token = decoded[len(decoded_prev) :]

        if MultiByteBuffer.is_multibyte_token(token):
            multibyte_buffer_logprobs.append(cast(float, logprobs))
            multibyte_buffer_entropy.append(cast(float, entropy))
            multibyte_buffer_probability.append(cast(float, probability))

        else:
            if not multibyte_buffer_logprobs.is_empty():
                multibyte_buffer_logprobs.append(cast(float, logprobs))
                logprobs = multibyte_buffer_logprobs.mean()
            if not multibyte_buffer_entropy.is_empty():
                multibyte_buffer_entropy.append(cast(float, entropy))
                entropy = multibyte_buffer_entropy.mean()
            if not multibyte_buffer_probability.is_empty():
                multibyte_buffer_probability.append(cast(float, probability))
                probability = multibyte_buffer_probability.mean()
            decoded_prev = decoded
            multibyte_buffer_logprobs.clear()
            multibyte_buffer_entropy.clear()
            multibyte_buffer_probability.clear()
            result.append_logprobs(logprobs)
            result.append_token(token)
            result.append_entropy(entropy)
            result.append_probability(probability)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

    return result


@with_spinner("Generating text...")
def generate_text(model: CachingModel, tokenizer: PreTrainedTokenizer, prompt: str, max_length=1000):
    """
    CachingModelを受け入れシンプルにテキスト生成を行う関数
    """
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
        system_prompt_template("醤油味を好む男性をロールプレイしてください。"),
        user_prompt_template("朝食べるのが多いのは、ご飯 or パン？"),
    ]

    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, max_length=1000)

    caching_model = CachingModel(model)

    generate_text(caching_model, tokenizer, str(inputs))

    result = text_to_llm_metrics(caching_model, tokenizer, str(inputs))

    visualize_metrics(result)


if __name__ == "__main__":
    main()
