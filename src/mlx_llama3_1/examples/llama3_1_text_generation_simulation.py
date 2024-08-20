from typing import List, Tuple, cast

import mlx.core as mx
from transformers import PreTrainedTokenizer, TensorType

from src.mlx_llama3_1.caching_model import CachingModel
from src.mlx_llama3_1.llama3_1 import load_llama3_1
from src.mlx_llama3_1.utils import epsilon, get_next_token_logits


def compare_responses(
    model: CachingModel, tokenizer: PreTrainedTokenizer, prompt: str, responses: List[str]
) -> List[Tuple[str, float]]:
    prompt_ids = mx.array(tokenizer.encode(prompt, return_tensors=TensorType("mlx")))
    prompt_length = prompt_ids.shape[1]

    results = []

    for response in responses:
        full_text = prompt + response
        input_ids = mx.array(tokenizer.encode(full_text, return_tensors=TensorType("mlx")))

        log_likelihood = 0.0
        state = input_ids[:, :prompt_length]

        for i in range(prompt_length, len(input_ids[0])):
            logits = model(state)
            probs = mx.softmax(get_next_token_logits(logits), axis=-1)
            next_token_id = input_ids[0, i].item()

            log_prob = mx.log(probs[next_token_id] + epsilon).item()
            log_likelihood += cast(float, log_prob)

            state = mx.concatenate([state, input_ids[:, i : i + 1]], axis=1)

        results.append((response, log_likelihood))

    return sorted(results, key=lambda x: x[1], reverse=True)


# 使用例
prompt = "あなたは日本人男性です。朝食には何を食べますか？"
responses = ["白米を食べます。", "パンケーキを食べます。"]

model, tokenizer = load_llama3_1()  # モデルとトークナイザーのロード
caching_model = CachingModel(model)

comparison_results = compare_responses(caching_model, tokenizer, prompt, responses)

print("Prompt:", prompt)

print("---")

for response, log_likelihood in comparison_results:
    print(f"Response: {response}")
    print(f"Log Likelihood: {log_likelihood}")
    # print(f"Normalized Log Likelihood: {log_likelihood / len(tokenizer.encode(response))}")
    print()
