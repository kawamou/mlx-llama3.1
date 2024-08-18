from typing import cast

from mlx_lm.utils import generate

from src.mlx_llama3_1.llama3_1 import load_llama3_1
from src.mlx_llama3_1.utils import system_prompt_template, user_prompt_template


def main():
    model, tokenizer = load_llama3_1()

    messages = [
        system_prompt_template("あなたはユーザーフレンドリーなチャットボットです"),
        user_prompt_template("動物に関する豆知識を教えて下さい"),
    ]

    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = cast(str, inputs)

    res = generate(model, tokenizer, inputs, verbose=True, max_tokens=1000)


if __name__ == "__main__":
    main()
