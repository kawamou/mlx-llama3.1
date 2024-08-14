from mlx_lm.utils import generate
from typing import cast
from src.mlx_llama3_1.llama3_1 import load_llama3_1


def system_prompt_template(content: str) -> dict:
    return {"role": "system", "content": content}


def user_prompt_template(content: str) -> dict:
    return {"role": "user", "content": content}


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
