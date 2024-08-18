import mlx.core as mx

# TODO 適切なファイル名に振り分け

epsilon = 1e-5  # float16の場合にlog(0)を防ぐための微小値 TODO: 今回fp16だが別の重みの場合は要検討


def system_prompt_template(content: str) -> dict:
    return {"role": "system", "content": content}


def user_prompt_template(content: str) -> dict:
    return {"role": "user", "content": content}


def get_next_token_logits(logits: mx.array) -> mx.array:
    """バッチの最初の要素から次のトークン（語彙全体）に対する予測を取得"""
    return logits[0, -1, :]


def get_context_logits(logits: mx.array) -> mx.array:
    """バッチの最初の要素からこれまでに与えたトークン（語彙全体）に対する予測を取得"""
    return logits[0, :-1, :]
