from mlx_lm.utils import load
from transformers import PreTrainedTokenizer
from mlx.nn import Module
from typing import Tuple, cast

DEFAULT_LLAMA_3_1_MODEL_ID = "mlx-community/Meta-Llama-3.1-8B-Instruct-bf16"


def load_llama3_1(
    model_name: str = DEFAULT_LLAMA_3_1_MODEL_ID,
) -> Tuple[Module, PreTrainedTokenizer]:
    model, tokenizer = load(model_name)
    return model, cast(PreTrainedTokenizer, tokenizer)
