import hashlib
from typing import Dict, List, cast

import mlx.core as mx
from mlx_lm.models.llama import Model

type Cache = Dict[str, mx.array]


class CachingModel:
    """LLMの出力logitsを全て記憶しておくクラス"""

    def __init__(self, model: Model):
        self._model = model
        self._cache: Cache = {}

    def __call__(self, input_ids: mx.array) -> mx.array:
        input_ids_str = ",".join(map(str, cast(List, input_ids.tolist())[0]))
        key_ = hashlib.sha256(input_ids_str.encode())
        key = key_.hexdigest()
        if key not in self._cache:  # TODO キャッシュは後から計算するためなので正確にはこのif文は不要
            logits = self._model(input_ids)
            self._cache[key] = logits
        return self._cache[key]

    def clear(self):
        self._cache = {}
