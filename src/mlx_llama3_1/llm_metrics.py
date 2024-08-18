from typing import List

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class LLMMetrics:

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
