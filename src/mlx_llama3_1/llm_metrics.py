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

from src.mlx_llama3_1.caching_model import CachingModel
from src.mlx_llama3_1.llama3_1 import load_llama3_1
from src.mlx_llama3_1.visualizer import rich_print, with_spinner


def get_color(v: float, cmap=plt.get_cmap(), scale=1.0, min_=0.4, max_=0.8):
    """print(_v, ":" ,token)で確認できるがEntropyは暗いほど良い（値が小さい）.
    Probabilityは明るいほど確信度が高い（値が大きい）.
    """
    v_scaled = np.clip(v / scale, min_, max_)
    c = cmap(v_scaled)
    c = np.array(c)[:3] * 255
    c = "rgb(" + ",".join([str(v) for v in c.astype(int)]) + ")"
    return c


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

        # layout["table"].update(table)
        output = Group(prompt_panel, entropy_panel, prob_panel, table)

        # 出力
        console.print(Padding(output, (1, 1, 1, 1)))
