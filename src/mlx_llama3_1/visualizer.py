import time
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console, Group
from rich.padding import Padding, PaddingDimensions
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text

from .llm_metrics import LLMMetrics

console = Console()


def with_spinner(message="Processing..."):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with console.status(message, spinner="dots") as status:
                try:
                    result = func(*args, **kwargs)
                    status.update(status="Done!", spinner="monkey")
                    time.sleep(0.5)  # 完了メッセージを少し表示
                    return result
                except Exception as e:
                    status.update(status=f"Error: {str(e)}")
                    time.sleep(0.5)  # エラーメッセージを少し表示
                    raise

        return wrapper

    return decorator


def get_color(v: float, cmap=plt.get_cmap(), scale=1.0, min_=0.4, max_=0.8):
    """print(_v, ":" ,token)で確認できるがEntropyは暗いほど良い（値が小さい）.
    Probabilityは明るいほど確信度が高い（値が大きい）.
    """
    v_scaled = np.clip(v / scale, min_, max_)
    c = cmap(v_scaled)
    c = np.array(c)[:3] * 255
    c = "rgb(" + ",".join([str(v) for v in c.astype(int)]) + ")"
    return c


def visualize_metrics(llm_metrics: LLMMetrics):
    console = Console()
    width = console.width - 100

    # プロンプトセクション
    prompt_panel = Panel(
        Text(llm_metrics._prompt, overflow="fold"), title="[bold cyan]Prompt", border_style="cyan", width=width
    )

    # トークンとエントロピーの可視化
    entropy_vis = Text(overflow="fold")
    for token, entropy in zip(llm_metrics._tokens, llm_metrics._entropies):
        entropy_vis.append(token, style=f"on {get_color(entropy, scale=max(llm_metrics._entropies))}")

    # トークンと確率の可視化
    prob_vis = Text(overflow="fold")
    for token, prob in zip(llm_metrics._tokens, llm_metrics._probabilities):
        prob_vis.append(token, style=f"on {get_color(prob, scale=1.0)}")

    # エントロピーと確率のパネルを作成
    entropy_panel = Panel(entropy_vis, title="[bold green]Entropy-based coloring", border_style="green", width=width)
    prob_panel = Panel(prob_vis, title="[bold yellow]Probability-based coloring", border_style="yellow", width=width)

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

    for token, entropy, prob, logprob in zip(
        llm_metrics._tokens, llm_metrics._entropies, llm_metrics._probabilities, llm_metrics._logprobs
    ):
        table.add_row(token, f"{entropy:.4f}", f"{prob:.4f}", f"{logprob:.4f}")
    table.add_section()
    table.add_row("Total", "---", "---", f"{llm_metrics.get_totallogprobs():.4f}", style="bold")
    table.add_row(
        "Average",
        f"{np.mean(llm_metrics._entropies):.4f}",
        f"{np.exp(np.mean(llm_metrics._logprobs)):.4f}",
        f"{np.mean(llm_metrics._logprobs):.4f}",
        style="bold",
    )
    table.add_row("Perplexity", "---", "---", f"{llm_metrics.get_perplexity():.4f}", style="bold")

    # layout["table"].update(table)
    output = Group(prompt_panel, entropy_panel, prob_panel, table)

    # 出力
    console.print(Padding(output, (1, 1, 1, 1)))
