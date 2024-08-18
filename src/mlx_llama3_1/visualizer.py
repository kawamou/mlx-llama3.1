import time
from functools import wraps
from typing import List, Literal, Optional, cast

from rich.console import Console
from rich.padding import Padding, PaddingDimensions
from rich.style import Style

console = Console()


def rich_print(txt: str, padding: PaddingDimensions = (1, 3), style: Optional[Style] = None, **kwargs):
    console.print(
        Padding("[bold]" + txt.strip() + "[/]", padding),
        style=style,
        justify="left",
        crop=False,
        **kwargs,
    )


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
