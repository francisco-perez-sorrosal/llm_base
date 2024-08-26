import functools
import time

from llm_foundation import logger


def timer(fn):
    """Calculate the time to run a function"""
    @functools.wraps(fn)
    def decorator_timer(*args, **kwargs):
        start_time = time.perf_counter()
        return_value = fn(*args, **kwargs)
        end_time = time.perf_counter()
        runtime = end_time - start_time
        logger.info(f"Finished {fn.__name__}() in {runtime:.4f} secs")
        return return_value
    return decorator_timer


def show_banner(text, level=1, width=120):
    """
    Print text in a banner style with hierarchy representation.

    :param text: The text to be printed within the banner.
    :param level: Hierarchy level of the banner (1 for top-level).
    :param width: The width of the banner including the borders.
    """
    if len(text) + 4 > width:  # Ensure text fits in the banner
        width = len(text) + 4

    border_chars = ["*", "=", "-", "."]  # Different chars for different levels

    # Choose border character based on level, defaulting to the last item for higher levels
    border_char = border_chars[min(level - 1, len(border_chars) - 1)] * width

    padding = (width - len(text) - 2) // 2
    padded_text = " " * padding + text + " " * padding
    if len(text) % 2 != 0:  # Adjust for odd-length text
        padded_text += " "

    logger.info(border_char)
    logger.info(border_char[0] + padded_text + border_char[0])
    logger.info(border_char)


def banner(text, level=1, width=120, mark_fn_end=False):
    def decorator_banner(fn):
        @functools.wraps(fn)
        def wrapper_fn(*args, **kwargs):
           show_banner(text, level=level, width=width)
           return_value = fn(*args, **kwargs)
           if mark_fn_end:
               show_banner(f"End of {text}", level=level, width=width)
           return return_value
        return wrapper_fn 
    return decorator_banner
