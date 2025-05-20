import inspect
import logging
import os
import sys
from typing import Any, Dict

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Custom theme for logger
LOGGER_THEME = Theme({
    "info": "bold cyan",
    "warning": "bold yellow",
    "error": "bold red",
    "critical": "bold white on red",
    "debug": "dim white",
    "function": "bold green",
})

# Create console with custom theme
console = Console(theme=LOGGER_THEME)

def find_caller_details():
    """Find details about the actual calling function."""
    # Walk the stack to find the first frame that's in user code
    # and not in logging or our custom logger
    frame = sys._getframe(0)
    found_logging = False

    while frame:
        frame_info = inspect.getframeinfo(frame)
        module_path = frame_info.filename

        # Skip frames from logging modules or our custom logger
        if ("logging" in module_path or
            "custom_logger.py" in module_path):
            found_logging = True
        # If we found a logging frame before and now we're outside,
        # this is likely the user's calling code
        elif found_logging:
            # Return caller information
            return {
                "function": frame_info.function,
                "filename": os.path.basename(module_path),
                "lineno": frame_info.lineno,
                "module": os.path.splitext(os.path.basename(module_path))[0],
            }

        if frame.f_back:
            frame = frame.f_back
        else:
            break

    # Fallback if we couldn't find a good caller
    return {
        "function": "unknown",
        "filename": "unknown",
        "lineno": 0,
        "module": "unknown",
    }

class CustomRichHandler(RichHandler):
    """Custom Rich handler that shows function names."""
    def emit(self, record) -> None:
        # Add caller details to the record before rendering
        caller = find_caller_details()
        record.function_name = caller["function"]
        record.filename = caller["filename"]
        record.module_name = caller["module"]

        # Now process the record normally
        super().emit(record)

    def render(self, *, record, traceback, message_renderable):
        # Add function name prefix
        if hasattr(record, "function_name") and record.function_name != "unknown":
            prefix = f"[function]{record.function_name}()[/function] "
            message = f"{prefix}{message_renderable}"
        else:
            message = message_renderable

        return super().render(
            record=record,
            traceback=traceback,
            message_renderable=message,
        )

def get_logger(
    name: str,
    level: int = logging.INFO,
    log_to_file: bool = False,
    log_file: str | None = None,
    format_string: str = "%(message)s",
    rich_kwargs: Dict[str, Any] | None = None,
) -> logging.Logger:
    """Create a custom logger that displays function name in a clean format.

    Args:
        name: Logger name, typically __name__
        level: Logging level
        log_to_file: Whether to also log to a file
        log_file: Path to log file, defaults to <name>.log if None
        format_string: Format string for the log message
        rich_kwargs: Additional kwargs for RichHandler

    Returns:
        Configured logger instance
    """
    # Default handler settings
    if rich_kwargs is None:
        rich_kwargs = {
            "rich_tracebacks": True,
            "markup": True,
            "show_time": True,
            "show_level": True,
            "show_path": False,  # We'll handle this in our custom format
            "console": console,
        }

    # Create the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create and add the custom handler
    rich_handler = CustomRichHandler(**rich_kwargs)
    logger.addHandler(rich_handler)

    # Add file handler if requested
    if log_to_file:
        if log_file is None:
            log_file = f"{name.replace('.', '_')}.log"

        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(function_name)s - %(message)s",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

def set_all_loggers_level(level: int) -> None:
    """Set all existing loggers to the specified level."""
    for logger_name in logging.root.manager.loggerDict:
        logging.getLogger(logger_name).setLevel(level)

# Example usage:
# logger = get_logger(__name__)
# logger.info("Processing data...")
# logger.warning("This is a warning", extra={"markup": True})
# logger.error("[bold]Error:[/bold] Something went wrong")
