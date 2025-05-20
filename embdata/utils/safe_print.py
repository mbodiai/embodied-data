from __future__ import annotations

import asyncio
import signal
import sys
import threading
from time import sleep
from typing import Protocol, cast

from rich.console import Console
from rich.live import Live
from rich.progress import Progress
from rich.spinner import Spinner as RichSpinner
from typing_extensions import (
    TYPE_CHECKING,
    Any,
    Callable,
    Concatenate,
    Generator,
    ParamSpec,
    Type,
    TypeVar,
    TypeVarTuple,
    overload,
)

if TYPE_CHECKING:
    from functools import partial
    from threading import Thread
    from types import CellType, CodeType, NoneType

P = ParamSpec("P")
V = TypeVar("V")
U = TypeVar("U")
T = TypeVar("T")
R = TypeVar("R")
Q = ParamSpec("Q")

WRAPPER_ASSIGNMENTS = ("__module__", "__name__", "__qualname__", "__doc__", "__annotations__")
WRAPPER_UPDATES = ("__dict__", "__doc__", "__annotations__")

# Global variables
_progress: "Progress | None" = None  # type: ignore # noqa
_spinner: "Spinner | None" = None  # type: ignore # noqa
_console: "Console | None" = None  # type: ignore # noqa

signal.signal(signal.SIGINT, signal.SIG_DFL)

class Spinner:
    def __init__(self, text: str = "Working...", spinner_type: str = "dots2", console=None):
        self.text = text
        self.spinner_type = spinner_type
        self.spinning = False
        self.stop_requested = False
        self._spinner = RichSpinner(spinner_type, text)
        self._console = console or Console()
        self._live = Live(self._spinner, refresh_per_second=20, transient=True, console=self._console)

        self._thread: Thread | None = None
        self._stop_event = threading.Event()

        import atexit
        atexit.register(self.cleanup)

    def _spin(self):
        with self._live:
            try:
                while not self._stop_event.is_set() and not self.stop_requested:
                    sleep(0.1)
                    self._live.update(self._spinner)
                self.spinning = False
            except KeyboardInterrupt:
                self.stop_requested = True
                self._live.console.clear_live()
                getconsole().clear_live()
                self._live.stop()
                self.stop()

    async def astart(self) -> None:
        await asyncio.to_thread(self.start)

    def start(self) -> None:
        if self.spinning:
            return
        self.spinning = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    async def astop(self) -> None:
        if not self.spinning or self.stop_requested:
            return
        self.stop()

    def stop(self) -> None:
        self._live.stop()
        self._live.console.clear_live()
        getconsole().clear_live()
        if not self.spinning:
            return
        self.stop_requested = True
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join()
            self._thread = None
        self._console.clear_live()
        self._live.stop()
        self.spinning = False
        self._spinner = None
        global _spinner
        _spinner = None

    def cleanup(self) -> None:
        self.stop()

def SPINNER() -> Spinner:
    global _spinner
    if _spinner:
        return _spinner
    _spinner = Spinner()
    return _spinner


def getconsole() -> Console:
    """Get the current console."""
    global _console
    if _console is None:
        _console = Console()
    return _console


def setconsole(console: Console) -> None:
    """Set the current console."""
    global _console
    _console = console


class function(Protocol):
    @property
    def __closure__(self) -> tuple[CellType, ...] | None: ...

    __code__: CodeType
    __defaults__: tuple[Any, ...] | None
    __dict__: dict[str, Any]

    @property
    def __globals__(self) -> dict[str, Any]: ...

    __name__: str
    __qualname__: str
    __annotations__: dict[str, Any]
    __kwdefaults__: dict[str, Any]

    @property
    def __builtins__(self) -> dict[str, Any]: ...

    if sys.version_info >= (3, 12):
        __type_params__: tuple[TypeVar | ParamSpec | TypeVarTuple, ...]
    __module__: str

    def __get__(self, instance: object, owner: type | None = None, /) -> Any: ...

@overload
def wrapafter(
    wrapped: Callable[Concatenate[Any, P], Any] | Type[Any],
    returns: Type[T] | Any = Type[Any],
    assigned=WRAPPER_ASSIGNMENTS,
    updated=WRAPPER_UPDATES,
) -> Callable[[Callable[..., Any]], Callable[P, T]]: ...


@overload
def wrapafter(
    wrapped: Callable[Concatenate[Any, P], Any] | Type[Any],
    returns: None | NoneType = None,
    assigned=WRAPPER_ASSIGNMENTS,
    updated=WRAPPER_UPDATES,
) -> Callable[[Callable[..., Any]], Callable[P, NoneType]]: ...


@overload
def wrapafter(
    wrapped: Callable[Concatenate[V | None, P], U],
    returns: Type[T] | Any = Any,
    assigned=WRAPPER_ASSIGNMENTS,
    updated=WRAPPER_UPDATES,
) -> Callable[[Callable[Concatenate[U | None, Q], Any]], Callable[Concatenate[U, P], T]]: ...


@overload
def wrapafter(
    wrapped: Callable[Concatenate[Type[V], P], U],
    returns: T = Type[Any],
    assigned=WRAPPER_ASSIGNMENTS,
    updated=WRAPPER_UPDATES,
) -> Callable[[Callable[Concatenate[U | None, Q], Any]], Callable[Concatenate[U, P], T]]: ...


def wrapafter(
    wrapped: Callable[Concatenate[V | None, P], U] | Callable[Concatenate[Type[V], P], U] | Any,
    returns=None,
    assigned=WRAPPER_ASSIGNMENTS,
    updated=WRAPPER_UPDATES,
) -> Any:
    """Decorate update_wrapper() to a wrapper function.

    Returns a decorator that invokes update_wrapper() with the decorated
    function as the wrapper argument and the arguments to wraps() as the
    remaining arguments. Default arguments are as for update_wrapper().
    This is a convenience function to simplify applying partial() to
    update_wrapper().
    """
    returns = returns or [Any]
    uw = cast("Callable[..., Callable[..., Any]]", lambda f: f)
    if TYPE_CHECKING:
        return cast("function", partial(uw, wrapped=wrapped, assigned=assigned, updated=updated))
    return lambda f: f


def setspinner(spinner) -> None:
    global _spinner
    _spinner = spinner


def getspinner() -> Spinner:
    global _spinner
    if not _spinner:
        _spinner = SPINNER()
    return _spinner



@wrapafter(Progress, returns=Progress)
def getprogress(*args, **kwargs) -> Progress:
    kw = {"expand": True, "transient": True, "speed_estimate_period": 0.1}
    kw.update(kwargs)
    global _progress
    if not _progress:
        from rich.progress import Progress

        _progress = Progress(*args, **kw)
        setprogress(_progress)
        setconsole(_progress.console)
    return _progress


def setprogress(progress: Progress) -> None:
    global _progress
    _progress = progress


@wrapafter(Console.print, returns=None)
def safe_print(*args, **kwargs) -> None:
    getspinner().stop()
    getprogress().stop()
    args = tuple(list(arg) if isinstance(arg, Generator) else arg for arg in args)
    getconsole().print(*args, **kwargs)
