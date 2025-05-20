import os

from numpy import array_str, ndarray, set_printoptions
from numpy._typing import _NestedSequence, _SupportsArray
from typing_extensions import TypeVar

_T_co = TypeVar("_T_co", covariant=True)


class SequenceLike(_NestedSequence, _SupportsArray, list[_T_co]):
    pass

def display(
    obj,
    max_length=50,
    max_string=100,
    indent_guides=False,
    overflow="ellipsis",
):
    if isinstance(obj,SequenceLike) and os.getenv("NO_RICH"):
        return array_str(obj)
    from io import StringIO

    from rich.console import Console
    from rich.pretty import Pretty
    strio = StringIO()
    c = Console(record=True, soft_wrap=True, file=strio)
    c.print(
        Pretty(
            obj,
            max_length=max_length,
            max_string=max_string,
            indent_guides=indent_guides,
            overflow=overflow,
        ),
    )
    return c.export_text(styles=False).strip()[:max_string]
