import logging
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Literal, get_origin

import numpy as np

from embdata.utils.import_utils import smart_import

if TYPE_CHECKING:
    from types import ModuleType
    spaces: ModuleType = smart_import("gymnasium.spaces", "type_safe_lazy")
    Space = spaces.Space
else:
    spaces = smart_import("gymnasium.spaces", "type_safe_lazy")


def space_for(
    value: Any,
    max_text_length: int = 1000,
    info: Dict[str, Any] | None = None,
) -> "Space":
    """Default Gym space generation for a given value.

    Only used for subclasses that do not override the space method.
    """
    spaces = smart_import("gymnasium.spaces")
    if isinstance(value, Enum) or get_origin(value) == Literal:
        return spaces.Discrete(len(value.__args__))
    if isinstance(value, bool):
        return spaces.Discrete(2)
    if hasattr(value, "dump") and hasattr(value, "dict"):
        value = value.dict()
    if isinstance(value, dict):
        return spaces.Dict(
            {k: space_for(v, max_text_length, info) for k, v in value.items()},
        )
    if isinstance(value, str):
        return spaces.Text(max_length=max_text_length)

    if isinstance(value, int | float | list | tuple | np.ndarray | np.number):
        bounds = None
        dtype = None
        shape = None
        if info is not None:
            shape = info.get("shape")
            bounds = info.get("bounds")
            dtype = info.get("dtype")
        try:
            if not hasattr(value, "shape") and not hasattr(value, "__len__"):
                shape = ()
                dtype = type(value)
                low, high = bounds or (-np.inf, np.inf)
                return spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
            value = np.asarray(value, dtype=float)
            shape = shape or value.shape
            dtype = dtype or value.dtype
            if bounds is None:
                low = np.full(shape, -np.inf, dtype=dtype)
                high = np.full(shape, np.inf, dtype=dtype)
            else:
                low, high = bounds
            return spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
        except Exception as e:  # noqa: BLE001
            logging.info(f"Could not convert value {value} to numpy array: {e}")  # noqa: G004
            if hasattr(value, "dump"):
                value = value.dump()
            if len(value) > 0 and isinstance(value[0], dict):
                return spaces.Tuple(
                    [spaces.Dict(space_for(v, max_text_length, info)) for v in value],
                )
            return spaces.Tuple(
                [spaces.Dict(space_for(value[0], max_text_length, info)) for value in value[:1]],
            )
    msg = f"Unsupported object {value} of type: {type(value)} for space generation"
    raise ValueError(msg)
