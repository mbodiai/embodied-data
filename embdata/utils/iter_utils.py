import logging
import re
from itertools import zip_longest
from typing import TYPE_CHECKING, Any, Callable, Union

import numpy as np

from embdata.utils.import_utils import smart_import

if TYPE_CHECKING:
    try:
        import torch
    except ImportError:
        torch = Any
    from datasets import Dataset, IterableDataset

    from embdata.sample import DictLike, ListLike, P, Sample
else:
    Sample = Any
    DictLike = Any
    ListLike = Any
    P = Any
    Dataset = smart_import("datasets.Dataset", "type_safe_lazy")
    IterableDataset = smart_import("torch.utils.data.IterableDataset", "type_safe_lazy")

logging = logging.getLogger(" ")

MAX_FLATTENED_SIZE = 64

exists_iter = lambda k, c: c is not None and len(c) > 0 and (hasattr(c[0], k) or k in c[0])
"""Does the first element in the iterable have the specified key?"""

get_iter = lambda k, c: None if not exists_iter(k, c) else c[0][k] if k in c[0] else getattr(c[0], k)
"""Get the value of the specified key from the first element in the iterable."""

get_iter_class = lambda k, c: get_iter(k, c).__class__ if get_iter(k, c) is not None else None
"""Get the class of the value of the specified key from the first element in the iterable."""

get_iter_size = lambda k, c: len(get_iter(k, c)) if get_iter(k, c) is not None else None
"""Get the size of the value of the specified key."""

get_iter_item = lambda k, c, i: get_iter(k, c)[i] if get_iter(k, c) is not None else None
"""Get the item at the specified index from the value of the specified key."""

map_iter = lambda k, c, f: list(map(f, get_iter(k, c))) if get_iter(k, c) is not None else None
"""Map the specified function to the value of the specified key."""

pack_iters = lambda f, *c: [f(*step) for step in zip(*c, strict=False)]
"""Pack the iterables into steps and apply the specified function to each step."""

pack_iters_longest = lambda f, *c: [f(*step) for step in zip_longest(*c, strict=False, fillvalue="Sample")]
"""Pack the iterables into steps and apply the specified function to each step with fill values."""

strip_iter = lambda k, c: ([step.pop(k) for step in c], c)
"""Strip or peel off the specified index from each step in the iterable."""

unstrip_iter = lambda f, c, s: [f(*step, s) for step, s in zip_longest(*c, s, strict=False, fillvalue=s)]
"""Undo the strip or peel operation by adding back the stripped iterable."""

shave_iter = lambda c, s: [step[:-s] for step in c]
"""Like strip but removes elements by indices."""

unshave_iter = lambda f, c, s: [f(*step, s) for step, s in zip_longest(*c, s, strict=False, fillvalue=s)]
"""Like unstrip but adds elements back by indices."""


def map_nested(fn: Callable, sample: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {k: map_nested(fn, v) if isinstance(v, dict) else fn(v) for k, v in sample.items()}


def map_nested_with_keys(fn: Callable, sample: dict, keys: tuple = (), leaf_predicate: Callable[[tuple, Any], bool]| None = None) -> dict:
    """Maps a function over a nested dictionary. Uses leaf_predicate to determine if the function should be applied.

    Default leaf_predicate is whether the value is a dictionary or not.
    """
    leaf_predicate = leaf_predicate or (lambda _, v: not hasattr(v, "items"))
    if leaf_predicate(keys, sample) or not isinstance(sample, dict):
        return fn(keys, sample)
    return {
        k: map_nested_with_keys(fn, v, (*keys, k), leaf_predicate)
        for k, v in sample.items()
    }


def is_exact_match(key: tuple, keys: tuple, sep=".") -> bool:
    return any(key.endswith(k.split(sep)[-1]) for k in keys)


def replace_ints_with_wildcard(s, sep=".") -> str:
    pattern = rf"(?<=^{sep})\d+|(?<={sep})\d+(?={sep})|\d+(?={sep}|$)"
    return re.sub(pattern, "*", s).rstrip(f"{sep}*").lstrip(f"{sep}*")


def is_excluded(key: str, exclude: None | set = None, sep=".") -> bool:
    """Check if the key should be excluded."""
    return any(e == replace_ints_with_wildcard(key, sep) for e in (exclude if exclude is not None else []))


logged_large_tensor = False

def update(d:Union["DictLike", "ListLike"], *args, **kwargs):
    """Update a nested dictionary or list with the specified key-value pairs.

    Examples:
    >>> d = Sample()
    >>> update(d, key="value")
    {'key': 'value'}
    >>> update(d, {"key2": "value2"})
    {'key': 'value', 'key2': 'value2'}
    """
    from embdata.sample import DictLike, ListLike # noqa

    obj = d
    new_kwargs = {}
    new_kwargs.update(zip([f"arg{i}" for i in range(len(args))], args, strict=False), **kwargs)
    kwargs = new_kwargs
    if len(kwargs) == 1 and isinstance(arg := kwargs[next(iter(kwargs.keys()))], DictLike):
        for k, v in arg.items():
            d[k] = v
        return obj
    if len(kwargs) == 1:
        msg = "Expected a dictionary-like object as the first argument."
        raise ValueError(msg)
    if len(kwargs) in (2, 3):
        if "key" in kwargs:
            key = kwargs.pop("key")
            if "delimiter" in kwargs:
                delimiter = kwargs.pop("delimiter")
                value = next(iter(kwargs.values()))
            elif len(kwargs) == 1:
                value = next(iter(kwargs.values()))
                delimiter = "."
            else:
                msg = "Only one value should be provided if 'key' is specified."
                raise ValueError(msg)
            update_kwargs = {key: value}
        else:
            keys = list(kwargs.keys())
            if keys[0] == "arg0":
                key = kwargs.pop("arg0")
                value = kwargs.pop(keys[1])

            if "delimiter" in kwargs:
                delimiter = kwargs.pop("delimiter")
                value = next(iter(kwargs.values()))
            elif len(kwargs) == 1:
                value = next(iter(kwargs.values()))
                delimiter = "."
            else:
                msg = "Only one value should be provided if 'key' is specified."
                raise ValueError(msg)
            update_kwargs = {key: value}
    elif len(args) > 0:
        msg = "Cannot update Sample with both positional and keyword arguments."
        raise ValueError(msg)
    else:
        delimiter = kwargs.pop("delimiter", ".")
        update_kwargs: DictLike = kwargs

    for key, value in update_kwargs.items():
        keys = key.split(delimiter)
        for _i, key in enumerate(keys[:-1]):
            if key.isdigit():  # Handle list index
                key = int(key)
                if isinstance(d, list) and len(d) <= key:
                    d.extend([Sample()] * (key - len(d) + 1))
            else:
                msg = f"Key {key} is not a valid index for a list."
                raise ValueError(msg)
            d = d[key] if isinstance(d, list) else d.setdefault(key, {})

        final_key = keys[-1]
        if final_key.isdigit():
            final_key = int(final_key)
            while len(d) <= final_key:
                d.append({})
            d[final_key] = value
        else:
            d[final_key] = value
    return obj

def flatten_recursive(
    obj,
    exclude: None | set = None,
    non_numerical="allow",
    sep=".",
    include=None,
) -> tuple[list[str], list]:
    """Flatten a nested dictionary or list into a list of keys and a list of values."""
    include = include or []

    def _flatten(obj, prefix=""):
        if isinstance(obj, np.ndarray):
            if len(np.ravel(obj)) > MAX_FLATTENED_SIZE:
                global logged_large_tensor  # noqa
                if not logged_large_tensor:
                    logging.warning(
                        "Large tensor encountered, skipping flattening. %s shape %s include: %s",
                        prefix,
                        obj.shape,
                        include,
                    )
                    logged_large_tensor = True
                if is_exact_match(prefix.rstrip(sep), include, sep):
                    # logging.debug(f"Exact match Key: {prefix}, include: {include}")
                    return [prefix.rstrip(sep)], [obj]
                return [], []
            # logging.debug(f"Converting tensor to numpy array: {obj}")
            obj = obj.tolist()
        out = []
        keys = []
        if isinstance(obj, dict) or (hasattr(obj, "items") and callable(obj.items)):
            for k, v in obj.items():
                new_key = f"{prefix}{k}" if prefix else k
                # logging.debug(f"Key: {new_key}, include: {include}")

                if is_excluded(new_key, exclude, sep):
                    continue
                if include and is_exact_match(new_key, include, sep):
                    # logging.debug(f"Exact match Key: {k}, include: {include}")
                    out.append(v)
                    keys.append(new_key)
                    continue
                subkeys, subouts = _flatten(v, f"{new_key}{sep}")
                out.extend(subouts)
                keys.extend(subkeys)
        elif isinstance(obj, list | Dataset | IterableDataset):
            for i, v in enumerate(obj):
                # logging.debug("Prefix: %s, i: , %s", prefix + str(i), i)
                subkeys, subouts = _flatten(v, f"{prefix}{i}{sep}")
                out.extend(subouts)
                keys.extend(subkeys)
        else:
            if non_numerical == "forbid" and not isinstance(obj, int | float | np.number):
                msg = f"Non-numerical value encountered: {obj}"
                raise TypeError(msg)
            if non_numerical == "ignore" and not isinstance(obj, int | float | np.number):
                return [], []
            out.append(obj)
            keys.append(prefix.rstrip(sep))
        return keys, out

    return _flatten(obj)

    # def rearrange(self, pattern: str, **kwargs) -> Any:
    #     """Pack, unpack, flatten, select indices according to an einops-style pattern.

    #     rearrange('(b s) [action state] -> b s [actions state]', s=32) will select the action and state keys
    #      and pack them into batches of size 32.
    #     """
    #     # Parse the input and output patterns
    #     input_pattern, output_pattern = pattern.split('->')
    #     input_pattern = input_pattern.strip()
    #     output_pattern = output_pattern.strip()

    #     # Extract keys from square brackets
    #     input_keys = re.findall(r'\[([^\]]+)\]', input_pattern)
    #     output_keys = re.findall(r'\[([^\]]+)\]', output_pattern)

    #     # Flatten the sample and select only the required keys
    #     flattened = self.flatten(to="dict")
    #     selected_data = {key: flattened[key] for key in input_keys[0].split() if key in flattened}

    #     # Convert selected data to numpy arrays
    #     np_data = {k: np.array(v) for k, v in selected_data.items()}

    #     # Apply einops rearrange
    #     rearranged_data = einops_rearrange(np_data, pattern, **kwargs)

    #     if isinstance(rearranged_data, dict):
    #         # If the output is a dictionary, directly assign it to the output Sample
    #         for k, v in rearranged_data.items():
    #             setattr(output_sample, k, v.tolist() if isinstance(v, np.ndarray) else v)
    #     else:
    #         # If the output is not a dictionary, we need to reconstruct it based on the output pattern
    #         output_keys = output_keys[0].split() if output_keys else input_keys[0].split()
    #         for i, key in enumerate(output_keys):
    #             setattr(output_sample, key, rearranged_data[..., i].tolist())

    #     return output_sample

    # def setdefault(self, key: str, default: Any, nest=True) -> Any:
    #     """Set the default value for the attribute with the specified key."""
    #     if not nest:
    #         if key in self:
    #             return self[key]
    #         self[key] = default
    #         return default
    #     keys = key.split(".")
    #     obj = self
    #     for k in keys[:-1]:
    #         k = "_items" if k == "items" else k
    #         if k == "*":
    #             return obj
    #         if isinstance(obj, dict):
    #             obj = obj.setdefault(k, {})
    #             try:
    #                 index = int(k)
    #                 if index >= len(obj):
    #                     obj.extend([None] * (index - len(obj) + 1))
    #                 if obj[index] is None:
    #                     obj[index] = {}
    #                 obj = obj[index]
    #             except ValueError:
    #                 raise AttributeError(f"Invalid list index: {k}")
    #         elif not isinstance(obj, Sample) and not hasattr(obj, k):
    #             new_obj = Sample()
    #             obj[k] = new_obj
    #             obj = new_obj
    #         elif hasattr(obj, k):
    #             obj = getattr(obj, k)
    #         else:
    #             obj[k] = Sample()
    #             obj = getattr(obj, k)
    #     if isinstance(obj, dict):
    #         if keys[-1] == "*":
    #             return obj.setdefault("*", default if isinstance(default, list) else [default])
    #         return obj.setdefault(keys[-1], default)
    #     if isinstance(obj, list):
    #         if keys[-1] == "*":
    #             return obj
    #         try:
    #             index = int(keys[-1])
    #             if index >= len(obj):
    #                 obj.extend([None] * (index - len(obj) + 1))
    #             if obj[index] is None:
    #                 obj[index] = default
    #             return obj[index]
    #         except ValueError:
    #             raise AttributeError(f"Invalid list index: {keys[-1]}")
    #     if not hasattr(obj, keys[-1]):
    #         if keys[-1] == "*":
    #             setattr(obj, keys[-1], default if isinstance(default, list) else [default])
    #         else:
    #             setattr(obj, keys[-1], default)
    #     return getattr(obj, keys[-1])
