# Copyright 2024 Mbodi AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A base model class for serializing, recording, and manipulating arbitray data.

It was designed to be extensible, flexible, yet strongly typed. In addition to
supporting any json API out of the box, it can be used to represent
arbitrary action and observation spaces in robotics and integrates seemlessly with H5, Gym, Arrow,
PyTorch, numpy, and HuggingFace Datasets.

Methods:
    schema: Get a simplified json schema of your data.
    to: Convert the Sample instance to a different container type:
        -
    default_value: Get the default value for the Sample instance.
    unflatten: Unflatten a one-dimensional array or dictionary into a Sample instance.
    flatten: Flatten the Sample instance into a one-dimensional array or dictionary.
    space_for: Default Gym space generation for a given value.
    init_from: Initialize a Sample instance from a given value.
    from_space: Generate a Sample instance from a Gym space.
    pack_from: Pack a list of samples into a single sample with lists for attributes.
    unpack: Unpack the packed Sample object into a list of Sample objects or dictionaries.
    dict: Return the Sample object as a dictionary with None values excluded.
    field_info: Get the FieldInfo for a given attribute key.
    space: Return the corresponding Gym space for the Sample instance based on its instance attributes.
    random_sample: Generate a random Sample instance based on its instance attributes.

Examples:
    >>> sample = Sample(x=1, y=2, z={"a": 3, "b": 4}, extra_field=5)
    >>> sample.flatten()
    [1, 2, 3, 4, 5]
    >>> sample.schema()
    {'type': 'object',
        'properties': {
            'x': {'type': 'number'},
            'y': {'type': 'number'},
            'z': {'type': 'object'},
        'properties':
        {
        'a':{'type': 'number'},
        'b': {'type': 'number'}
        }
    },
    'extra_field': {
        'type': 'number'
    }
    >>> Sample.unflatten(flat_list, schema)
    Sample(x=1, y=2, z={'a': 3, 'b': 4}, extra_field=5)
"""

import logging as log
import traceback
from functools import cached_property
from itertools import zip_longest

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic.fields import FieldInfo
from rich.pretty import pretty_repr
from typing_extensions import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Dict,
    Generator,
    ItemsView,
    Iterable,
    List,
    Literal,
    Union,
)

from embdata.describe import describe, describe_keys, full_paths
from embdata.utils import iter_utils, schema_utils
from embdata.utils.hooks import Dataset, Features, spaces, torch
from embdata.utils.import_utils import smart_import
from embdata.utils.space_utils import space_for

OneDimensional = Annotated[Literal["dict", "np", "pt", "list", "sample"], "Numpy, PyTorch, list, sample, or dict"]

logged_recurse = False

if TYPE_CHECKING:
    lists = Any
    dicts = Any
    pytorch = Any
    samples = Any
    pt = Any
    sample = Any
    ignore = Any
    forbid = Any
    allow = Any
    descriptions = Any
    exclude = Any
    recurse = Any
    info = Any
    simple = Any
    tensor = Any
    longest = Any
    truncate = Any
    shallow = Any
    python = Any
    json = Any






class Sample(BaseModel):
    """A base model class for serializing, recording, and manipulating arbitray data."""

    model_config = ConfigDict(
        use_enum_values=False,
        validate_assignment=False,
        extra="allow",
        arbitrary_types_allowed=True,
        populate_by_name=True,
        from_attributes=True,
    )

    def __init__(self, wrapped=None, **data) -> None:
        """A base model class for serializing, recording, and manipulating arbitray data.

        It accepts any keyword arguments and endows them with the following methods:

        Methods:
            schema: Get a simplified json schema of your data.
            to: Convert the Sample instance to a different container type:
                -
            default_value: Get the default value for the Sample instance.
            unflatten: Unflatten a one-dimensional array or dictionary into a Sample instance.
            flatten: Flatten the Sample instance into a one-dimensional array or dictionary.
            space_for: Default Gym space generation for a given value.
            init_from: Initialize a Sample instance from a given value.
            from_space: Generate a Sample instance from a Gym space.
            pack_from: Pack a list of samples into a single sample with lists for attributes.
            unpack: Unpack the packed Sample object into a list of Sample objects or dictionaries.
            dict: Return the Sample object as a dictionary with None values excluded.
            field_info: Get the FieldInfo for a given attribute key.
            space: Return the corresponding Gym space for the Sample instance based on its instance attributes.
            random_sample: Generate a random Sample instance based on its instance attributes.

        Examples:
            >>> sample = Sample(x=1, y=2, z={"a": 3, "b": 4}, extra_field=5)
            >>> sample.flatten()
            [1, 2, 3, 4, 5]
            >>> sample.schema()
            {'type': 'object',
                'properties': {
                    'x': {'type': 'number'},
                    'y': {'type': 'number'},
                    'z': {'type': 'object'},
                'properties':
                {
                'a':{'type': 'number'},
                'b': {'type': 'number'}
                }
            },
            'extra_field': {
                'type': 'number'
            }
            >>> Sample.unflatten(flat_list, schema)
            Sample(x=1, y=2, z={'a': 3, 'b': 4}, extra_field=5)
        """
        if isinstance(wrapped, Sample):
            # Only wrap if no other data is provided.
            if not data:
                data = {k: v for k, v in wrapped.model_dump() if not k.startswith("_")}
        elif isinstance(wrapped, dict):
            # Only wrap if no other data is provided.
            if not data:
                data = {
                    k: Sample(**v) if isinstance(v, dict) else v for k, v in wrapped.items() if not k.startswith("_")
                }
        elif self.__class__ == Sample:
            # Only the Sample class can wrap an arbitrary type.
            if hasattr(wrapped, "__len__") and hasattr(wrapped, "__getitem__") and not isinstance(wrapped, str | bytes | dict | Sample):
                # There is no schema to unflatten from, just have it as an attribute
                data["wrapped"] = wrapped
            elif wrapped is not None:
                data["wrapped"] = wrapped
        elif hasattr(wrapped, "__len__") and hasattr(wrapped, "__getitem__") and not isinstance(wrapped, str | bytes | dict | Sample):
            # Derived classes have a schema to unflatten from.
            d = self.__class__.unflatten(wrapped).model_dump()
            if not d and wrapped:
                msg = f"Could not unflatten {type(wrapped)} with schema {self.schema()} for {self.__class__}"
                raise ValueError(msg)
            data.update(d)
        elif isinstance(wrapped, spaces.Space):
            data.update(self.from_space(wrapped).model_dump())
        elif "items" in data:
            data["_items"] = data.pop("items")
        super().__init__(**data)
        self.__post_init__()

    def __len__(self) -> int:
        """Return the number of attributes in the Sample instance."""
        return len(list(self.keys()))

    def shape(self) -> Dict[str, int]:
        """Return the number of attributes and lengths of the longest nested attributes."""
        shape = {"attributes": len(self)}
        for k, v in super().__iter__():
            if isinstance(v, Sample):
                shape[k] = v.shape()
            elif isinstance(v, list | tuple | np.ndarray):
                shape[k] = len(v)
        return shape

    def __getitem__(self, key: str | int) -> Any:  # noqa
        """Implements nested or flat key access for the Sample object.

        If the key is an integer and the Sample object wraps a list, the value is returned at the specified index.
        If the key is a string and contains a separator ('.' or '/'), the value is returned at the specified nested key.
        Otherwise, the value is returned as an attribute of the Sample object.
        """
        og_key = key
        if isinstance(key, int) and hasattr(self, "_items"):
            return self._items[key]
        if isinstance(key, int) and hasattr(self, "wrapped") and isinstance(self.wrapped, List | Dataset):
            return self.wrapped[key]

        if self.__class__ == Sample:
            if isinstance(key, int):
                if hasattr(self, "_items"):
                    return self._items[key]
                if hasattr(self, "wrapped") and isinstance(self.wrapped, ListLike):
                    return self.wrapped[key]

                items = getattr(self, "items", None)
                items = [] if items is None else self._items if hasattr(self, "_items") else self.values()
                if isinstance(items, Generator):
                    items = list(items)
                if callable(items):
                    items = list(items())
                if len(items) < key or key < 0:
                    msg = f"Index out of range: {key} (expected 0-{len(items) - 1})"
                    raise IndexError(msg)
                try:
                    return items[key]
                except Exception as e:
                    msg = f"Indexing not supported for {type(items)}: {items}"
                    raise TypeError(msg) from e
            if isinstance(key, int):
                msg = f"Sample object does not wrap a list but index was requested: {key}. Did you mean to call items? "
                raise TypeError(msg)
        try:
            if isinstance(key, str) and any(c in key for c in "./*"):
                sep = "." if "." in key else "/"
                key = key.replace("*", "").replace(f"{sep}{sep}", sep)
                keys = key.split(sep)
                obj = self
                for k in keys[:-1]:
                    if k:
                        k = "_items" if k == "items" else k
                        obj = obj[k]
                k = keys[-1] if keys[-1] != "items" else "_items"
                return obj[k] if k is not None else obj
            return getattr(self, key)
        except (AttributeError, KeyError, TypeError) as e:
            if hasattr(self, "_extra"):
                sep = "." if "." in key else "/"
                keys = og_key.replace("*", "all").replace(f"{sep}{sep}", sep).split(sep)
                key = "__nest__".join(keys)
                return getattr(self._extra, key)
            msg = f"Key: `{key}` not found in Sample {str(self)[:20]}. Try using sample[key] instead if key is an integer or contains special characters."
            raise KeyError(msg) from e

    def __setattr__(self, key: str, value: Any) -> None:
        """Set the value of the attribute with the specified key."""
        if self.__class__ == Sample and key == "items":
            super().__setattr__("_items", value)
        else:
            super().__setattr__(key, value)

    def __setitem__(self, key: str | int, value: Any) -> None:
        """Set the value of the attribute with the specified key.

        If the key is an integer and the Sample object wraps a list, the value is set at the specified index.
        If the key is a string and contains a separator ('.' or '/'), the value is set at the specified nested key.
        Otherwise, the value is set as an attribute of the Sample object.
        """
        if self.__class__ == Sample:
            if isinstance(key, int) and hasattr(self, "wrapped") and isinstance(self.wrapped, List | Dataset):
                self.wrapped[key] = value
            elif isinstance(key, int) and len(self) > key:
                msg = f"Index out of range: {key} (expected 0-{len(self) - 1})"
                raise IndexError(msg)
            if isinstance(key, int):
                msg = f"Sample object does not wrap a list but index was requested: {key}"
                raise TypeError(msg)
        if key in super().__iter__():
            setattr(self, key, value)
            return
        if any(c in key for c in "./*"):
            sep = "." if "." in key else "/"
            keys = key.replace("*", "").replace(f"{sep}{sep}", sep).split(sep)
            obj = self
            for k in keys[:-1]:
                if k:
                    k = "_items" if k == "items" else k
                    if not hasattr(obj, k):
                        setattr(obj, k, Sample())
                    obj = obj[k]
            key = keys[-1] if keys[-1] != "items" else "_items"
            if isinstance(obj, dict):
                obj[key] = value
            else:
                setattr(obj, key, value)
        else:
            key = "_items" if key == "items" else key
            setattr(self, key, value)

    def __post_init__(self) -> None:
        if self.__class__ == Sample:
            self._extra: BaseModel = create_model(
                "Sample",
                __doc__=self.__class__.__doc__,
                __config__=self.model_config,
                **{
                    k.replace(".", "__nest__").replace("*", "all"): Annotated[
                        list[type(v[0])] if isinstance(v, list) and len(v) > 0 else type(v),
                        Field(default_factory=lambda: v),
                    ]
                    for k, v in self.dump().items()
                    if not k.startswith("_")
                },
            )()
            self._extra.__getitem__ = self.__class__.__getitem__
            self._extra.__setitem__ = self.__class__.__setitem__
            for k, v in self.dump().items():
                if not k.startswith("_"):
                    setattr(self._extra, k, v)

    def __hash__(self) -> int:
        """Return a hash of the Sample instance."""

        def hash_helper(obj):
            if isinstance(obj, list | tuple):
                return hash(tuple(hash_helper(item) for item in obj))
            if isinstance(obj, dict):
                return hash(tuple((k, hash_helper(v)) for k, v in sorted(obj.items())))
            if isinstance(obj, Sample):
                return hash(tuple(hash_helper(v) for v in obj.dump().values()))

            return hash(obj)

        return hash_helper(self.dump())

    def __str__(self) -> str:
        """Return a string representation of the Sample instance."""
        try:
            unnested = {"_items"}
            for k, _ in [(k, v) for k, v in super().__dict__.items() if not k.startswith("_")]:
                if "." in k:
                    unnested.add(k.split(".")[0])
                elif "/" in k:
                    unnested.add(k.split("/")[0])
            return pretty_repr(self, max_length=10, max_string=50)
        except Exception:  # noqa
            traceback.print_exc()
            try:
                return f"{self.__class__.__name__}({self.dump()})"
            except Exception:
                return super().__str__()

    def __repr__(self) -> str:
        """Return a string representation of the Sample instance."""
        return str(self)

    def update(self, other: Union[dict, "Sample"]) -> None:
        """Update the Sample instance with the attributes of another Sample instance or dictionary."""
        for k, v in other.items():
            self[k] = v
        return self

    def __contains__(self, key: str) -> bool:
        """Check if the Sample instance contains the specified attribute."""
        return key in list(self.keys())

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value of the attribute with the specified key or the default value if it does not exist."""
        try:
            return self[key]
        except Exception:
            return default

    def pop(self, key: str, default: Any = None) -> Any:
        """Remove the attribute with the specified key and return its value or the default value if it does not exist."""
        try:
            value = self[key]
            delattr(self, key)
            return value
        except Exception:
            return default
        else:
            return value

    def _dump(
        self,
        exclude: set[str] | str | None = "None",
        as_field: str | None = None,
        mode: Literal["json", "python"] = "python",
        depth: Literal["recurse", "shallow"] = "recurse",
    ) -> Dict[str, Any] | Any:
        out = {}
        exclude = set() if exclude is None else exclude if isinstance(exclude, set) else {exclude}
        for k, v in self._iter():
            if as_field is not None and k == as_field:
                return v
            if exclude and "None" in exclude and v is None:
                continue
            if exclude and k in exclude:
                continue
            if isinstance(v, Sample):
                out[k] = v.dump(exclude=exclude, as_field=as_field, mode=mode, depth=depth) if depth == "recurse" else v
            elif (
                depth == "recurse" and isinstance(v, list | tuple | Dataset) and len(v) > 0 and isinstance(v[0], Sample)
            ):
                out[k] = [item.dump(exclude, as_field, depth=depth, mode=mode) for item in v]
            else:
                out[k] = v
        return {k: v for k, v in out.items() if v is not None or "None" not in exclude}

    def dump(
        self,
        exclude: set[str] | str | None = "None",
        as_field: str | None = None,
        depth: Literal["recurse", "shallow"] = "recurse",
        mode: Literal["json", "python"] = "python",
    ) -> Dict[str, Any] | Any:
        """Dump the Sample instance to a dictionary or value at a specific field if present.

        Args:
            exclude (set[str], optional): Attributes to exclude. Defaults to "None". Indicating to exclude None values.
            as_field (str, optional): The attribute to return as a field. Defaults to None.
            mode (Literal["recurse", "shallow"], optional): Whether to dump nested Sample instances. Defaults to "recurse".

        Returns:
            Dict[str, Any]: Dictionary representation of the Sample object.
        """
        return self._dump(exclude=exclude, as_field=as_field, mode=mode, depth=depth)

    def values(self) -> Iterable[Any]:
        ignore = set()
        for k, _ in super().__iter__():
            if "." in k:
                ignore.add(k.split(".")[0])
            elif "/" in k:
                ignore.add(k.split("/")[0])
        for k, v in super().__iter__():
            if k not in ignore:
                yield v

    def _iter(self) -> Generator[tuple[str, Any], None, None]:
        if hasattr(self, "__pydantic_extra__"):
            yield from super().__iter__()
        else:
            yield from [(k, v) for k, v in super().__dict__.items() if not k.startswith("_")]

    def keys(self) -> Iterable[str]:
        ignore = set()
        for k, _ in self._iter():
            if "." in k:
                ignore.add(k.split(".")[0])
            elif "/" in k:
                ignore.add(k.split("/")[0])

        for k, _ in self._iter():
            if k not in ignore:
                yield k

    def items(self) -> Iterable[ItemsView]:
        ignore = set()
        for k, _ in self._iter():
            if "." in k:
                ignore.add(k.split(".")[0])
            elif "/" in k:
                ignore.add(k.split("/")[0])
        for k, v in self._iter():
            if k not in ignore:
                yield k, v

    def dict(
        self, exclude: set[str] | None | str = "None", mode: Literal["json", "python"] = "python", depth: Literal["recurse", "shallow"] = "recurse",
    ) -> Dict[str, Any]:
        """Return a dictionary representation of the Sample instance.

        Args:
            exclude_none (bool, optional): Whether to exclude None values. Defaults to True.
            exclude (set[str], optional): Set of attribute names to exclude. Defaults to None.
            mode (Literal["recurse", "shallow"], optional): Whether to dump nested Sample instances. Defaults to "recurse".

        Returns:
            Dict[str, Any]: Dictionary representation of the Sample object.
        """
        exclude = exclude or set()
        if depth == "shallow":
            return {
                k: v
                for k, v in self
                if k not in exclude and not k.startswith("_") and (v is not None or "None" not in exclude)
            }
        return self.dump(exclude=exclude, mode=mode, depth=depth)

    @classmethod
    def unflatten(cls, one_d_array_or_dict, schema=None) -> "Sample":
        """Unflatten a one-dimensional array or dictionary into a Sample instance.

        Args:
            one_d_array_or_dict: A one-dimensional array, dictionary, or tensor to unflatten.
            schema: A dictionary representing the JSON schema. Defaults to using the class's schema.

        Returns:
            Sample: The unflattened Sample instance.

        Examples:
            >>> sample = Sample(x=1, y=2, z={"a": 3, "b": 4}, extra_field=5)
            >>> flat_dict = sample.flatten(to="dict")
            >>> print(flat_dict)
            {'x': 1, 'y': 2, 'z': {'a': 3, 'b': 4}, 'extra_field': 5}
            >>> Sample.unflatten(flat_dict, sample.schema())
            Sample(x=1, y=2, z={'a': 3, 'b': 4}, extra_field=5)
        """
        if schema is None:
            try:
                try:
                    schema = cls().schema()
                except (AttributeError, KeyError, TypeError):
                    log.debug(f"Error getting schema for {cls}")
                    schema = schema_utils.resolve_refs(cls.model_json_schema())
                schema = schema_utils.simplify(schema, {}, cls.model_config.get("title", cls.__name__), include="simple")
            except Exception:
                global logged_recurse
                if not logged_recurse:
                    log.warning(f"Error unflattening schema: {schema}")
                    logged_recurse = True
                schema = schema_utils.resolve_refs(cls.model_json_schema())


        if not schema.get("properties"):
            kwargs = one_d_array_or_dict if isinstance(one_d_array_or_dict, dict) else {"items": one_d_array_or_dict}
            return cls(**kwargs)
        return cls(**schema_utils.unflatten_from_schema(one_d_array_or_dict, schema, cls))

    def flatten(  # noqa
        self,
        to: Literal[
            "list",
            "lists",
            "dict",
            "dicts",
            "np",
            "numpy",
            "pt",
            "torch",
            "pytorch",
            "sample",
            "samples",
        ] = "list",
        include: str | List[str] | None = None,
        exclude: str | set[str] | None = None,
        non_numerical: Literal["ignore", "forbid", "allow"] = "allow",
        sep: str = ".",
    ) -> Union[Dict[str, Any], np.ndarray, "torch.Tensor", List, Any]:
        """Flatten the Sample instance into a strictly one-dimensional or two-dimensional structure.

        **Note** A dimension is defined as a primitive or whatever is returned by the `include` keys.

        For nested lists use the '*' wildcard to select all elements.
        Use plural output types to return a list of lists or dictionaries.

        `include` can be any nested key however the output will be undefined if it exists in multiple places.
        Its order will be preserved along the second dimension.

        Example:
        - "a.b.*.c" will select all 'c' keys of dicts in the list at 'a.b'.
        - "a.*.b" will select all 'b' keys of dicts in the list at 'a'.
        - "a.b" will select all 'b' keys of any dict at 'a'.

        **Caution** If Both "c.a.b" and "d.a.b" exist, the selection "a.b" will be ambiguous.

        Integer indices are not currently supported although that may change in the future.

        Args:
            to : str, optional (default="list")

                Specifies the type of the return value if `include` is not provied or the second dimension if `include` is provided.
                Options are:
                - "list(s)": Returns a single flat list (list of lists).
                - "dict(s)": Returns a flattened dictionary (list of flatened dictionaries).
                - "np", "numpy": Returns a numpy array with non-numerical values excluded.
                - "pt, "pytorch", "torch": Returns a PyTorch tensor with non-numerical values excluded.

            non_numerical : str, optional (default="ignore")
                Determines how non-numerical values are handled. Options are:
                - "ignore": Non-numerical values are excluded from the output.
                - "forbid": Raises a ValueError if non-numerical values are encountered.
                - "allow": Includes non-numerical values in the output.

            exclude : set[str], optional (default=None)
                Set of keys to ignore during flattening.
            sep : str, optional (default=".")

            Separator used for nested keys in the flattened output.

        include : str | set[str] | List[str], optional (default=None)

            Specifies which keys to include in the output. Can be any nested key.

        Returns:
        Dict[str, Any] | np.ndarray | torch.Tensor | List
            The one or two-dimensional flattened output.

        Examples:
            >>> sample = Sample(a=1, b={"c": 2, "d": [3, 4]}, e=Sample(f=5))
            >>> sample.flatten()
            [1, 2, 3, 4, 5]
            >>> sample.flatten(to="dict")
            {'a': 1, 'b.c': 2, 'b.d.0': 3, 'b.d.1': 4, 'e.f': 5}
            >>> sample.flatten(ignore={"b"})
            [1, 5]
        """
        has_include = include is not None and len(include) > 0
        include = [] if include is None else [include] if isinstance(include, str) else include
        exclude = {} if exclude is None else {exclude} if isinstance(exclude, str) else exclude

        full_includes = {k: v for k, v in full_paths(self, include, show=False).items() if k in include}
        if not full_includes:
            full_includes = {v: v for k, v in describe_keys(self, show=False).items() if v in include}
        # Get the full paths of the selected keys. e.g. c-> a.b.*.c
        if not has_include and not exclude:
            full_excludes = []
        elif has_include:
            full_excludes = set(describe_keys(self).values()) - (
                set(full_includes.values()) | set(describe_keys(self, include).keys())
            )
        else:
            full_excludes = set(full_paths(self, exclude, show=False).values())

        for ex in full_excludes.copy():
            if any(ex.startswith(inc) for inc in full_includes.values()):
                full_excludes.remove(ex)
            if any(inc.startswith(ex) for inc in full_includes.values()):
                full_excludes.remove(ex)

        if to in ["numpy", "np", "torch", "pt"] and non_numerical != "forbid":
            non_numerical = "ignore"

        flattened_keys, flattened = iter_utils.flatten_recursive(
            self,
            exclude=full_excludes,
            non_numerical=non_numerical,
            sep=sep,
            include=include.copy(),
        )


        if not has_include and to == "list":
            return flattened
        if not has_include or to in ["dict", "sample"]:
            zipped = zip(flattened_keys, flattened, strict=False)
            if to == "sample":
                return Sample(**dict(zipped))
            if to == "dict":
                return dict(zipped)
            if to in ["np", "numpy"]:
                return np.array(flattened, dtype=object)
            if to in ["pt", "torch", "pytorch"]:
                torch = smart_import("torch")
                return torch.tensor(flattened, dtype=torch.float32)

            return flattened
        result = []
        current_group = {k: [] for k in include}
        ninclude_processed = dict.fromkeys(include, 0)
        flattened_keys = [iter_utils.replace_ints_with_wildcard(k, sep=sep) for k in flattened_keys]
        for flattened_key, value in zip(flattened_keys, flattened, strict=False):
            for selected_key, full_selected_key in full_includes.items():
                # e.g.: a.b.*.c was selected and a.b.0.c.d should be flattened to the c part of a row
                if full_selected_key in flattened_key and not any(
                    ignore_key == flattened_key for ignore_key in full_excludes
                ):
                    current_group[selected_key].append(value)
                    ninclude_processed[selected_key] += 1

            # All keys have been processed, add a new row.
            if all(
                num_processed == ninclude_processed[next(iter(include))]
                for num_processed in ninclude_processed.values()
            ) and all(len(processed_items) > 0 for processed_items in current_group.values()):
                # Ensure that we limit to two dimensions.
                current_group = {k: v[0] if len(v) == 1 else v for k, v in current_group.items() if k not in exclude}
                if to in ["dicts", "samples"]:
                    # Short circuit to avoid unnecessary processing.
                    result.append(Sample(**current_group) if to == "samples" else current_group)
                elif to in ["list"]:
                    flat_key, flattened = iter_utils.flatten_recursive(
                        current_group,
                        non_numerical=non_numerical,
                        sep=sep,
                        include=include,
                    )

                    result.extend(flattened)
                else:
                    flat_key, flattened = iter_utils.flatten_recursive(
                        current_group,
                        non_numerical=non_numerical,
                        sep=sep,
                    )
                    match to:
                        case "dicts":
                            result.append(dict(zip(flat_key, flattened, strict=False)))
                        case "samples":
                            result.append(Sample(**dict(zip(flat_key, flattened, strict=False))))
                        case _:
                            result.append(flattened)

            if all(
                num_processed == ninclude_processed[next(iter(include))]
                for num_processed in ninclude_processed.values()
            ):
                current_group = {k: [] for k in include}
                ninclude_processed = dict.fromkeys(include, 0)
        flattened = list(result.values()) if to in ["dicts", "samples"] and not isinstance(result, list) else result
        if to == "np":
            return np.array(flattened, dtype=float)
        if to == "pt":
            torch = smart_import("torch")
            return torch.tensor(flattened, dtype=float)
        return flattened


    def schema(self, include: Literal["all", "descriptions", "info", "simple", "tensor"] = "info") -> Dict:
        """Returns a simplified json schema.

        Args:
            include ("all", "descriptions", "info", "simple", optional): The level of detail to include in the schema.
                Defaults to "info".
                for "all", send the full pydantic schema.
                for "descriptions", send the simplified schema with descriptions.
                for "info", send the simplified schema with model info only.
                for "simple", send the simplified schema which has:
                    - references resolved
                    - descriptions removed
                    - additionalProperties removed
                    - items removed
                    - allOf removed
                    - anyOf resolved

        Returns:
            dict: A simplified JSON schema.
        """
        schema = self._extra.model_json_schema() if hasattr(self, "_extra") else self.model_json_schema()
        if include == "all":
            return schema

        schema = schema_utils.resolve_refs(schema)
        try:
            schema_utils.simplify(schema, self, include=include, target_model=self.__class__)
        except Exception:
            global logged_recurse
            if not logged_recurse:
                log.warning(f"Error simplifying schema for {self.keys()}")
                log.warning(f"Schema: {schema}")
                log.warning(f"Model: {self.keys()}")
                logged_recurse = True
        return schema



    def to(self, container: Any, **kwargs) -> Any:
        """Convert the Sample instance to a different container type.

        Args:
            container (Any): The container type, class, or callable to convert to.
            **kwargs: Additional keyword arguments.

        Args:
            container (Any): The container type, class, or callable to convert to.

            If a string, convert to one of:
            -'dict', 'list', 'np', 'pt' (pytorch), 'space' (gym.space),
            -'schema', 'json', 'hf' (datasets.Dataset)

            If a class, of type Sample, convert to that class.
            If a callable, convert to the output of the callable.

        Returns:
            Any: The converted container.

        Examples:
        >>> sample = Sample(x=1, y=2, z={"a": 3, "b": 4}, extra_field=5)
        >>> sample.to("features")
        {'x': Value(dtype='float32', id=None), 'y': Value(dtype='float32', id=None), 'z': {'a': Value(dtype='float32', id=None), 'b': Value(dtype='float32', id=None)}, 'extra_field': Value(dtype='float32', id=None)}
        """
        if isinstance(container, type) and issubclass(container, Sample):
            return container.unflatten(self.flatten())

        if container == "dict":
            return self.dump()
        if container == "list":
            return self.tolist()
        if container in ["np", "numpy"]:
            return self.numpy()
        if container in ["pt", "torch", "pytorch"]:
            return self.torch()
        if container == "space":
            return self.space()
        if container == "schema":
            return self.schema()
        if container == "json":
            return self.model_dump_json()
        if container in ["hf", "huggingface", "dataset", "datasets"]:
            return self.dataset()
        if container == "features":
            return Features(self.infer_features_dict())
        if container == "sample":
            return self.flatten(to="sample", **kwargs)
        try:
            log.debug(f"No matching container found for {type(container) if not isinstance(container, str) else '`' + container + '`'}. Attempting nested conversion.")
            for k, v in super().__iter__():
                if isinstance(v, Sample):
                    self[k] = v.to(container, **kwargs)
        except Exception as e:  # noqa
            try:
                return container(self)
            except Exception as e1:  # noqa
                try:
                    return container(self.dump(), **kwargs)
                except Exception as e2:  # noqa
                    msg = f"Unsupported container type: {type(container)}"
                    raise ValueError(msg) from e
        return self

    @classmethod
    def default_value(cls) -> "Sample":
        """Get the default value for the Sample instance.

        Returns:
            Sample: The default value for the Sample instance.
        """
        return cls()

    @classmethod
    def space_for(
        cls,
        value: Any,
        max_text_length: int = 1000,
        info: Dict[str, Any] | None = None,
    ) -> spaces.Space:
        """Default Gym space generation for a given value.

        Only used for subclasses that do not override the space method.
        """
        return space_for(value, max_text_length=max_text_length, info=info)

    @classmethod
    def from_space(cls, space: spaces.Space) -> "Sample":
        """Generate a Sample instance from a Gym space."""
        sampled = space.sample()
        if isinstance(sampled, dict):
            return cls(**sampled)
        if isinstance(sampled, np.ndarray | "torch.Tensor" | list | tuple):
            sampled = np.asarray(sampled)
            if len(sampled.shape) > 0 and isinstance(sampled[0], dict | Sample):
                return cls.unpack_from(sampled)
        return cls(sampled)

    @classmethod
    def unpack_from(
        cls,
        samples: List[Union["Sample", Dict]],
        padding: Literal["truncate", "longest"] = "longest",
        pad_value: Any = None,
    ) -> "Sample":
        """Pack a list of samples or dicts into a single sample with lists of samples for attributes.

        [Sample(a=1, b=4),  ->  Sample(a=[1, 2, 3],
         Sample(a=2, b=5),             b=[4, 5, 6])
         Sample(a=3, b=6)]

        This is equivalent to a zip operation on a list of dicts or transpose on the first two dimensions.

        Args:
            samples: List of Sample objects or dictionaries to pack.
            padding: Strategy for handling samples of different lengths.
                     "truncate" will truncate to the shortest sample,
                     "longest" will pad shorter samples to match the longest.
            pad_value: Value to use for padding when padding="longest". If None, uses cls.default_value().

        Returns:
            A new Sample object with packed attributes.

        Raises:
            ValueError: If the input list is empty.
            TypeError: If the input list contains items that are neither Sample nor dict.
        """
        if not samples:
            msg = "Cannot pack an empty list of samples"
            raise ValueError(msg)

        if isinstance(samples[0], dict):
            attributes = list(samples[0].keys())
        elif isinstance(samples[0], Sample):
            attributes = list(samples[0].dump().keys())
        elif samples[0] is None:
            msg = "Cannot pack a list containing None"
            raise ValueError(msg)
        else:
            msg = f"Cannot determine attributes from the first sample: {samples[0]}"
            raise TypeError(msg)

        pad_value = pad_value if pad_value is not None else cls.default_value()
        if padding == "truncate":
            attributes = [attr for attr in attributes if all(attr in sample for sample in samples)]

        return cls(
            **{
                attr: [
                    sample.get(attr, pad_value) if isinstance(sample, dict) else getattr(sample, attr, pad_value)
                    for sample in samples
                ]
                for attr in attributes
            },
        )

    def pack(
        self,
        to: Literal["dicts", "samples"] = "samples",
        padding: Literal["truncate", "longest"] = "truncate",
        pad_value: Any = None,
    ) -> list["Sample"] | List[Dict[str, Any]]:
        """Unpack the packed Sample object into a list of Sample objects or dictionaries.

        Sample(a=[1, 3, 5],   ->    [Sample(a=1, b=2),
               b=[2, 4, 6]           Sample(a=3, b=4),
                                     Sample(a=5, b=6)]
        )

        This is the inverse of the unpack method (analagous to a permute operation on the first two dimensions).

        Args:
            to: Specifies the output type, either "samples" or "dicts".
            padding: Strategy for handling attributes of different lengths.
                     "truncate" will truncate to the shortest attribute,
                     "longest" will pad shorter attributes to match the longest.
            pad_value: Value to use for padding when padding="longest". If None, uses self.default_value().

        Returns:
            A list of Sample objects or dictionaries, depending on the 'to' parameter.

        Example:
        >>> Sample(a=[1, 3, 5], b=[2, 4, 6]).pack()
        [Sample(a=1, b=2), Sample(a=3, b=4), Sample(a=5, b=6)]
        """
        if not any(isinstance(v, list) for k, v in self):
            return []

        pad_value = pad_value if pad_value is not None else self.default_value()
        attributes = list(self.keys())

        if padding == "truncate":
            min_length = min(len(v) for v in self.values() if isinstance(v, list))
            data = {k: v[:min_length] if isinstance(v, list) else v for k, v in self.items()}
        else:
            data = self
        max_length = max(len(v) if isinstance(v, list) else 1 for v in data.values())

        unzipped = zip_longest(
            *[data[attr] if isinstance(data[attr], list) else [data[attr]] * max_length for attr in attributes],
            fillvalue=pad_value,
        )
        mapper = (lambda items: self.__class__(**dict(items))) if to == "samples" else dict
        return [mapper(zip(attributes, values, strict=False)) for values in unzipped]

    def unpack(self, to: Literal["dicts", "samples", "lists"] = "samples") -> List[Union["Sample", Dict]]:
        """Unpack the Sample object into a list of Sample objects or dictionaries.

        Example:
            Sample(steps=[               [
                Sample(a=1, b=1), ->       [Sample(a=1), Sample(a=2)],
                Sample(a=2, b=2),          [Sample(b=1), Sample(b=2)],
            ])                            ]
            ).
        """
        return [[x.dump() if to == "dicts" else x for x in samples] for _, samples in self]

    @classmethod
    def pack_from(
        cls,
        *args,
        packed_field: str = "steps",
        padding: Literal["truncate", "longest"] = "truncate",
        pad_value: Any | None = None,
    ) -> "Sample":
        """Pack an iterable of Sample objects or dictionaries into a single Sample object with a single list attribute of Samples.

        [Sample(a=1)], Sample(a=2)], -> Sample(steps=[Sample(a=1,b=1),
        [Sample(b=1), Sample(b=2)]                    Sample(a=2,b=2)])

        This is equivalent to a zip operation on the list of samples (or transpose on Row, Col dimensions where
        Args:
            args: Iterable of Sample objects or dictionaries to pack.
            packed_field: The attribute name to pack the samples into.
            padding: Strategy for handling samples of different lengths.
                     "truncate" will truncate to the shortest sample,
                     "longest" will pad shorter samples to match the longest.
            pad_value: Value to use for padding when padding="longest". If None, uses cls.default_value().

        Returns:
            A new Sample object with a single list attribute containing the packed samples.


        """
        if not args:
            msg = "Cannot pack an empty list of samples"
            raise ValueError(msg)
        if len(args) == 1 and isinstance(args[0], list | tuple):
            args = args[0]
        if not all(isinstance(arg, Sample | dict) for arg in args):
            msg = "All arguments must be Sample objects or dictionaries"
            raise TypeError(msg)

        if padding == "longest":
            return cls(**{packed_field: list(zip_longest(*args, fillvalue=pad_value))})
        return cls(**{packed_field: list(zip(*args, strict=False))})

    def __unpack__(self):
        return self.unpack("dicts")

    @classmethod
    def default_space(cls) -> spaces.Dict:
        """Return the Gym space for the Sample class based on its class attributes."""
        return cls().space()

    @classmethod
    def default_sample(cls) -> Union["Sample", Dict[str, Any]]:
        """Generate a default Sample instance from its class attributes. Useful for padding.

        This is the "no-op" instance and should be overriden as needed.
        """
        return cls()

    def model_info(self) -> Dict[str, Any]:
        """Get the model information.

        This includes various metadata such as shape, bounds, and other information.
        """
        out = {}
        for key, value in self._iter():
            info = self.field_info(key) if not isinstance(value, Sample) else value.model_info()
            if info:
                out[key] = info
        return out

    def field_info(self, key: str) -> Dict[str, Any]:
        """Get the extra json values set from a FieldInfo for a given attribute key.

        This can include bounds, shape, and other information.
        """
        info = {}
        if self.model_extra and key in self.model_extra:
            info = FieldInfo(annotation=self.model_extra[key]).json_schema_extra or {}
        if key in self.model_fields:
            info = self.model_fields[key].json_schema_extra or {}
        return info.get("_info", {})

    def set_field_info(self, field_key, info_key, value) -> None:
        """Add or set a field info value to the Sample instance."""
        if self.model_extra and field_key in self.model_extra:
            info = FieldInfo(annotation=self.model_extra[field_key]).json_schema_extra or {}
            info.update({info_key: value})
            self.model_extra[field_key] = info
        elif field_key in self.model_fields:
            info = self.model_fields[field_key].json_schema_extra or {}
            info.update({info_key: value})
        else:
            msg = f"Field {field_key} not found in model fields or extra."
            raise ValueError(msg)

    def space(self) -> spaces.Dict:
        """Return the corresponding Gym space for the Sample instance based on its instance attributes.

        Omits None values.

        Override this method in subclasses to customize the space generation.
        """
        space_dict = {}
        for key, value in self.model_dump(exclude_none=True).items():
            info = self.field_info(key)
            space_dict[key] = self.space_for(value, info=info)
        return spaces.Dict(space_dict)

    def random_sample(self) -> "Sample":
        """Generate a random Sample instance based on its instance attributes. Omits None values.

        Override this method in subclasses to customize the sample generation.
        """
        return self.__class__.model_validate(self.space().sample())

    @cached_property
    def _numpy(self) -> np.ndarray:
        """Convert the Sample instance to a numpy array."""
        return self.flatten("np").astype(float)
        return self.flatten("np").astype(float)

    @cached_property
    def _tolist(self) -> list:
        """Convert the Sample instance to a list."""
        return self.flatten("list")

    @cached_property
    def _torch(self) -> "torch.Tensor":
        """Convert the Sample instance to a PyTorch tensor."""
        return self.flatten("pt")

    @cached_property
    def _json(self) -> str:
        """Convert the Sample instance to a JSON string."""
        return self.model_dump_json()

    def numpy(self) -> np.ndarray:
        """Return the numpy array representation of the Sample instance."""
        return self._numpy

    def tolist(self) -> list:
        """Return the list representation of the Sample instance."""
        return self._tolist

    def torch(self) -> "torch.Tensor":
        """Return the PyTorch tensor representation of the Sample instance."""
        return self._torch

    def json(self) -> str:
        """Return the JSON string representation of the Sample instance."""
        return self._json

    def features(self) -> Features:
        """Convert the Sample instance to a HuggingFace Features object."""
        return Features(self.infer_features_dict())

    def dataset(self) -> "Dataset":
        """Convert the Sample instance to a HuggingFace Dataset object."""
        smart_import("datasets")

        # HuggingFace datasets require pillow images to be converted to bytes.
        data = self.wrapped if hasattr(self, "wrapped") and self.wrapped is not None else self.dump(as_field="pil")
        if isinstance(data, list):
            return Dataset.from_list(data, features=self.features())
        if isinstance(data, dict):
            return Dataset.from_dict(data, features=self.features())
        if isinstance(data, Generator):
            return Dataset.from_generator(data, features=self.features())

        msg = f"Unsupported data type {type(data)} for conversion to Dataset."
        raise ValueError(msg)

    def describe(self) -> str:
        """Return a string description of the Sample instance."""
        return describe(self, compact=True, name=type(self).__name__)

    def copy(self) -> "Sample":
        """Return a deep copy of the Sample instance."""
        return smart_import("copy").deepcopy(self)


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)

    s = Sample(x=1, y=2, z={"a": 3, "b": 4, "c": np.array([1, 2, 3])}, extra_field=5)
