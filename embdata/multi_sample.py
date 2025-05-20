from functools import wraps

from multidict import MultiDict
from pydantic import PrivateAttr
from typing_extensions import (
    Any,
    Generic,
    ItemsView,
    Iterable,
    Iterator,
    List,
    Self,
    Type,
    TypeAlias,
    TypeVar,
    overload,
)

from embdata.sample import Sample

T= TypeVar("T", bound=Sample)

class MultiSample(Sample, Generic[T]):
    """Model for a collection of values. Iterating over the collection will yield all values unlike Sample.

    Methods:
        add: Add a new value to the collection.
        getone: Get the first value for a key.s
        getall: Get all values for a key.
    """
    _store: MultiDict[str, T] = PrivateAttr(default_factory=MultiDict)
    _object_type: Type[Sample] = PrivateAttr(default_factory=lambda: Sample)

    def __class_getitem__(cls, item: Any) -> TypeAlias:
        cls._object_type = item
        return cls

    @wraps(MultiDict.add)
    def append(self, value: T) -> Self:
        self._store.add(value.name if hasattr(value, "name") else value.__class__.__name__, value)
        return self

    @wraps(MultiDict.add)
    def add(self, key: str, value: T) -> Self:
        self._store.add(key, value)
        return self

    def get_map(self) -> dict[str, T | List[T]]:
        return self._store

    @wraps(MultiDict.getone)
    def getone(self, key: str) -> T:
        return self._store.getone(key)

    @wraps(MultiDict.popone)
    def popone(self, key: str) -> T:
        return self._store.popone(key)

    @wraps(MultiDict.getall)
    def getall(self, key: str) -> List[T]:
        return self._store.getall(key)

    def __iter__(self) -> Iterator[T]:
        for value in self._store.values():
            if isinstance(value, list):
                yield from value
            else:
                yield value

    def __len__(self) -> int:
        return len(list(iter(self)))

    def __getattr__(self, item: str) -> T | List[T]:
        if any(a in item for a in ["_", "__", "pydantic", "getitem", "getattr", "setattr", "delattr"]):
            return super().__getattr__(item)
        return self._store.__getattribute__(item)

    def __contains__(self, item: str) -> bool:
        return item in self._store

    def __setitem__(self, key: str, value: T) -> None:
        self._store[key] = value


    def __getitem__(self, item: int | str) -> T | List[T]:
        if isinstance(item, int):
            return list(iter(self))[item]
        return self._store[item]

    def items(self) -> Iterable[ItemsView]:
        return self._store.items()

    def values(self) -> Iterable[T]:
        return self._store.values()

    def keys(self) -> Iterable[str]:
        return self._store.keys()

    @staticmethod
    def concat(collections: List["MultiSample[T]"]) -> "MultiSample[T]":
        if len(collections) in [0, 1]:
            return collections[0] if collections else MultiSample()
        result = collections[0]
        for collection in collections[1:]:
            for key, value in collection.items():
                if isinstance(value, list):
                    if key in result:
                        if isinstance(result[key], list):
                            result[key].extend(value)
                        else:
                            result[key] = [result[key], *value]
                    else:
                        result[key] = value
                elif key in result:
                    if isinstance(result[key], list):
                        result[key].append(value)
                    else:
                        result[key] = [result[key], value]
                else:
                    result[key] = value

        return result

    @staticmethod
    def from_list(data: List[T]) -> "MultiSample[T]":
        collection = Collection[T]()
        for item in data:
            key = item.name if hasattr(item, "name") else item.__class__.__name__
            collection.add(key, item)
        return collection

    @overload
    def __init__(self, data_list: list[T]) -> None:
        ...

    @overload
    def __init__(self, data_dict: dict[str, T]) -> None:
        ...

    @overload
    def __init__(self, **data: dict[str, T] | list[T]) -> None:
        ...

    def __init__(self, arg: list[T] | dict[str, T] | None = None, **data: dict[str, T]) -> None:
        """Initialize the MultiSample object with the given data.

        The data can be a list of values, a dictionary of values, or keyword arguments.
        """
        arg = arg or {}
        if isinstance(arg, list):
            objects = {}
            for i, item in enumerate(arg):
                # Determine the key for the object
                key = item.name if hasattr(item, "name") else item.get("name", f"item_{i}")
                # Append or initialize the objects under the key
                if key in objects:
                    if isinstance(objects[key], list):
                        objects[key].append(item)
                    else:
                        objects[key] = [objects[key], item]
                else:
                    objects[key] = item  # Initialize with the first item
            arg = objects

        data.update(arg)
        ObjectType = self._object_type
        # Ensure that we correctly handle lists and dictionaries
        final_data = {}
        for k, v in data.items():
            if isinstance(v, ObjectType):
                final_data[k] = v
            elif isinstance(v, dict):
                final_data[k] = ObjectType(**v)
            elif isinstance(v, list):
                final_data[k] = [ObjectType(**item) if isinstance(item, dict) else ObjectType(item) for item in v]
            else:
                final_data[k] = ObjectType(v)

        super().__init__(**final_data)
        # # Add references to multidict
        for key, value in final_data.items():
            self.add(key, value)

    def __eq__(self, other: object) -> bool:
        return all(
            obj == other_obj or (obj is None and other_obj is None)
            for obj, other_obj in zip(self.values(), other.values(), strict=False)
        )


Collection: TypeAlias = MultiSample[T]
