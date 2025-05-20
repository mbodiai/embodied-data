import traceback
from typing import Any, ClassVar, Dict, Generic, Literal, Type

from pydantic import PrivateAttr, TypeAdapter, model_serializer, model_validator
from pydantic.json_schema import DEFAULT_REF_TEMPLATE, GenerateJsonSchema, JsonSchemaMode
from typing_extensions import TypeVar

from embdata.sample import Sample

T = TypeVar("T", bound=Sample)

class List(Sample, Generic[T]):
    _list: list[T] = PrivateAttr(default_factory=list[T])
    _adapter: TypeAdapter[list[T]] = PrivateAttr(default_factory=lambda:TypeAdapter(list[T]))
    _item_class: ClassVar[Type[T]] = PrivateAttr(default_factory=lambda: Sample)

    def __init__(self, items: list[T]):
        super().__init__(_list=items)
        self._list = items


    @model_serializer(when_used="always")
    def serialize(self) -> list[dict]:
        """Serialize the list of MyModel objects to a list of dictionaries."""
        return [item.model_dump() for item in self._list]

    @model_validator(mode="before")
    @classmethod
    def validate(cls, value: list[T]) -> list[T]:
        return {"_list": value}

    @classmethod
    def model_validate(cls, obj: Any, strict: bool = False, from_attributes: bool = False, context: dict[str, Any] = ...) -> T:
        return cls(cls._adapter.validate_python(obj, strict=strict, from_attributes=from_attributes, context=context))

    @classmethod
    def model_json_schema(cls,
        by_alias: bool = True,
        ref_template: str = DEFAULT_REF_TEMPLATE,
        schema_generator: type[GenerateJsonSchema] = GenerateJsonSchema,
        mode: JsonSchemaMode = "validation",
    ) -> dict[str, Any]:
        return cls._adapter.json_schema(by_alias=by_alias, ref_template=ref_template, schema_generator=schema_generator, mode=mode)

    @classmethod
    def model_validate_json(cls, json_data: str | bytes | bytearray, *, strict: bool | None = None, context: Any | None = None) -> "List":
        if json_data.startswith("{") and json_data.endswith("}"):
            json_data = json_data[json_data.find("["):json_data.rfind("]")+1]
        return cls(cls._adapter.validate_json(json_data, strict=strict, context=context))

    def model_dump_json(self) -> str:
        return self._adapter.dump_json(self._list)

    def model_dump(self) -> list[T]:
        return self._adapter.dump_python(self._list)

    @classmethod
    def schema(cls, include: Literal["all", "descriptions", "info", "simple", "tensor"] = "info") -> Dict:
        """Returns an instance of the class with a default item of _item_class.

        Returns:
            An instance of the List class with one default item.
        """
        if not hasattr(cls, "_item_class"):
            msg = f"{cls.__name__} does not have _item_class defined."
            raise AttributeError(msg)
        required_fields = {}
        try:
            required_fields = {name: info.annotation() for name, info in cls._item_class.model_fields.items()}
        except Exception as e:
            msg = f"Nested Required Fields are not Supported for {cls._item_class.__name__}"
            traceback.print_exc()
            raise ValueError(msg) from e

        return Sample.schema(cls(items=[cls._item_class(**required_fields)]), include=include)

    @classmethod
    def __class_getitem__(cls, typevar_values: type[Any] | tuple[type[Any], ...]) -> "List":
        """This method allows for dynamic setting of the _item_class when using the
        List with a specific type, e.g., List[MyModel].
        """
        cls._item_class = typevar_values  # Set the item class to the provided type
        cls._adapter = TypeAdapter(list[typevar_values])  # Update the adapter for the type
        return cls


    def __repr__(self) -> str:
        return f"{self._list}"[:50]

    def __str__(self) -> str:
        return f"{self._list}"

    def __getitem__(self, index: int) -> T:
        return self._list[index]

    def __len__(self) -> int:
        return len(self._list)

    def __iter__(self):
        return iter(self._list)

    def append(self, item: T):
        self._list.append(item)
        return self

    def extend(self, items: list[T]):
        self._list.extend(items)
        return self

    def pop(self, index: int = -1) -> T:
        """Pop an item from the list at the given index and return it."""
        return self._list.pop(index)

    def remove(self, item: T):
        self._list.remove(item)
        return self

    def clear(self):
        self._list.clear()
        return self

    def reverse(self):
        self._list.reverse()
        return self

    def sort(self, key=None, reverse=False):
        self._list.sort(key=key, reverse=reverse)
        return self

    def insert(self, index: int, item: T):
        self._list.insert(index, item)
        return self

    def count(self, item: T) -> int:
        self._list.count(item)
        return self

    def index(self, item: T, start: int = 0, stop: int = 9223372036854775807) -> int:
        self._list.index(item, start=start, stop=stop)
        return self

    def copy(self) -> "List":
        self._list.copy()
        return self

    def __add__(self, other: "List") -> "List":
        self._list += other._list
        return self

    def __iadd__(self, other: "List") -> "List":
        self._list += other._list
        return self

    def __mul__(self, n: int) -> "List":
        self._list *= n
        return self

    def reversed(self) -> "List":
        self._list.reverse()
        return self

