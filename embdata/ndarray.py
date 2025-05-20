from __future__ import annotations

import pickle as pickle_pkg
import sys
import types
from dataclasses import dataclass
from functools import lru_cache, partial, singledispatch
from pathlib import Path
from types import EllipsisType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    Iterable,
    LiteralString,
    Self,
    TypeAlias,
    TypeVar,
    TypeVarTuple,
    Unpack,
    cast,
    get_args,
    overload,
    runtime_checkable,
)

import compress_pickle
import numpy as np
import numpy.typing as npt
from einops import rearrange, reduce
from numpy.lib.npyio import NpzFile
from packaging.version import Version
from pydantic import (
    BaseModel,
    DirectoryPath,
    FilePath,
    GetJsonSchemaHandler,
    PositiveInt,
    computed_field,
    validate_call,
)

try:
    from numpy.core._exceptions import UFuncTypeError
except Exception:
    try:
        from numpy._core._exceptions import UFuncTypeError
    except Exception:
        try:
            from numpy._core.exceptions import UFuncTypeError
        except Exception:
            UFuncTypeError = Exception

from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticCustomError, core_schema
from ruamel import yaml
from typing_extensions import (
    Annotated,
    List,
    Literal,
    Protocol,
    Sequence,
    SupportsIndex,
    Tuple,
    TypedDict,
    TypeVar,
    get_args,
)

from embdata import SequenceLike, display

SupportedDTypes = (
    type[np.generic]
    | type[np.number]
    | type[np.bool_]
    | type[np.int64]
    | type[np.dtypes.Int64DType]
    | type[np.uint64]
    | type[np.dtypes.UInt64DType]
    | type[np.float64]
    | type[np.timedelta64]
    | type[np.datetime64]
)


if TYPE_CHECKING:
    from numbers import Number

    from numpy.typing import NDArray
    from pydantic.json_schema import JsonSchemaValue
    from pydantic.types import DirectoryPath


class PydanticNumpyMultiArrayNumpyFileOnFilePathError(Exception):
    pass


# Type aliases for literals
sz   = Literal
idx  = Literal
digit = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# String-bound TypeVars
lT = TypeVar("lT", bound=LiteralString)
lU = TypeVar("lU", bound=LiteralString)
lV = TypeVar("lV", bound=LiteralString)
lW = TypeVar("lW", bound=LiteralString)
O = TypeVar("O") # noqa
U = TypeVar("U")
V = TypeVar("V")
T = TypeVar("T")
S = TypeVar("S")
R = TypeVar("R")

# Cache for shape and dtype
_shape_and_dtype_cache: dict[Any, Any] = {}

# Shape and dtype TypeVars
ShapeT = TypeVarTuple("ShapeT")
DTypeT = TypeVar("DTypeT", bound=np.dtype[Any])

# Covariant TypeVars
U_co = TypeVar("U_co", bound=Any, covariant=True)
T_co = TypeVar("T_co", bound=Any, covariant=True)
S_co = TypeVar("S_co", bound=Any, covariant=True)

# YAML configuration
yaml = yaml.YAML(typ="safe")

# Float type aliases
Float64: TypeAlias = np.float64
Float32: TypeAlias = np.float32
Float16: TypeAlias = np.float16
Float: TypeAlias = np.float64 | np.float32 | np.float16 | float

@dataclass(frozen=True)
class MultiArrayNumpyFile:
    path: FilePath
    key: str
    cached_load: bool = False

    def load(self) -> NDArray:
        """Load the NDArray stored in the given path within the given key.

        Returns:
        -------
        NDArray
        """
        loaded = _cached_np_array_load(self.path) if self.cached_load else np.load(self.path)
        try:
            return loaded[self.key]
        except IndexError as e:
            msg = f"The given path points to an uncompressed numpy file, which only has one array in it: {self.path}"
            raise AttributeError(msg) from e


@runtime_checkable
class slice(Protocol[U_co, T_co, S_co]):
    @property
    def start(self) -> U_co: ...
    @property
    def step(self) -> S_co: ...
    @property
    def stop(self) -> T_co: ...
    @overload
    def __new__(cls:slice[U, T, S], stop: T, /) -> slice[U_co, T_co, S_co]: ...
    @overload
    def __new__(cls:slice[U, T, S], start: U, stop: T, step: S = ..., /): ...
    def __eq__(self, value: object, /) -> bool: ...

    __hash__: ClassVar[None]  # type: ignore[assignment]

    def indices(self:slice[U, T, S], len: SupportsIndex, /) -> tuple[U, T, S]: ...

def validate_numpy_array_file(v: FilePath) -> NDArray:
    """Validate file path to numpy file by loading and return the respective numpy array."""
    result = np.load(v)

    if isinstance(result, NpzFile):
        files = result.files
        if len(files) > 1:
            msg = (
                f"The provided file path is a multi array NpzFile, which is not supported; "
                f"convert to single array NpzFiles.\n"
                f"Path to multi array file: {result}\n"
                f"Array keys: {', '.join(result.files)}\n"
                f"Use embdata.ndarray.{MultiArrayNumpyFile.__name__} instead of a PathLike alone"
            )
            raise PydanticNumpyMultiArrayNumpyFileOnFilePathError(msg)
        result = result[files[0]]

    return result


def validate_multi_array_numpy_file(v: MultiArrayNumpyFile) -> npt.NDArray:
    """Validation function for loading numpy array from a name mapping numpy file.

    Parameters
    ----------
    v: MultiArrayNumpyFile
        MultiArrayNumpyFile to load

    Returns:
    -------
    NDArray from MultiArrayNumpyFile
    """
    return v.load()


def np_general_all_close(arr_a: npt.NDArray, arr_b: npt.NDArray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    """Data type agnostic function to define if two numpy array have elements that are close.

    Parameters
    ----------
    arr_a: npt.NDArray
    arr_b: npt.NDArray
    rtol: float
        See np.allclose
    atol: float
        See np.allclose

    Returns:
    -------
    Bool
    """
    return _np_general_all_close(arr_a, arr_b, rtol, atol)


def shape_and_dtype(*params:tuple[Unpack[tuple[Unpack[ShapeT],DTypeT|Any|EllipsisType]]],cls:type[np.ndarray[Any,Any]]|None=None)->tuple[tuple[int, ...],type[Any]]:
    key = (params,cls)
    if key in _shape_and_dtype_cache:
        return _shape_and_dtype_cache[key]
    while hasattr(params, "__len__") and len(params) == 1:
        params = params[0]

    _shape = None
    _dtype = None
    _labels = None
    if params is None or params in ("*", Any, (Any,)):
        params = ("*",)
    if not isinstance(params, tuple):
        params = (params,)
    if len(params) == 1:
        if isinstance(params[0], type):
            _dtype = params[0]
    else:
        *_shape, _dtype = params
        _shape = tuple(s if s not in ("*", Any) else -1 for s in _shape)

    _labels = []
    if isinstance(_dtype, int) or _dtype == "*":
        _shape += (_dtype,)
        _dtype = Any
    _shape = _shape or ()
    for s in _shape:
        if isinstance(s, str):
            if s.isnumeric():
                _labels.append(int(s))
            elif s in ("*", Any):
                _labels.append(-1)
            elif "=" in s:
                s = s.split("=")[1]  # noqa: PLW2901
                if not s.isnumeric():
                    msg = f"Invalid shape parameter: {s}"
                    raise ValueError(msg)
                _labels.append(int(s))
            else:
                msg = f"Invalid shape parameter: {s}"
                raise ValueError(msg)
    if _dtype is int:
        _dtype= np.int64
    elif _dtype is float:
        _dtype = np.float64

    if _shape == ():
        _shape = None
    shape = _shape if isinstance(_shape,tuple|list) else (_shape,)
    shape = tuple(getattr(s,"__args__",[s])[0] for s in shape)
    _shape_and_dtype_cache[key] = shape, _dtype
    return shape, _dtype


if Version(np.version.version) < Version("1.25.0"):

    def _np_general_all_close(arr_a: npt.NDArray, arr_b: npt.NDArray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
        try:
            return np.allclose(arr_a, arr_b, rtol=rtol, atol=atol, equal_nan=True)
        except UFuncTypeError:
            return np.allclose(arr_a.astype(np.float64), arr_b.astype(np.float64), rtol=rtol, atol=atol, equal_nan=True)
        except TypeError:
            return bool(np.all(arr_a == arr_b))

else:
    from numpy.exceptions import DTypePromotionError

    def _np_general_all_close(arr_a: npt.NDArray, arr_b: npt.NDArray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
        try:
            return np.allclose(arr_a, arr_b, rtol=rtol, atol=atol, equal_nan=True)
        except UFuncTypeError:
            return np.allclose(arr_a.astype(np.float64), arr_b.astype(np.float64), rtol=rtol, atol=atol, equal_nan=True)
        except DTypePromotionError:
            return bool(np.all(arr_a == arr_b))


class NumpyModel(BaseModel):
    _dump_compression: ClassVar[str] = "lz4"
    _dump_numpy_savez_file_name: ClassVar[str] = "arrays.npz"
    _dump_non_array_file_stem: ClassVar[str] = "object_info"

    _directory_suffix: ClassVar[str] = ".pdnp"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseModel):
            return NotImplemented  # delegate to the other item in the comparison

        self_type = self.__pydantic_generic_metadata__["origin"] or self.__class__
        other_type = other.__pydantic_generic_metadata__["origin"] or other.__class__

        if not (
            self_type == other_type
            and getattr(self, "__pydantic_private__", None) == getattr(other, "__pydantic_private__", None)
            and self.__pydantic_extra__ == other.__pydantic_extra__
        ):
            return False

        if isinstance(other, NumpyModel):
            self_ndarray_field_to_array, self_other_field_to_value = self._dump_numpy_split_dict()
            other_ndarray_field_to_array, other_other_field_to_value = other._dump_numpy_split_dict()

            return self_other_field_to_value == other_other_field_to_value and _compare_np_array_dicts(
                self_ndarray_field_to_array,
                other_ndarray_field_to_array,
            )

        # Self is NumpyModel, other is not; likely unequal; checking anyway.
        return super().__eq__(other)

    @classmethod
    @validate_call
    def model_directory_path(cls, output_directory: DirectoryPath, object_id: str) -> DirectoryPath:
        return output_directory / f"{object_id}.{cls.__name__}{cls._directory_suffix}"

    @classmethod
    @validate_call
    def load(
        cls,
        output_directory: DirectoryPath,
        object_id: str,
        *,
        pre_load_modifier: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ):
        """Load NumpyModel instance.

        Parameters
        ----------
        output_directory: DirectoryPath
            The root directory where all model instances of interest are stored
        object_id: String
            The ID of the model instance
        pre_load_modifier: Callable[[dict[str, Any]], dict[str, Any]] | None
            Optional function that modifies the loaded arrays

        Returns:
        -------
        NumpyModel instance
        """
        object_directory_path = cls.model_directory_path(output_directory, object_id)

        npz_file = np.load(object_directory_path / cls._dump_numpy_savez_file_name)

        other_path: FilePath
        if (other_path := object_directory_path / cls._dump_compressed_pickle_file_name).exists():  # pyright: ignore
            other_field_to_value = compress_pickle.load(other_path)
        elif (other_path := object_directory_path / cls._dump_pickle_file_name).exists():  # pyright: ignore
            with Path(other_path).open("rb") as in_pickle:
                other_field_to_value = pickle_pkg.load(in_pickle) # pyright: ignore # noqa
        elif (other_path := object_directory_path / cls._dump_non_array_yaml_name).exists():  # pyright: ignore
            with Path(other_path).open() as in_yaml:
                other_field_to_value = yaml.load(in_yaml)
        else:
            other_field_to_value = {}

        field_to_value = {**npz_file, **other_field_to_value}
        if pre_load_modifier:
            field_to_value = pre_load_modifier(field_to_value)

        return cls(**field_to_value)

    @validate_call
    def dump(
        self,
        output_directory: Path,
        object_id: str,
        *,
        compress: bool = True,
        pickle: bool = False,
    ) -> DirectoryPath:
        assert "arbitrary_types_allowed" not in self.model_config or (
            self.model_config["arbitrary_types_allowed"] and pickle
        ), "Arbitrary types are only supported in pickle mode"

        dump_directory_path = self.model_directory_path(output_directory, object_id)
        dump_directory_path.mkdir(parents=True, exist_ok=True)

        ndarray_field_to_array, other_field_to_value = self._dump_numpy_split_dict()

        if ndarray_field_to_array:
            (np.savez_compressed if compress else np.savez)(
                dump_directory_path / self._dump_numpy_savez_file_name,
                **ndarray_field_to_array,
            )

        if other_field_to_value:
            if pickle:
                if compress:
                    compress_pickle.dump(
                        other_field_to_value,
                        dump_directory_path / self._dump_compressed_pickle_file_name,  # pyright: ignore
                        compression=self._dump_compression,
                    )
                else:
                    with open(dump_directory_path / self._dump_pickle_file_name, "wb") as out_pickle:  # pyright: ignore
                        pickle_pkg.dump(other_field_to_value, out_pickle)

            else:
                with open(dump_directory_path / self._dump_non_array_yaml_name, "w") as out_yaml:  # pyright: ignore
                    yaml.dump(other_field_to_value, out_yaml)

        return dump_directory_path

    def _dump_numpy_split_dict(self) -> tuple[dict, dict]:
        ndarray_field_to_array = {}
        other_field_to_value = {}

        for k, v in self.model_dump().items():
            if isinstance(v, np.ndarray):
                ndarray_field_to_array[k] = v
            elif v:
                other_field_to_value[k] = v

        return ndarray_field_to_array, other_field_to_value

    @classmethod  # type: ignore[misc]
    @computed_field(return_type=str)
    @property
    def _dump_compressed_pickle_file_name(cls) -> str:
        return f"{cls._dump_non_array_file_stem}.pickle.{cls._dump_compression}"

    @classmethod  # type: ignore[misc]
    @computed_field(return_type=str)
    @property
    def _dump_pickle_file_name(cls) -> str:
        return f"{cls._dump_non_array_file_stem}.pickle"

    @classmethod  # type: ignore[misc]
    @computed_field(return_type=str)
    @property
    def _dump_non_array_yaml_name(cls) -> str:
        return f"{cls._dump_non_array_file_stem}.yaml"


def model_agnostic_load(
    output_directory: DirectoryPath,
    object_id: str,
    models: Iterable[type[NumpyModel]],
    not_found_error: bool = False,
    **load_kwargs,
) -> NumpyModel | None:
    """Provided an Iterable containing possible models, and the directory where they have been dumped.

     Load the first
    instance of model that matches the provided object ID.

    Parameters
    ----------
    output_directory: DirectoryPath
        The root directory where all model instances of interest are stored
    object_id: String
        The ID of the model instance
    models: Iterable[type[NumpyModel]]
        All NumpyModel instances of interest, note that they should have differing names
    not_found_error: bool
        If True, throw error when the respective model instance was not found
    load_kwargs
        Key-word arguments to pass to the load function

    Returns:
    -------
    NumpyModel instance if found
    """
    for model in models:
        if model.model_directory_path(output_directory, object_id).exists():
            return model.load(output_directory, object_id, **load_kwargs)

    if not_found_error:
        msg = (
            f"Could not find NumpyModel with {object_id} in {output_directory}."
            f"Tried from following classes:\n{', '.join(model.__name__ for model in models)}"
        )
        raise FileNotFoundError(
            msg,
        )

    return None


@lru_cache
def _cached_np_array_load(path: FilePath):
    """Store the loaded numpy object within LRU cache in case we need it several times.

    Parameters
    ----------
    path: FilePath
        Path to the numpy file

    Returns:
    -------
    Same as np.load
    """
    return np.load(path)


def _compare_np_array_dicts(
    dict_a: dict[str, npt.NDArray],
    dict_b: dict[str, npt.NDArray],
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> bool:
    """Compare two dictionaries containing numpy arrays as values.

    Parameters:
    dict_a, dict_b: dictionaries to compare. They should have same keys.
    rtol, atol: relative and absolute tolerances for np.isclose()

    Returns:
    Boolean value for each key, True if corresponding arrays are close, else False.
    """
    keys1 = frozenset(dict_a.keys())
    keys2 = frozenset(dict_b.keys())

    if keys1 != keys2:
        return False

    for key in keys1:
        arr_a = dict_a[key]
        arr_b = dict_b[key]

        if arr_a.shape != arr_b.shape or not np_general_all_close(arr_a, arr_b, rtol, atol):
            return False

    return True


class NumpyDataDict(TypedDict):
    data: List
    data_type: SupportedDTypes | str
    shape: Tuple[int, ...]


if sys.version_info < (3, 11):

    def array_validator(array: np.ndarray, shape: Tuple[int, ...] | None, dtype: SupportedDTypes | None) -> npt.NDArray:
        if shape is not None:
            expected_ndim = len(shape)
            actual_ndim = array.ndim
            if actual_ndim != expected_ndim:
                details = f"Array has {actual_ndim} dimensions, expected {expected_ndim}"
                msg = "ShapeError"
                raise PydanticCustomError(msg, details)
            for i, (expected, actual) in enumerate(zip(shape, array.shape, strict=False)):
                if expected != -1 and expected is not None and expected != actual:
                    details = f"Dimension {i} has size {actual}, expected {expected}"
                    msg = "ShapeError"
                    raise PydanticCustomError(msg, details)

        if (
            dtype
            and array.dtype.type != dtype
            and issubclass(dtype, np.integer)
            and issubclass(array.dtype.type, np.floating)
        ):
            array = np.round(array).astype(dtype, copy=False)
        if dtype and issubclass(dtype, np.dtypes.UInt64DType | np.dtypes.Int64DType):
            dtype = np.int64
            array = array.astype(dtype, copy=True)
        return array
else:

    @singledispatch
    def array_validator(
        array: np.ndarray,
        shape: Tuple[int, ...] | None,
        dtype: SupportedDTypes | None,
        labels: List[str] | None,
    ) -> npt.NDArray:
        if (
            shape
            and hasattr(shape, "__len__")
            and shape
            and isinstance(shape[-1], type | np.dtype | str)
            and shape[-1] != "*"
        ):
            dtype = shape[-1]
            shape = shape[:-1]
        if hasattr(shape, "__len__") and shape:
            shape = tuple([s if isinstance(s, int) else -1 for s in shape])

        if shape is not None and shape != (-1,):
            expected_ndim = len(shape)
            actual_ndim = array.ndim
            if actual_ndim != expected_ndim:
                details = f"Array of shape {array.shape} has {actual_ndim} dimensions, expected shape {shape} with {expected_ndim}  dimensions"
                msg = "ShapeError"
                raise PydanticCustomError(msg, details)
            for i, (expected, actual) in enumerate(zip(shape, array.shape, strict=False)):
                if expected != -1 and expected is not None and expected != actual:
                    details = f"Dimension {i} has size {actual}, expected {expected}"
                    msg = "ShapeError"
                    raise PydanticCustomError(msg, details)
        dtype = type(dtype) if not isinstance(dtype, type) else dtype
        if dtype and array.dtype != dtype and issubclass(dtype, np.integer) and issubclass(array.dtype, np.floating):
            array = np.round(array).astype(dtype, copy=False)
        if dtype and issubclass(dtype, np.dtypes.UInt64DType | np.dtypes.Int64DType):
            dtype = np.int64
            array = array.astype(dtype, copy=True)
        return array

    @array_validator.register
    def list_tuple_validator(
        array: list | tuple,
        shape: Tuple[int, ...] | None,
        dtype: SupportedDTypes | None,
    ) -> npt.NDArray:
        return array_validator.dispatch(np.ndarray)(np.asarray(array), shape, dtype)

    @array_validator.register
    def dict_validator(
        array: dict,
        shape: Tuple[int, ...] | None,
        dtype: SupportedDTypes | None,
        labels: List[str] | None,
    ) -> npt.NDArray:
        array = np.array(array["data"]).astype(array["data_type"]).reshape(array["shape"])
        return array_validator.dispatch(np.ndarray)(array, shape, dtype, labels)


def create_array_validator(
    shape: Tuple[int, ...] | None,
    dtype: SupportedDTypes | None,
    labels: List[str] | None,
) -> Callable[[Any], npt.NDArray]:
    """Creates a validator function for NumPy arrays with a specified shape and data type."""
    return partial(array_validator, shape=shape, dtype=dtype, labels=labels)


@validate_call
def _deserialize_numpy_array_from_data_dict(data_dict: NumpyDataDict) -> np.ndarray:
    return np.array(data_dict["data"]).astype(data_dict["data_type"]).reshape(data_dict["shape"])


_common_numpy_array_validator = core_schema.union_schema(
    [
        core_schema.chain_schema(
            [
                core_schema.is_instance_schema(Path),
                core_schema.no_info_plain_validator_function(validate_numpy_array_file),
            ],
        ),
        core_schema.chain_schema(
            [
                core_schema.is_instance_schema(MultiArrayNumpyFile),
                core_schema.no_info_plain_validator_function(validate_multi_array_numpy_file),
            ],
        ),
        core_schema.is_instance_schema(np.ndarray),
        core_schema.chain_schema(
            [
                core_schema.is_instance_schema(Sequence),
                core_schema.no_info_plain_validator_function(lambda v: np.asarray(v)),
            ],
        ),
        core_schema.chain_schema(
            [
                core_schema.is_instance_schema(dict),
                core_schema.no_info_plain_validator_function(_deserialize_numpy_array_from_data_dict),
            ],
        ),
    ],
)


def get_numpy_json_schema(
    _field_core_schema: core_schema.CoreSchema,
    _handler: GetJsonSchemaHandler,
    shape: List[PositiveInt] | None = None,
    data_type: SupportedDTypes | None = None,
    labels: List[str] | None = None,
) -> JsonSchemaValue:
    """Generates a JSON schema for a NumPy array field within a Pydantic model.

    This function constructs a JSON schema definition compatible with Pydantic models
    that are intended to validate NumPy array inputs. It supports specifying the data type
    and dimensions of the NumPy array, which are used to construct a schema that ensures
    input data matches the expected structure and type.

    Parameters
    ----------
    _field_core_schema : core_schema.CoreSchema
        The core schema component of the Pydantic model, used for building basic schema structures.
    _handler : GetJsonSchemaHandler
        A handler function or object responsible for converting Python types to JSON schema components.
    shape : Optional[List[PositiveInt]], optional
        The expected shape of the NumPy array. If specified, the schema will enforce that the input
    data_type : Optional[SupportedDTypes], optional
        The expected data type of the NumPy array elements. If specified, the schema will enforce
        that the input array's data type is compatible with this. If `None`, any data type is allowed,
        by default None.

    Returns:
    -------
    JsonSchemaValue
        A dictionary representing the JSON schema for a NumPy array field within a Pydantic model.
        This schema includes details about the expected array dimensions and data type.
    """
    _handler(_common_numpy_array_validator)
    array_shape = shape if shape else "Any"


    if data_type:
        array_data_type = getattr(data_type,"__name__", type(data_type).__name__)
        item_schema = core_schema.list_schema(
            items_schema=core_schema.any_schema(metadata=f"Must be compatible with numpy.dtype: {array_data_type}"),
        )
    else:
        array_data_type = "Any"
        item_schema = core_schema.list_schema(items_schema=core_schema.any_schema())

    if shape:
        for dim in reversed(shape):
            if isinstance(dim, int):
                item_schema = core_schema.list_schema(items_schema=item_schema, min_length=dim, max_length=dim)
            else:
                item_schema = core_schema.list_schema(items_schema=item_schema)

    data_schema = item_schema

    return {
        "title": "ndarray",
        "type": f"np.ndarray[{array_shape}, np.dtype[{array_data_type}]]",
        "required": ["data_type", "data"],
        "properties": {
            "data_type": {"title": "dtype", "default": array_data_type, "type": "string"},
            "shape": {"title": "shape", "default": array_shape, "type": "array"},
            "data": data_schema,
        },
    }


def array_to_data_dict_serializer(array: npt.NDArray) -> NumpyDataDict:
    array = np.array(array)

    if issubclass(array.dtype.type, np.timedelta64) or issubclass(array.dtype.type, np.datetime64):
        data = array.astype(int).tolist()
    else:
        data = array.astype(float).tolist()
    dtype = str(array.dtype) if hasattr(array, "dtype") else "np.float64"
    return NumpyDataDict(data=data, data_type=dtype, shape=array.shape)


T = TypeVar("T", bound=int)
U = TypeVar("U", bound=int)
V = TypeVar("V", bound=int)
W = TypeVar("W")
X = TypeVar("X")
Y = TypeVar("Y")
Z = TypeVar("Z")
M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)
O = TypeVar("O", bound=int)
P = TypeVar("P", bound=int)

_Ts = TypeVarTuple("_Ts")
Ts = TypeVarTuple("Ts")
DT = TypeVar("DT")
ET = TypeVar("ET")
RedT = TypeVar("RedT")




def resolve_type_vars(item: types.GenericAlias, seen: set | None = None) -> Any:
    seen = seen or set()
    if item in seen:
        return item
    seen.add(item)

    if hasattr(item, "__args__") and item.__args__:
        return tuple(resolve_type_vars(arg, seen) for arg in item.__args__)
    if hasattr(item, "__iter__"):
        return tuple(resolve_type_vars(arg, seen) for arg in item)
    return 1 if isinstance(item, TypeVar) or item is Any else item

def get_dtype(item, dtype: DT) -> DT:
    if dtype is not None:
        return dtype
    if hasattr(item, "dtype"):
        return item.dtype
    if hasattr(item, "__len__") and len(item) > 0:
        if not hasattr(item, "__iter__"):
            return np.dtype(type(item))
        while hasattr(item, "__iter__"):
            item = item[0]
        return np.dtype(type(item))
    return np.dtype(np.float32)



def is_shape_like(item: Any) -> bool:
    """Only accepts tuples of integers."""
    if item is None:
        return False
    return isinstance(item, tuple) and all(isinstance(i, int) for i in item)



def first_exists(item: None | Iterable) -> Any:
    if item is None:
        return None
    return next(iter(item), None)


class ndarray(np.ndarray, Generic[*Ts, DT]): # noqa
    """A lightweight NumPy wrapper for type hinting array shapes, data types, and supporting Pydantic serialization.

    Simply pass the shape in as class args and optionally the data type as the last arg: `ndarray[224, 224, 3, np.uint8]`.
    At runtime, this class behaves identically to a numpy array with a more convenient constructor.

    ### Examples
    ```python
    from samples import ndarray, BatchDim as B, sz

    img: ndarray[640, 480, 3, np.uint8] = ndarray()  # Equivalent to np.zeros(...)
    transposed = img.T  # Intellisense shows `ndarray[480, 640, 3, np.uint8]`
    auto_hinted = ndarray[224, 224, 3, np.uint8]()  # Intellisense shows `ndarray[224, 224, 3, np.uint8]`
    ```
    ### Advanced usage
    ```python
     depth_map_batch: ndarray[64, 224, 224, np.uint16] = ndarray()
     def detect_6Dpose(any_depth_map: ndarray[B, "*", "*", np.int16]) -> ndarray[B, sz[6], np.float32]:...

     poses = detect_6Dpose(depth_map_batch)  # Intellisense shows `ndarray[64, 6, np.float32]`
     avg_pose = poses.mean(axis=0)  # Intellisense shows `ndarray[6, np.float32]`
    ```

    ### Features:
        - At runtime this class has identical behavior to a numpy array with a more convenient constructor.
        - Use the first class args for the shape and optionally the last arg for the data type: `ndarray[224, 224, 3, np.uint8]`.
        - Wildcard shape dimensions are supported: `must_be_2d: ndarray["*", "*", np.float16] = ndarray()`.
        - When used in Pydantic models, this class will automatically serialize and deserialize numpy arrays.
        - Supports Ellipses: `ndarray[3, ..., np.float16]` but unfortunately, `# type: ignore` must be used to suppress pyright.
        - Type hint shapes by specifying the dimensions in the subscript and optionally, the data type in the final subscript.
        - When present in pydantic models, this class will automatically serialize and deserialize numpy arrays.
    ```
    """

    if TYPE_CHECKING:
        data: np.ndarray | None = None
        dtype: DT | None = None
        labels: List[str] | None = None
    else:
        _shape: Tuple[*Ts] | None = None
        _data: np.ndarray | None = None
        _dtype: DT | None = None
        _labels: List[str] | None = None
    @overload
    def __new__(
        cls: type[ndarray[*Ts, DT]],
        data: list[list|int | float] | None = None,
        dtype: DT | None = None,
    ) -> ndarray[*Ts, DT]: ...
    @overload
    def __new__(
        cls: ndarray[*Ts, DT],
        shape: Tuple[*Ts] | None = None,
        dtype: DT | None = None,
        data: SequenceLike | None = None,
    ) -> ndarray[*Ts, DT]: ...

    @overload
    def __new__(
        cls: ndarray[*Ts, DT],
        data: ndarray[*Ts, ET] | np.ndarray,
        target_shape: Tuple[*Ts] | None = None,
        dtype: DT | None = None,
    ) -> ndarray[*Ts, DT]: ...

    def __new__(cls, *args, **kwargs) -> ndarray[*Ts, DT]:
        """Create a new ndarray instance with the given shape, data type, and data.

        If the target_shape does not match the data shape, the data will be reshaped to match the target_shape.
        """
        # First arg is data or shape only if it is a tuple of integers and cls._shape is not set
        shape = kwargs.get("shape", getattr(cls, "_shape", None))
        data = kwargs.get("data")
        if shape is None and is_shape_like(first_exists(args)):
            shape, *args = args
        if data is None and isinstance(first_exists(args), np.ndarray | ndarray | list):
            data, *args = args
        dtype = args[-1] if hasattr(args, "__len__") and len(args) > 0 else kwargs.get("dtype", getattr(cls, "_dtype", get_dtype(data, None)))
        if not (isinstance(dtype, type | np.dtype)):
            dtype = np.float64
        data = np.asarray(data, dtype=dtype)
        if data is None or (len(data.shape) == 0 and len(shape) > 0):
            data = np.zeros(shape, dtype=dtype)
        obj = np.asarray(data, dtype=dtype).view(cls)
        obj._data = data
        obj._shape = shape
        obj._dtype = dtype
        return obj

    @property
    def shape(self) -> Tuple[*Ts]:
        return cast(Tuple[*Ts], np.asarray(self.data).shape)

    @shape.setter
    def shape(self, value: Tuple[*Ts]) -> None:
        self._shape = value

    @overload
    def __init__(self, data: list[list | int | float] | None = None, dtype: DT | None = None): ...
    @overload
    def __init__(self, shape: Tuple[*Ts], dtype: DT, data: np.ndarray | None = None): ...
    @overload
    def __init__(self, shape: Tuple[*Ts], dtype: DT, data: list | None = None): ...
    @overload
    def __init__(self, shape: Tuple[*Ts], dtype: DT, data: dict | None = None): ...

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def __class_getitem__(cls, params=None):
        _shape = None
        _dtype = None
        _labels = None

        if params is None or params in ("*", Any, (Any,)):
            params = ("*",)
        if not isinstance(params, tuple):
            params = (params,)
        if len(params) == 1:
            if isinstance(params[0], type):
                _dtype = params[0]
            else:
                _shape = (params[0],)
        else:
            *_shape, _dtype = params
            _shape = tuple(s if s not in ("*", Any) else -1 for s in _shape)

        _labels = []
        if isinstance(_dtype, int) or _dtype == "*":
            _shape += (_dtype,)
            _dtype = Any
        _shape = _shape or ()
        for s in _shape:
            if isinstance(s, str):
                if s.isnumeric():
                    _labels.append(int(s))
                elif s in ("*", Any):
                    _labels.append(-1)
                elif "=" in s:
                    s = s.split("=")[1]
                    if not s.isnumeric():
                        msg = f"Invalid shape parameter: {s}"
                        raise ValueError(msg)
                    _labels.append(int(s))
                else:
                    msg = f"Invalid shape parameter: {s}"
                    raise ValueError(msg)
            elif isinstance(s, int):
                _labels.append(s)
            else:
                _labels.append(-1)
        if _dtype is int:
            _dtype = np.dtype(np.int32)
        elif _dtype is float:
            _dtype = np.dtype(np.float32)
        elif _dtype is not None and _dtype not in ("*", Any) and isinstance(_dtype, type):
            _dtype: DT = np.dtype(np.float32)
        if _shape == ():
            _shape = None

        # Create a new subclass with the specified shape and dtype
        new_cls_name = f"{cls.__name__}[{', '.join(map(str, params))}]"
        bases = (cls, np.ndarray)
        namespace = {
            "_shape": _shape,
            "_dtype": _dtype,
            "_labels": _labels,
            "__annotations__": {"_shape": _shape, "_dtype": _dtype, "_labels": _labels},
            "__get_pydantic_core_schema__": cls.__get_pydantic_core_schema__,
            "__get_pydantic_json_schema__": cls.__get_pydantic_json_schema__,
        }
        new_cls = types.new_class(new_cls_name, bases, {}, lambda ns: ns.update(namespace))
        new_cls.__module__ = cls.__module__
        new_cls._shape = resolve_type_vars(_shape)
        new_cls._dtype = _dtype
        new_cls._labels = _labels
        shape = _shape
        dtype = _dtype
        labels = _labels

        class ShapeDtypeData(cls, Generic[*_Ts, T]):
            model_config = {"arbitrary_types_allowed": True}
            _shape: Tuple[*_Ts] = shape
            _dtype: T = np.dtype(np.float32) if dtype and len(get_args(dtype)) > 1 else dtype
            _labels: List[str] = labels
            _data: list[Number]

        return Annotated[new_cls, np.ndarray | FilePath | MultiArrayNumpyFile, ShapeDtypeData]

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: Callable[[Any], core_schema.CoreSchema],
    ) -> core_schema.CoreSchema:
        shape = cls._shape
        dtype = cls._dtype

        np_array_validator = create_array_validator(shape, dtype, None)
        np_array_schema = core_schema.no_info_plain_validator_function(np_array_validator)

        return core_schema.json_or_python_schema(
            python_schema=core_schema.chain_schema(
                [
                    core_schema.union_schema(
                        [
                            core_schema.is_instance_schema(np.ndarray),
                            core_schema.is_instance_schema(list),
                            core_schema.is_instance_schema(tuple),
                            core_schema.is_instance_schema(dict),
                        ],
                    ),
                    _common_numpy_array_validator,
                    np_array_schema,
                ],
            ),
            json_schema=core_schema.chain_schema(
                [
                    core_schema.union_schema(
                        [
                            core_schema.list_schema(),
                            core_schema.dict_schema(),
                        ],
                    ),
                    np_array_schema,
                ],
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                array_to_data_dict_serializer,
                when_used="json-unless-none",
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        field_core_schema: core_schema.CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        return get_numpy_json_schema(
            field_core_schema, handler, getattr(cls, "_shape", None), getattr(cls, "_dtype", None),
        )

    def transpose(self: ndarray[U,T,*_Ts, DT], axes) -> ndarray[T,U,*_Ts, DT]:
        transposed = super().transpose(axes)
        return transposed.view(type(self))

    def rearrange(self, pattern: str):
        return rearrange(tensor=self._data, pattern=pattern)

    # TODO(zaid): Add support for labels
    @overload
    def permute(self: ndarray[T,U,*_Ts, DT], *axes: Tuple[idx[1], idx[0]]) -> ndarray[U,T,*_Ts, DT]: ...
    @overload
    def permute(self: ndarray[T,U,V,*_Ts, DT], *axes: Tuple[idx[0], idx[2], idx[1]]) -> ndarray[T,V,U,*_Ts, DT]: ...
    @overload
    def permute(self: ndarray[T,U,V,*_Ts, DT], *axes: Tuple[idx[2], idx[0], idx[1]]) -> ndarray[V,T,U,*_Ts, DT]: ...
    @overload
    def permute(self: ndarray[T,U,V,*_Ts, DT], *axes: Tuple[idx[1], idx[2], idx[0]]) -> ndarray[U,V,T,*_Ts, DT]: ...
    @overload
    def permute(self: ndarray[T,U,V,*_Ts, DT], *axes: Tuple[idx[2], idx[1], idx[0]]) -> ndarray[V,U,T,*_Ts, DT]: ...
    @overload
    def permute(self: ndarray[T,U,*_Ts, DT], axes: Tuple[idx[1], idx[0]]) -> ndarray[U,T,*_Ts, DT]: ...
    @overload
    def permute(self: ndarray[T,U,V,*_Ts, DT], axes: Tuple[idx[0], idx[2], idx[1]]) -> ndarray[T,V,U,*_Ts, DT]: ...
    @overload
    def permute(self: ndarray[T,U,V,*_Ts, DT], axes: Tuple[idx[2], idx[0], idx[1]]) -> ndarray[V,T,U,*_Ts, DT]: ...
    @overload
    def permute(self: ndarray[T,U,V,*_Ts, DT], axes: Tuple[idx[1], idx[2], idx[0]]) -> ndarray[U,V,T,*_Ts, DT]: ...
    @overload
    def permute(self: ndarray[T,U,V,*_Ts, DT], axes: Tuple[idx[2], idx[1], idx[0]]) -> ndarray[V,U,T,*_Ts, DT]: ...
    def permute(self, axes):
        """Permute the axes of the array (general axis manipulation)."""
        if len(axes) > len(self.shape):
            msg = "Axes length must be less than or equal to the number of dimensions"
            raise ValueError(msg)

        if any(a > 2 for a in axes):
            """Type hints are not supported for permutation axes greater than 2"""
            return rearrange(
                tensor=self,
                pattern=f"{' '.join(['...'] * (len(self.shape) - 1))} -> {' '.join(['...'] * (len(self.shape) - 1))}",
            )
        dims = ["a", "b", "c"][: len(axes)] + ["..."] if len(axes) > 3 else []
        permuted = [dims[a] for a in axes]
        return rearrange(tensor=self, pattern=f"{' '.join(dims)} -> {' '.join(permuted)}")


    @overload
    def __imatmul__(self: ndarray[U,T, DT], other: ndarray[T,V, DT]) -> ndarray[U,V, DT]: ...
    @overload
    def __imatmul__(self: ndarray[T,U, DT], other: ndarray[U,V, DT]) -> ndarray[T,V, DT]: ...
    def __imatmul__(self, other):
        """Perform in-place matrix multiplication (self @= other).

        Modifies self in-place if the underlying data is writable.

        Raises:
            ValueError: If the underlying array data is not writable.
        """
        if not self.data.flags.writeable:
            msg = "Cannot perform in-place matrix multiplication on a read-only array."
            raise ValueError(msg)

        # Ensure 'other' is treated as a NumPy array for matmul
        other_np = np.asarray(other)
        # Calculate result
        result_data = np.matmul(self.data, other_np)

        # Check if shapes match for in-place update
        if result_data.shape != self.shape:
             # If shapes don't match, NumPy's matmul might return a new array
             # that cannot be assigned back in-place. Reassign internal data.
             # Note: This deviates slightly from true in-place if shape changes,
             # but handles cases like multiplying by a broadcastable scalar/vector implicitly.
             # A stricter check might be needed depending on desired behavior for shape mismatches.
             self._data = result_data
             # Update internal shape if necessary (though __new__ might handle this better)
             self._shape = result_data.shape
        else:
             # Perform in-place update if shapes match
             np.matmul(self.data, other_np, out=self.data)

        # Update internal dtype if necessary (though usually stays the same)
        self._dtype = self.data.dtype
        return self

    @overload
    def __matmul__(self: ndarray[sz[1],U, DT], other: ndarray[U,*_Ts, DT]) -> ndarray[*_Ts, DT]: ...
    @overload
    def __matmul__(self: ndarray[*_Ts,T, DT], other: ndarray[T, sz[1], DT]) -> ndarray[*_Ts, DT]: ...
    @overload
    def __matmul__(self: ndarray[*_Ts,T, DT], other: ndarray[T,U, DT]) -> ndarray[*_Ts,U, DT]: ...
    @overload
    def __matmul__(self: ndarray[T, U, DT], other: ndarray[U, *_Ts, DT]) -> ndarray[T, *_Ts, DT]: ...
    def __matmul__(self, other):
        """Perform matrix multiplication (self @ other).

        Returns:
            A new ndarray instance containing the result.
        """
        # Ensure 'other' is treated as a NumPy array for matmul
        other_np = np.asarray(other)
        # Calculate result using the underlying .data attribute
        result_data = np.matmul(self.data, other_np)
        # Return a new instance of this class
        # Use result_data.dtype as the type might change (e.g., int @ float -> float)
        return self.__class__(data=result_data, dtype=result_data.dtype)

    @overload
    def __rmatmul__(self: ndarray[T,U, DT], other: ndarray[*_Ts,T, DT]) -> ndarray[*_Ts,U, DT]: ...
    @overload
    def __rmatmul__(self: ndarray[T, *_Ts, DT], other: ndarray[U, T, DT]) -> ndarray[U, *_Ts, DT]: ...
    def __rmatmul__(self, other):
        """Perform right matrix multiplication (other @ self).

        Returns:
            A new ndarray instance containing the result.
        """
        # Ensure 'other' is treated as a NumPy array for matmul
        other_np = np.asarray(other)
        # Calculate result using the underlying .data attribute
        result_data = np.matmul(other_np, self.data)
        # Return a new instance of this class
        # Use result_data.dtype as the type might change
        return self.__class__(data=result_data, dtype=result_data.dtype)

    @overload
    def reshape(
        self: ndarray[sz[2], sz[3], DT],
        shape: Tuple[sz[3], sz[2]],
    ) -> ndarray[sz[3], sz[2], DT]: ...

    @overload
    def reshape(
        self: ndarray[sz[2], sz[3], DT],
        shape: Tuple[sz[6]],
    ) -> ndarray[sz[6], DT]: ...

    @overload
    def reshape(
        self: ndarray[sz[2], sz[4], DT],
        shape: Tuple[sz[4], sz[2]],
    ) -> ndarray[sz[4], sz[2], DT]: ...

    @overload
    def reshape(
        self: ndarray[sz[2], sz[4], DT],
        shape: Tuple[sz[8]],
    ) -> ndarray[sz[8], DT]: ...

    @overload
    def reshape(
        self: ndarray[sz[3], sz[3], DT],
        shape: Tuple[sz[6]],
    ) -> ndarray[sz[6], DT]: ...

    @overload
    def reshape(
        self: ndarray[sz[3], sz[3], DT],
        shape: Tuple[sz[3], sz[3]],
    ) -> ndarray[sz[3], sz[3], DT]: ...

    @overload
    def reshape(
        self: ndarray[sz[4], sz[4], DT],
        shape: Tuple[sz[8]],
    ) -> ndarray[sz[8], DT]: ...

    @overload
    def reshape(
        self: ndarray[sz[4], sz[4], DT],
        shape: Tuple[sz[4], sz[4]],
    ) -> ndarray[sz[4], sz[4], DT]: ...

    @overload
    def reshape(
        self: ndarray[sz[5], sz[5], DT],
        shape: Tuple[sz[10]],
    ) -> ndarray[sz[10], DT]: ...

    @overload
    def reshape(
        self: ndarray[sz[5], sz[5], DT],
        shape: Tuple[sz[5], sz[5]],
    ) -> ndarray[sz[5], sz[5], DT]: ...

    def reshape(self, shape: Tuple[Any, ...]):
        """Reshape the array to the specified shape.

        Args:
            shape (Tuple[int, ...]): The new shape.

        Returns:
            ndarray: A new ndarray instance with the specified shape.
        """
        return super().reshape(shape).view(type(self))



    def reduce(self,pattern: str, reduction: str = "mean"):
        """Combination of reordering and reduction using reader-friendly notation from `einops`.

        Examples:
        ```python
        >>> x = np.random.randn(100, 32, 64)
        ```
        Perform max-reduction on the first axis
        ```python
        >>> y = reduce(x, 't b c -> b c', 'max')
        ```
        Same as previous, but with clearer axes meaning
        ```
        >>> y = reduce(x, 'time batch channel -> batch channel', 'max')
        >>> x = np.random.randn(10, 20, 30, 40)
        ```
        2d max-pooling with kernel size = 2 * 2 for image processing
        ```python
        >>> y1 = reduce(x, 'b c (h1 h2) (w1 w2) -> b c h1 w1', 'max', h2=2, w2=2)
        ```
        If one wants to go back to the original height and width, depth-to-space trick can be applied
        ```python
        >>> y2 = rearrange(y1, 'b (c h2 w2) h1 w1 -> b c (h1 h2) (w1 w2)', h2=2, w2=2)
        >>> assert parse_shape(x, 'b _ h w') == parse_shape(y2, 'b _ h w')
        ```
        Adaptive 2d max-pooling to 3 * 4 grid
        ```python
        >>> reduce(x, 'b c (h1 h2) (w1 w2) -> b c h1 w1', 'max', h1=3, w1=4).shape
        (10, 20, 3, 4)
        ```
        Global average pooling
        ```python
        >>> reduce(x, 'b c h w -> b c', 'mean').shape
        (10, 20)
        ```
        Subtracting mean over batch for each channel
        ```python
        >>> y = x - reduce(x, 'b c h w -> () c () ()', 'mean')
        ```
        Subtracting per-image mean for each channel
        ```python
        >>> y = x - reduce(x, 'b c h w -> b c () ()', 'mean')
        ```
        """
        if "mean" in reduction:
            arr = self.astype(np.float32)
        elif reduction not in ("max", "min", "sum"):
            msg = f"Reduction operation '{reduction}' not supported. Use 'max', 'min', 'sum', or 'mean'."
            raise ValueError(msg)
        red = reduce(tensor=arr, pattern=pattern, reduction=reduction)
        return ndarray(data=red, dtype=self.dtype)


    @overload
    def mean(self: ndarray[RedT,*_Ts,DT], axis: idx[0]) -> ndarray[*_Ts, DT]: ...
    @overload
    def mean(self: ndarray[U,RedT,*_Ts, DT], axis: idx[1]) -> ndarray[U,*_Ts, DT]: ...
    @overload
    def mean(self: ndarray[U, T,RedT, *_Ts, DT], axis: idx[2]) -> ndarray[U, T, *_Ts, DT]: ...
    @overload
    def mean(self: ndarray[U, T,V, RedT, *_Ts, DT], axis: idx[3]) -> ndarray[U, T, V, *_Ts, DT]: ...
    @overload
    def mean(self: ndarray[U, T,V, W, RedT, *_Ts, DT], axis: idx[4]) -> ndarray[U, T, V, W, *_Ts, DT]: ...
    @overload
    def mean(self: ndarray[U, T,V, W, X, RedT, *_Ts, DT], axis: idx[5]) -> ndarray[U, T, V, W, X, *_Ts, DT]: ...
    @overload
    def mean(self: ndarray[*_Ts, RedT,T,U,V,W, DT], axis: idx[-5] | idx[6]) -> ndarray[*_Ts, T, U, V, W, DT]: ...
    @overload
    def mean(self: ndarray[*_Ts, RedT,U,V,W, DT], axis: idx[-4] | idx[7]) -> ndarray[*_Ts, U, V, W, DT]: ...
    @overload
    def mean(self: ndarray[*_Ts, RedT,V,W, DT], axis: idx[-3] | idx[8]) -> ndarray[*_Ts, V, W, DT]: ...
    @overload
    def mean(self: ndarray[*_Ts, RedT,W, DT], axis: idx[-2] | idx[9]) -> ndarray[*_Ts, W, DT]: ...
    @overload
    def mean(self: ndarray[*_Ts, RedT, DT], axis: idx[-1] | idx[10]) -> ndarray[*_Ts, DT]: ...

    def mean(self, axis=None, dtype=None, out=None,*, keepdims=False, where=True):
        # Use the parent class's mean method to avoid recursion
        mean_result = np.ndarray.mean(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where)
        # Pass mean_result as the 'data' keyword argument
        return ndarray(data=mean_result, dtype=self._dtype)

    # Addition
    @overload
    def __add__(self: ndarray[U, T, DT], other: ndarray[U, T, DT]) -> ndarray[U, T, DT]: ...
    @overload
    def __add__(self: ndarray[U, T, DT], other: ndarray[U, sz[1], DT]) -> ndarray[U, T, DT]: ...
    @overload
    def __add__(self: ndarray[U, T, DT], other: ndarray[sz[1], T, DT]) -> ndarray[U, T, DT]: ...

    def __add__(self, other) -> Self:
        result = np.add(self, other)
        return ndarray(data=result, dtype=self.dtype)

    # Subtraction
    @overload
    def __sub__(self: ndarray[U,T, DT], other: ndarray[U,T, DT]) -> ndarray[U,T, DT]: ...
    @overload
    def __sub__(self: ndarray[U,T, DT], other: ndarray[U, sz[1], DT]) -> ndarray[U,T, DT]: ...
    @overload
    def __sub__(self: ndarray[U,T, DT], other: ndarray[sz[1], T, DT]) -> ndarray[U,T, DT]: ...

    def __sub__(self, other):
        result = np.subtract(self, other)
        return ndarray(data=result, dtype=self.dtype)

    # Multiplication
    @overload
    def __mul__(self: ndarray[U,T, DT], other: ndarray[U,T, DT]) -> ndarray[U,T, DT]: ...
    @overload
    def __mul__(self: ndarray[U,T, DT], other: ndarray[U, sz[1], DT]) -> ndarray[U,T, DT]: ...
    @overload
    def __mul__(self: ndarray[U,T, DT], other: ndarray[sz[1], T, DT]) -> ndarray[U,T, DT]: ...

    def __mul__(self, other):
        result = np.multiply(self, other)
        return ndarray(data=result, dtype=self.dtype)

    # Division
    @overload
    def __div__(self: ndarray[U,T, DT], other: ndarray[U,T, DT]) -> ndarray[U,T, DT]: ...
    @overload
    def __div__(self: ndarray[U,T, DT], other: ndarray[U, sz[1], DT]) -> ndarray[U,T, DT]: ...
    @overload
    def __div__(self: ndarray[U,T, DT], other: ndarray[sz[1], T, DT]) -> ndarray[U,T, DT]: ...

    def __div__(self, other):
        result = np.divide(self, other)
        return ndarray(data=result, dtype=self.dtype)

    # True Division
    @overload
    def __truediv__(self: ndarray[U,T, DT], other: ndarray[U,T, DT]) -> ndarray[U,T, DT]: ...
    @overload
    def __truediv__(self: ndarray[U,T, DT], other: ndarray[U, sz[1], DT]) -> ndarray[U,T, DT]: ...
    @overload
    def __truediv__(self: ndarray[U,T, DT], other: ndarray[sz[1], T, DT]) -> ndarray[U,T, DT]: ...

    def __truediv__(self, other):
        result = np.true_divide(self, other)
        return ndarray(data=result, dtype=self.dtype)

    # Floor Division
    @overload
    def __floordiv__(self: ndarray[U,T, DT], other: ndarray[U,T, DT]) -> ndarray[U,T, DT]: ...
    @overload
    def __floordiv__(self: ndarray[U,T, DT], other: ndarray[U, sz[1], DT]) -> ndarray[U,T, DT]: ...
    @overload
    def __floordiv__(self: ndarray[U,T, DT], other: ndarray[sz[1], T, DT]) -> ndarray[U,T, DT]: ...

    def __floordiv__(self, other):
        result = np.floor_divide(self, other)
        return ndarray(data=result, dtype=self.dtype)

    # Modulus
    @overload
    def __mod__(self: ndarray[U,T, DT], other: ndarray[U,T, DT]) -> ndarray[U,T, DT]: ...
    @overload
    def __mod__(self: ndarray[U,T, DT], other: ndarray[U, sz[1], DT]) -> ndarray[U,T, DT]: ...
    @overload
    def __mod__(self: ndarray[U,T, DT], other: ndarray[sz[1], T, DT]) -> ndarray[U,T, DT]: ...

    def __mod__(self, other):
        result = np.mod(self, other)
        return ndarray(data=result, dtype=self.dtype)

    # Covariance Overloads
    @overload
    def cov(self: ndarray[M, N, DT],
        other: None = None,
        rowvar: Literal[True] = True,
        bias: bool = False,
        ddof: int | None = None,
    ) -> ndarray[M, M, DT]:
        ...

    @overload
    def cov(
        self: ndarray[N, M, DT],
        other: None = None,
        rowvar: Literal[False] = False,
        bias: bool = False,
        ddof: int | None = None,
    ) -> ndarray[M, M, DT]:
        ...

    @overload
    def cov(
        self: ndarray[M, N, DT],
        other: ndarray[M, N, DT],
        rowvar: Literal[True] = True,
        bias: bool = False,
        ddof: int | None = None,
    ) -> ndarray[2 * M, 2 * M, DT]:
        ...

    @overload
    def cov(
        self: ndarray[N, M, DT],
        other: ndarray[N, M, DT],
        rowvar: Literal[False] = False,
        bias: bool = False,
        ddof: int | None = None,
    ) -> ndarray[2 * M, 2 * M, DT]:
        ...

    @overload
    def cov(
        self: ndarray[Any, Any, DT],
        other: ndarray[Any, Any, DT] | None = None,
        rowvar: bool = ...,
        bias: bool = False,
        ddof: int | None = None,
    ) -> ndarray[Any, Any, DT]:
        ...

    def cov(
        self,
        other: ndarray[Any, Any, DT] | None = None,
        rowvar: bool = True,
        bias: bool = False,
        ddof: int | None = None,
    ) -> ndarray[Any, Any, DT]:
        """Calculate the covariance matrix.

        Args:
            other (Optional[ndarray]): Optional additional dataset to include in the covariance calculation.
            rowvar (bool): If True, each row represents a variable, with observations in columns. Defaults to True.
            bias (bool): If True, normalizes by `N` (number of observations). Defaults to False.
            ddof (Optional[int]): Delta degrees of freedom for normalization, overriding default if specified.

        Returns:
            ndarray: Covariance matrix as a new ndarray instance.
        """
        cov_result = np.cov(self, y=other, rowvar=rowvar, bias=bias, ddof=ddof)
        return ndarray(data=cov_result, dtype=self.dtype)

    # Horizontal Stack
    @overload
    def hstack(self: ndarray[U, sz[1], DT], others: Sequence[ndarray[U, sz[1], DT]]) -> ndarray[U, sz[2], DT]: ...
    @overload
    def hstack(self: ndarray[U, sz[2], DT], others: Sequence[ndarray[U, sz[2], DT]]) -> ndarray[U, sz[4], DT]: ...
    @overload
    def hstack(self: ndarray[U, sz[3], DT], others: Sequence[ndarray[U, sz[3], DT]]) -> ndarray[U, sz[6], DT]: ...
    @overload
    def hstack(self: ndarray[U, sz[4], DT], others: Sequence[ndarray[U, sz[4], DT]]) -> ndarray[U, sz[8], DT]: ...
    @overload
    def hstack(self: ndarray[U, sz[5], DT], others: Sequence[ndarray[U, sz[5], DT]]) -> ndarray[U, sz[10], DT]: ...
    @overload
    def hstack(self: ndarray[sz[1], T, DT], others: Sequence[ndarray[sz[1], T, DT]]) -> ndarray[sz[2], T, DT]: ...
    @overload
    def hstack(self: ndarray[sz[2], T, DT], others: Sequence[ndarray[sz[2], T, DT]]) -> ndarray[sz[4], T, DT]: ...
    @overload
    def hstack(self: ndarray[sz[3], T, DT], others: Sequence[ndarray[sz[3], T, DT]]) -> ndarray[sz[6], T, DT]: ...
    @overload
    def hstack(self: ndarray[sz[4], T, DT], others: Sequence[ndarray[sz[4], T, DT]]) -> ndarray[sz[8], T, DT]: ...
    @overload
    def hstack(self: ndarray[sz[5], T, DT], others: Sequence[ndarray[sz[5], T, DT]]) -> ndarray[sz[10], T, DT]: ...

    def hstack(self, others):
        stacked = np.hstack([self, *others])
        return ndarray(data=stacked, dtype=stacked.dtype)

    # Vertical Stack
    @overload
    def vstack(self: ndarray[U, sz[1], DT], others: Sequence[ndarray[U, sz[1], DT]]) -> ndarray[sz[2], sz[1], DT]: ...
    @overload
    def vstack(self: ndarray[U, sz[2], DT], others: Sequence[ndarray[U, sz[2], DT]]) -> ndarray[sz[4], sz[2], DT]: ...
    @overload
    def vstack(self: ndarray[U, sz[3], DT], others: Sequence[ndarray[U, sz[3], DT]]) -> ndarray[sz[6], sz[3], DT]: ...
    @overload
    def vstack(self: ndarray[U, sz[4], DT], others: Sequence[ndarray[U, sz[4], DT]]) -> ndarray[sz[8], sz[4], DT]: ...
    @overload
    def vstack(self: ndarray[U, sz[5], DT], others: Sequence[ndarray[U, sz[5], DT]]) -> ndarray[sz[10], sz[5], DT]: ...
    @overload
    def vstack(self: ndarray[sz[1], T, DT], others: Sequence[ndarray[sz[1], T, DT]]) -> ndarray[sz[1], sz[2], DT]: ...
    @overload
    def vstack(self: ndarray[sz[2], T, DT], others: Sequence[ndarray[sz[2], T, DT]]) -> ndarray[sz[2], sz[4], DT]: ...
    @overload
    def vstack(self: ndarray[sz[3], T, DT], others: Sequence[ndarray[sz[3], T, DT]]) -> ndarray[sz[3], sz[6], DT]: ...
    @overload
    def vstack(self: ndarray[sz[4], T, DT], others: Sequence[ndarray[sz[4], T, DT]]) -> ndarray[sz[4], sz[8], DT]: ...
    @overload
    def vstack(self: ndarray[sz[5], T, DT], others: Sequence[ndarray[sz[5], T, DT]]) -> ndarray[sz[5], sz[10], DT]: ...

    def vstack(self, others):
        stacked = np.vstack([self, *others])
        return ndarray(data=stacked, dtype=stacked.dtype)

    # Concatenate
    @overload
    def concatenate(self: ndarray[U, sz[1], DT], others: Sequence[ndarray[U, sz[1], DT]], axis: idx[0]) -> ndarray[U, sz[2], DT]: ...
    @overload
    def concatenate(self: ndarray[U, sz[2], DT], others: Sequence[ndarray[U, sz[2], DT]], axis: idx[1]) -> ndarray[U, sz[4], DT]: ...
    @overload
    def concatenate(self: ndarray[U, sz[3], DT], others: Sequence[ndarray[U, sz[3], DT]], axis: idx[2]) -> ndarray[U, sz[6], DT]: ...
    @overload
    def concatenate(self: ndarray[U, sz[4], DT], others: Sequence[ndarray[U, sz[4], DT]], axis: idx[3]) -> ndarray[U, sz[8], DT]: ...
    @overload
    def concatenate(self: ndarray[U, sz[5], DT], others: Sequence[ndarray[U, sz[5], DT]], axis: idx[4]) -> ndarray[U, sz[10], DT]: ...
    @overload
    def concatenate(self: ndarray[sz[1], T, DT], others: Sequence[ndarray[sz[1], T, DT]], axis: idx[1]) -> ndarray[sz[2], T, DT]: ...
    @overload
    def concatenate(self: ndarray[sz[2], T, DT], others: Sequence[ndarray[sz[2], T, DT]], axis: idx[2]) -> ndarray[sz[4], T, DT]: ...
    @overload
    def concatenate(self: ndarray[sz[3], T, DT], others: Sequence[ndarray[sz[3], T, DT]], axis: idx[3]) -> ndarray[sz[6], T, DT]: ...
    @overload
    def concatenate(self: ndarray[sz[4], T, DT], others: Sequence[ndarray[sz[4], T, DT]], axis: idx[4]) -> ndarray[sz[8], T, DT]: ...
    @overload
    def concatenate(self: ndarray[sz[5], T, DT], others: Sequence[ndarray[sz[5], T, DT]], axis: idx[5]) -> ndarray[sz[10], T, DT]: ...

    def concatenate(self, others, axis):
        concatenated = np.concatenate([self, *others], axis=axis)
        return ndarray(data=concatenated, dtype=concatenated.dtype)

    # Inverse
    @overload
    def inv(self: ndarray[U, U, DT]) -> ndarray[U, U, DT]: ...

    def inv(self):
        inversed = np.linalg.inv(self)
        return ndarray(data=inversed, dtype=inversed.dtype)

    # Dot Product
    @overload
    def dot(self: ndarray[U, T, DT], other: ndarray[T, V, DT]) -> ndarray[U, V, DT]: ...

    def dot(self, other):
        dotted = np.dot(self, other)
        return ndarray(data=dotted, dtype=dotted.dtype)

    #TODO: reduce, rearrange
    @overload
    def min(self: ndarray[RedT,*_Ts,DT], axis: idx[0]) -> ndarray[*_Ts, DT]: ...
    @overload
    def min(self: ndarray[U,RedT,*_Ts, DT], axis: idx[1]) -> ndarray[U,*_Ts, DT]: ...
    @overload
    def min(self: ndarray[U, T,RedT, *_Ts, DT], axis: idx[2]) -> ndarray[U, T, *_Ts, DT]: ...
    @overload
    def min(self: ndarray[U, T,V, RedT, *_Ts, DT], axis: idx[3]) -> ndarray[U, T, V, *_Ts, DT]: ...
    @overload
    def min(self: ndarray[U, T,V, W, RedT, *_Ts, DT], axis: idx[4]) -> ndarray[U, T, V, W, *_Ts, DT]: ...
    @overload
    def min(self: ndarray[U, T,V, W, X, RedT, *_Ts, DT], axis: idx[5]) -> ndarray[U, T, V, W, X, *_Ts, DT]: ...
    @overload
    def min(self: ndarray[*_Ts, RedT,T,U,V,W, DT], axis: idx[-5] | idx[6]) -> ndarray[*_Ts, T, U, V, W, DT]: ...
    @overload
    def min(self: ndarray[*_Ts, RedT,U,V,W, DT], axis: idx[-4] | idx[7]) -> ndarray[*_Ts, U, V, W, DT]: ...
    @overload
    def min(self: ndarray[*_Ts, RedT,V,W, DT], axis: idx[-3] | idx[8]) -> ndarray[*_Ts, V, W, DT]: ...
    @overload
    def min(self: ndarray[*_Ts, RedT,W, DT], axis: idx[-2] | idx[9]) -> ndarray[*_Ts, W, DT]: ...
    @overload
    def min(self: ndarray[*_Ts, RedT, DT], axis: idx[-1] | idx[10]) -> ndarray[*_Ts, DT]: ...
    def min(
        self,
        axis,
        dtype: DT | None = None,
        out: ndarray | None = None,
        *,
        keepdims: bool = False,
        initial: ndarray | None = None,
        where: ndarray | None = None,
    ):
        self.data = np.min(self, axis, dtype, out, keepdims, initial, where)
        return self

    @overload
    def max(self: ndarray[RedT,*_Ts,DT], axis: idx[0]) -> ndarray[*_Ts, DT]: ...
    @overload
    def max(self: ndarray[U,RedT,*_Ts, DT], axis: idx[1]) -> ndarray[U,*_Ts, DT]: ...
    @overload
    def max(self: ndarray[U, T,RedT, *_Ts, DT], axis: idx[2]) -> ndarray[U, T, *_Ts, DT]: ...
    @overload
    def max(self: ndarray[U, T,V, RedT, *_Ts, DT], axis: idx[3]) -> ndarray[U, T, V, *_Ts, DT]: ...
    @overload
    def max(self: ndarray[U, T,V, W, RedT, *_Ts, DT], axis: idx[4]) -> ndarray[U, T, V, W, *_Ts, DT]: ...
    @overload
    def max(self: ndarray[U, T,V, W, X, RedT, *_Ts, DT], axis: idx[5]) -> ndarray[U, T, V, W, X, *_Ts, DT]: ...
    @overload
    def max(self: ndarray[*_Ts, RedT,T,U,V,W, DT], axis: idx[-5] | idx[6]) -> ndarray[*_Ts, T, U, V, W, DT]: ...
    @overload
    def max(self: ndarray[*_Ts, RedT,U,V,W, DT], axis: idx[-4] | idx[7]) -> ndarray[*_Ts, U, V, W, DT]: ...
    @overload
    def max(self: ndarray[*_Ts, RedT,V,W, DT], axis: idx[-3] | idx[8]) -> ndarray[*_Ts, V, W, DT]: ...
    @overload
    def max(self: ndarray[*_Ts, RedT,W, DT], axis: idx[-2] | idx[9]) -> ndarray[*_Ts, W, DT]: ...
    @overload
    def max(self: ndarray[*_Ts, RedT, DT], axis: idx[-1] | idx[10]) -> ndarray[*_Ts, DT]: ...
    def max(self, axis, out=None,*, keepdims=False, initial=None, where=True):
        reduced = np.max(super(), axis, out, keepdims, initial, where)
        return reduced.view(type(self))

    @overload
    def squeeze(self: ndarray[*_Ts, sz[1], DT]) -> ndarray[*_Ts, DT]: ...
    @overload
    def squeeze(self: ndarray[sz[1],*_Ts, DT]) -> ndarray[*_Ts, DT]: ...
    def squeeze(self):
        self.data = np.squeeze(self)
        return self

    @overload
    def unsqueeze(self: ndarray[*_Ts, DT], axis: idx[0]) -> ndarray[sz[1],*_Ts, DT]: ...
    @overload
    def unsqueeze(self: ndarray[*_Ts, DT], axis: idx[-1]) -> ndarray[*_Ts, sz[1], DT]: ...
    def unsqueeze(self, axis):
        self.data = np.expand_dims(self, axis)
        return self

    @property
    def T(self: "ndarray[T,U,*_Ts, DT]") -> "ndarray[U,T,*_Ts, DT]": # noqa
        return super().transpose().view(type(self))

    @property
    def Inv(self: Self) -> Self: # noqa
        self.data = np.linalg.inv(self)
        return self

    def __str__(self):
        return display(self)

def usage() -> None:
    arr = ndarray(shape=(1, 2), dtype=float, data=[[1, 2]])
    assert arr.shape == (1, 2), f"Expected shape (1, 2) but got {arr.shape}"
    assert np.array_equal(arr, [[1, 2]]), f"Expected [[1, 2]] but got {arr}"
    b: ndarray[1, 3, float] = ndarray[1, 3, float](data=[[1, 2, 3]])
    assert b.shape == (1, 3), f"Expected shape (1, 3) but got {b.shape}"
    assert np.array_equal(b, [[1, 2, 3]]), f"Expected [[1, 2, 3]] but got {b}"

    transpose_type = b.T
    assert transpose_type.shape == (3, 1), f"Expected shape (3, 1) but got {transpose_type.shape}"
    reshape_type = b.reshape((3, 1))

    b.reshape((3,))
    b.reshape((1, 3))
    assert reshape_type.shape == (3,), f"Expected shape (3,) but got {reshape_type.shape}"
    reduce_type = b.mean(1)
    assert reduce_type.shape == (1,), f"Expected shape (1,) but got {reduce_type.shape}"

    c = ndarray[3, 2, 1]()
    assert c.shape == (3, 2, 1), f"Expected shape (3, 2, 1) but got {c.shape}"

    d = c.reduce("a b c -> a c")
    assert d.shape == (3, 1), f"Expected shape (3, 1) but got {d.shape}"
    Shape = TypeVarTuple("Shape")
    Batch = TypeVar("Batch")

    b = ndarray[300, 3, float]()

    class Base(Generic[T, *Shape, DT]): ...

    class MyMLPipeline(Base):
        @classmethod
        def batch_of_images_to_bbox(cls, x: ndarray[Batch, *Ts, DT]):
            return ndarray[Batch, 4, float]()

    def batch_of_images_to_bbox(x: ndarray[Batch, *Ts, DT]):
        return cast("ndarray[Batch, sz[4], float]", ndarray[x.shape[0], 4, float]())

    a = batch_of_images_to_bbox(b)
    assert a.shape == (300, 4), f"Expected shape (300, 4) but got {a.shape}"

    ta = a.T
    assert ta.shape == (4, 300), f"Expected shape (4, 300) but got {ta.shape}"

    ta.reduce("a b -> b")

    d5_array = ndarray[1, 2, 3, 4, 5, float]()
    red0 = d5_array.mean(0)
    assert red0.shape == (2, 3, 4, 5), f"Expected shape (2, 3, 4, 5) but got {red0.shape}"
    red1 = d5_array.mean(1)
    assert red1.shape == (1, 3, 4, 5), f"Expected shape (1, 3, 4, 5) but got {red1.shape}"
    red2 = d5_array.mean(2)
    assert red2.shape == (1, 2, 4, 5), f"Expected shape (1, 2, 4, 5) but got {red2.shape}"
    red3 = d5_array.mean(3)
    assert red3.shape == (1, 2, 3, 5), f"Expected shape (1, 2, 3, 5) but got {red3.shape}"
    red4forward = d5_array.mean(4)
    assert red4forward.shape == (1, 2, 3, 4), f"Expected shape (1, 2, 3, 4) but got {red4forward.shape}"
    red4 = d5_array.mean(-1)
    assert red4.shape == (1, 2, 3, 4), f"Expected shape (1, 2, 3, 4) but got {red4.shape}"
    tts = ndarray[3,3,float]([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert tts.shape == (3, 3), f"Expected shape (3, 3) but got {tts.shape}"
    reduced = tts.mean(0)
    assert np.array_equal(reduced, [4.0, 5.0, 6.0]), f"Expected [[4, 5, 6]] but got {reduced}"
    s = tts.mean(-1)
    assert np.array_equal(s, [2, 5, 8]), f"Expected [[2], [5], [8]] but got {s}"


if __name__ == "__main__":
    # usage()

    # Example Usage:

    # Creating a sample ndarray instance
    x = ndarray[2, 3, float](data=np.random.randn(2, 3).astype(np.float32))

    # Reshaping from (2, 3) to (3, 2)
    y = x.reshape((3, 2))  # Expected type: ndarray[sz3, sz2, float]

    # Reshaping from (2, 3) to (6,)
    z = x.reshape((6,))  # Expected type: ndarray[sz6, float]




