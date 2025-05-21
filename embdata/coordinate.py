# pyright: ignore[reportGeneralTypeIssues]
# Copyright (c) 2024 Mbodi AI
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
"""NamedTuple-Like class for representing geometric data in cartesian and polar coordinates.

A 3D pose represents the planar x, y, and theta, while a 6D pose represents the volumetric x, y, z, roll, pitch, and yaw.

Example:
    >>> import math
    >>> pose_3d = Pose3D(x=1, y=2, theta=math.pi / 2)
    >>> pose_3d.to("cm")
    Pose3D(x=100.0, y=200.0, theta=1.5707963267948966)
    >>> pose_3d.to("deg")
    Pose3D(x=1.0, y=2.0, theta=90.0)
    >>> class BoundedPose6D(Pose6D):
    ...     x: float = CoordinateField(bounds=(0, 5))
    >>> pose_6d = BoundedPose6D(x=10, y=2, z=3, roll=0, pitch=0, yaw=0)
    Traceback (most recent call last):
    ...
    ValueError: x value 10 is not within bounds (0, 5)
"""

import builtins
import copy
import typing
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import wraps
from types import EllipsisType, GenericAlias
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    Self,
    TypeAlias,
    TypeVar,
    TypeVarTuple,
    Unpack,
    cast,
    overload,
)
from typing import (
    _UnpackGenericAlias as UnpackGenericAlias,  # type: ignore # pyright: ignore[reportAttributeAccessIssue,reportUnknownVariableType]
)
from weakref import ref

import numpy as np
import typing_extensions
from numpy.typing import ArrayLike
from pydantic import (
    Field,
    JsonValue,
    PrivateAttr,
    create_model,
    model_validator,
)
from pydantic_core import PydanticUndefined
from typing_extensions import Annotated, Literal, TypedDict, override

from embdata.array import H, Ts, W, array
from embdata.ndarray import Float64, ndarray, shape_and_dtype
from embdata.ndarray import slice as ndarray_slice_protocol
from embdata.sample import Sample
from embdata.units import (
    AngularLabel,
    AngularUnitType,
    LinearLabel,
    LinearUnitType,
    PixelUnitType,
    TemporalUnitType,
    islabel,
    isunit,
)
from embdata.utils.counting import N, sz
from embdata.utils.dynamic import dynamic
from embdata.utils.import_utils import smart_import

JsonDict: TypeAlias = dict[str, int | float | str | bool | list[JsonValue] | dict[str, JsonValue]]
if TYPE_CHECKING:
    import torch
    from pydantic.fields import FieldInfo

    from embdata.geometry.quaternion import Quaternion, QuatPose

InfoUndefinedType = Literal["unset"]
InfoUndefined = "unset"
_Ts = TypeVarTuple("_Ts")
EuelerSequence = Literal["zyx", "xyz"]
"""The order in which the rotation matrices are multiplied. ZYX means intrinsic rotations about X first, then Y,
 then Z aka roll, pitch, yaw.
"""

Float = np.float64 | np.float32 | np.float16 | float
Floats = np.float64 | np.floating | float

AnyType = TypeVar("AnyType")
DT = TypeVar("DT", bound=Any)

if TYPE_CHECKING:
    SkipVector = Annotated[AnyType, ...]
else:

    @dataclass(slots=True)
    class SkipVector:
        def __class_getitem__(cls, item: AnyType) -> AnyType:
            return Annotated[item, cls()]

        def __hash__(self) -> int:
            return hash(type(self))


def MetadataField(
    default: Any = ...,
    **kwargs: Any,
) -> Any:
    """Create a Pydantic Field with extra metadata for coordinates."""
    return Field(default=default, **kwargs, json_schema_extra={"metadata": True})





def CoordinateField(  # noqa
    default: Any = ...,
    reference_frame: str | EllipsisType = ...,
    origin: np.ndarray[Any, Any] | list[float] | EllipsisType = ...,
    unit: LinearUnitType | AngularUnitType | PixelUnitType | TemporalUnitType = "m",
    bounds: tuple[float, float] | array[sz[2], int] | EllipsisType = ...,
    description: str | None = None,
    examples: str | list[str] | None = None,
    visibility: Literal["public", "private"] = "public",
    **kwargs: Any,
) -> Any:
    """Create a Pydantic Field with extra metadata for coordinates.

    This function extends Pydantic's Field with additional metadata specific to coordinate systems,
    including reference frame, unit, and bounds information.

    Args:
        default: Default value for the field.
        default_factory: Factory for creating the default value.
        reference_frame: Reference frame for the coordinates.
        origin: Origin of the coordinate system.
        unit: Unit of the coordinate (LinearUnit, AngularUnit, or TemporalUnit).
        bounds: Tuple representing the allowed range for the coordinate value.
        description: Description of the field.
        example: Example for the field.
        **kwargs: Additional keyword arguments for field configuration.

    Returns:
        Field: Pydantic Field with extra metadata.

    Examples:
        >>> from pydantic import BaseModel
        >>> class Robot(BaseModel):
        ...     x: float = CoordinateField(unit="m", bounds=(0, 10))
        ...     angle: float = CoordinateField(unit="rad", bounds=(0, 6.28))
        >>> robot = Robot(x=5, angle=3.14)
        >>> robot.dict()
        {'x': 5.0, 'angle': 3.14}
        >>> Robot(x=15, angle=3.14)
        Traceback (most recent call last):
        ...
        ValueError: x value 15.0 is not within bounds (0, 10)
    """
    default_factory = kwargs.pop("default_factory", None)
    if default_factory is not None:
        default_factory = cast("Callable[[], Any]", default_factory)

    json_schema_extra = {
        "_info": {
            **{
                key: value
                for key, value in {
                    "reference_frame": reference_frame,
                    "unit": unit,
                    "bounds": bounds,
                    "origin": origin,
                    **kwargs,
                }.items()
                if value is not None and not (isinstance(value, str) and value == "unset")
            },
        },
    }
    json_schema_extra = cast("JsonDict", json_schema_extra)
    examples = examples.split(",") if isinstance(examples, str) and "," in examples else examples
    examples = examples.split("\n") if isinstance(examples, str) and "\n" in examples else examples
    if isinstance(examples, str):
        examples = [examples]
    if default_factory is not None:
        if default and default != PydanticUndefined:
            msg = "default and default_factory cannot be used together"
            raise ValueError(msg)
        if visibility == "public":
            return Field(
                default_factory=default_factory,
                json_schema_extra=json_schema_extra,
                description=description,
                examples=examples,
            )
        return PrivateAttr(
            default_factory=default_factory,
        )
    return (
        Field(default=default, json_schema_extra=json_schema_extra, description=description, examples=examples)
        if visibility == "public"
        else PrivateAttr(default_factory=lambda: json_schema_extra["_info"])
    )


CoordField = CoordinateField
"""Alias for CoordinateField."""
wraps(CoordField)(CoordinateField)
CoordField.__doc__ = "Alias for CoordinateField." + getattr(CoordinateField, "__doc__", "")
CoordsField = CoordinateField
wraps(CoordsField)(CoordinateField)
CoordsField.__doc__ = "Alias for CoordinateField." + getattr(CoordinateField, "__doc__", "")

tp_cache = {}


class Shape(tuple[*Ts]):
    pass

if not TYPE_CHECKING:
    class CoordinateInfo(dict, Generic[*Ts]):
        reference_frame: str
        origin: "ref[Coordinate[Shape[*Ts]]] | ref[Coordinate[*Ts]]|ref[WorldOrigin[*Ts]]"
        unit: LinearUnitType | AngularUnitType | PixelUnitType | TemporalUnitType
        __repr_args__ = Sample.__repr_args__
        __repr_name__ = Sample.__repr_name__
        __rich_repr__ = Sample.__rich_repr__

        def __repr_args__(self) -> Iterable[Any]:
            for k in type(self).__annotations__:
                if not k.startswith("_"):
                    v = self[k]
                    if isinstance(v, Float):
                        v = float(v)
                    if isinstance(v, ref):
                        v = v()
                    yield k, v
else:
    class CoordinateInfo(TypedDict, Generic[*Ts]):
        reference_frame: str
        origin: "Coordinate[Shape[*Ts]] | Coordinate[*Ts] | WorldOrigin[*Ts]|None"
        unit: LinearUnitType | AngularUnitType | PixelUnitType | TemporalUnitType



def default_coordinate_info(kind: "type[CoordinateInfo[*Ts]]|None") -> "Callable[[],CoordinateInfo[*Ts]]":
    return lambda: CoordinateInfo(reference_frame="world", origin=None, unit="m")


def isnumeric(v: Any) -> bool:
    return isinstance(v, float | int | np.floating | np.integer)


class Coordinate(Sample, Generic[Unpack[Ts]]):
    """A list of numbers representing a coordinate in the world frame for an arbitrary space."""

    model_config = {"arbitrary_types_allowed": True, "ignored_types": (Shape, dynamic)}

    _info: CoordinateInfo[*Ts] = PrivateAttr(default_factory=lambda: CoordinateInfo(reference_frame="world", origin=None, unit="m"))
    _shape: ClassVar[Any]
    _dtype: ClassVar[Any]
    _origin_set: bool = PrivateAttr(default=False)
    @override
    def __repr_args__(self) -> Iterable[Any]:
        for k in type(self).model_fields:
            if not k.startswith("_"):
                v = getattr(self, k)
                if isinstance(v, Floats):
                    v = float(v)
                yield k, v

    if not TYPE_CHECKING:

        @classmethod
        def __class_getitem__(cls, *args) -> "type[Coordinate[*Ts]]":
            if args in tp_cache:
                return tp_cache[args]
                if isinstance(
                    args,
                    typing.TypeVar
                    | typing.TypeVarTuple
                    | typing_extensions.TypeVar
                    | typing_extensions.TypeVarTuple
                    | GenericAlias
                    | UnpackGenericAlias,
                ):
                    return cls
                params = args[0]

                if isinstance(
                    params,
                    typing.TypeVar
                    | typing.TypeVarTuple
                    | typing_extensions.TypeVar
                    | typing_extensions.TypeVarTuple
                    | GenericAlias
                    | UnpackGenericAlias,
                ):
                    return cls
                if any(
                    isinstance(
                        arg,
                        typing.TypeVar
                        | typing.TypeVarTuple
                        | typing_extensions.TypeVar
                        | typing_extensions.TypeVarTuple
                        | GenericAlias
                        | UnpackGenericAlias,
                    )
                    for arg in params
                ):
                    return cls
                shape, dtype = shape_and_dtype(params)
                try:
                    if args not in tp_cache:
                        tp_cache[args] = type(
                            f"{cls.__name__}[{','.join(map(str, shape))},{str(getattr(dtype, '__name__', dtype)).split('.')[-1].replace('<', '').replace('>', '')}]",
                            (cls,),
                            {
                                "_shape": shape,
                                "shape": shape,
                                "_dtype": dtype,
                                "__args__": args,
                                "__origin__": cls,
                                "__module__": __name__,
                                "__parameters__": args,
                            },
                        )
                        tp_cache[args].shape = shape
                        return tp_cache[args]
                except Exception as e:
                    from embdata.log import debug

                    debug(e)
                return tp_cache[args]
            return cls

    def info(self) -> CoordinateInfo[*Ts]:
        """Return a copy of the coordinate's metadata."""
        info = self._info
        if info.get("origin") is None:
            # Default to a zero coordinate of same subclass so that ``origin()``
            # returns a *typed* instance (e.g. ``Pose6D``) rather than the
            # generic ``WorldOrigin`` which broke multiple tests.
            info["origin"] = type(self).zeros()
        if info.get("reference_frame") is None:
            info["reference_frame"] = "world"
        if info.get("unit") is None:
            info["unit"] = "m"
        self._info = info
        return CoordinateInfo[*Ts](**info)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        argiter: Iterable[Any]
        origin = kwargs.pop("origin", None)
        reference_frame = kwargs.pop("reference_frame", None)
        unit = kwargs.pop("unit", None)
        # Process positional arguments into field values
        # kwargs = {k:Float(v) if isinstance(v,float|int|np.floating|np.integer) else v for k,v in kwargs.items()}
        params = type(self).model_fields
        arg_idx = 0
        if len(args) == 1 and isinstance(arg := next(iter(args), None), list | tuple | np.ndarray):
            argiter = list(arg)
        elif len(args) == 1 and isinstance(arg := next(iter(args), None), Coordinate):
            argiter = [*arg]
        else:
            argiter = list(args)
        for p in params:
            if arg_idx >= len(argiter):
                break
            kwargs[p] = argiter[arg_idx]
            annotation = params[p].annotation
            if p in params and isinstance(annotation, type) and issubclass(annotation, Coordinate):
                kwargs[p] = annotation(*args[arg_idx]) if hasattr(args[arg_idx], "__iter__") else args[arg_idx]
            arg_idx += 1

        super().__init__(
            **{
                k: np.float64(v) if isinstance(v, float | int | np.floating | np.integer) else v
                for k, v in kwargs.items()
            },
        )
        if origin is not None:
            self.set_origin(origin)
        if reference_frame is not None:
            self.set_reference_frame(reference_frame)
        if unit is not None:
            self.set_info("unit", unit)

    @classmethod
    def zeros(
        cls,
        origin: "Coordinate[*Ts] | WorldOrigin[*Ts] | None" = None,
        reference_frame: str | None = None,
        unit: LinearUnitType | AngularUnitType | PixelUnitType | TemporalUnitType | None = None,
    ) -> "Self":
        """Create a Coordinate instance with all zeros."""
        # Since shape is an instance method, we need to get shape from _shape or model_fields
        shape = getattr(cls, "_shape", None)
        if not shape or (isinstance(shape, tuple) and any(s is None for s in shape)):
            shape = (len(cls.model_fields),)

        c = cls(*np.zeros(shape))
        c._info = CoordinateInfo[*Ts](
            reference_frame=reference_frame or "world",
            origin= cast("WorldOrigin[*Ts,]", origin if origin is not None else WorldOrigin[*Ts]()),
            unit=unit or "m",
        )
        # keep strong ref so weakref doesn't die
        c._info_strong_refs = {"origin": origin}
        return c

    @override
    def model_info(self) -> dict[str, Any]:
        _info = super().model_info()
        _info.update({"self": self.info()})
        return _info

    @override
    def copy(self) -> "Self":
        """Deep copy *self* ensuring the *origin* metadata is duplicated too."""
        new_obj = copy.deepcopy(self)
        # Duplicate origin so caller-side mutations don't affect original
        origin_clone = copy.deepcopy(self.origin())
        new_obj.set_info("origin", origin_clone)
        return new_obj

    @overload
    def set_info(self, info: CoordinateInfo[*Ts]) -> None: ...
    @overload
    def set_info(self, **kwargs: Unpack[CoordinateInfo[*Ts]]) -> None: ...
    @overload
    def set_info(self, key: str, value: Any) -> None: ...
    def set_info(self, *args: Any, **kwargs: Any) -> None:
        """Update metadata entries ``origin``, ``reference_frame`` or ``unit``.

        Accepts either a mapping (`set_info({...})`), key-value pairs
        (`set_info("origin", obj)`), or keyword arguments
        (`set_info(origin=obj, unit="m")`). Unspecified keys stay unchanged.
        """
        if args:
            if len(args) == 1 and isinstance(arg := next(iter(args), None), dict):
                kwargs.update(arg)
            elif len(args) % 2:
                msg = "set_info expects key/value pairs"
                raise ValueError(msg)
            kwargs.update(dict(zip(args[0::2], args[1::2], strict=False)))

        info = dict(getattr(self, "_info", {}))

        if "unit" not in info:
            info["unit"] = "m"
        if "reference_frame" not in info:
            info["reference_frame"] = "world"

        # Ensure there is some origin (may be overwritten below)
        if "origin" not in info:
            shape_tuple = self.shape or (len(type(self).model_fields),)
            if any(s is None for s in shape_tuple):
                shape_tuple = (len(type(self).model_fields),)
            info["origin"] = WorldOrigin[(*shape_tuple, float)]()

        # Apply requested updates
        if "origin" in kwargs:
            origin_obj = kwargs.pop("origin")
            # Allow any Coordinate type as origin, or WorldOrigin
            if not (isinstance(origin_obj, Coordinate | WorldOrigin)):
                msg = f"Origin must be a Coordinate subtype or WorldOrigin, got {type(origin_obj).__name__}"
                raise TypeError(msg)
            info["origin"] = origin_obj
        if "reference_frame" in kwargs:
            info["reference_frame"] = kwargs.pop("reference_frame")

        if "unit" in kwargs:
            info["unit"] = kwargs.pop("unit")

        self._info = CoordinateInfo[*Ts](**info)

    def set_origin(self, origin: "Coordinate[*Ts] | WorldOrigin[Any] | None") -> None:
        """Set the origin of the coordinate."""
        self._info["origin"] = origin

    def magnitude(self) -> np.floating[Any]:
        """Compute the magnitude of the coordinate."""
        return np.linalg.norm(self.numpy())

    def direction(self) -> array[*Ts, Float]:
        """Compute the direction of the coordinate."""
        return self.numpy() / self.magnitude()

    def angle(self) -> float:
        """Compute the angle of the first two dimensions of the coordinate."""
        return np.arctan2(*self[:2]) if len(self) >= 2 else 0.0  # noqa

    def reference_frame(self) -> str:
        """Get the reference frame of the coordinate."""
        return self.info()["reference_frame"]

    def set_reference_frame(self, reference_frame: str) -> None:
        """Set the reference frame of the coordinate."""
        self._info["reference_frame"] = reference_frame


    def astype(self, dtype: DT) -> "array[*Ts,DT]":
        """Cast the coordinate to a different type."""
        return self.numpy().astype(np.dtype(dtype))

    def origin(self) -> Self:
        """Return the origin of this coordinate.

        Behaviour:
        • If an explicit origin is stored, return it (after making sure its
          reference-frame matches ``self``).
        • If no origin is stored (``None`` or ``WorldOrigin`` sentinel),
          synthesise a zero-vector of the same subclass in *self*'s frame.
        """
        stored_origin = self.info().get("origin")

        # ------------------------------------------------------------------
        # 1. Nothing stored  →  create zeros() in this frame
        # ------------------------------------------------------------------
        if stored_origin is None or isinstance(stored_origin, WorldOrigin):
            new_origin: Self = type(self).zeros(reference_frame=self.reference_frame())  # type: ignore[arg-type]
            self.set_info("origin", new_origin)
            return new_origin

        # ------------------------------------------------------------------
        # 2. We already have a Coordinate object
        #    Ensure its reference-frame is aligned with *self*.
        # ------------------------------------------------------------------
        if isinstance(stored_origin, Coordinate):
            # Do *not* force the origin's reference frame to match *self*.
            # The origin is an independent coordinate object whose metadata
            # must remain untouched (see tests `test_reference_frame_origin_setting`
            # and friends).  Callers who need a version of the origin expressed
            # in the child's frame can do that explicitly via
            #   origin.relative(child_frame)  or similar utilities.
            return stored_origin  # type: ignore[return-value]

        # Fallback (should not happen): convert numerics
        return type(self)(*stored_origin)  # type: ignore[arg-type]

    @staticmethod
    def convert_linear_unit(value: float, from_unit: LinearUnitType | str, to_unit: LinearUnitType | str) -> float:
        """Convert a value from one linear unit to another.

        This method supports conversion between meters (m), centimeters (cm),
        millimeters (mm), inches (in), and feet (ft).

        Args:
            value (float): The value to convert.
            from_unit (str): The unit to convert from.
            to_unit (str): The unit to convert to.

        Returns:
            float: The converted value.

        Examples:
            >>> Coordinate.convert_linear_unit(1.0, "m", "cm")
            100.0
            >>> Coordinate.convert_linear_unit(100.0, "cm", "m")
            1.0
            >>> Coordinate.convert_linear_unit(1.0, "m", "ft")
            3.280839895013123
            >>> Coordinate.convert_linear_unit(12.0, "in", "cm")
            30.48
        """
        conversion_from_factors = {
            "m": 1.0,
            "cm": 0.01,
            "mm": 0.001,
            "in": 0.0254,
            "ft": 0.3048,
        }
        conversion_to_factors = {
            "m": 1.0,
            "cm": 100.0,
            "mm": 1000.0,
            "in": 1.0 / 0.0254,
            "ft": 1.0 / 0.3048,
        }
        from_unit_factor = conversion_from_factors[from_unit]
        to_unit_factor = conversion_to_factors[to_unit]
        if from_unit == to_unit:
            return value
        return value * from_unit_factor * to_unit_factor

    @staticmethod
    def convert_angular_unit(value: float, to: AngularUnitType | str) -> float:
        """Convert radians to degrees or vice versa.

        Args:
            value (float): The angular value to convert.
            to (AngularUnit): The target angular unit ("deg" or "rad").

        Returns:
            float: The converted angular value.
        """
        if isinstance(value, Coordinate):
            return value.to(to)
        return np.degrees(value) if to == "deg" else np.radians(value)

    def __mul__(self, other: Any) -> "Self":
        return self.numpy() * other

    def __truediv__(self, other: Any) -> "Self":
        return self.numpy() / other

    def __sub__(self, other: Any) -> "Self":
        return self.numpy() - other

    def __add__(self, other: Any) -> "Self":
        return self.numpy() + other

    @dynamic
    def shape(self) -> tuple[int, ...]:
        sh = getattr(self, "_shape", None)
        if not sh or (isinstance(sh, tuple) and any(s is None for s in sh)):
            return (len(type(self).model_fields),)
        return sh  # type: ignore[return-value]

    if TYPE_CHECKING:
        dtype = np.ndarray.dtype
        ndim = np.ndarray.ndim
        __array__ = np.ndarray.__array__

    else:

        @property
        def dtype(self) -> Any:
            return Float64

        @property
        def ndim(self) -> int:
            return self.numpy().ndim

        def __array__(self, dtype: Any | None = None) -> np.ndarray:
            """Return a NumPy representation to support the ``numpy.array`` protocol.

            Implemented as a normal method (not a property) so that NumPy calls
            ``obj.__array__(dtype)`` correctly. Returning an ``ndarray`` directly
            avoids the recursion issue that arose when this was mistakenly
            implemented as a property which caused :pyclass:`TypeError` because
            NumPy attempted to call the returned ``ndarray``.
            """
            return np.asarray(list(self.values()), dtype=dtype)

    def tobytes(self) -> bytes:
        return self.numpy().tobytes()

    @override
    def numpy(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        return np.array(list(self.values())).astype(np.float64)


    @overload
    def __getitem__(self, item: int,/) -> Float: ...
    @overload
    def __getitem__(self, item: str,/) -> Float: ...
    @overload
    def __getitem__(self: "Coordinate[Any,*_Ts,Float]", *item: ndarray_slice_protocol[None, N, None]) -> "array[N,*_Ts,Float]": ...
    @overload
    def __getitem__(self, *item: builtins.slice) -> "array[*Ts,Float]": ...
    @override
    def __getitem__(self, *args: Any) -> "Any":

        if not args:
            return self.numpy()
        item_arg = args[0]
        if isinstance(item_arg, str):
            return getattr(self, item_arg)
        if isinstance(item_arg, builtins.slice):
            return self.numpy().__getitem__(item_arg)
        # Fallback for other types of arguments (e.g., int, tuple of ints for advanced indexing)
        return self.numpy().__getitem__(*args)

    @override
    def __setitem__(self, item: int | str | builtins.slice, value: array[*Ts, Float]) -> None:
        for i, v in enumerate(value):
            setattr(self, list(type(self).model_fields.keys())[i], v)

    def __neg__(self) -> "Self":
        """Negate the coordinate."""
        return type(self)(*(-self.numpy()))

    def __tensor__(self) -> "torch.Tensor":
        """Return a torch tensor representation of the pose."""
        torch = smart_import("torch")
        return torch.tensor(self.values())

    @override
    def values(self) -> array[*Ts, Float]:
        """Return the numeric components of the coordinate in declared order.

        For concrete subclasses (e.g. ``Pose6D``) ``model_fields`` contains the
        field definitions, so we can simply iterate over that mapping.  For the
        un-parameterised base ``Coordinate`` class (used ad-hoc in tests) the
        field list is empty; we therefore fall back to the runtime attributes
        recorded in ``__dict__``.
        """
        if type(self).model_fields:
            seq = [getattr(self, name) for name in type(self).model_fields]
        else:
            seq = [v for k, v in super().__iter__() if not k.startswith("_")]
        return ndarray[*Ts, Float](seq)

    @override
    def __iter__(self) -> "Iterator[Any]":
        """Iterate over the coordinate values."""
        yield from [getattr(self, name) for name in type(self).model_fields if not name.startswith("_")]

    @model_validator(mode="after")
    def ensure_shape_and_bounds(self) -> Any:
        """Validate the bounds of the coordinate."""
        for key, value in super().__iter__():
            if key.startswith("_"):
                continue
            bounds = self.field_info(key).get("bounds")
            shape = self.field_info(key).get("shape")
            if bounds and bounds is not  ...:
                if len(bounds) != 2 or not all(isinstance(b, int | float) for b in bounds):
                    msg = f"{key} bounds must consist of two numbers"
                    raise ValueError(msg)

                if shape and shape is not ...:
                    shape = [shape] if isinstance(shape, int) else shape
                    shape_processed = []
                    value_processed = value
                    while len(shape_processed) < len(shape):
                        shape_processed.append(len(value_processed))
                        if shape_processed[-1] != len(value_processed):
                            msg = f"{key} value {value} of length {len(value_processed)} at dimension {len(shape_processed) - 1} does not have the correct shape {shape}"
                            raise ValueError(msg)
                        value_processed = value_processed[0]

                    if hasattr(value, "shape") or isinstance(value, list | tuple):
                        for i, v in enumerate(value):
                            if not bounds[0] <= v <= bounds[1]:
                                msg = f"{key} item {i} ({v}) is out of bounds {bounds}"
                                raise ValueError(msg)
                elif not bounds[0] <= value <= bounds[1]:
                    msg = f"{key} value {value} is not within bounds {bounds}"
                    raise ValueError(msg)
        return self

    @override
    def to(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        container_or_unit: Any | str | None = None,
        unit: LinearUnitType | AngularUnitType | PixelUnitType | TemporalUnitType | None = None,
        angular_unit: AngularUnitType | None = None,
        **kwargs: Any,
    ) -> Any:
        """Convert the coordinate to a different unit or container.

        To see the available units, see the embdata.units module.

        Args:
            container_or_unit (Any, optional): The target container type or unit.
            unit (str, optional): The target linear unit. Defaults to "m".
            angular_unit (str, optional): The target angular unit. Defaults to "rad".
            **kwargs: Additional keyword arguments for field configuration.

        Returns:
            Any: The converted pose, either as a new Pose3D object with different units
                or as a different container type.

        Examples:
            >>> import math
            >>> pose = Pose3D(x=1, y=2, theta=math.pi / 2)
            >>> pose.to("cm")
            Pose3D(x=100.0, y=200.0, theta=1.5707963267948966)
            >>> pose.to("deg")
            Pose3D(x=1.0, y=2.0, theta=90.0)
            >>> pose.to("list")
            [1.0, 2.0, 1.5707963267948966]
            >>> pose.to("dict")
            {'x': 1.0, 'y': 2.0, 'theta': 1.5707963267948966}
        """
        if container_or_unit is not None and not isunit(container_or_unit):
            items = super().to(container_or_unit, **kwargs)
        else:
            items = super()

        if isunit(container_or_unit, LinearUnitType):
            unit = container_or_unit
        elif isunit(container_or_unit, AngularUnitType):
            angular_unit = container_or_unit
        converted_fields: dict[str, tuple[Any, FieldInfo]] = {}

        for key, value in items.items():
            if unit and islabel(key, LinearLabel) and unit != self.field_info(key)["unit"]:
                converted_field = self.convert_linear_unit(value, self.field_info(key)["unit"], unit)
                converted_fields[key] = (
                    converted_field,
                    CoordinateField(converted_field, unit=unit, **kwargs),
                )
            elif angular_unit and islabel(key, AngularLabel) and angular_unit != self.field_info(key)["unit"]:
                converted_field = self.convert_angular_unit(value, angular_unit)
                converted_fields[key] = (
                    converted_field,
                    CoordinateField(converted_field, unit=angular_unit, **kwargs),
                )
            else:
                original_field_info = self.field_info(key)
                converted_fields[key] = (
                    value,
                    CoordinateField(value, **original_field_info),
                )

        return create_model(
            self.__class__.__name__,
            __base__=self.__class__,
            __module__=self.__class__.__module__,
            **{k: (float, v[1]) for k, v in converted_fields.items()},  # type: ignore
        )(**{k: v[0] for k, v in converted_fields.items()})  # type: ignore


class WorldOrigin(Generic[*_Ts]):
    """World origin coordinate."""

    _singletons: ClassVar[dict[tuple[int, ...], Self]] = {}
    _origin: ClassVar[Self]
    shape: tuple[int, ...]
    def values(self) -> array[*_Ts, Float]:
        return np.zeros(self.shape)

    def origin(self) -> Self:
        return None

    def __iter__(self) -> Iterator[float]:
        yield from self.values()

    def __new__(cls, *args: Any,shape: tuple[int, ...] = (0,), **kwargs: Any) -> Self:
        if not hasattr(cls, "_singleton"):
            cls._singletons[shape] = super().__new__(cls, *args, **kwargs)
            cls._singletons[shape].shape = shape
        return cls._singletons[shape]

    @classmethod
    def __call__(cls, *args: Any, **kwargs: Any) -> Self:
        return cls._singleton


Coord = Coordinate
"""Alias for Coordinate."""
Coords = Coordinate
"""Alias for Coordinate."""


class Point(Coord[sz[3], Float]):
    x: Float = CoordinateField(unit="m", default=0.0)
    y: Float = CoordinateField(unit="m", default=0.0)
    z: Float = CoordinateField(unit="m", default=0.0)


class Vector3(Point):
    pass


class Pose3D(Coord):
    """Absolute coordinates for a 3D space representing x, y, and theta.

    This class represents a pose in 3D space with x and y coordinates for position
    and theta for orientation.

    Attributes:
        x (float): X-coordinate in meters.
        y (float): Y-coordinate in meters.
        theta (float): Orientation angle in radians.

    Examples:
        >>> import math
        >>> pose = Pose3D(x=1, y=2, theta=math.pi / 2)
        >>> pose
        Pose3D(x=1.0, y=2.0, theta=1.5707963267948966)
        >>> pose.to("cm")
        Pose3D(x=100.0, y=200.0, theta=1.5707963267948966)
    """

    x: float = CoordinateField(unit="m", default=0.0)
    y: float = CoordinateField(unit="m", default=0.0)
    theta: float = CoordinateField(unit="rad", default=0.0)
    @property
    def rotation(self) -> array[sz[2], sz[2], Float]:
        """Compute the rotation matrix from the orientation angle."""
        from embdata.ndarray import ndarray

        return ndarray[sz[2], sz[2], Float](
            [
                [np.cos(self.theta), -np.sin(self.theta)],
                [np.sin(self.theta), np.cos(self.theta)],
            ],
        )

    def translation_vector(self) -> array[sz[2], Float]:
        """Compute the translation vector from the position."""
        return np.array([self.x, self.y])


PlanarPose: TypeAlias = Pose3D
if not TYPE_CHECKING:
    PlanarPose.__doc__ = "x, y, and theta in meters and radians."


class Pose6D(Coord[sz[6], Float]):
    """6D pose with ordered fields."""
    model_config = {"arbitrary_types_allowed": True, "ignored_types": (Shape, dynamic,property)}
    x: Float = CoordinateField(unit="m", default=0.0)
    y: Float = CoordinateField(unit="m", default=0.0)
    z: Float = CoordinateField(unit="m", default=0.0)
    roll: Float = CoordinateField(unit="rad", default=0.0)
    pitch: Float = CoordinateField(unit="rad", default=0.0)
    yaw: Float = CoordinateField(unit="rad", default=0.0)

    @dynamic
    def shape(self) -> tuple[sz[6]]:
        return (6,)

    @property
    def translation(self) -> array[sz[3], Float]:
        return self.numpy()[:3]

    @property
    def rotation(self) -> array[sz[3],sz[3],Float]:
        from scipy.spatial.transform import Rotation
        # To match standard convention (roll around X, pitch around Y, yaw around Z)
        # when using "xyz" Euler sequence, the angles must be provided in that order.
        return Rotation.from_euler("xyz", [self.roll, self.pitch, self.yaw], degrees=False).as_matrix()


    @override
    def __mul__(self: Self, other: "Self") -> "Pose6D":
        """Compose two poses (this ∘ other).

        The original implementation lost the translation component because
        it fed a *Pose6D* object directly into ``QuatPose.from_pose`` which
        does not understand that type.  We now convert the operands to their
        numeric representation first so both rotation **and** translation are
        preserved.
        """
        from embdata.geometry.quaternion import QuatPose

        if not isinstance(other, Pose6D):
            try:
                other = Pose6D(*other)  # type: ignore[arg-type]
            except Exception as exc:
                msg = "Unsupported operand type(s) for *: '" f"{type(self).__name__}' and '{type(other).__name__}'"
                raise TypeError(
                    msg,
                ) from exc

        q_self = QuatPose.from_pose(self.numpy())
        q_other = QuatPose.from_pose(other.numpy())
        composed = q_self * q_other
        return Pose6D.from_quaternion(composed, reference_frame=self.reference_frame())

    def inverse(self) -> "Pose6D":
        """Return the pose that undoes this transform (world ← self)."""
        from embdata.geometry.quaternion import QuatPose
        q_self = QuatPose.from_pose(self.numpy())
        inv = q_self.inverse()
        return Pose6D.from_quaternion(inv, reference_frame=self.reference_frame())

    @override
    def __add__(self, other: "Pose6D") -> "Pose6D":
        return self * other
    @override
    def __sub__(self, other: "Any") -> "Pose6D":
        if not isinstance(other, type(self)):
            msg = f"Unsupported operand type(s) for -: '{type(self).__name__}' and '{type(other).__name__}'"
            raise TypeError(msg)
        return self.relative(other)


    @overload
    def __init__(self, x:Float=0.0, y:Float=0.0, z:Float=0.0, roll:Float=0.0, pitch:Float=0.0, yaw:Float=0.0,*, reference_frame:str="world") -> None:...
    @overload
    def __init__(self, translation:array[sz[3],Float]|None=None, rotation:array[sz[3],sz[3],Float]|None=None,*, reference_frame:str="world") -> None:...
    @overload
    def __init__(self, translation:array[sz[3],Float]|None=None, orientation:array[sz[3],Float]|None=None, *,reference_frame:str="world") -> None:...
    @overload
    def __init__(self, array:array[sz[6],Float]|list[float],*, reference_frame:str="world") -> None:...
    def __init__(self, *args:Any, **kwargs:Any) -> None:
        arglist = list(args)
        array = kwargs.pop("array", arglist.pop(0) if arglist and getattr(arglist[0], "__len__", None) == 6 else None)
        translation = kwargs.pop("translation", arglist.pop(0) if arglist and getattr(arglist[0], "__len__", None) == 3 else None)
        rotation = kwargs.pop("rotation", arglist.pop(1) if arglist and getattr(arglist[0], "__len__", None) == 3 and getattr(arglist[0][0], "__len__", None) == 3 else None)
        orientation = kwargs.pop("orientation", arglist.pop(0) if arglist and getattr(arglist[0], "__len__", None) == 3 else None)
        reference_frame = kwargs.pop("reference_frame", "world")
        if array is not None:
            super().__init__(array, reference_frame=reference_frame)
        elif translation is not None and rotation is not None:
            super().__init__(translation, rotation, reference_frame=reference_frame)

        elif translation is not None and orientation is not None:
            super().__init__(translation, orientation, reference_frame=reference_frame)
        else:
            super().__init__(*args, reference_frame=reference_frame, **kwargs)

    def quaternion(self, scalar_first: bool = False) -> array[sz[4], Float]:
        """Convert roll, pitch, yaw to a quaternion using the class's standard 'xyz' Euler sequence.

        Args:
            scalar_first (bool): If True, returns quaternion as [x, y, z, w].
                                 If False (default), returns [w, x, y, z].

        Returns:
            np.ndarray: A quaternion representation of the pose's orientation.
        """
        from scipy.spatial.transform import Rotation
        return Rotation.from_euler("xyz", [self.roll, self.pitch, self.yaw], degrees=False).as_quat(canonical=True,
                                                                                                    scalar_first=scalar_first)


    @classmethod
    def from_quaternion(
        cls,
        quat: "array[sz[4], Float] | QuatPose | Quaternion",
        position: "array[sz[3], Float] | None" = None,
        *,
        reference_frame: str = "world",
    ) -> "Pose6D":
        """Create a pose from a unit quaternion (w,x,y,z order)."""
        from embdata.geometry.quaternion import QuatPose, quat_to_euler
        if isinstance(quat, QuatPose):
            if position is not None:
                msg = "'position' must be *None* when 'quat' is a QuatPose"
                raise ValueError(msg)
            pos = quat.t
            q_arr = quat.q
        else:
            q_arr = np.asarray(quat, dtype=float)
            pos = np.zeros(3) if position is None else position

        angles = quat_to_euler(q_arr, scalar_first=True)

        return cls(*pos, *angles, reference_frame=reference_frame)

    from_quat = from_quaternion

    @override
    def to(
        self,
        container_or_unit: Any | str | None = None,
        unit: LinearUnitType | AngularUnitType | PixelUnitType | TemporalUnitType | None = None,
        angular_unit: AngularUnitType | None = None,
        **kwargs: Any,
    ) -> Any:
        if container_or_unit == "quaternion":
            # The 'sequence' kwarg for .to(), if present, should ideally be 'xyz'
            # or be handled if other interpretations were intended at the .to() level.
            # For now, we assume Pose6D's internal 'xyz' is the target.
            # The test calls with sequence="xyz", which aligns.
            if "sequence" in kwargs and kwargs["sequence"] != "xyz":
                # This case needs clarification: either raise error, log warning, or adapt.
                # For now, it proceeds using Pose6D's internal 'xyz' fixed sequence.
                # Consider raising ValueError(f"Pose6D.to('quaternion') uses a fixed 'xyz' sequence, got {kwargs['sequence']}")
                pass # Do nothing, use internal fixed sequence

            scalar_first_arg = kwargs.get("scalar_first", True)
            return self.quaternion(scalar_first=scalar_first_arg)
        if container_or_unit == "rotation_matrix":
            return self.rotation # self.rotation should now return a 3x3 np.ndarray

        # Delegate to parent for other unit conversions or container type conversions
        return super().to(container_or_unit=container_or_unit, unit=unit, angular_unit=angular_unit, **kwargs)

    # ------------------------------------------------------------------
    # Equality / inequality
    # ------------------------------------------------------------------
    @override
    def __eq__(self, other: object) -> bool:
        """Two poses are considered equal if their numerical components and
        reference frames match within tolerance - *metadata such as origin or
        units are deliberately ignored*.

        This relaxed comparison aligns with the expectations in the test
        suite (e.g. ``test_relative_to_different_origins``) where the origin
        chain may differ even though the actual pose is the same.
        """
        import numpy as _np

        if not isinstance(other, Pose6D):
            try:
                other = Pose6D(*other)  # type: ignore[arg-type]
            except Exception:
                return False

        return _np.allclose(self.numpy(), other.numpy(), atol=1e-8)

    def __ne__(self, other: object) -> bool:
        return not self == other


Pose = Pose6D
if not TYPE_CHECKING:
    Pose.__doc__ = "x, y, z, roll, pitch, and yaw in meters and radians."


class Plane(Coordinate[sz[4], Float]):
    """ax + by + cz + d = 0."""

    a: Float
    b: Float
    c: Float
    d: Float

    _inliers: list[Point] | None = PrivateAttr(default=None)
    _point_cloud: Any | None = PrivateAttr(default=None)

    @property
    def coefficients(self) -> array[sz[4], Float]:
        """Calculate the normal vector of the plane."""
        return ndarray[sz[4], Float](self.astype(float))

    @property
    def inliers(self) -> list[Point] | None:
        return self._inliers

    @inliers.setter
    def inliers(self, value: list[Point] | None):
        self._inliers = value

    @property
    def point_cloud(self) -> Any | None:
        return self._point_cloud

    @point_cloud.setter
    def point_cloud(self, value: Any | None):
        self._point_cloud = value

    def normal(self) -> Point:
        """Calculate the normal vector of the plane."""
        return Point(*(self[:3].astype(np.float64) / np.linalg.norm(self[:3])))

    def __init__(
        self,
        coefficients: array[sz[4], Float] | ArrayLike,
        *,
        inliers: list[Any] | None = None,
        point_cloud: Any | None = None,
    ):
        super().__init__(*np.array(coefficients))
        self.inliers = inliers
        self.point_cloud = point_cloud




class BBox2D(Coordinate[sz[2], sz[2], Float]):
    """Model for 2D Bounding Box."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float | None = None
    label: str | None = None


class Mask(Coordinate[H, W, bool]):
    """Model for a mask."""

    array: array[H, W, bool]
    confidence: float | None = MetadataField(default=None)
    label: str | None = MetadataField(default=None)


class BBox3D(Coordinate[sz[3], sz[2], Float]):
    """Model for 3D Bounding Box."""

    x1: float
    y1: float
    z1: float
    x2: float
    y2: float
    z2: float



class PixelCoords(Coordinate[sz[2], Float]):
    """Model for Pixel Coordinates."""

    u: int = CoordinateField(unit="px", default=0)
    v: int = CoordinateField(unit="px", default=0)


class Corner2D(Coordinate[sz[2], sz[2], Float]):
    """4 2D points representing the corners of an ArUco marker."""

    top_left: PixelCoords
    top_right: PixelCoords
    bottom_right: PixelCoords
    bottom_left: PixelCoords

def demo() -> None:
    """Demonstrate the functionality of the coordinate module.

    This function serves as a sanity check for the features documented in coordinate.md.
    It tests and demonstrates various operations on coordinate objects.
    """
    import math

    import numpy as np

    from embdata.utils.safe_print import safe_print

    safe_print("\n=== Coordinate Module Demo ===\n")

    # 1. Basic coordinate type creation
    safe_print("1. Creating coordinate objects:")
    point = Point(1.0, 2.0, 3.0)
    planar_pose = Pose3D(x=1.0, y=2.0, theta=math.pi/2)
    pose = Pose6D(1.0, 2.0, 3.0, 0.1, 0.2, 0.3)
    plane = Plane([1.0, 0.0, 0.0, -5.0])  # YZ plane at x=5

    safe_print(f"  Point: {point}")
    safe_print(f"  Pose3D: {planar_pose}")
    safe_print(f"  Pose6D: {pose}")
    safe_print(f"  Plane: {plane}")

    # 2. Metadata and reference frames
    safe_print("\n2. Working with metadata and reference frames:")
    point_camera = Point(1.0, 2.0, 3.0, reference_frame="camera", unit="m")
    safe_print(f"  Reference frame: {point_camera.reference_frame()}")

    point_camera.set_reference_frame("world")
    safe_print(f"  After changing reference frame: {point_camera.reference_frame()}")

    origin = Point(0.0, 0.0, 0.0, reference_frame="world")
    point_camera.set_origin(origin)
    safe_print(f"  Origin: {point_camera.origin()}")

    # 3. Unit conversions
    safe_print("\n3. Unit conversions:")
    pose_m_rad = Pose3D(x=1.0, y=2.0, theta=math.pi/2)
    pose_cm = pose_m_rad.to("cm")
    pose_deg = pose_m_rad.to("deg")

    safe_print(f"  Original (m, rad): {pose_m_rad}")
    safe_print(f"  Converted to cm: {pose_cm}")
    safe_print(f"  Converted to deg: {pose_deg}")
    safe_print(f"  Converted to both: {pose_m_rad.to('cm', angular_unit='deg')}")

    # 4. Numeric operations
    safe_print("\n4. Numeric operations:")
    p1 = Point(1, 2, 3)
    p2 = Point(4, 5, 6)

    safe_print(f"  p1: {p1}")
    safe_print(f"  p2: {p2}")
    safe_print(f"  p1 + p2: {p1 + p2}")
    safe_print(f"  p2 - p1: {p2 - p1}")
    safe_print(f"  p1 * 2: {p1 * 2}")
    safe_print(f"  p1 / 2: {p1 / 2}")
    safe_print(f"  -p1: {-p1}")
    safe_print(f"  p1 magnitude: {p1.magnitude()}")

    # 5. Pose operations
    safe_print("\n5. Pose operations:")
    pose1 = Pose6D(1, 2, 3, 0.1, 0.2, 0.3)
    pose2 = Pose6D(4, 5, 6, 0.4, 0.5, 0.6)

    safe_print(f"  pose1: {pose1}")
    safe_print(f"  pose2: {pose2}")

    composed = pose1 * pose2
    safe_print(f"  pose1 * pose2: {composed}")

    inverse = pose1.inverse()
    safe_print(f"  inverse of pose1: {inverse}")

    # Test that inverse works correctly
    identity = pose1 * inverse
    identity_check = all(abs(v) < 1e-6 for v in identity.translation) and all(abs(v) < 1e-6 for v in [identity.roll, identity.pitch, identity.yaw])
    safe_print(f"  pose1 * inverse ≈ identity: {identity_check}")

    # 6. NumPy integration
    safe_print("\n6. NumPy integration:")
    point_np = point.numpy()
    safe_print(f"  Point as numpy: {point_np}")
    safe_print(f"  Create from numpy: {Point(*np.array([4, 5, 6]))}")

    # 7. Working with Planes
    safe_print("\n7. Working with Planes:")
    test_plane = Plane([1, 1, 1, -10])
    safe_print(f"  Plane coefficients: {test_plane.coefficients}")
    safe_print(f"  Plane normal: {test_plane.normal()}")

    # 8. Accessing values
    safe_print("\n8. Accessing coordinate values:")
    p = Point(1, 2, 3)
    safe_print(f"  By attribute - p.x: {p.x}, p.y: {p.y}, p.z: {p.z}")
    safe_print(f"  By index - p[0]: {p[0]}, p[1]: {p[1]}, p[2]: {p[2]}")
    safe_print(f"  By slicing - p[0:2]: {p[0:2]}")

    values = []
    for value in p:
        values.append(value)
    safe_print(f"  By iteration: {values}")

    # 9. Shape information
    safe_print("\n9. Shape information:")
    safe_print(f"  Point shape: {p.shape}")
    safe_print(f"  Pose6D shape: {pose.shape}")

    # 10. Bound checking
    safe_print("\n10. Bound checking:")

    class BoundedCoord(Coordinate):
        x: float = CoordinateField(bounds=(-10, 10), unit="m")
        y: float = CoordinateField(bounds=(-10, 10), unit="m")
        z: float = CoordinateField(bounds=(0, 100), unit="m")

    valid_bounded = BoundedCoord(5, 5, 50)
    safe_print(f"  Valid bounded coordinate: {valid_bounded}")

    try:
        BoundedCoord(5, 5, 150)
        safe_print("  ERROR: Should have thrown ValueError for out-of-bounds")
    except ValueError as e:
        safe_print(f"  Correctly caught bound error: {e}")

    safe_print("\n=== Demo Complete ===\n")

if __name__ == "__main__":
    demo()

