# Copyright (c) 2024 Mbodi AI
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


from typing_extensions import Any, Literal, get_args

LinearUnit = Literal["m", "cm", "mm", "km", "in", "ft", "yd", "mi"]
AngularUnit = Literal["rad", "deg", "radians", "degrees"]
AngularUnitType = Literal["rad", "deg", "radians", "degrees"]
LinearUnitType = Literal["m", "cm", "mm", "km", "in", "ft", "yd", "mi"]
PixelUnitType = Literal["px", "pixel", "pixels"]
TemporalUnitType = Literal[
    "s",
    "ms",
    "us",
    "ns",
    "sec",
    "min",
    "hour",
    "day",
    "week",
    "month",
    "year",
    "seconds",
    "minutes",
    "hours",
    "days",
    "weeks",
    "months",
    "years",
]

LinearLabel = Literal["x", "y", "z", "length", "width", "height", "radius", "l", "w", "h", "r"]
AngularLabel = Literal["roll", "pitch", "yaw", "theta", "phi", "psi", "delta", "gamma", "beta", "alpha"]

def isunit(unit: Any, unit_type: Any | None = None) -> bool:
    unit_types = get_args(unit_type) if unit_type else [get_args(u) for u in [LinearUnitType, AngularUnitType, TemporalUnitType]]
    return unit in unit_types


def islabel(label: Any, label_type: Any | None = None) -> bool:
    label_types = get_args(label_type) if label_type else [get_args(l) for l in [LinearLabel, AngularLabel]]
    return label in label_types
