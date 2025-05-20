# Dynamic Coordinate Objects

The `Coordinate` class in the `embdata` package allows for creating dynamic coordinate objects with flexible attributes. This document provides examples of how to use these objects, convert between different types, and check bounds.

## Creating Dynamic Coordinate Objects

You can create coordinate objects with different attributes on the fly:

```python
from embdata.coordinate import Coordinate

# Create a 2D point
point_2d = Coordinate(x=1, y=2)

# Create a 3D point
point_3d = Coordinate(x=3, y=4, z=5)

# Create a custom coordinate
custom_coord = Coordinate(latitude=40.7128, longitude=-74.0060)
```

## Converting Between Coordinate Types

You can easily convert between different coordinate types:

```python
# Convert 2D point to 3D point
point_3d = Coordinate(x=point_2d.x, y=point_2d.y, z=0)

# Convert custom coordinate to 2D point
point_2d = Coordinate(x=custom_coord.longitude, y=custom_coord.latitude)
```

## Checking Bounds

The `Coordinate` class supports bound checking for its attributes using `CoordinateField`:

```python
from embdata.coordinate import Coordinate, CoordinateField

class BoundedCoordinate(Coordinate):
    x: float = CoordinateField(bounds=(-10, 10))
    y: float = CoordinateField(bounds=(-10, 10))

# This will work
valid_coord = BoundedCoordinate(x=5, y=5)

# This will raise a ValueError
try:
    invalid_coord = BoundedCoordinate(x=15, y=5)
except ValueError as e:
    print(f"Validation error: {e}")
```

The `validate_bounds` method in the `Coordinate` class checks the bounds for each field defined as a `CoordinateField`. If a value is outside the specified bounds, it raises a `ValueError` with a descriptive message.

## Using Different Coordinate Types

You can create specific coordinate types for different use cases:

```python
class Point2D(Coordinate):
    x: float
    y: float

class Point3D(Coordinate):
    x: float
    y: float
    z: float

class GPSCoordinate(Coordinate):
    latitude: float
    longitude: float

# Usage
p2d = Point2D(x=1, y=2)
p3d = Point3D(x=3, y=4, z=5)
gps = GPSCoordinate(latitude=40.7128, longitude=-74.0060)
```

These examples demonstrate the flexibility and power of the `Coordinate` class in creating and working with various types of coordinate objects.
