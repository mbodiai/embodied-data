# Dynamic Coordinate Objects

## Core Concepts

The coordinate system in `embdata` is built around several core concepts:

- **Coordinates**: Base class for all coordinate types, supporting generic N-dimensional coordinates
- **Reference Frames**: Each coordinate exists within a specific reference frame
- **Origin**: Coordinates can have an origin from which they are measured
- **Units**: Linear, angular, pixel, and temporal units are supported with built-in conversions

## Basic Coordinate Types

The package provides several specialized coordinate types:

```python
from embdata.coordinate import Point, Pose3D, Pose6D, Plane

# 3D point (x, y, z)
point = Point(1.0, 2.0, 3.0)

# 3D planar pose (x, y, theta)
planar_pose = Pose3D(x=1.0, y=2.0, theta=0.5)

# 6D pose (x, y, z, roll, pitch, yaw)
pose = Pose6D(1.0, 2.0, 3.0, 0.1, 0.2, 0.3)

# Plane (ax + by + cz + d = 0)
plane = Plane([1.0, 0.0, 0.0, -5.0])  # YZ plane at x=5
```

## Metadata and Reference Frames

Each coordinate carries metadata about its reference frame, origin, and units:

```python
# Create a point with specific metadata
point = Point(1.0, 2.0, 3.0, reference_frame="camera", unit="m")

# Get and set reference frame
print(point.reference_frame())  # "camera"
point.set_reference_frame("world")

# Set origin - another coordinate in the same system
origin = Point(0.0, 0.0, 0.0, reference_frame="world")
point.set_origin(origin)

# Access origin
print(point.origin())  # Point(x=0.0, y=0.0, z=0.0)
```

## Units and Conversions

Coordinates support unit conversions for both linear and angular dimensions:

```python
from embdata.coordinate import Pose3D
import math

# Create a pose in meters and radians
pose = Pose3D(x=1.0, y=2.0, theta=math.pi/2)

# Convert to centimeters (linear units)
pose_cm = pose.to("cm")
print(pose_cm)  # Pose3D(x=100.0, y=200.0, theta=1.5707963267948966)

# Convert to degrees (angular units)
pose_deg = pose.to("deg")
print(pose_deg)  # Pose3D(x=1.0, y=2.0, theta=90.0)

# Convert both
pose_cm_deg = pose.to("cm", angular_unit="deg")
print(pose_cm_deg)  # Pose3D(x=100.0, y=200.0, theta=90.0)
```

## CoordinateField and Bounds Checking

The `CoordinateField` function allows for defining fields with specific metadata and bounds:

```python
from embdata.coordinate import Coordinate, CoordinateField

class BoundedCoordinate(Coordinate):
    x: float = CoordinateField(bounds=(-10, 10), unit="m")
    y: float = CoordinateField(bounds=(-10, 10), unit="m")
    z: float = CoordinateField(bounds=(0, 100), unit="m")

# This works
valid_coord = BoundedCoordinate(5, 5, 50)

# This raises a ValueError (z out of bounds)
try:
    invalid_coord = BoundedCoordinate(5, 5, 150)
except ValueError as e:
    print(f"Validation error: {e}")
```

## Numeric Operations

Coordinates support various arithmetic operations:

```python
from embdata.coordinate import Point

p1 = Point(1, 2, 3)
p2 = Point(4, 5, 6)

# Addition
p3 = p1 + p2  # ndarray([5, 7, 9])

# Subtraction
p4 = p2 - p1  # ndarray([3, 3, 3])

# Scalar multiplication
p5 = p1 * 2  # ndarray([2, 4, 6])

# Division
p6 = p1 / 2  # ndarray([0.5, 1, 1.5])

# Negation
p7 = -p1  # ndarray([-1, -2, -3])
```

## Pose Operations

The `Pose6D` class provides special operations for transformations:

```python
from embdata.coordinate import Pose6D
import numpy as np

# Create two poses
pose1 = Pose6D(1, 2, 3, 0.1, 0.2, 0.3)
pose2 = Pose6D(4, 5, 6, 0.4, 0.5, 0.6)

# Compose poses (pose1 * pose2)
composed = pose1 * pose2

# Get inverse of a pose
inverse = pose1.inverse()

# Convert to different representations
quat = pose1.to("quaternion")  # Get quaternion representation
rot_matrix = pose1.to("rotation_matrix")  # Get rotation matrix
```

## NumPy and Tensor Integration

Coordinates integrate with NumPy and PyTorch:

```python
from embdata.coordinate import Point
import numpy as np

point = Point(1, 2, 3)

# Convert to numpy array
np_array = point.numpy()
print(np_array)  # array([1., 2., 3.])

# Create from numpy array
point2 = Point(*np.array([4, 5, 6]))

# Convert to tensor (if torch is available)
tensor = point.__tensor__()
```

## Working with Planes

The `Plane` class provides functionality for plane equations:

```python
from embdata.coordinate import Plane, Point

# Create a plane with coefficients [a, b, c, d]
# For equation ax + by + cz + d = 0
plane = Plane([1, 1, 1, -10])  # Plane with normal [1,1,1] at distance 10/sqrt(3) from origin

# Get plane normal
normal = plane.normal()
print(normal)  # Point with normalized coefficients

# Get all coefficients
coeffs = plane.coefficients
print(coeffs)  # array([1., 1., 1., -10.])
```

## Advanced Features

### Accessing Values

Coordinates can be accessed like arrays or by attribute names:

```python
from embdata.coordinate import Point

p = Point(1, 2, 3)

# By attribute
print(p.x, p.y, p.z)  # 1.0 2.0 3.0

# By index
print(p[0], p[1], p[2])  # 1.0 2.0 3.0

# Slicing
print(p[0:2])  # array([1., 2.])

# Iteration
for value in p:
    print(value)  # 1.0, 2.0, 3.0
```

### Shape Information

Coordinates track their shape:

```python
from embdata.coordinate import Point, Pose6D

p = Point(1, 2, 3)
pose = Pose6D(1, 2, 3, 0.1, 0.2, 0.3)

print(p.shape)  # (3,)
print(pose.shape)  # (6,)
```

These examples demonstrate the powerful and flexible coordinate system provided by the `embdata` package, supporting a wide range of geometric operations, transformations, and conversions.

## Demo Function

For a complete demonstration of the coordinate functionality, the package includes a demo function:

```python
from embdata.coordinate import demo

# Run the demo to see working examples of all coordinate features
demo()
```

This will output a comprehensive demonstration of coordinate creation, transformations, operations, and more. It's a great way to see how the various features work together in practice.
