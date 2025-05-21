# Affine Transformations

## Overview

The transformation classes are built on the foundation of the `Coordinate` system and provide:

- Representation of transformations with rotation and translation components
- Composition of transformations (combining multiple transforms)
- Inverse operations
- Point transformation
- Conversion between different representations (matrices, poses, quaternions)

## Basic Usage

### Affine2D Transformations

`Affine2D` represents 2D transformations with a 2×2 rotation matrix and a 2D translation vector:

```python
from embdata.geometry.affine import Affine2D
import numpy as np

# Create an identity transformation
transform = Affine2D()
print(transform.rotation)     # Identity rotation matrix
print(transform.translation)  # Zero translation vector

# Create with specific rotation and translation
rotation = np.array([[0, -1], [1, 0]])  # 90-degree rotation
translation = np.array([1, 2])
transform = Affine2D(rotation=rotation, translation=translation)

# Get as homogeneous transformation matrix (3×3)
matrix = transform.matrix()
print(matrix)
# Output:
# [[ 0. -1.  1.]
#  [ 1.  0.  2.]
#  [ 0.  0.  1.]]
```

### Affine3D Transformations

`Affine3D` represents 3D transformations with a 3×3 rotation matrix and a 3D translation vector:

```python
from embdata.geometry.affine import Affine3D
import numpy as np

# Create an identity transformation
transform = Affine3D()
print(transform.rotation)     # Identity rotation matrix
print(transform.translation)  # Zero translation vector

# Create with specific rotation and translation
rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # Rotation around Z-axis
translation = np.array([1, 2, 3])
transform = Affine3D(rotation=rotation, translation=translation)

# Get as homogeneous transformation matrix (4×4)
matrix = transform.matrix()
print(matrix)
# Output:
# [[ 0. -1.  0.  1.]
#  [ 1.  0.  0.  2.]
#  [ 0.  0.  1.  3.]
#  [ 0.  0.  0.  1.]]
```

## Creating Transformations

Affine transformations can be created from various sources:

### From Pose Objects

```python
from embdata.coordinate import Pose
from embdata.geometry.affine import Affine3D

# Create from individual components
transform = Affine3D.from_pose(x=1, y=2, z=3, roll=0.1, pitch=0.2, yaw=0.3)
print(transform.rotation)
print(transform.translation)  # [1. 2. 3.]

# Create from a Pose object
pose = Pose(1, 2, 3, 0.1, 0.2, 0.3)
transform = Affine3D.from_pose(pose)
```

### From Rotation Representations

```python
import numpy as np
from embdata.geometry.affine import Affine3D

# From roll, pitch, yaw angles
rpy = np.array([0.1, 0.2, 0.3])  # roll, pitch, yaw in radians
transform = Affine3D.from_rpy(rpy)

# From quaternion (x, y, z, w format)
quat = np.array([0.1, 0.2, 0.3, 0.9])  # Normalized automatically
transform = Affine3D.from_quat_pose(quat, translation=[1, 2, 3])

# From homogeneous transformation matrix
matrix = np.eye(4)  # 4×4 identity matrix
matrix[:3, 3] = [1, 2, 3]  # Set translation
transform = Affine3D.from_array(matrix)
```

## Transformation Operations

### Composition of Transformations

Transformations can be combined using the `@` operator (matrix multiplication):

```python
from embdata.geometry.affine import Affine3D
import numpy as np

# Define two transformations
transform1 = Affine3D(
    rotation=np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    translation=np.array([1, 2, 3])
)

transform2 = Affine3D(
    rotation=np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
    translation=np.array([4, 5, 6])
)

# Compose transformations (transform1 followed by transform2)
composed = transform1 @ transform2
print(composed.rotation)
# Output:
# [[0 0 1]
#  [1 0 0]
#  [0 1 0]]
print(composed.translation)
# Output:
# [-4  6  9]
```

### Inverse Transformations

Computing the inverse of a transformation:

```python
# Create a transformation
transform = Affine3D.from_pose(x=1, y=2, z=3, roll=0.1, pitch=0.2, yaw=0.3)

# Compute the inverse of a transformation
inverse = transform.inverse()

# Verify: transform @ inverse ≈ identity
identity = transform @ inverse
is_identity = np.allclose(identity.rotation, np.eye(3), atol=1e-6) and np.allclose(identity.translation, np.zeros(3), atol=1e-6)
print(f"Is identity: {is_identity}")  # True
```

## Transforming Points

Apply transformations to points:

```python
import numpy as np
from embdata.geometry.affine import Affine3D

# Create a transformation
transform = Affine3D.from_pose(x=1, y=2, z=3, roll=0.1, pitch=0.2, yaw=0.3)

# Transform a single point
point = np.array([1, 2, 3])
transformed_point1 = transform.transform_points(point)
# Or use the @ operator
transformed_point2 = transform @ point

# Results are identical
print(np.allclose(transformed_point1, transformed_point2))  # True

# Transform multiple points at once (batch transformation)
points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
transformed_points = transform.transform_points(points)
```

## Converting Between Representations

### To Pose

Convert an affine transformation back to a pose representation:

```python
from embdata.geometry.affine import Affine3D

# Create a transformation
transform = Affine3D.from_pose(x=1, y=2, z=3, roll=0.1, pitch=0.2, yaw=0.3)

# Convert back to a Pose object
pose = transform.pose()
print(pose)  # Pose6D with x, y, z, roll, pitch, yaw
```

### To Quaternion

Convert the rotation component to a quaternion representation:

```python
# Get quaternion representation (x, y, z, w format)
quat = transform.quaternion()
print(quat)  # [0.0342708  0.10602051 0.14357218 0.98334744]
```

### Extracting RPY

Extract roll, pitch, yaw angles directly from a rotation matrix:

```python
from embdata.geometry.utils import rotation_matrix_to_rpy

# Extract roll, pitch, yaw from rotation matrix
rpy = rotation_matrix_to_rpy(transform.rotation)
print(rpy)  # [0.1, 0.2, 0.3]
```

## Integration with Coordinate System

Affine transformations are built on the `Coordinate` class, enabling them to use reference frames and other coordinate features:

```python
from embdata.geometry.affine import Affine3D

# Create a transformation with a reference frame
transform = Affine3D.from_pose(x=1, y=2, z=3, roll=0.1, pitch=0.2, yaw=0.3)
transform.set_reference_frame("camera")

# The resulting pose inherits the reference frame
pose = transform.pose()
print(pose.reference_frame())  # "camera"
```

## Practical Examples

### Camera Extrinsics

Represent a camera's position and orientation in world space:

```python
import math
from embdata.geometry.affine import Affine3D
import numpy as np

# Camera is at position [2, 1, 0.5] looking along the X-axis
camera_to_world = Affine3D.from_pose(x=2, y=1, z=0.5, roll=0, pitch=0, yaw=math.pi/2)

# Transform a point from camera coordinates to world coordinates
point_camera = np.array([0.5, 0, 0])  # Point 0.5m in front of camera
point_world = camera_to_world @ point_camera
print(point_world)  # [2.  1.5 0.5]

# Transform from world to camera
world_to_camera = camera_to_world.inverse()
point_in_camera_frame = world_to_camera @ point_world
print(np.allclose(point_camera, point_in_camera_frame))  # True
```

### Robot Kinematics

Represent a robot's end-effector position relative to its base:

```python
import math
from embdata.geometry.affine import Affine3D
import numpy as np

# Base to end-effector transform
base_to_end = Affine3D.from_pose(x=0.5, y=0, z=0.8, roll=0, pitch=math.pi/2, yaw=0)

# Transform a point in end-effector frame to base frame
point_end = np.array([0.1, 0, 0])  # 10cm in front of end-effector
point_base = base_to_end @ point_end
print(point_base)  # [0.5 0.  0.7]

# Define a transformation from world to base
world_to_base = Affine3D.from_pose(x=1, y=1, z=0, roll=0, pitch=0, yaw=math.pi/4)

# Combine transformations to get end-effector in world frame
world_to_end = world_to_base @ base_to_end
point_world = world_to_end @ point_end
print(point_world)  # [1.35355339 1.35355339 0.7]
```

## Demo Function

The `embdata.geometry.affine` module includes a comprehensive demo function that showcases all the functionality described in this document. You can run it directly to see the transformations in action:

```python
from embdata.geometry.affine import demo

# Run the demo to see all functionality in action
demo()
```

This will output a step-by-step demonstration of all the affine transformation capabilities, including creating transformations, composing them, computing inverses, transforming points, and practical examples.
