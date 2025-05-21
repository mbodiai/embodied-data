# Filters

## Overview

Filters in `embdata` follow a common pattern:
- They are implemented as `FilterActor` subclasses
- Each filter has a corresponding configuration class
- They process input data and return processed output
- They can be combined into filter pipelines

## Working with Point Clouds

Most filters in `embdata` operate on point clouds. You can create point clouds in several ways:

### Converting Depth Images to Point Clouds

A common source of point cloud data is depth cameras. The `embdata` package provides tools to convert depth images to point clouds:

```python
from importlib.resources import files
from embdata.sense.camera_config import CAMERA_D435_1
from embdata.sense.depth import Depth
from embdata.sense.image import Image

# Load RGB and depth images
RGB_FILE = str(files("embdata")/"resources"/"color_image.png")
DEPTH_FILE = str(files("embdata")/"resources"/"depth_image.png")

# Load images
rgb_image = Image(path=RGB_FILE, encoding="png", mode="RGB")
depth = Depth(
    path=DEPTH_FILE,
    encoding="png",
    mode="I",
    size=(1280, 720),
    camera=CAMERA_D435_1,
    unit="mm",
    rgb=rgb_image
)

# Convert to point cloud
point_cloud = depth.to_pointcloud(camera=CAMERA_D435_1, backend="open3d")
```

### Loading Point Clouds Directly

You can also load point clouds directly from files:

```python
import open3d as o3d

# Load point cloud from file
point_cloud = o3d.io.read_point_cloud("example.ply")
```

## Plane Segmentation Filter

The `PlaneSegmentationFilter` identifies and segments the largest plane in a point cloud using RANSAC. It returns a `Plane` object containing the segmented plane's coefficients, inlier points, and the filtered point cloud.

### Configuration

Configure the filter using `PlaneSegmentationConfig`:

```python
from embdata.sense.filters.segmentation import PlaneSegmentationConfig

config = PlaneSegmentationConfig(
    distance_threshold=0.01,  # Max distance (m) from plane to be considered an inlier
    ransac_n=3,               # Number of points to sample for plane estimation
    num_iterations=1000,      # Number of RANSAC iterations
    invert_selection=True,    # If True, return points NOT in the plane
)
```

Parameters:
- `distance_threshold`: Maximum distance a point can be from the plane model to be considered an inlier (in meters)
- `ransac_n`: Number of points to randomly sample for RANSAC plane estimation (minimum 3)
- `num_iterations`: Number of RANSAC iterations to perform
- `invert_selection`: If True, return points not in the plane (outliers); if False, return points in the plane (inliers)

### Complete Example

This example shows the complete workflow from loading depth images to segmenting a plane:

```python
from importlib.resources import files
from embdata.sense.camera_config import CAMERA_D435_1
from embdata.sense.depth import Depth
from embdata.sense.image import Image
from embdata.sense.filters.segmentation import PlaneSegmentationFilter, PlaneSegmentationConfig
from embdata.utils.safe_print import safe_print

# Step 1: Load RGB and depth images
RGB_FILE = str(files("embdata")/"resources"/"color_image.png")
DEPTH_FILE = str(files("embdata")/"resources"/"depth_image.png")

rgb_image = Image(path=RGB_FILE, encoding="png", mode="RGB")
depth = Depth(
    path=DEPTH_FILE,
    encoding="png",
    mode="I",
    size=(1280, 720),
    camera=CAMERA_D435_1,
    unit="mm",
    rgb=rgb_image
)

# Step 2: Convert depth to point cloud
point_cloud = depth.to_pointcloud(camera=CAMERA_D435_1, backend="open3d")

# Step 3: Create and configure the plane segmentation filter
plane_filter = PlaneSegmentationFilter(
    config=PlaneSegmentationConfig(
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=1000,
        invert_selection=True,  # Remove the plane points
    )
)

# Step 4: Apply the filter
plane_result = plane_filter.act(observation=point_cloud)

# Step 5: Access the results
safe_print(f"Plane coefficients: {plane_result.coefficients}")
safe_print(f"Inlier points: {len(plane_result.inliers)}")
filtered_point_cloud = plane_result.point_cloud
safe_print(f"Filtered point cloud size: {len(filtered_point_cloud.points)}")
```

### Visualizing Results

You can visualize the segmentation results using Open3D:

```python
import open3d as o3d

# Original point cloud
o3d.visualization.draw_geometries([point_cloud])

# Filtered point cloud (with plane removed)
o3d.visualization.draw_geometries([plane_result.point_cloud])

# Visualize just the plane points
inlier_cloud = point_cloud.select_by_index(plane_result.inliers)
inlier_cloud.paint_uniform_color([1, 0, 0])  # Color the plane points red
o3d.visualization.draw_geometries([inlier_cloud])
```
