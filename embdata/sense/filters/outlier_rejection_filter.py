from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import open3d as o3d
from pydantic import Field

from embdata.sample import Sample
from embdata.sense.filters.config import FilterConfig
from embdata.sense.filters.filter import FilterActor
from embdata.utils.custom_logger import get_logger

if TYPE_CHECKING:
    from embdata.actor import ConfiguredActorProto

logger = get_logger(__name__)

class OutlierRejectionFilterConfig(FilterConfig):
    """Configuration for Outlier Removal Filter."""
    # Type of outlier removal method to use
    method: Literal["statistical", "radius"] = Field(default="statistical", description="Outlier removal method to use: 'statistical' or 'radius'")

    # Parameters for statistical outlier removal
    nb_neighbors: int = Field(default=20, gt=0, description="Number of neighbors to analyze for each point (for statistical method).")
    std_ratio: float = Field(default=2.0, gt=0, description="Standard deviation ratio threshold (for statistical method).")

    # Parameters for radius outlier removal
    radius: float = Field(default=0.05, gt=0, description="Radius of the sphere to determine if a point has neighbors (for radius method).")
    min_points: int = Field(default=3, ge=1, description="Minimum number of points required within the radius (for radius method).")

class OutlierRejectionFilter(
    FilterActor[
        o3d.geometry.PointCloud,
        o3d.geometry.PointCloud,
        Sample,
        bool,
        OutlierRejectionFilterConfig,
    ],
):
    """Removes outlier points from a point cloud using either:
    1. Statistical method - removes points that are further away from their neighbors compared to the average
    2. Radius method - removes points that have fewer than min_points neighbors within a given radius.
    """

    def __init__(self, config: OutlierRejectionFilterConfig, parent: ConfiguredActorProto | None = None):
        super().__init__(config, parent)

    def act(self, observation: o3d.geometry.PointCloud, **kwargs) -> o3d.geometry.PointCloud:
        """Removes outliers from the point cloud using the configured method."""
        if not observation.has_points():
            logger.warning(f"[{self.name}] Input point cloud has no points. Skipping outlier rejection.")
            return observation

        # Choose the outlier removal method based on config
        if self.config.method == "statistical":
            logger.info(f"[{self.name}] Applying statistical outlier removal (nb_neighbors={self.config.nb_neighbors}, std_ratio={self.config.std_ratio})")
            filtered_pcd, _ = observation.remove_statistical_outlier(
                nb_neighbors=self.config.nb_neighbors,
                std_ratio=self.config.std_ratio,
            )
        elif self.config.method == "radius":
            logger.info(f"[{self.name}] Applying radius outlier removal (radius={self.config.radius}, min_points={self.config.min_points})")
            filtered_pcd, _ = observation.remove_radius_outlier(
                nb_points=self.config.min_points,
                radius=self.config.radius,
            )
        else:
            # This shouldn't happen due to Literal type, but just in case
            logger.warning(f"[{self.name}] Unknown outlier removal method: {self.config.method}. Returning original point cloud.")
            return observation

        return filtered_pcd


def demo() -> None:
    """Demonstrate OutlierRejectionFilter usage."""
    from importlib.resources import files

    from embdata.sense.camera_config import CAMERA_D435_1
    from embdata.sense.depth import Depth
    from embdata.sense.image import Image
    from embdata.utils.safe_print import safe_print

    RGB_FILE = str(files("embdata")/"resources"/"color_image.png")
    DEPTH_FILE = str(files("embdata")/"resources"/"depth_image.png")

    rgb_image = Image(path=RGB_FILE, encoding="png", mode="RGB")
    depth = Depth(path=DEPTH_FILE,
                  encoding="png",
                  mode="I",
                  size=(1280, 720),
                  camera=CAMERA_D435_1,
                  unit="mm",
                  rgb=rgb_image)

    point_cloud = depth.to_pointcloud(camera=CAMERA_D435_1, backend="open3d")
    safe_print(f"Original point cloud: {len(point_cloud.points)} points")

    # Example using statistical outlier removal
    stat_filter = OutlierRejectionFilter(config=OutlierRejectionFilterConfig(
        method="statistical",
        nb_neighbors=20,
        std_ratio=2.0,
    ))
    filtered_cloud_stat = stat_filter.act(observation=point_cloud)
    safe_print(f"Statistical outlier removal result: {len(filtered_cloud_stat.points)} points")

    # Example using radius outlier removal
    radius_filter = OutlierRejectionFilter(config=OutlierRejectionFilterConfig(
        method="radius",
        radius=0.05,
        min_points=3,
    ))
    filtered_cloud_radius = radius_filter.act(observation=point_cloud)
    safe_print(f"Radius outlier removal result: {len(filtered_cloud_radius.points)} points")

if __name__ == "__main__":
    demo()
