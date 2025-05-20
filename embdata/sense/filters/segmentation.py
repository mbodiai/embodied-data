from __future__ import annotations

from typing import TYPE_CHECKING

import open3d as o3d
from dotenv import load_dotenv
from pydantic import Field

from embdata.coordinate import Plane
from embdata.sample import Sample
from embdata.sense.filters.config import FilterConfig
from embdata.sense.filters.filter import FilterActor

if TYPE_CHECKING:
    from embdata.actor import ConfiguredActorProto

o3dg = o3d.geometry
load_dotenv()

class PlaneSegmentationConfig(FilterConfig):
    """Configuration for Plane Segmentation/Removal Filter."""
    distance_threshold: float = Field(default=0.01, gt=0, description="Max distance a point can be from the plane model to be considered an inlier.")
    ransac_n: int = Field(3, gt=0, description="Number of points to randomly sample for RANSAC plane estimation.")
    num_iterations: int = Field(1000, gt=0, description="Number of RANSAC iterations.")
    invert_selection: bool = Field(True, description="If True, return points *not* in the plane (outliers). If False, return points *in* the plane (inliers).")

class PlaneSegmentationFilter(FilterActor[o3dg.PointCloud, Plane, Sample, bool, PlaneSegmentationConfig]):
    """A filter that segments the largest plane using RANSAC and returns either the plane inliers or outliers."""

    def __init__(self, config: PlaneSegmentationConfig, parent: ConfiguredActorProto | None = None):
        super().__init__(config, parent)


    def act(self, observation: o3dg.PointCloud, **kwargs) -> Plane:
        """Performs plane segmentation and selects points based on config."""
        plane_model, inlier_indices = observation.segment_plane(
            distance_threshold=self.config.distance_threshold,
            ransac_n=self.config.ransac_n,
            num_iterations=self.config.num_iterations,
        )

        # Select points based on the invert_selection flag
        selected_pcd = observation.select_by_index(inlier_indices, invert=self.config.invert_selection)

        return Plane(coefficients=plane_model,
                      inliers=inlier_indices,
                      point_cloud=selected_pcd)


def demo() -> None:
    """Demonstrate PlaneSegmentationFilter usage."""
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
    plane_filter = PlaneSegmentationFilter(config=PlaneSegmentationConfig(
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=1000,
        invert_selection=True,
    ))
    plane = plane_filter.act(observation=point_cloud)
    safe_print(plane.coefficients)


if __name__ == "__main__":
    demo()
