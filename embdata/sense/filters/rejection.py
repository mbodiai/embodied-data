from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import open3d as o3d
from pydantic import Field

from embdata.coordinate import Plane
from embdata.sample import Sample
from embdata.sense.filters.config import FilterConfig
from embdata.sense.filters.filter import FilterActor
from embdata.utils.custom_logger import get_logger

if TYPE_CHECKING:
    from embdata.actor import ConfiguredActorProto

o3dg = o3d.geometry
logger = get_logger(__name__)


class PlaneRejectionConfig(FilterConfig):
    """Configuration for rejecting points based on a plane from a previous step."""
    keep_positive_side: bool = Field(default=True, description="If True, keeps points where ax+by+cz+d > distance_offset. If False, keeps points where < -distance_offset.")
    keep_min_distance: float = Field(0.0, description="Max distance from the plane. Points must be further than this max_distance (in the chosen direction) to be kept.")


class PlaneRejectionFilter(FilterActor[Plane, o3dg.PointCloud, Sample, bool, PlaneRejectionConfig]):
    """Rejects points from the point cloud contained within the input Plane object.

    Rejects points based on their distance and side relative to the plane defined in the input Plane object.
    """

    def __init__(self, config: PlaneRejectionConfig, parent: ConfiguredActorProto | None = None):
        super().__init__(config, parent)

    def act(
        self,
        observation: Plane,
        **kwargs,
    ) -> o3dg.PointCloud:
        """Filters points from observation.point_cloud based on signed distance to observation.coefficients."""
        point_cloud = observation.point_cloud
        coefficients = observation.coefficients
        if not point_cloud or not point_cloud.has_points():
            msg = f"[{self.name}] Input point cloud (from Plane object) has no points. Cannot perform plane rejection."
            raise ValueError(msg)

        if coefficients is None or len(coefficients) != 4:
            msg = f"[{self.name}] Invalid plane coefficients ({coefficients}) from Plane object. Cannot perform rejection."
            raise ValueError(msg)

        a, b, c, d = coefficients
        distances = point_cloud.points @ np.array([a, b, c]) + d

        if self.config.keep_positive_side:
            keep_indices = cast(list[int], np.where(distances > self.config.keep_min_distance)[0])
        else:
            keep_indices = cast(list[int], np.where(distances < -self.config.keep_min_distance)[0])

        selected_pcd: o3dg.PointCloud = point_cloud.select_by_index(keep_indices)
        return selected_pcd


def demo() -> None:
    """Demonstrate PlaneRejectionFilter usage."""
    from importlib.resources import files

    from embdata.sense.camera_config import CAMERA_D435_1
    from embdata.sense.depth import Depth
    from embdata.sense.filters.segmentation import (
        PlaneSegmentationConfig,
        PlaneSegmentationFilter,
    )
    from embdata.sense.image import Image
    from embdata.utils.safe_print import safe_print

    RGB_FILE = str(files("embdata")/"resources"/"color_image.png")
    DEPTH_FILE = str(files("embdata")/"resources"/"depth_image.png")

    rgb_image = Image(path=RGB_FILE, encoding="png", mode="RGB")
    depth = Depth(path=DEPTH_FILE, encoding="png", mode="I", size=(1280, 720), camera=CAMERA_D435_1, unit="mm", rgb=rgb_image)
    point_cloud = depth.to_pointcloud(camera=CAMERA_D435_1, backend="open3d")

    plane_seg_config = PlaneSegmentationConfig(
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=100,
        invert_selection=False,
    )

    plane_segmenter = PlaneSegmentationFilter(config=plane_seg_config)
    plane_object_output: Plane = plane_segmenter.act(observation=point_cloud)

    if plane_object_output.coefficients is None or len(plane_object_output.coefficients) != 4:
        safe_print("Plane segmentation did not produce a valid plane or inliers. Skipping rejection test.")
    else:
        safe_print(f"Plane segmented: {plane_object_output.coefficients}, with {len(plane_object_output.point_cloud.points)} points in its PCD.")
        plane_rejection_config = PlaneRejectionConfig(
            name="plane_rejector_test",
            keep_positive_side=True,
            keep_min_distance=0.01,
        )
        plane_rejection_filter = PlaneRejectionFilter(config=plane_rejection_config)

        filtered_cloud = plane_rejection_filter.act(observation=plane_object_output)
        safe_print(f"Plane rejection resulted in {len(filtered_cloud.points)} points.")


if __name__ == "__main__":
    demo()
