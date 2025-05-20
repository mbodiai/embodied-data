from __future__ import annotations

from importlib.resources import files
from typing import TYPE_CHECKING

import numpy as np
import open3d as o3d
from pydantic import Field

from embdata.sample import Sample
from embdata.sense.filters.config import FilterConfig
from embdata.sense.filters.filter import FilterActor
from embdata.utils.custom_logger import get_logger
from embdata.utils.safe_print import safe_print

if TYPE_CHECKING:
    from embdata.actor import ConfiguredActorProto

logger = get_logger(__name__)

class ClusterConfig(FilterConfig):
    """Configuration for DBSCAN Clustering Filter."""
    eps: float = Field(default=0.01, gt=0, description="Maximum distance between points to be considered neighbors.")
    min_points: int = Field(default=10, gt=0, description="Minimum number of points required to form a dense region (core point).")
    print_progress: bool = Field(default=False, description="If True, print progress messages during clustering.")
    return_largest_cluster: bool = Field(default=True, description="If True, return the largest cluster.")

class ClusterFilter(FilterActor[o3d.geometry.PointCloud, o3d.geometry.PointCloud, Sample, bool, ClusterConfig]):
    """Applies DBSCAN clustering to a point cloud.

    Can return either the largest cluster or all non-noise clusters based on configuration.
    Points labeled as noise (-1) are removed.
    """

    def __init__(self, config: ClusterConfig, parent: ConfiguredActorProto | None = None):
        super().__init__(config, parent)

    def act(self, observation: o3d.geometry.PointCloud, **kwargs) -> o3d.geometry.PointCloud:
        """Performs DBSCAN clustering.

        Returns the largest cluster if config.return_largest_cluster is True.
        Otherwise, returns all points belonging to any cluster (non-noise).
        """
        # Perform DBSCAN clustering
        labels = np.array(observation.cluster_dbscan(
            eps=self.config.eps,
            min_points=self.config.min_points,
            print_progress=self.config.print_progress,
        ))

        # Check if any non-noise points exist or if the input cloud was empty
        if labels.size == 0 or labels.max() < 0:
            logger.info(f"[{self.name}] DBSCAN found no clusters or only noise. Returning empty cloud.")
            return o3d.geometry.PointCloud()

        if self.config.return_largest_cluster:
            # Find the largest cluster (excluding noise)
            non_noise_labels = labels[labels != -1]
            # At this point, non_noise_labels.size > 0 is guaranteed because labels.max() >= 0

            unique_cluster_labels, counts = np.unique(non_noise_labels, return_counts=True)

            largest_cluster_idx = np.argmax(counts)
            largest_cluster_label = unique_cluster_labels[largest_cluster_idx]
            largest_cluster_size = counts[largest_cluster_idx]
            logger.debug(f"[{self.name}] Returning largest cluster (label: {largest_cluster_label}) with {largest_cluster_size} points.")
            indices_to_keep = np.where(labels == largest_cluster_label)[0]
        else:
            # Return all points belonging to any cluster (i.e., not noise)
            indices_to_keep = np.where(labels != -1)[0]
            # At this point, indices_to_keep.size > 0 is guaranteed because labels.max() >= 0
            num_distinct_clusters = np.unique(labels[labels != -1]).size
            logger.debug(f"[{self.name}] Returning all {num_distinct_clusters} cluster(s) (total {indices_to_keep.size} points).")

        return observation.select_by_index(indices_to_keep)


def demo() -> None:

    from embdata.sense.camera_config import CAMERA_D435_1
    from embdata.sense.depth import Depth
    from embdata.sense.image import Image

    RGB_FILE = str(files("embdata")/"resources"/"color_image.png")
    DEPTH_FILE = str(files("embdata")/"resources"/"depth_image.png")
    CAMERA = CAMERA_D435_1


    rgb_image = Image(path=RGB_FILE, encoding="png", mode="RGB")
    depth = Depth(path=DEPTH_FILE,
                    encoding="png",
                    mode="I",
                    size=(1280, 720), # Example size, adjust if needed
                    camera=CAMERA,
                    unit="mm",
                    rgb=rgb_image)
    point_cloud = depth.to_pointcloud(camera=CAMERA_D435_1, backend="open3d")
    if not point_cloud.has_points():
        safe_print("[Error] Initial point cloud is empty. Check image paths and depth data.")
    safe_print(f"Initial point cloud has {len(point_cloud.points)} points.")

    # Scenario 1: Return only the largest cluster
    config_largest_only = ClusterConfig(
        eps=0.05,
        min_points=20,
        print_progress=True,
        return_largest_cluster=True,
    )
    cluster_filter_largest = ClusterFilter(config=config_largest_only)
    filtered_cloud_largest = cluster_filter_largest.act(observation=point_cloud)
    safe_print(f"Scenario 1 (Largest Cluster Only): Number of points = {len(filtered_cloud_largest.points)}")

    # Scenario 2: Return all clusters (non-noise)
    config_all_clusters = ClusterConfig(
        eps=0.05, # Keep same eps as above for comparison
        min_points=20, # Keep same min_points
        print_progress=True,
        return_largest_cluster=False,
    )
    cluster_filter_all = ClusterFilter(config=config_all_clusters)
    filtered_cloud_all = cluster_filter_all.act(observation=point_cloud)
    safe_print(f"Scenario 2 (All Clusters): Number of points = {len(filtered_cloud_all.points)}")

if __name__ == "__main__":
    demo()
