from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import open3d as o3d
from pydantic import Field

from embdata.sample import Sample
from embdata.sense.filters.config import FilterConfig
from embdata.sense.filters.filter import FilterActor
from embdata.utils.custom_logger import get_logger

if TYPE_CHECKING:
    from embdata.actor import ConfiguredActorProto

logger = get_logger(__name__)


class SearchMode(str, Enum):
    """Defines which point cloud's points are kept in the result."""
    KEEP_SOURCE = "keep_source"  # Keep points from source point cloud
    KEEP_TARGET = "keep_target"  # Keep points from target point cloud


class KDTreeSearchMethod(str, Enum):
    """Available KDTree search methods in Open3D."""
    KNN = "knn"           # K-nearest neighbors search
    RADIUS = "radius"     # Radius search (all points within a radius)
    HYBRID = "hybrid"     # Hybrid search (k-nearest neighbors within a radius)


class KDTreeConfig(FilterConfig):
    """Configuration for KDTree-based point cloud filtering."""
    # Which points to keep
    search_mode: SearchMode = Field(default=SearchMode.KEEP_SOURCE,
                                   description="Which points to keep in the result.")

    # Which search method to use
    search_method: KDTreeSearchMethod = Field(default=KDTreeSearchMethod.KNN,
                                             description="Which KDTree search method to use.")

    # Parameters for each search method
    max_distance: float = Field(default=0.025, gt=0,
                              description="Maximum distance threshold for considering a point a neighbor.")
    knn_k: int = Field(default=1, ge=1,
                     description="Number of nearest neighbors to search for in KNN and hybrid search.")

    # Parameters for hybrid search only
    hybrid_sorted: bool = Field(default=True,
                              description="Whether to sort results by distance in hybrid search.")


class KDTreeObservation(Sample):
    """Combined observation containing both source and target point clouds for KDTreeFilter."""
    source: o3d.geometry.PointCloud
    target: o3d.geometry.PointCloud


class KDTreeFilter(FilterActor[KDTreeObservation, o3d.geometry.PointCloud, Sample, bool, KDTreeConfig]):
    """KDTree-based filter that finds and keeps points based on nearest neighbor relationships.

    Given a source point cloud and a target point cloud in the observation, this filter:
    1. Builds a KDTree on one of the point clouds
    2. Searches for nearest neighbors between the two point clouds using the configured search method
    3. Returns a new point cloud containing only the filtered points
    """
    def __init__(self, config: KDTreeConfig | None = None, parent: ConfiguredActorProto | None = None):
        super().__init__(config or KDTreeConfig(), parent)

    def act(self, observation: KDTreeObservation, **kwargs) -> o3d.geometry.PointCloud:
        """Filter points based on KDTree nearest neighbor search.

        Args:
            observation: A KDTreeObservation containing both source and target point clouds.

        Returns:
            A new point cloud containing only the filtered points.

        Notes:
            - Which point cloud's points are kept depends on the search_mode configuration.
            - Which search method is used depends on the search_method configuration.
            - Only points meeting the distance criteria are kept.
        """
        source_pcd = observation.source
        target_pcd = observation.target

        # Validate input point clouds
        if not isinstance(source_pcd, o3d.geometry.PointCloud) or not source_pcd.has_points():
            logger.warning(f"[{self.name}] Source point cloud is invalid or empty.")
            return o3d.geometry.PointCloud()

        if not isinstance(target_pcd, o3d.geometry.PointCloud) or not target_pcd.has_points():
            logger.warning(f"[{self.name}] Target point cloud is invalid or empty.")
            return o3d.geometry.PointCloud()

        source_points = np.asarray(source_pcd.points)
        target_points = np.asarray(target_pcd.points)

        # Determine which point cloud to build the KDTree on based on search mode
        if self.config.search_mode == SearchMode.KEEP_SOURCE:
            # Build KDTree on target, query from source
            tree_pcd = target_pcd
            query_points = source_points
            original_pcd = source_pcd
            logger.info(f"[{self.name}] Building KDTree on target ({len(target_points)} points), querying from source ({len(source_points)} points).")
        else:
            tree_pcd = source_pcd
            query_points = target_points
            original_pcd = target_pcd
            logger.info(f"[{self.name}] Building KDTree on source ({len(source_points)} points), querying from target ({len(target_points)} points).")

        # Build the KDTree
        kdtree = o3d.geometry.KDTreeFlann(tree_pcd)

        # Filter points based on nearest neighbor search
        filtered_indices = []
        max_dist_squared = self.config.max_distance**2

        # Search based on the selected method
        for i, point in enumerate(query_points):
            if self.config.search_method == KDTreeSearchMethod.KNN:
                # K-nearest neighbors search
                k, idx, squared_distances = kdtree.search_knn_vector_3d(point, self.config.knn_k)
                if k > 0 and squared_distances[0] <= max_dist_squared:
                    filtered_indices.append(i)

            elif self.config.search_method == KDTreeSearchMethod.RADIUS:
                # Radius search
                k, idx, squared_distances = kdtree.search_radius_vector_3d(point, self.config.max_distance)
                if k > 0:
                    filtered_indices.append(i)

            elif self.config.search_method == KDTreeSearchMethod.HYBRID:
                # Hybrid search (k nearest neighbors within radius)
                k, idx, squared_distances = kdtree.search_hybrid_vector_3d(
                    point,
                    radius=self.config.max_distance,
                    max_nn=self.config.knn_k,
                )
                if k > 0:
                    filtered_indices.append(i)

        # Create a new point cloud with only the filtered points
        result_pcd = o3d.geometry.PointCloud()

        if filtered_indices:
            # Extract the filtered points and their colors (if available)
            filtered_points = query_points[filtered_indices]
            result_pcd.points = o3d.utility.Vector3dVector(filtered_points)

            if original_pcd.has_colors():
                original_colors = np.asarray(original_pcd.colors)
                result_pcd.colors = o3d.utility.Vector3dVector(original_colors[filtered_indices])

        logger.info(f"[{self.name}] Filtered {len(query_points)} points down to {len(filtered_indices)} points using {self.config.search_method} search.")
        return result_pcd


def demo() -> None:
    """Demonstrate KDTreeFilter usage with different search methods."""
    from importlib.resources import files

    from embdata.sense.camera_config import CAMERA_D435_1
    from embdata.sense.depth import Depth
    from embdata.sense.image import Image
    from embdata.utils.safe_print import safe_print

    # Load test data
    RGB_FILE = str(files("embdata")/"resources"/"color_image.png")
    DEPTH_FILE = str(files("embdata")/"resources"/"depth_image.png")

    # Create a point cloud from depth image
    rgb_image = Image(path=RGB_FILE, encoding="png", mode="RGB")
    depth = Depth(path=DEPTH_FILE,
                 encoding="png",
                 mode="I",
                 size=(1280, 720),
                 camera=CAMERA_D435_1,
                 unit="mm",
                 rgb=rgb_image)

    # Create full point cloud
    full_cloud = depth.to_pointcloud(camera=CAMERA_D435_1, backend="open3d")
    safe_print(f"Full point cloud has {len(full_cloud.points)} points.")

    # Create a downsampled version to use as source
    from embdata.sense.filters.downsample import DownSampleConfig, DownSampleFilter, VoxelConfig
    downsampler = DownSampleFilter(DownSampleConfig(voxel=VoxelConfig(voxel_size=0.05), order=["voxel"]))
    downsampled_cloud = downsampler.act(full_cloud)
    safe_print(f"Downsampled source point cloud has {len(downsampled_cloud.points)} points.")

    # Create combined observation
    combined_obs = KDTreeObservation(source=downsampled_cloud, target=full_cloud)

    # Example 1: KNN search (default, k=1)
    knn_filter = KDTreeFilter(KDTreeConfig(
        max_distance=0.01,
        search_mode=SearchMode.KEEP_SOURCE,
        search_method=KDTreeSearchMethod.KNN,
        knn_k=1,
    ))
    knn_result = knn_filter.act(combined_obs)
    safe_print(f"KNN search result (k=1): {len(knn_result.points)} points")

    # Example 2: KNN search with k=5
    knn5_filter = KDTreeFilter(KDTreeConfig(
        max_distance=0.01,
        search_mode=SearchMode.KEEP_SOURCE,
        search_method=KDTreeSearchMethod.KNN,
        knn_k=5,
    ))
    knn5_result = knn5_filter.act(combined_obs)
    safe_print(f"KNN search result (k=5): {len(knn5_result.points)} points")

    # Example 3: Radius search
    radius_filter = KDTreeFilter(KDTreeConfig(
        max_distance=0.01,
        search_mode=SearchMode.KEEP_SOURCE,
        search_method=KDTreeSearchMethod.RADIUS,
    ))
    radius_result = radius_filter.act(combined_obs)
    safe_print(f"Radius search result: {len(radius_result.points)} points")

    # Example 4: Hybrid search
    hybrid_filter = KDTreeFilter(KDTreeConfig(
        max_distance=0.01,
        search_mode=SearchMode.KEEP_SOURCE,
        search_method=KDTreeSearchMethod.HYBRID,
        knn_k=5,
    ))
    hybrid_result = hybrid_filter.act(combined_obs)
    safe_print(f"Hybrid search result (k=5, radius=0.1): {len(hybrid_result.points)} points")

    # Example 5: Search keeping target points instead
    target_filter = KDTreeFilter(KDTreeConfig(
        max_distance=0.01,
        search_mode=SearchMode.KEEP_TARGET,
        search_method=KDTreeSearchMethod.KNN,
        knn_k=1,
    ))
    target_result = target_filter.act(combined_obs)
    safe_print(f"Search keeping target points: {len(target_result.points)} points")


if __name__ == "__main__":
    demo()
