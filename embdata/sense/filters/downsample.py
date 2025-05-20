from typing import Literal, Unpack, overload

import numpy as np
import open3d as o3d
from pydantic import Field
from typing_extensions import TypedDict

from embdata.actor import ConfiguredActorProto
from embdata.sample import Sample
from embdata.sense.filters.config import FilterConfig
from embdata.sense.filters.filter import FilterActor

o3dg = o3d.geometry


class VoxelConfig(FilterConfig):
    voxel_size: float = 0.002


class RadiusConfig(FilterConfig):
    radius: float = 0.01
    min_points: int = 20


class StatisticalOutlierConfig(FilterConfig):

    min_neighbors: int = 20
    std_ratio: float = 1.0


class DownSampleConfig(FilterConfig):
    voxel: VoxelConfig = Field(default_factory=VoxelConfig)
    radius: RadiusConfig = Field(default_factory=RadiusConfig)
    outlier: StatisticalOutlierConfig = Field(default_factory=StatisticalOutlierConfig)
    order: list[Literal["voxel", "radius", "outlier"]] = Field(default_factory=list)
    return_inlier_indices: bool = False

    class Kwargs(TypedDict,total=False):
        voxel_size: float
        radius: float
        min_points: int
        min_neighbors: int
        std_ratio: float


class DownSampleFilter(FilterActor[o3dg.PointCloud, o3dg.PointCloud, Sample, bool, DownSampleConfig]):

    def __init__(self, config: DownSampleConfig|None = None, parent: ConfiguredActorProto | None = None):
        super().__init__(config or DownSampleConfig(), parent)
        self.inlier_indices = None
    @overload
    def act(self, observation: o3dg.PointCloud, **config:Unpack[DownSampleConfig.Kwargs]) -> o3dg.PointCloud:...
    @overload
    def act(self, observation: o3dg.PointCloud, **config:Unpack[DownSampleConfig.Kwargs]) -> tuple[o3dg.PointCloud, list[int]]:...
    def act(self, observation: o3dg.PointCloud, **config:Unpack[DownSampleConfig.Kwargs]) -> o3dg.PointCloud | tuple[o3dg.PointCloud, list[int]]:
        pcd = observation
        for order in self.config.order:
            if order == "voxel":
                pcd = pcd.voxel_down_sample(voxel_size=self.config.voxel.voxel_size)
                inlier_indices = np.arange(len(pcd.points)).tolist()
            elif order == "radius":
                pcd,inlier_indices = pcd.remove_radius_outlier(nb_points=self.config.radius.min_points, radius=self.config.radius.radius)
            elif order == "outlier":
                pcd,inlier_indices = pcd.remove_statistical_outlier(nb_neighbors=self.config.outlier.min_neighbors, std_ratio=self.config.outlier.std_ratio)
        if config.get("return_inlier_indices",False):
            return pcd,inlier_indices
        return pcd


def demo() -> None:

    from importlib.resources import files

    from embdata.sense.camera_config import CAMERA_D435_1
    from embdata.sense.depth import Depth
    from embdata.sense.image import Image
    from embdata.utils.safe_print import safe_print

    RGB_FILE = str(files("embdata")/"resources"/"color_image.png")
    DEPTH_FILE = str(files("embdata")/"resources"/"depth_image.png")
    CAMERA = CAMERA_D435_1

    image = Image(path=RGB_FILE, encoding="png", mode="RGB")
    depth = Depth(path=DEPTH_FILE, encoding="png", mode="I", size=(1280, 720), camera=CAMERA, unit="mm", rgb=image)

    points = depth.to_pointcloud(CAMERA)
    safe_print(f"Initial point cloud has {len(points.points)} points.")

    down_sample_filter = DownSampleFilter(DownSampleConfig(voxel=VoxelConfig(voxel_size=0.005),
                                                           order=["voxel"]))
    downsampled_pcd = down_sample_filter.act(points)
    safe_print(f"Downsampled point cloud has {len(downsampled_pcd.points)} points.")

if __name__ == "__main__":
    demo()
