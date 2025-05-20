from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, Unpack, cast

import numpy as np
import open3d as o3d
from pydantic import Field

from embdata.sample import Sample
from embdata.sense.filters.config import FilterConfig
from embdata.sense.filters.downsample import DownSampleConfig, DownSampleFilter
from embdata.sense.filters.filter import FilterActor
from embdata.utils.camera_utils import RGBDSensorReading
from embdata.utils.custom_logger import get_logger

if TYPE_CHECKING:
    from embdata.actor import ConfiguredActorProto

o3dg = o3d.geometry
logger = get_logger(__name__)


class FusionConfig(FilterConfig):
    primary_sensor_name: str = Field(default="cam0")
    prefilter: DownSampleConfig | None = Field(default=None)
    postfilter: DownSampleConfig | None = Field(default=None)

    class Dict(TypedDict):
        primary_sensor_name: str
        prefilter: DownSampleConfig | None
        postfilter: DownSampleConfig | None

    class Kwargs(TypedDict,total=False):
        primary_sensor_name: str
        prefilter: DownSampleConfig | None
        postfilter: DownSampleConfig | None

class FusionFilter(FilterActor[dict[str, RGBDSensorReading], o3dg.PointCloud, Sample, bool, FusionConfig]):
    def __init__(self, config: FusionConfig, parent: ConfiguredActorProto | None = None):
        super().__init__(config, parent)
        self.filters = {
            "prefilter": DownSampleFilter(config.prefilter,parent=self),
            "postfilter": DownSampleFilter(config.postfilter,parent=self),
        }

    def act(self, rgbdsensor_readings: dict[str, RGBDSensorReading], **fusion_kwargs:Unpack[FusionConfig.Kwargs]) -> o3dg.PointCloud:
        """Assumes that each camera's extrinsic is in the same frame."""
        config: FusionConfig = FusionConfig(**{**self.config.model_dump(), **fusion_kwargs})
        combined_pcd = o3dg.PointCloud()
        if config.prefilter:
            combined_pcd = self.filters[config.prefilter.name](combined_pcd, **config.prefilter.model_dump())

        for k, v in rgbdsensor_readings.items():
            T_marker_camK = v.camera.extrinsic.matrix()
            pcd = v.depth.to_pointcloud(v.camera)
            pcd = cast(o3dg.PointCloud, pcd.transform(cast(np.ndarray, T_marker_camK)))
            combined_pcd += pcd
            logger.debug(f"[{self.name}] Transformed and added PCD from '{k}'")

        if config.postfilter:
            combined_pcd = self.filters[config.postfilter.name](combined_pcd, **config.postfilter.model_dump())
        logger.info(f"[{self.name}] Combined {len(rgbdsensor_readings)} PCDs. Total points: {len(combined_pcd.points)}.")
        return combined_pcd



def demo() -> None:

    from embdata.agents.config import DefaultCameras
    from embdata.sense.aruco import ArucoDetectionConfig, ArucoDetector
    from embdata.sense.meshcat_visualizer import initialize_visualization, visualize_point_cloud
    from embdata.tests.test_data import sample_data
    aruco_config = ArucoDetectionConfig(
        camera = DefaultCameras.CAM_0,
        marker_size = 0.2,
    )
    readings = sample_data.readings
    for reading in readings.values():
        reading.camera.extrinsic = ArucoDetector(config=aruco_config).detect_extrinsics(reading.image, reading.depth)
    fusion_filter = FusionFilter(config=FusionConfig(primary_sensor_name="cam0"))
    combined: o3dg.PointCloud = fusion_filter.act(rgbdsensor_readings=readings)

    vis_session = initialize_visualization()
    if vis_session and combined:
        visualize_point_cloud(vis_session, np.asarray(combined.points), name="combined_points", colors=np.asarray(combined.colors) if combined.has_colors() else [0.8, 0.2, 0.2])

    if vis_session:
        input("Press Enter to exit after viewing visualization...")

if __name__ == "__main__":
    demo()

