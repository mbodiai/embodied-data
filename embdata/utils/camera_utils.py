from pydantic import BaseModel

from embdata.coordinate import Pose
from embdata.sample import Sample
from embdata.sense.camera import Camera
from embdata.sense.depth import Depth
from embdata.sense.image import Image


class CameraPose(BaseModel):
    """Represents the extrinsics (position and orientation) of a marker in each camera's frame.

    Attributes:
        camera: Camera object with updated extrinsics
        pose: Pose object representing the marker's pose in the camera's frame
    """
    camera: Camera
    pose: Pose

    def __iter__(self):
        return iter([self.camera, self.pose])


class RGBDSensorReading(Sample):

    image: Image
    depth: Depth
    camera: Camera | None = None

