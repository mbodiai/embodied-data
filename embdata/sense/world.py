import logging
import warnings
from typing import overload

from pydantic import Field, model_validator
from typing_extensions import (
    Any,
    List,
    Literal,
)

from embdata.coordinate import Plane, Pose6D
from embdata.sample import Sample
from embdata.sense.aruco import Aruco
from embdata.sense.camera import Camera
from embdata.sense.depth import Depth
from embdata.sense.image import Image
from embdata.sense.world_object import Collection, WorldObject

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sz = Literal[3]


class World(Sample):
    """Model for World Data.

    To keep things simple, always keep the objects in the camera frame. Perform transformations during access.
    """

    image: Image | None = None
    depth: Depth | None = None
    annotated: Image | None = None
    objects: Collection[WorldObject] = Field(default_factory=Collection[WorldObject],
                                             description="List of scene objects")

    camera: Camera | None = Field(default_factory=Camera, description="Camera parameters of the scene")
    aruco: Aruco | None = None
    plane: Plane | None = Field(default=None, description="Plane detected in the scene")

    def __getitem__(self, item):
        # Access the underlying dictionary directly to avoid recursion
        if item in self.objects:
            return self.objects[item]
        return getattr(self, item)

    def object_names(self) -> List[str]:
        return list({obj.name for obj in self.objects} | {"plane", "camera"})

    @model_validator(mode="before")
    @classmethod
    def validate(cls, v: Any | None) -> Any:
        if v is None:
            v = {}

        if "objects" not in v or v["objects"] is None:
            v["objects"] = Collection[WorldObject]()

        if v and v.get("objects") is not None and not isinstance(v["objects"], Collection[WorldObject]):
            v["objects"] = Collection[WorldObject](v["objects"])

        # Add the camera object
        reference_obj = WorldObject(name="camera",
                                    pose=Pose6D(),
                                    bbox_2d=None,
                                    pixel_coords=None,
                                    mask=None,
                                    bbox_3d=None,
                                    volume=None)

        reference_obj.pose.set_reference_frame("camera")
        v["objects"].append(reference_obj)

        return v

    @overload
    def add_object(self, obj: WorldObject) -> None:
        ...

    @overload
    def add_object(self, obj: List[WorldObject]) -> None:
        ...

    def add_object(self, obj: WorldObject | List[WorldObject]) -> None:
        """Add one or more objects to the world.

        Args:
            obj (Union[WorldObject, List[WorldObject]]): Single object or list of objects to add.

        Warnings:
            UserWarning: If an object name is missing.
        """
        if isinstance(obj, list):
            for single_obj in obj:
                if single_obj.name == "":
                    warnings.warn("Object name is missing. The object will not be added.", UserWarning, stacklevel=2)
                    continue
                self.objects.append(single_obj)
        else:
            if obj.name == "":
                warnings.warn("Object name is missing. The object will not be added.", UserWarning, stacklevel=2)
                return
            self.objects.append(obj)
