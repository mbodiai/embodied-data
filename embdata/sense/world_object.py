from typing import TypeVar
from weakref import ReferenceType, ref

import numpy as np
from pydantic import model_validator
from typing_extensions import (
    Any,
    Literal,
    Type,
    TypeAlias,
)

from embdata.coordinate import BBox2D, BBox3D, Mask, PixelCoords, Plane, Point, Pose6D
from embdata.geometry.pca import get_intrinsic_orientation, pca
from embdata.multi_sample import MultiSample
from embdata.ndarray import ndarray
from embdata.sample import Sample
from embdata.sense.image import Image

T = TypeVar("T", bound=Sample)
Collection: TypeAlias = MultiSample[T]

InfoUndefinedType = Literal["unset"]
InfoUndefined = "unset"


def process_field(value: Any, field_type: Type[Sample]) -> Sample | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return field_type(**value)
    return field_type(value) if not isinstance(value, field_type) else value


class WorldObject2D(Sample):
    """Model for world Object. It describes the objects in the scene.

    Attributes:
        name (str): The name of the object.
        bbox_2d (BBox2D | None): The 2D bounding box of the object.
        pixel_coords (PixelCoords | None): The pixel coordinates of the object.
        mask (ndarray | None): The mask of the object.
        image (ReferenceType[Image] | None): Weak reference to the parent image.
    """

    name: str = ""
    bbox_2d: BBox2D | None = None
    pixel_coords: PixelCoords | None = None
    mask: Mask | None = None
    image: ReferenceType[Image] | None = None

    @model_validator(mode="before")
    @classmethod
    def validate(cls, v: Any | None) -> Any:
        v["bbox_2d"] = process_field(v.get("bbox_2d"), BBox2D)
        if v["bbox_2d"] is None:
            del v["bbox_2d"]
        v["pixel_coords"] = process_field(v.get("pixel_coords"), PixelCoords)
        if v["pixel_coords"] is None:
            del v["pixel_coords"]
        v["mask"] = process_field(v.get("mask"), Mask)
        if v["mask"] is None:
            del v["mask"]
        if "image" in v and v["image"] is not None:
            v["image"] = ref(v["image"])
        return v

    def __eq__(self, other):
        if not isinstance(other, WorldObject2D):
            return False

        # Get actual image objects if they exist
        self_image = self.image() if self.image is not None else None
        other_image = other.image() if other.image is not None else None

        return (
            self.name == other.name
            and (
                self.bbox_2d == other.bbox_2d
                or (self.bbox_2d is None and other.bbox_2d is None)
            )
            and self.pixel_coords == other.pixel_coords
            and (self.mask == other.mask or (self.mask is None and other.mask is None))
            and (self_image == other_image)
        )


class WorldObject(WorldObject2D):
    """Model for world Object. It describes the objects in the scene.

    Attributes:
        name (str): The name of the object.
        bbox_2d (BBox2D): The 2D bounding box of the object.
        bbox_3d (BBox3D): The 3D bounding box of the object.
        pose (Pose6D): The pose of the object.
        pixel_coords (PixelCoords): The pixel coordinates of the object.
        mask (Mask): The mask of the object.
        volume (float): The volume of the object.
    """
    pose: Pose6D | None = None
    bbox_3d: BBox3D | None = None
    volume: float | None = None

    points: ndarray | None = None
    xyz_min: ndarray | None = None
    xyz_max: ndarray | None = None
    pca_vectors: ndarray | None = None


    @model_validator(mode="before")
    @classmethod
    def validate(cls, v: Any | None) -> Any:
        # Process WorldObject2D fields first
        v = super().validate(v)

        # Process additional fields
        v["bbox_3d"] = process_field(v.get("bbox_3d"), BBox3D)
        v["pose"] = process_field(v.get("pose"), Pose6D)
        return v

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and self.bbox_3d == other.bbox_3d
            and self.volume == other.volume
            and self.pose == other.pose
        )


    def mask_to_points(self, depth) -> None:
        """Convert a mask to a set of 3D points.

        Args:
            depth: Depth object containing depth image and camera information

        Raises:
            ValueError: If mask is not available
        """
        if self.mask is None:
            msg = f"Mask is not available for {self.name}"
            raise ValueError(msg)

        # Get the mask array directly
        mask_array = self.mask.mask
        mask_array = np.squeeze(mask_array)

        # Find non-zero points in the mask
        indices = np.argwhere(mask_array == 1)
        points_3d = depth.camera.deproject(uv=indices, depth_image=depth.array)
        valid_points_3d = points_3d[~np.all(points_3d == 0, axis=1)]

        # Store points on the object
        self.points = valid_points_3d


    def pixel_to_point(self, depth) -> None:
        """Convert pixel coordinates to 3D points.

        Args:
            depth: Depth object containing depth image and camera information
        """
        if self.pixel_coords is None:
            msg = f"Pixel coordinates are not available for {self.name}"
            raise ValueError(msg)

        point_3d = depth.camera.deproject(self.pixel_coords, depth_image=depth.array)
        self.points = point_3d


    def get_pose_in_origin(self,
                           plane: Plane,
                           origin_pose: Pose6D) -> None:
        """Compute object pose, orientation and bounds from 3D points or pixel coordinates.

        Args:
            plane: Plane object for orientation computation
            origin_pose: Pose6D object for origin

        Raises:
            ValueError: If neither points nor pixel_coords are available
        """
        # Case 1: Use 3D points from mask
        if self.points is not None and len(self.points) > 0:
            self.xyz_min: Point = Point(*(np.percentile(self.points, 5, axis=0)))
            self.xyz_max: Point = Point(*(np.percentile(self.points, 95, axis=0)))

            # Perform PCA and store the vectors
            pca_vecs, pca_eigen_values = pca(self.points, num_components=3)
            normal = plane.normal(plane.coefficients)
            negative_normal = np.array([-normal.x, -normal.y, -normal.z])

            # Determine PCA orientation
            pca_matrix = get_intrinsic_orientation(
                pca_vecs,
                pca_eigen_values,
                negative_normal,
            )

            self.pca_vectors = pca_matrix

            self.pose = Pose6D(
                x=(self.xyz_min.x + self.xyz_max.x) / 2,
                y=(self.xyz_min.y + self.xyz_max.y) / 2,
                z=(self.xyz_min.z + self.xyz_max.z) / 2,
                roll=0,
                pitch=0,
                yaw=0,
            )

        # Case 2: If there is only one point, use it as the pose
        elif self.points is not None and len(self.points) == 1:
            self.pose = Pose6D(
                *self.points,
                roll=0,
                pitch=0,
                yaw=0,
            )
        else:
            msg = f"Neither points nor pixel_coords are available for {self.name}"
            raise ValueError(msg)

        # Set the reference frame of the object to camera
        if self.pose is not None:
            self.pose.set_reference_frame("camera")
            self.pose.set_info("origin", origin_pose)

