import base64
import colorsys
import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Literal, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from embdata.coordinate import Mask
from embdata.sense.camera import Camera
from embdata.sense.world import World
from embdata.sense.world_object import WorldObject

logger = logging.getLogger(__name__)

class ColorScheme(Enum):
    """Color scheme options for visualization."""

    DISTINCT = auto()
    SEQUENTIAL = auto()
    PAIRED = auto()


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""

    masks: bool = True
    labels: bool = True
    bbox2d: bool = True
    bbox3d: bool = True
    axes: bool = True
    matches: bool = True
    remove_background: bool = False
    color_scheme: ColorScheme = ColorScheme.DISTINCT
    font_scale: float = 0.5
    line_thickness: int = 2
    source_image_channels: Literal["rgb", "bgr"] = "rgb"
    return_type: Literal["data", "image"] = "data"  # New field to control return type


class Visualize:
    """Class for visualizing object detections, poses, and annotations."""

    def __init__(self, config: VisualizationConfig) -> None:
        """Initialize visualization settings.

        Args:
            config: Visualization configuration settings
        """
        self.config = config
        self.colors = self._generate_colors(20)  # Generate 20 distinct colors

    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Generate n visually distinct colors.

        Args:
            n: Number of colors to generate

        Returns:
            List of RGB color tuples
        """
        colors = []
        for i in range(n):
            hue = i / n
            sat = 0.9
            val = 0.9
            rgb = colorsys.hsv_to_rgb(hue, sat, val)
            colors.append(tuple(int(255 * c) for c in rgb))
        return colors

    def _get_rotation_vector(self, obj: WorldObject) -> Tuple[NDArray | None, NDArray | None]:
        """Get rotation and translation vectors from pose."""
        if obj.pose is None:
            return None, None

        # Get rotation matrix from euler angles
        rpy = [obj.pose.roll, obj.pose.pitch, obj.pose.yaw]
        rotation_matrix = Rotation.from_euler("xyz", rpy).as_matrix()

        # Convert to OpenCV convention
        rvec, _ = cv2.Rodrigues(rotation_matrix)
        tvec = obj.pose.numpy()[:3].reshape((3, 1))

        return rvec, tvec

    def draw_mask(self, image: NDArray, obj: WorldObject, color_idx: int) -> dict | NDArray:
        """Draw segmentation mask."""
        try:
            # Handle Mask object type
            if isinstance(obj.mask, Mask):
                mask_array = obj.mask.mask
            else:
                logger.warning(f"Unexpected mask type: {type(obj.mask)}")
                return {"bytes": "", "shape": []} if self.config.return_type == "data" else image

            # Ensure binary mask
            mask_array = (mask_array > 0.5).astype(np.uint8)

            if self.config.return_type == "data":
                # Convert mask to bytes and encode in base64
                mask_bytes = mask_array.tobytes()
                mask_base64 = base64.b64encode(mask_bytes).decode("utf-8")

                return {
                    "bytes": mask_base64,
                    "shape": list(mask_array.shape),
                }
            # For image return type, create colored overlay
            color = self.colors[color_idx % len(self.colors)]
            overlay = np.zeros_like(image)
            overlay[mask_array == 1] = color

            # Create alpha channel
            alpha = np.zeros(image.shape[:2], dtype=np.float32)
            alpha[mask_array == 1] = 0.5

            # Blend using alpha channel
            blended = image.copy()
            for c in range(3):
                blended[:, :, c] = image[:, :, c] * (1 - alpha) + overlay[:, :, c] * alpha

            return blended.astype(np.uint8)

        except Exception as e:
            logger.exception(f"Error in draw_mask: {e!s}")
            logger.exception("Full traceback:")
            if self.config.return_type == "data":
                return {"bytes": "", "shape": []}
            return image

    def draw_bbox2d(self, image: NDArray, obj: WorldObject, color_idx: int) -> List[float]:
        """Draw 2D bounding box with anti-aliased lines.

        Args:
            image: Input RGB image
            obj: WorldObject containing 2D bbox
            color_idx: Index for color selection

        Returns:
            Image with 2D bbox overlay
        """
        if obj.bbox_2d is None:
            return []

        result = image.copy()
        color = self.colors[color_idx % len(self.colors)]

        # Convert bbox coordinates to integers
        x1 = int(obj.bbox_2d.x1)
        y1 = int(obj.bbox_2d.y1)
        x2 = int(obj.bbox_2d.x2)
        y2 = int(obj.bbox_2d.y2)

        if self.config.return_type == "data":
            return [x1, y1, x2, y2]

        if self.config.return_type == "image":
            # Create a larger image for anti-aliasing
            scale = 4
            h, w = result.shape[:2]
            large_img = cv2.resize(
                result, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC,
            )

            # Draw thick lines on larger image
            cv2.rectangle(
                large_img,
                (x1 * scale, y1 * scale),
                (x2 * scale, y2 * scale),
                color,
                thickness=self.config.line_thickness * scale,
                lineType=cv2.LINE_AA,
            )

            # Resize back to original size
            return cv2.resize(large_img, (w, h), interpolation=cv2.INTER_AREA)

        msg = "Invalid return type"
        raise ValueError(msg)

    def draw_bbox3d(self,
                    image: NDArray,
                    obj: WorldObject,
                    camera: Camera,
                    color_idx: int) -> List[Tuple[float, float]] | NDArray:
        """Draw 3D bounding box with occlusion handling.

        Args:
            image: Input RGB image
            obj: WorldObject containing 3D bbox
            camera: Camera object with intrinsics
            color_idx: Index for color selection

        Returns:
            Either list of corner points (data) or annotated image (image)
        """
        if obj.bbox_3d is None or obj.pose is None:
            return [] if self.config.return_type == "data" else image

        # Get rotation and translation vectors
        rvec, tvec = self._get_rotation_vector(obj)
        if rvec is None or tvec is None:
            return [] if self.config.return_type == "data" else image

        result = image.copy()
        color = self.colors[color_idx % len(self.colors)]

        # Calculate dimensions in object's local frame
        length = abs(obj.bbox_3d.z2 - obj.bbox_3d.z1)  # Longest dimension (was Z)
        height = abs(obj.bbox_3d.y2 - obj.bbox_3d.y1)  # Height (Y stays the same)
        width = abs(obj.bbox_3d.x2 - obj.bbox_3d.x1)  # Width (was X)

        # Create corners in object's local frame
        corners = np.float32(
            [
                [-length / 2, -width / 2, height / 2],  # front top left
                [length / 2, -width / 2, height / 2],  # front top right
                [length / 2, -width / 2, -height / 2],  # front bottom right
                [-length / 2, -width / 2, -height / 2],  # front bottom left
                [-length / 2, width / 2, height / 2],  # back top left
                [length / 2, width / 2, height / 2],  # back top right
                [length / 2, width / 2, -height / 2],  # back bottom right
                [-length / 2, width / 2, -height / 2],  # back bottom left
            ],
        )

        try:
            # Project corners to image plane
            corners_2d, _ = cv2.projectPoints(
                corners,
                rvec,
                tvec,
                camera.intrinsic.matrix.astype(np.float32),
                camera.distortion.numpy().astype(np.float32),
            )
            corners_2d = corners_2d.reshape(-1, 2).astype(int)

            if self.config.return_type == "data":
                return [tuple(point) for point in corners_2d]

            # Draw edges with different colors for front/back faces
            edges_front = [(0, 1), (1, 2), (2, 3), (3, 0)]  # front face
            edges_back = [(4, 5), (5, 6), (6, 7), (7, 4)]  # back face
            edges_connect = [(0, 4), (1, 5), (2, 6), (3, 7)]  # connecting edges

            # Draw front face (solid)
            for edge in edges_front:
                pt1 = corners_2d[edge[0]]
                pt2 = corners_2d[edge[1]]
                if (
                    0 <= pt1[0] < image.shape[1]
                    and 0 <= pt1[1] < image.shape[0]
                    and 0 <= pt2[0] < image.shape[1]
                    and 0 <= pt2[1] < image.shape[0]
                ):
                    cv2.line(
                        result,
                        tuple(pt1),
                        tuple(pt2),
                        color,
                        thickness=self.config.line_thickness // 2,
                        lineType=cv2.LINE_AA,
                    )

            # Draw back face (dashed)
            for edge in edges_back:
                pt1 = corners_2d[edge[0]]
                pt2 = corners_2d[edge[1]]
                if (
                    0 <= pt1[0] < image.shape[1]
                    and 0 <= pt1[1] < image.shape[0]
                    and 0 <= pt2[0] < image.shape[1]
                    and 0 <= pt2[1] < image.shape[0]
                ):
                    cv2.line(
                        result,
                        tuple(pt1),
                        tuple(pt2),
                        color,
                        thickness=self.config.line_thickness // 2,
                        lineType=cv2.LINE_AA,
                    )

            # Draw connecting edges
            for edge in edges_connect:
                pt1 = corners_2d[edge[0]]
                pt2 = corners_2d[edge[1]]
                if (
                    0 <= pt1[0] < image.shape[1]
                    and 0 <= pt1[1] < image.shape[0]
                    and 0 <= pt2[0] < image.shape[1]
                    and 0 <= pt2[1] < image.shape[0]
                ):
                    cv2.line(
                        result,
                        tuple(pt1),
                        tuple(pt2),
                        color,
                        thickness=self.config.line_thickness // 2,
                        lineType=cv2.LINE_AA,
                    )

            return result

        except cv2.error as e:
            logger.debug("OpenCV Error in draw_bbox3d: %s", e)
            return [] if self.config.return_type == "data" else image

    def draw_axes(self, image: NDArray, obj: WorldObject, camera: Camera) -> dict | NDArray:
        """Draw coordinate axes for pose visualization.

        Args:
            image: Input RGB image
            obj: WorldObject containing pose
            camera: Camera object with intrinsics

        Returns:
            Either axis points data (data) or annotated image (image)
        """
        if obj.pose is None:
            return {} if self.config.return_type == "data" else image

        # Get rotation and translation vectors
        rvec, tvec = self._get_rotation_vector(obj)
        if rvec is None or tvec is None:
            return {} if self.config.return_type == "data" else image

        # Project axis endpoints for labels
        axis_points = np.float32([
            [0, 0, 0],      # origin
            [0.1, 0, 0],    # x-axis
            [0, 0.1, 0],    # y-axis
            [0, 0, 0.1],    # z-axis
        ]).reshape(-1, 3)

        points_2d, _ = cv2.projectPoints(
            axis_points,
            rvec,
            tvec,
            camera.intrinsic.matrix,
            camera.distortion.numpy(),
        )
        points_2d = points_2d.reshape(-1, 2)

        if self.config.return_type == "data":
            return {
                "origin": tuple(points_2d[0]),
                "x_axis": tuple(points_2d[1]),
                "y_axis": tuple(points_2d[2]),
                "z_axis": tuple(points_2d[3]),
            }

        # If return_type is "image", draw the axes
        result = image.copy()

        # Draw main axes
        result = cv2.drawFrameAxes(
            image=result,
            cameraMatrix=camera.intrinsic.matrix,
            distCoeffs=camera.distortion.numpy(),
            rvec=rvec,
            tvec=tvec,
            length=0.1,
            thickness=self.config.line_thickness,
        )

        # Get endpoints for labels
        x_end = tuple(points_2d[1].astype(int))
        y_end = tuple(points_2d[2].astype(int))
        z_end = tuple(points_2d[3].astype(int))

        # Add labels with offset
        offset = 10
        if all(
            0 <= p[0] < image.shape[1] and 0 <= p[1] < image.shape[0]
            for p in [x_end, y_end, z_end]
        ):
            # Use color indices: 0 for X (red), 1 for Y (green), 2 for Z (blue)
            result = self.draw_label(result, "X", (x_end[0] + offset, x_end[1] + offset), 0)
            result = self.draw_label(result, "Y", (y_end[0] + offset, y_end[1] + offset), 1)
            result = self.draw_label(result, "Z", (z_end[0] + offset, z_end[1] + offset), 2)

        return result

    def draw_label(self, image: NDArray, text: str, position: Tuple[int, int], color_idx: int) -> dict | NDArray:
        """Draw text label with outline and background.

        Args:
            image: Input RGB image
            text: Text to draw
            position: Position to draw text (x, y)
            color_idx: Index for color selection

        Returns:
            Either label data (data) or annotated image (image)
        """
        if self.config.return_type == "data":
            return {
                "text": text,
                "position": position,
                "color_idx": color_idx,
            }

        result = image.copy()
        color = self.colors[color_idx % len(self.colors)]

        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.config.font_scale
        thickness = self.config.line_thickness
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness,
        )

        x, y = position

        # Draw text outline
        cv2.putText(
            result,
            text,
            (x, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness * 3,
            cv2.LINE_AA,
        )

        # Draw main text
        cv2.putText(
            result,
            text,
            (x, y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        return result

    def draw_matches(self, image: NDArray, matches: List, color_idx: int = 0) -> List[dict] | NDArray:
        """Draw feature matches between reference and target images.

        Args:
            image: Input RGB image
            matches: List of matches
            color_idx: Index for color selection

        Returns:
            Either list of match points (data) or annotated image (image)
        """
        if not matches or not self.config.matches:
            return [] if self.config.return_type == "data" else image

        if self.config.return_type == "data":
            return [
                {
                    "source": (int(match["source"][0]), int(match["source"][1])),
                    "target": (int(match["target"][0]), int(match["target"][1])),
                }
                for match in matches
            ]

        result = image.copy()
        color = self.colors[color_idx % len(self.colors)]

        # Draw lines between matches
        for match in matches:
            source = tuple(map(int, match["source"]))
            target = tuple(map(int, match["target"]))
            cv2.line(
                result,
                source,
                target,
                color,
                thickness=self.config.line_thickness,
                lineType=cv2.LINE_AA,
            )

        return result


    def show(self, world: World, matches: List | None = None) -> dict | NDArray:
        """Main visualization function.

        Args:
            world: World object containing image, camera, and objects
            matches: Optional list of matches from one-shot detection

        Returns:
            Either visualization data (data) or annotated image (image)
        """
        results = {
            "masks": [],
            "bbox2d": [],
            "bbox3d": [],
            "axes": [],
            "labels": [],
            "matches": [],
        }

        # Process matches if provided
        if matches is not None and self.config.matches:
            results["matches"] = self.draw_matches(world.image.array, matches)

        # Process each object in the world
        for idx, obj in enumerate(world.objects):
            if obj.name in ["camera", "aruco", "plane", "person"]:
                continue

            # Get object pose in camera frame
            world.get_object(obj.name, reference="camera")

            # Collect visualization data based on config
            if self.config.masks and obj.mask is not None:
                mask_data = self.draw_mask(world.image.array, obj, idx)
                results["masks"].append(mask_data)

            if self.config.bbox2d and obj.bbox_2d is not None:
                bbox2d_data = self.draw_bbox2d(world.image.array, obj, idx)
                results["bbox2d"].append(bbox2d_data)

            if self.config.bbox3d and obj.bbox_3d is not None:
                bbox3d_data = self.draw_bbox3d(world.image.array, obj, world.camera, idx)
                results["bbox3d"].append(bbox3d_data)

            if self.config.axes and obj.pose is not None:
                axes_data = self.draw_axes(world.image.array, obj, world.camera)
                results["axes"].append(axes_data)

            if self.config.labels:
                # Project object center for label placement
                rvec, tvec = self._get_rotation_vector(obj)
                if rvec is not None and tvec is not None:
                    center_3d = obj.pose.numpy()[:3].reshape(1, 3)
                    center_2d, _ = cv2.projectPoints(
                        center_3d,
                        rvec,
                        tvec,
                        world.camera.intrinsic.matrix,
                        world.camera.distortion.numpy(),
                    )
                    center_2d = center_2d.reshape(-1, 2)[0].astype(int)
                    label_pos = (center_2d[0], center_2d[1] - 20)
                    label_data = self.draw_label(world.image.array, obj.name, label_pos, idx)
                    results["labels"].append(label_data)

        if self.config.return_type == "data":
            return results

        # If return_type is "image", draw everything on the image
        annotated_image = world.image.array.copy()
        if self.config.source_image_channels == "rgb":
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        # Draw all visualizations
        for idx, obj in enumerate(world.objects):
            if obj.name in ["camera", "aruco", "plane"]:
                continue

            if self.config.masks and obj.mask is not None:
                annotated_image = self.draw_mask(annotated_image, obj, idx)

            if self.config.bbox2d and obj.bbox_2d is not None:
                annotated_image = self.draw_bbox2d(annotated_image, obj, idx)

            if self.config.bbox3d and obj.bbox_3d is not None:
                annotated_image = self.draw_bbox3d(annotated_image, obj, world.camera, idx)

            if self.config.axes and obj.pose is not None:
                annotated_image = self.draw_axes(annotated_image, obj, world.camera)

            if self.config.labels:
                rvec, tvec = self._get_rotation_vector(obj)
                if rvec is not None and tvec is not None:
                    center_3d = obj.pose.numpy()[:3].reshape(1, 3)
                    center_2d, _ = cv2.projectPoints(
                        center_3d,
                        rvec,
                        tvec,
                        world.camera.intrinsic.matrix,
                        world.camera.distortion.numpy(),
                    )
                    center_2d = center_2d.reshape(-1, 2)[0].astype(int)
                    label_pos = (center_2d[0], center_2d[1] - 20)
                    annotated_image = self.draw_label(annotated_image, obj.name, label_pos, idx)

        # Draw matches if provided
        if matches is not None and self.config.matches:
            annotated_image = self.draw_matches(annotated_image, matches)

        # Convert back to RGB for final output
        if self.config.source_image_channels == "rgb":
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        return annotated_image
