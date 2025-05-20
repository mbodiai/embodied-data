# Copyright 2024 mbodi ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Motions to control a robot.

This module defines the motions to control a robot as pydantic models.

Motions:
    JointControl: A joint value, typically an angle.
    FullJointControl: A lsit of joint values.
    HandControl: A 7D space representing x, y, z, roll, pitch, yaw, and openness of the hand.
    HeadControl: Tilt and pan.
    MobileSingleArmControl: Control for a robot that can move in planar space with a single arm.
    BimanualArmControl: Control for a robot that can move planar space with two arms.
    HumanoidControl: Control for a robot with two arms, two legs, and a head.

Example:
    To create a new Pydantic model for a motion, inherit from the Motion class and define pydantic fields with the MotionField,
    function as you would with any other Pydantic field.

Example:
    from embdata.motion import Motion, AbsoluteMotionField, MotionField, MotionType, VelocityMotionField
    from embdata.coordinate import PlanarPose, Pose

    class HandControl(Motion):
        pose: Pose = RelativeMotionField(default_factory=Pose, description="Pose of the robot hand.", motion_type="relative")
        grasp: float = AbsoluteMotionField(
            default=0,
            bounds=[-1,1],
            description="Openness of the robot hand. -1 is closed, 1 is open.",
            )

    You can also use the RelativeMotionField and VelocityMotionField or TorqueMotionField for different types of motions.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import ConfigDict

from embdata.coordinate import PlanarPose, Pose
from embdata.motion import AbsoluteMotionField, Motion, MotionField, RelativeMotionField
from embdata.ndarray import ndarray


class AnyMotionControl(Motion):
    """Motion Control with arbitrary fields but minimal validation. Should not be subclassed. Subclass Motion instead for validation.

    Pass in names, joints, and any other fields to create a motion control.

    Example:
        >>> class ArmControl(MotionControl):
        ...     names: list[str] = MotionField(default_factory=list, description="Names of the joints.")
        ...     joints: list[float] = MotionField(
        ...         default_factory=lambda: np.zeros(3), bounds=[-1.0, 1.0], shape=(3,), description="Values of the joints."
        ...     )
        >>> arm_control = ArmControl(names=["shoulder", "elbow", "wrist"], joints=[0.1, 0.2])
        Traceback (most recent call last):
            ...
        ValueError: Number of joints 2 does not match number of names 3
        >>> arm_control = ArmControl(names=["shoulder", "elbow", "wrist"], joints=[3.0, 2.0, 1.0])
        Traceback (most recent call last):
            ...
        ValueError: joints item 0 (3.0) is out of bounds [-1.0, 1.0]
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow", populate_by_name=True)

    names: list[str] | None = None
    joints: list[float] | ndarray | Any = None


class HandControl(Motion):
    """Action for a 7D space representing x, y, z, roll, pitch, yaw, and openness of the hand.

    This class represents the control for a robot hand, including its pose and grasp state.

    Attributes:
        pose (Pose): The pose of the robot hand, including position and orientation.
        grasp (float): The openness of the robot hand, ranging from 0 (closed) to 1 (open).

    Example:
        ```python
        from embdata.coordinate import Pose
        from embdata.motion.control import HandControl

        # Create a HandControl instance
        hand_control = HandControl(pose=Pose(position=[0.1, 0.2, 0.3], orientation=[0, 0, 0, 1]), grasp=0.5)

        # Access and modify the hand control
        hand_control.grasp = 0.8
        ```
    """

    pose: Pose = MotionField(default_factory=Pose, description="Pose of the robot hand.")
    grasp: float = AbsoluteMotionField(
        default=0,
        bounds=[-1, 1],
        description="Openness of the robot hand. 0 is closed, 1 is open.",
    )


if not TYPE_CHECKING:
    HandControl.__doc__ = "Control for a robot hand."

class AbsoluteHandControl(Motion):
    pose: Pose = AbsoluteMotionField(default_factory=Pose, description="Pose of the robot hand.")
    grasp: float = AbsoluteMotionField(
        default=0,
        bounds=[-1, 1],
        description="Openness of the robot hand. 0 is closed, 1 is open.",
    )


class RelativePoseHandControl(Motion):
    pose: Pose = RelativeMotionField(default_factory=Pose, description="Pose of the robot hand.")
    grasp: float = AbsoluteMotionField(
        0,
        bounds=[-1, 1],  # [-1,1] so that 0 is a no-op.
        description="Openness of the robot hand. 0 is closed, 1 is open.",
    )

class MobileBaseControl(Motion, PlanarPose):
    """Control for a robot that can move in 2D space."""

    x: float = MotionField(0.0, description="X position of the robot on the ground.")
    y: float = MotionField(0.0, description="Y position of the robot on the ground.")
    theta: float = MotionField(0.0, description="Orientation of the robot on the ground.")

class HeadControl(Motion):
    tilt: float = MotionField(0.0, description="Tilt of the robot head in radians (down is negative).")
    pan: float = MotionField(0.0, description="Pan of the robot head in radians (left is negative).")


class MobileSingleHandControl(Motion):
    """Control for a robot that can move its base in 2D space with a 6D EEF control + grasp."""

    # Location of the robot on the ground.
    base: PlanarPose = MotionField(
        default_factory=PlanarPose,
        description="Location of the robot on the ground.",
        motion_type="relative",
    )
    hand: HandControl = MotionField(
        default_factory=HandControl,
        description="Control for the robot hand.",
        motion_type="relative",
    )
    head: HeadControl = MotionField(
        default_factory=HeadControl,
        description="Control for the robot head.",
        motion_type="relative",
    )


class MobileSingleArmControl(Motion):
    """Control for a robot that can move in 2D space with a single arm."""

    base: PlanarPose | None = MotionField(
        default_factory=PlanarPose,
        description="Location of the robot on the ground.",
    )
    arm: ndarray | None = MotionField(
        default_factory=lambda: np.zeros(7),
        description="Control for the robot arm.",
    )
    head: HeadControl | None = MotionField(default=None, description="Control for the robot head.")


class MobileBimanualArmControl(Motion):
    """Control for a robot that can move in 2D space with two arms."""

    base: PlanarPose | None = MotionField(
        default_factory=PlanarPose,
        description="Location of the robot on the ground.",
    )
    left_arm: ndarray | None = MotionField(
        default_factory=lambda: np.zeros(7),
        description="Control for the left robot arm.",
    )
    right_arm: ndarray | None = MotionField(
        default_factory=lambda: np.zeros(7),
        description="Control for the right robot arm.",
    )
    head: HeadControl | None = MotionField(default=None, description="Control for the robot head.")


class HumanoidControl(Motion):
    """Control for a humanoid robot."""

    left_arm: ndarray | None = MotionField(
        default_factory=lambda: np.zeros(7),
        description="Control for the left robot arm.",
    )
    right_arm: ndarray | None = MotionField(
        default_factory=lambda: np.zeros(7),
        description="Control for the right robot arm.",
    )
    left_leg: ndarray | None = MotionField(
        default_factory=lambda: np.zeros(6),
        description="Control for the left robot leg.",
    )
    right_leg: ndarray | None = MotionField(
        default_factory=lambda: np.zeros(6),
        description="Control for the right robot leg.",
    )
    head: HeadControl | None = MotionField(default=None, description="Control for the robot head.")


class HandTerminalControl(Motion):
    """Terminal state for a hand control task."""

    hand: HandControl | None = MotionField(
        default=None,
        description="State of the robot hand.",
    )
    done: bool | None = None
    """Whether the state is estimated to be terminal."""

