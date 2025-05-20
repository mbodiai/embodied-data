# Copyright 2024 Mbodi AI
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

import pytest
import numpy as np
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from embdata.motion.control import AnyMotionControl
from embdata.motion.control import (
    HandControl,
    HeadControl,
    MobileSingleArmControl,
)
from embdata.motion import Motion, MotionField

from embdata.coordinate import Pose6D
from embdata.coordinate import PlanarPose


@pytest.fixture(autouse=True)
def mock_file():
    with TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "test.h5"
        yield filepath


def test_location_angle_serialization():
    la = PlanarPose(x=0.5, y=-0.5, theta=1.57)
    json = la.dict()
    assert json == {"x": 0.5, "y": -0.5, "theta": 1.57}


def test_location_angle_deserialization():
    json_data = '{"x": 0.5, "y": -0.5, "theta": 1.57}'
    la = PlanarPose.model_validate_json(json_data)
    assert la.x == 0.5 and la.y == -0.5 and la.theta == 1.57


def test_pose6d_serialization():
    pose = Pose6D(x=1, y=0.9, z=0.9, roll=0.1, pitch=0.2, yaw=0.3)
    json_data = pose.model_dump_json()
    expected = {"x": 1, "y": 0.9, "z": 0.9, "roll": 0.1, "pitch": 0.2, "yaw": 0.3}
    assert json.loads(json_data) == expected


def test_pose6d_deserialization():
    json_data = '{"x": 1, "y": 0.9, "z": 0.9, "roll": 0.1, "pitch": 0.2, "yaw": 0.3}'
    pose = Pose6D.model_validate_json(json_data)
    assert (pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw) == (1, 0.9, 0.9, 0.1, 0.2, 0.3)


def test_full_joint_control_serialization():
    fjc = AnyMotionControl(joints=[2.5, -1.0], names=["elbow", "wrist"])
    expected = '{"names":["elbow","wrist"],"joints":[2.5,-1.0]}'
    assert fjc.model_dump_json() == expected


def test_full_joint_control_deserialization():
    json_data = '{"joints": [2.5, -1.0], "names": ["elbow", "wrist"]}'
    fjc = AnyMotionControl.model_validate_json(json_data)
    assert fjc.joints == [2.5, -1.0]
    assert fjc.names == ["elbow", "wrist"]


def test_mobile_single_arm_control_serialization():
    msac = MobileSingleArmControl(
        base=PlanarPose(x=0.5, y=-0.5, theta=1.57),
        arm=[2.5, -1.0],
        head=HeadControl(tilt=1.0, pan=-1.0),
    )
    json = msac.dict()
    expected = {
        "base": {"x": 0.5, "y": -0.5, "theta": 1.57},
        "arm": [2.5, -1.0],
        "head": {"tilt": 1.0, "pan": -1.0},
    }
    assert json["base"] == expected["base"]
    assert np.array_equal(json["arm"], expected["arm"])
    assert json["head"] == expected["head"]


def test_mobile_single_arm_control_deserialization():
    joints = [2.5, -1.0]
    names = ["elbow", "wrist"]

    motion = AnyMotionControl(joints=joints, names=names)

    json_data = '{"base": {"x": 0.5, "y": -0.5, "theta": 1.57}, "arm": [2.5,-1.0], "head": {"tilt": 1.0, "pan": -1.0}}'
    msac = MobileSingleArmControl.model_validate_json(json_data)
    assert (msac.base.x, msac.base.y, msac.base.theta) == (0.5, -0.5, 1.57)
    assert np.array_equal(msac.arm, [2.5, -1.0])
    assert msac.head.tilt == 1.0
    assert msac.head.pan == -1.0


def test_hand_control_serialization():
    hc = HandControl(pose=Pose6D(x=0.5, y=-0.5, z=0.5, roll=0.5, pitch=-0.5, yaw=0.5), grasp=1.0)
    json_data = json.dumps(hc.dict())
    expected = {
        "pose": {"x": 0.5, "y": -0.5, "z": 0.5, "roll": 0.5, "pitch": -0.5, "yaw": 0.5},
        "grasp": 1.0,
    }
    assert json.loads(json_data) == expected


def test_hand_control_deserialization():
    json_data = '{"pose": {"x": 0.5, "y": -0.5, "z": 0.5, "roll": 0.5, "pitch": -0.5, "yaw": 0.5}, "grasp": 1.0}'
    hc = HandControl.model_validate_json(json_data)
    assert (hc.pose.x, hc.pose.y, hc.pose.z) == (0.5, -0.5, 0.5)
    assert (hc.pose.roll, hc.pose.pitch, hc.pose.yaw) == (0.5, -0.5, 0.5)
    assert hc.grasp == 1.0


def test_unflatten():
    original_pose = PlanarPose(x=0.5, y=-0.5, theta=1.57)
    flattened_pose = original_pose.flatten(to="dict")

    schema = {
        "type": "object",
        "properties": {
            "x": {"type": "number"},
            "y": {"type": "number"},
            "theta": {"type": "number"},
        },
    }
    unflattened_pose = PlanarPose.unflatten(flattened_pose, schema)

    assert unflattened_pose.x == original_pose.x
    assert unflattened_pose.y == original_pose.y
    assert unflattened_pose.theta == original_pose.theta


def test_unflatten_hand_control():
    original_pose = HandControl(pose=Pose6D(x=0.5, y=-0.5, z=0.5, roll=0.5, pitch=-0.5, yaw=0.5), grasp=1.0)
    flattened_pose = original_pose.flatten(to="dict")

    schema = {
        "type": "object",
        "properties": {
            "pose": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "z": {"type": "number"},
                    "roll": {"type": "number"},
                    "pitch": {"type": "number"},
                    "yaw": {"type": "number"},
                },
            },
            "grasp": {"type": "number"},
        },
    }
    unflattened_pose = HandControl.unflatten(flattened_pose, schema)

    assert unflattened_pose.pose.x == original_pose.pose.x
    assert unflattened_pose.pose.y == original_pose.pose.y
    assert unflattened_pose.pose.z == original_pose.pose.z
    assert unflattened_pose.pose.roll == original_pose.pose.roll
    assert unflattened_pose.pose.pitch == original_pose.pose.pitch
    assert unflattened_pose.pose.yaw == original_pose.pose.yaw
    assert unflattened_pose.grasp == original_pose.grasp


def test_bounds():
    class XarmPose6D(Motion):
        """Movement for a 6D space representing x, y, z, roll, pitch, and yaw."""

        x: float = MotionField(
            default=0,
            description="X position in 3D space. +x is forward; -x is backward.",
            bounds=(-0.3, 0.4),
        )
        y: float = MotionField(
            default=0,
            description="Y position in 3D space. +y is left; -y is right.",
            bounds=(-0.4, 0.4),
        )
        z: float = MotionField(
            default=0,
            description="Z position in 3D space. +z is up; -z is down.",
            bounds=(-0.175, 0.4),
        )
        roll: float = MotionField(
            default=0,
            description="Roll about the X-axis in radians. Positive roll is clockwise.",
            bounds=(-np.pi / 2, np.pi / 2),
        )
        pitch: float = MotionField(
            default=0,
            description="Pitch about the Y-axis in radians. Positive pitch is down.",
            bounds=(-np.pi / 2, np.pi / 2),
        )
        yaw: float = MotionField(
            default=0,
            description="Yaw about the Z-axis in radians. Positive yaw is left.",
            bounds=(-np.pi, np.pi),
        )

    class XarmHandControl(HandControl):
        """Action for a 7D space representing x, y, z, roll, pitch, yaw, and oppenness of the hand."""

        pose: XarmPose6D = MotionField(default_factory=XarmPose6D, description="6D pose of the robot hand.")
        grasp: float = MotionField(
            default=0,
            description="Openness of the robot hand. 0 is closed, 1 is open.",
            bounds=(0, 1),
        )

    xarm = XarmHandControl()
    assert xarm.pose.field_info("x")["bounds"] == (-0.3, 0.4)
    assert xarm.pose.field_info("y")["bounds"] == (-0.4, 0.4)
    assert xarm.pose.field_info("z")["bounds"] == (-0.175, 0.4)
    assert xarm.pose.field_info("roll")["bounds"] == (-np.pi / 2, np.pi / 2)
    assert xarm.pose.field_info("pitch")["bounds"] == (-np.pi / 2, np.pi / 2)
    assert xarm.pose.field_info("yaw")["bounds"] == (-np.pi, np.pi)
    assert xarm.field_info("grasp")["bounds"] == (0, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
