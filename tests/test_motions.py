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
from embdata.motion import AnyMotionControl
from embdata.motion.control import (
    HandControl,
    HeadControl,
    MobileSingleArmControl,
)
from embdata.motion import Motion, MotionField

from embdata.geometry import Pose6D
from embdata.geometry import PlanarPose


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
    flattened_pose = original_pose.flatten(output_type="dict")

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


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
