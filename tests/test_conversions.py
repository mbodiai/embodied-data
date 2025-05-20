# Copyright (c) 2024 Mbodi AI
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import pytest

import numpy as np
from embdata.utils.geometry_utils import pose_to_transformation_matrix
from scipy.spatial.transform import Rotation


def test_relative_position_description():
    # Test case 1: object to the right, forward, and above
    pose1 = (0, 0, 0, 0, 0, 0)
    pose2 = (3, 4, 5, 0, 0, 0)
    matrix1 = pose_to_transformation_matrix(pose1)
    matrix2 = pose_to_transformation_matrix(pose2)
    relative_transformation = np.linalg.inv(matrix1) @ matrix2
    assert np.allclose(
        relative_transformation,
        np.array([[1.0, 0.0, 0.0, 3.0], [0.0, 1.0, 0.0, 4.0], [0.0, 0.0, 1.0, 5.0], [0.0, 0.0, 0.0, 1.0]]),
    )
    # Test case 2: object to the left, backward, and below
    pose1 = (0, 0, 0, 0, 0, 0)
    pose2 = (-3, -4, -5, 0, 0, 0)
    matrix1 = pose_to_transformation_matrix(pose1)
    matrix2 = pose_to_transformation_matrix(pose2)
    relative_transformation = np.linalg.inv(matrix1) @ matrix2
    assert np.allclose(
        relative_transformation,
        np.array([[1.0, 0.0, 0.0, -3.0], [0.0, 1.0, 0.0, -4.0], [0.0, 0.0, 1.0, -5.0], [0.0, 0.0, 0.0, 1.0]]),
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
