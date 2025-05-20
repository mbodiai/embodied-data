# This script demonstrates the usage of scipy.spatial.transform.Rotation
# based on its official documentation examples.

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R


def plot_rotated_axes(ax, r_instance, name=None, offset=(0, 0, 0), scale=1) -> None:
    """Helper function to plot a named rotated coordinate frame."""
    colors = ("#FF6666", "#005533", "#1199EE")  # R, G, B like colors for X, Y, Z
    loc = np.array(offset)

    axes_labels = ["X", "Y", "Z"]

    for i in range(3): # X, Y, Z axes
        # Define standard basis vector
        line = np.zeros((2, 3))
        line[1, i] = scale  # Vector from origin to 'scale' along i-th axis

        # Apply rotation
        line_rot = r_instance.apply(line)

        # Offset to plot position
        line_plot = line_rot + loc

        # Plot the rotated axis line
        ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], color=colors[i], linewidth=2)

        # Position for axis label
        text_loc_vec = np.zeros(3)
        text_loc_vec[i] = scale * 1.2 # Place label slightly beyond the axis line
        text_loc_rot = r_instance.apply(text_loc_vec)
        text_plot = text_loc_rot + loc

        ax.text(*text_plot, axes_labels[i], color=colors[i], va="center", ha="center", fontsize=9)

    if name:
        ax.text(*loc, name, color="k", va="center", ha="center",
                bbox={"fc": "w", "alpha": 0.8, "boxstyle": "round,pad=0.3"}, fontsize=10, zorder=10)


def run_all_examples() -> None:
    """Runs all examples from the scipy.spatial.transform.Rotation documentation."""
    # Example 1: Initialize from Quaternion
    quat_ex1 = np.array([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)], dtype=float)
    R.from_quat(quat_ex1)

    # Example 2: Initialize from Rotation Matrix
    matrix_ex2 = np.array([[0, -1, 0],
                             [1,  0, 0],
                             [0,  0, 1]], dtype=float)
    R.from_matrix(matrix_ex2)

    # Example 3: Initialize from Rotation Vector
    rotvec_ex3 = np.array([0, 0, np.pi/2], dtype=float)
    R.from_rotvec(rotvec_ex3)

    # Example 4: Initialize from Euler Angles (single axis)
    R.from_euler("z", [90.0], degrees=True)

    # Example 5: Initialize multiple rotations (Euler) and apply to vector
    rotations_euler_multi_list = [
        [90.0, 0.0, 0.0],
        [0.0, 45.0, 0.0],
        [45.0, 60.0, 30.0],
    ]
    rotations_euler_multi_np = np.array(rotations_euler_multi_list, dtype=float)
    r_ex5 = R.from_euler("zyx", rotations_euler_multi_np.tolist(), degrees=True)
    v_ex5 = np.array([1, 2, 3], dtype=float)
    r_ex5.apply(v_ex5)

    # Example 6: Indexing and Slicing Rotations
    r_ex5[0]
    r_ex5[1:3]

    # Example 7: Convert to NumPy array
    r_array_ex7 = np.asarray(r_ex5)
    if len(r_array_ex7) > 0:
        pass
    else:
        pass


    # Example 8: Rotation Composition
    r1_ex8 = R.from_euler("z", [90.0], degrees=True)
    r2_ex8 = R.from_rotvec(np.array([np.pi/4, 0, 0], dtype=float))
    v_ex8 = np.array([1, 2, 3], dtype=float)
    r2_ex8.apply(r1_ex8.apply(v_ex8))
    r3_ex8 = r2_ex8 * r1_ex8
    r3_ex8.apply(v_ex8)

    # Example 9: Rotation Self-Composition (Power)
    p_ex9 = R.from_rotvec(np.array([1, 0, 0], dtype=float))
    p_ex9 ** 2

    # Example 10: Rotation Inversion
    r1_ex10_angles = np.array([90.0, 45.0], dtype=float)
    r1_ex10 = R.from_euler("z", r1_ex10_angles.tolist(), degrees=True)
    r1_ex10.inv()

    # Plotting Example

    angles_plot_np = np.array([90.0, -30.0, 0.0], dtype=float)
    angles_plot_list = angles_plot_np.tolist()
    r0_plot = R.identity()
    r1_plot = R.from_euler("ZYX", angles_plot_list, degrees=True)  # intrinsic ZYX
    r2_plot = R.from_euler("zyx", angles_plot_list, degrees=True)  # extrinsic zyx
    r3_plot = R.from_euler("XYZ", angles_plot_list, degrees=True)  # intrinsic XYZ

    # Verify ZYX intrinsic vs zyx extrinsic equivalence for the same angles
    if np.allclose(r1_plot.as_matrix(), r2_plot.as_matrix()):
        pass
    else:
        pass

    # Compare with Intrinsic XYZ
    if np.allclose(r1_plot.as_matrix(), r3_plot.as_matrix()):
        pass
    else:
        pass


    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(111, projection="3d", proj_type="ortho")

    plot_rotated_axes(ax, r0_plot, name="r0 (Identity)", offset=(0, 0, 0), scale=1)
    plot_rotated_axes(ax, r1_plot, name="r1 (ZYX intrinsic)", offset=(2.5, 0, 0), scale=1)
    plot_rotated_axes(ax, r2_plot, name="r2 (zyx extrinsic)", offset=(5.0, 0, 0), scale=1)
    plot_rotated_axes(ax, r3_plot, name="r3 (XYZ intrinsic)", offset=(7.5, 0, 0), scale=1)

    annotation_text = (
        "r0: Identity Rotation\n"
        "r1: Intrinsic Euler (ZYX: [90, -30, 0])\n"
        "r2: Extrinsic Euler (zyx: [90, -30, 0])\n"
        "    (Note: r1 and r2 are equivalent)\n"
        "r3: Intrinsic Euler (XYZ: [90, -30, 0])"
    )
    ax.text(0.02, 0.98, annotation_text, transform=ax.transAxes, ha="left", va="top",
              bbox={"boxstyle": "round,pad=0.5", "fc": "aliceblue", "ec": "gray", "alpha": 0.9},
              fontsize=9)

    ax.set_xlabel("Global X-axis", labelpad=10)
    ax.set_ylabel("Global Y-axis", labelpad=10)
    ax.set_zlabel("Global Z-axis", labelpad=10)

    ax.set_xlim((-1.5, 9.0))
    ax.set_ylim((-1.5, 2.5))
    ax.set_zlim((-1.5, 1.5))

    ax_length = 1.0
    ax.plot([-ax_length, ax_length], [0,0], [0,0], color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.plot([0,0], [-ax_length, ax_length], [0,0], color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.plot([0,0], [0,0], [-ax_length, ax_length], color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.text(ax_length + 0.1, 0, 0, "X_g", color="gray", fontsize=8) # Shortened labels
    ax.text(0, ax_length + 0.1, 0, "Y_g", color="gray", fontsize=8)
    ax.text(0, 0, ax_length + 0.1, "Z_g", color="gray", fontsize=8)

    ax.set_title("Visualization of 3D Rotations", fontsize=14, pad=20)

    ax.view_init(elev=20, azim=-60)

    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout(rect=(0, 0, 1, 0.96))

def assert_xyz_is_zyx_is_ZYX() -> None:
    """Tests Euler angle sequence behaviors based on Scipy's `from_euler` conventions.
    - Confirms the equivalence: Intrinsic 'ZYX' with angles [a,b,c] is equivalent to
      Extrinsic 'xyz' with REVERSED angles [c,b,a].
    - Shows other non-equivalences for clarity.
    """
    # Generate random angles in radians
    angles_abc_np = np.random.uniform(-np.pi, np.pi, size=3)
    angles_abc_list = angles_abc_np.tolist()
    angles_cba_list = angles_abc_np[::-1].tolist() # Reversed angles [c,b,a]

    _a, _b, _c = angles_abc_list[0], angles_abc_list[1], angles_abc_list[2]

    # Calculate matrices for each sequence using these radian angles
    # Scipy conventions:
    # R.from_euler('ZYX', [a,b,c]) -> M = R_Z(a)R_Y(b)R_X(c)
    # R.from_euler('xyz', [j,k,l]) -> M = R_x(j)R_y(k)R_z(l)
    # R.from_euler('zyx', [a,b,c]) -> M = R_z(a)R_y(b)R_x(c)

    m_ZYX_intrinsic_abc = R.from_euler("ZYX", angles_abc_list, degrees=False).as_matrix()
    m_xyz_extrinsic_abc = R.from_euler("xyz", angles_abc_list, degrees=False).as_matrix()
    m_xyz_extrinsic_cba = R.from_euler("xyz", angles_cba_list, degrees=False).as_matrix()
    m_zyx_extrinsic_abc = R.from_euler("zyx", angles_abc_list, degrees=False).as_matrix()

    # Test 1: Intrinsic 'ZYX' [a,b,c] vs Extrinsic 'xyz' with REVERSED angles [c,b,a]
    # This is a standard equivalence: R_Z(a)R_Y(b)R_X(c) == R_x(c)R_y(b)R_z(a)
    # Scipy: R_x(angles_cba_list[0])R_y(angles_cba_list[1])R_z(angles_cba_list[2])
    # where angles_cba_list[0]=c, angles_cba_list[1]=b, angles_cba_list[2]=a.
    are_ZYX_abc_and_xyz_cba_close = np.allclose(m_ZYX_intrinsic_abc, m_xyz_extrinsic_cba)
    assert are_ZYX_abc_and_xyz_cba_close, \
        "Intrinsic 'ZYX' [a,b,c] and Extrinsic 'xyz' [c,b,a] (Scipy convention) should be equivalent."

    # Test 2: Intrinsic 'ZYX' [a,b,c] vs Extrinsic 'xyz' [a,b,c] (SAME angles)
    # R_Z(a)R_Y(b)R_X(c) vs R_x(a)R_y(b)R_z(c) - Generally NOT equivalent.
    are_ZYX_abc_and_xyz_abc_close = np.allclose(m_ZYX_intrinsic_abc, m_xyz_extrinsic_abc)
    if are_ZYX_abc_and_xyz_abc_close:
        pass
    else:
        pass

    # Test 3: Intrinsic 'ZYX' [a,b,c] vs Extrinsic 'zyx' [a,b,c] (Scipy definition for 'zyx')
    # ZYX intrinsic: R_Z(a)R_Y(b)R_X(c)
    # zyx extrinsic (Scipy): R_z(a)R_y(b)R_x(c) - Generally NOT equivalent.
    are_ZYX_abc_and_zyx_abc_close = np.allclose(m_ZYX_intrinsic_abc, m_zyx_extrinsic_abc)
    if are_ZYX_abc_and_zyx_abc_close:
        pass
    else:
        pass


    # Test User's Desired Convention
    # Intrinsic 'XYZ' with angles [roll, pitch, yaw]
    # vs.
    # Extrinsic 'zyx' with angles [yaw, pitch, roll]
    roll_val = np.random.uniform(-np.pi, np.pi)
    pitch_val = np.random.uniform(-np.pi, np.pi)
    yaw_val = np.random.uniform(-np.pi, np.pi)

    angles_rpy_list = [roll_val, pitch_val, yaw_val]
    angles_ypr_list = [yaw_val, pitch_val, roll_val]


    m_intrinsic_XYZ_rpy = R.from_euler("XYZ", angles_rpy_list, degrees=False).as_matrix()
    m_extrinsic_zyx_ypr = R.from_euler("zyx", angles_ypr_list, degrees=False).as_matrix()

    are_intrinsic_XYZ_rpy_and_extrinsic_zyx_ypr_close = np.allclose(m_intrinsic_XYZ_rpy, m_extrinsic_zyx_ypr)
    assert are_intrinsic_XYZ_rpy_and_extrinsic_zyx_ypr_close, \
        "Intrinsic 'XYZ' [r,p,y] should be equivalent to Extrinsic 'zyx' [y,p,r]."


if __name__ == "__main__":
    assert_xyz_is_zyx_is_ZYX()
    # run_all_examples()
    # plt.ion()
    # plt.show(block=True)

