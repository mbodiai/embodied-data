
from typing import Any, cast, overload

import numpy as np
from typing_extensions import Literal

from embdata.array import array
from embdata.ndarray import Float, M, N, P, sz


@overload
def get_intrinsic_orientation(
    pca_vectors: array[sz[3], sz[3],Float], pca_variances: array[sz[3],Float], plane_normal: array[sz[3],Float],
) -> array[sz[3], sz[3],Float]:...
@overload
def get_intrinsic_orientation(
    longest_to_shortest: array[sz[3], sz[3],Float], plane_normal: array[sz[3],Float],
) -> array[sz[3], sz[3],Float]:...
def get_intrinsic_orientation(*args:Any,**kwargs:Any)->Any:
    """Determine and align the orientation of principal components based on object shape.

    This function interprets eigenvalues and eigenvectors from a principal component analysis
    (PCA) to determine the orientation of an object relative to a reference plane. For objects
    classified as "tall," the primary component (v_x) aligns with both the largest principal component
    (v1) and the plane normal. For "flat" objects, the primary component remains the longest (v1),
    but the third component aligns with the plane normal.

    The alignment steps vary depending on object shape as follows:

    **Tall objects**:
    - v_x is aligned with the longest component (v1) and also with the plane normal.
    - v_y and v_z are selected to form a right-handed coordinate system, ensuring orthogonality.

    **Flat objects**:
    - v_x is assigned to the longest component (v1).
    - v_z aligns with the plane normal, choosing the most aligned component between v2 and v3.
    - v_y is computed to complete a right-handed coordinate system.

    Parameters:
        eigenvalues (np.ndarray): Array of sorted eigenvalues from the covariance matrix in descending order.
        eigenvectors (np.ndarray): Array of eigenvectors as column vectors representing principal directions.
        plane_normal (np.ndarray): Normal vector of the reference plane for alignment, assumed to be a unit vector.

    Returns:
        np.ndarray: A 3x3 rotation matrix representing the object's orientation, where each column corresponds
            to the x, y, and z axes of the object frame.

    Example:
        >>> eigenvalues = np.array([5.0, 2.0, 1.0])
        >>> eigenvectors = np.array([[0.7, 0.2, 0.1], [0.3, 0.9, 0.2], [0.6, 0.1, 0.8]])
        >>> plane_normal = np.array([0.0, 0.0, 1.0])
        >>> R_object = determine_pca_orientation(eigenvalues, eigenvectors, plane_normal)

    The function first assigns the primary axis (v_x) to the largest principal component (v1),
    assuming it represents the object's main extent. It then determines if this primary axis
    aligns with the plane normal based on the dot product. For "tall" objects, v_x is aligned
    with the plane normal, and v_y and v_z are assigned to form an orthogonal right-handed
    coordinate frame. For "flat" objects, the function checks v2 and v3 for the best alignment
    with the plane normal, assigns the closest as v_z, and calculates v_y to ensure a consistent
    orientation.

    Notes:
        - A threshold of 0.9 is used on the dot product between v_x and the plane normal to
          decide alignment. This is adjustable based on tolerance for near-alignment.
    """
    pca_vectors = pca_variances = plane_normal = None
    if kwargs.get("pca_variances") is not None:
        pca_variances = kwargs.get("pca_variances")
    if kwargs.get("pca_vectors") is not None:
        pca_vectors = kwargs.get("pca_vectors")
    if kwargs.get("longest_to_shortest") is not None:
        pca_vectors = kwargs.get("longest_to_shortest")
    if kwargs.get("plane_normal") is not None:
        plane_normal = kwargs.get("plane_normal")
    if not pca_vectors:
        if len(args) < 1:
            msg = "Either pca_vectors or longest_to_shortest must be provided."
            raise ValueError(msg)
        pca_vectors = args[0]
        if not pca_variances:
            if len(args) < 2:
                msg = "pca_variances must be provided."
                raise ValueError(msg)
            pca_variances = args[1]
        if not plane_normal:
            if len(args) < 3:
                msg = "plane_normal must be provided."
                raise ValueError(msg)
            plane_normal = args[2]
    elif not plane_normal:
        if len(args) < 1:
            msg = "plane_normal must be provided."
            raise ValueError(msg)
        plane_normal = args[0]

    # Extract principal components
    v1 = pca_vectors[:, 0]  # Largest variance
    v2 = pca_vectors[:, 1]  # Intermediate variance
    v3 = pca_vectors[:, 2]  # Smallest variance

    # Assign the primary axis (v_x) to the largest principal component (v1)
    v_x = v1

    # Check alignment of v_x with plane_normal
    dot_v1 = np.abs(np.dot(v_x, plane_normal))

    # Define a threshold to determine # D# Define a threshold to determine if v_x is aligned with plane_normal
    # For example, if the dot product is close to 1, they are aligned.
    threshold = 0.9
    if dot_v1 > threshold:
        # The object is tall, and v_x is along plane_normal
        # Assign v_x to v1 (already done)
        # For v_y and v_z, choose to form a right-handed coordinate system
        v_y = v2  # Use v2 for v_y
        v_z = np.cross(v_x, v_y)
        v_z /= np.linalg.norm(v_z)
        v_y = np.cross(v_z, v_x)
        v_y /= np.linalg.norm(v_y)
    else:
        # The object is flat or not aligned with plane_normal
        # Assign v_x to v1
        # Assign v_z to the principal component that aligns most with plane_normal among v2 and v3
        dot_v2 = np.abs(np.dot(v2, plane_normal))
        dot_v3 = np.abs(np.dot(v3, plane_normal))
        v_z = v2 if dot_v2 > dot_v3 else v3
        # Align v_z with plane_normal
        if np.dot(v_z, plane_normal) < 0:
            v_z = -v_z
        # Compute v_y to form a right-handed coordinate system
        v_y = np.cross(v_z, v_x)
        v_y /= np.linalg.norm(v_y)
        # Recompute v_z to ensure orthogonality
        v_z = np.cross(v_x, v_y)
        v_z /= np.linalg.norm(v_z)
        # v_x is already normalized since eigenvectors are unit vectors

    # Construct the rotation matrix from x, y, z unit vectors
    return np.column_stack((v_x, v_y, v_z))


def pca(samples: array[N,M,Float], num_components: P = 3) -> tuple[array[P,M,Float], array[P,Float]]:
    # Center the data
    center = samples.mean(axis=0)
    centered = samples - center

    # Compute covariance matrix
    covs = np.cov(centered, rowvar=False)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covs)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]

    # Select the top 'num_components' principal components
    principal_directions = eigenvectors[:, :num_components]  # No transpose

    # Ensure the directions are unit vectors
    principal_directions /= np.linalg.norm(principal_directions, axis=0, keepdims=True)


    # Enforce the right-hand rule (as before)
    matrix = principal_directions[:, :3]
    det = np.linalg.det(matrix)
    if det < 0:
        principal_directions[:, 0] = -principal_directions[:, 0]
        matrix = principal_directions[:, :3]

    # Check handedness
    v1 = principal_directions[:, 0]
    v2 = principal_directions[:, 1]
    v3 = principal_directions[:, 2]
    if check_handedness(v1, v2, v3) != "R":
        msg = "The principal directions do not form a right-handed frame."
        raise ValueError(msg)
    return principal_directions, cast("array[P, Float]", eigenvalues[:num_components])


def check_handedness(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> Literal["R", "L"]:
    """Check if three orthogonal vectors form a right-handed or left-handed frame.

    Args:
        v1 (np.ndarray): First vector.
        v2 (np.ndarray): Second vector.
        v3 (np.ndarray): Third vector.

    Returns:
        str: "right-handed" if the frame follows the right-hand rule,
             "left-handed" otherwise.

    Example:
        >>> v1 = np.array([1, 0, 0])
        >>> v2 = np.array([0, 1, 0])
        >>> v3 = np.array([0, 0, 1])
        >>> check_hand_frame(v1, v2, v3)
        'right-handed'
    """
    # Compute the cross product of the first two vectors
    cross_product = np.cross(v1, v2)

    # Check if the cross product matches the third vector
    if np.allclose(cross_product, v3, atol=1e-3):  # Relaxed tolerance
        return "R"
    return "L"


