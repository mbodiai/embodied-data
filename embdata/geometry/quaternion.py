from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from typing_extensions import overload, override

from embdata.array import ArrayLike, array
from embdata.ndarray import ndarray

sz = Literal
Float = np.float64 | np.floating | float


def xyzrpy_to_quatpose(xyzrpy: array[sz[6], Float]) -> "QuatPose":  # Use string for forward reference
    # This will call the __new__ of the actual QuatPose defined below
    return QuatPose(xyzrpy[:3], euler_to_quat(*xyzrpy[3:]))


def quatpose_to_xyzrpy(t: array[sz[3], Float], q: array[sz[4], Float]) -> array[sz[6], Float]:
    roll, pitch, yaw = quat_to_euler(q)
    return np.concatenate(np.array([t, [roll, pitch, yaw]]))

def rotation_matrix_to_quaternion(R: array[sz[3, 3], Float]) -> array[sz[4], Float]:
    # ≤ 10 lines standard conversion (not shown for brevity)
    w = np.sqrt(1 + np.trace(R)) / 2
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)
    return np.array([w, x, y, z], dtype=float)


@dataclass(frozen=True)
class QuatPose(tuple[array[sz[3], Float], array[sz[4], Float]]):
    t: np.ndarray[Any,Any]
    q: array[sz[4], Float]



    def relative(self, frame: "QuatPose") -> "QuatPose":
        """Pose of *self* expressed in the coordinate frame *frame*.
        (= frame⁻¹ ∘ self).
        """
        if not isinstance(frame, QuatPose):
            msg = "relative() expects a QuatPose frame"
            raise ValueError(msg)
        rel_t = quat_apply(quat_conj(frame.q), self.t - np.asarray(frame.t))
        rel_q = quat_mul(quat_conj(frame.q), self.q)
        return QuatPose(rel_t, rel_q)

    def absolute(self, frame: "QuatPose") -> "QuatPose":
        """Interpret *self* as living in *frame* and return it in the root/world
        frame.  (= frame ∘ self).
        """
        if not isinstance(frame, QuatPose):
            msg = "absolute() expects a QuatPose frame"
            raise ValueError(msg)
        abs_t = frame.t + quat_apply(frame.q, self.t)
        abs_q = quat_mul(frame.q, self.q)
        return QuatPose(abs_t, abs_q)

    def values(self) -> tuple[array[sz[3], Any], array[sz[4], Any]]:
        return self.t, self.q



    @overload
    @classmethod
    def from_pose(
        cls, x: Float = 0.0, y: Float = 0.0, z: Float = 0.0, roll: Float = 0.0, pitch: Float = 0.0, yaw: Float = 0.0,
    ) -> "QuatPose": ...
    @overload
    @classmethod
    def from_pose(cls, xyzrpy: ArrayLike[sz[6], Float]|Iterable[Float]) -> "QuatPose": ...

    @classmethod
    def from_pose(cls, *args: Any, **kwargs: Any) -> "QuatPose":
        first = next(iter(args), None)
        if first is not None and isinstance(first, list | tuple | np.ndarray) and len(first) == 6:
            xyzrpy_arr = np.asarray(first, dtype=float)
            return cls(
                t=xyzrpy_arr[:3],  # type: ignore
                q=euler_to_quat(*xyzrpy_arr[3:]),
            )
        if len(args) == 6:
            return cls(
                t=np.array([args[0], args[1], args[2]], dtype=float),  # type: ignore
                q=euler_to_quat(args[3], args[4], args[5]),
            )
        if len(args) == 3:
            return cls(
                t=np.array([args[0], args[1], args[2]], dtype=float),  # type: ignore
                q=euler_to_quat(0.0, 0.0, 0.0),  # Identity quaternion
            )
        if kwargs:
            x = kwargs.get("x", 0.0)
            y = kwargs.get("y", 0.0)
            z = kwargs.get("z", 0.0)
            roll = kwargs.get("roll", 0.0)
            pitch = kwargs.get("pitch", 0.0)
            yaw = kwargs.get("yaw", 0.0)
            return cls(
                t=np.array([x, y, z], dtype=float),  # type: ignore
                q=euler_to_quat(roll, pitch, yaw),
            )
        return cls(t=np.zeros(3, dtype=float), q=euler_to_quat(0.0, 0.0, 0.0))

    @classmethod
    def from_translation_rotation(cls, t: array[sz[3], Float], m: array[sz[3, 3], Float]) -> "QuatPose":
        return cls(t, rotation_matrix_to_quaternion(m))

    def __new__(cls, t: array[sz[3], Any], q: array[sz[4], Any]):
        # Ensure t and q are numpy arrays of float for internal consistency
        # The type hint array[sz[3],Any] is for flexibility from callers,
        # but internally we work with concrete numpy arrays.
        t_arr = np.asarray(t, dtype=float)
        q_arr = np.asarray(q, dtype=float)
        q_norm = np.linalg.norm(q_arr)
        if q_norm == 0:  # Avoid division by zero for zero quaternion
            q_arr_normalized = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)  # Default to identity
        else:
            q_arr_normalized = q_arr / q_norm  # keep unit length

        # super() for tuple needs to pass the tuple items directly
        return super().__new__(cls, (t_arr, q_arr_normalized))

    def numpy(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Return 6-vector [x y z w x y z] as a plain NumPy array."""
        return np.concatenate([np.asarray(self.t, dtype=np.float64), np.asarray(self.q, dtype=np.float64)])
    @override
    def __add__(self, other: "object") -> "QuatPose":
        if not isinstance(other, QuatPose):
            try:
                other = QuatPose.from_pose(other)
            except Exception:
                msg = f"Unsupported type: {type(other)}"
                raise ValueError(msg)
        return QuatPose(self.t + other.t, self.q * other.q)

    def __sub__(self, other: "object") -> "QuatPose":
        if not isinstance(other, QuatPose):
            try:
                other = QuatPose.from_pose(other)
            except Exception:
                msg = f"Unsupported type: {type(other)}"
                raise ValueError(msg)
        return QuatPose(self.t - other.t, self.q * other.q.T)

    def __mul__(self, other: "QuatPose") -> "QuatPose":
        return QuatPose(self.t + quat_apply(self.q, other.t), quat_mul(self.q, other.q))

    def __truediv__(self, other: "QuatPose") -> "QuatPose":
        return QuatPose(self.t - quat_apply(self.q, other.t), quat_mul(self.q, other.q.T))

    def __neg__(self) -> "QuatPose":
        return QuatPose(-self.t, -self.q)

    def __eq__(self, other: "object") -> bool:
        if not isinstance(other, QuatPose):
            try:
                other = QuatPose.from_pose(other)
            except Exception:
                msg = f"Unsupported type: {type(other)}"
                raise ValueError(msg)
        return np.allclose(self.t, other.t) and np.allclose(self.q, other.q)

    def __ne__(self, other: "QuatPose") -> bool:
        return not self == other

    def inverse(self) -> "QuatPose":
        """Return the transformation that undoes this pose (i.e. world ← self).

        For a pose represented by translation *t* and rotation *q* (unit quaternion),
        the inverse is (-q ⊗ t, q*) where q* is the quaternion conjugate.
        """
        inv_q: array[sz[4], Float] = quat_conj(self.q)
        inv_t: array[sz[3], Float] = -np.asarray(quat_apply(inv_q, self.t))
        return QuatPose(inv_t, inv_q)

@overload
def quat_to_euler(qx:Float,qy:Float,qz: Float, w: Float) -> array[sz[3], Float]: ...
@overload
def quat_to_euler(q: array[sz[4], Float] | QuatPose, scalar_first: bool = False) -> array[sz[3], Float]:...
def quat_to_euler(*args: Any,**kwargs: Any) -> array[sz[3], Float]:
    """Quaternion [w,x,y,z] → ZYX Euler angles (roll, pitch, yaw)."""
    scalar_first = kwargs.get("scalar_first", False)
    if kwargs.get("qx"):
        quat = kwargs.values()
        if kwargs.get("scalar_first"):
            msg = f"Cannot specify scalar value and keyword quaternion values. Recieved args: {args}, kw: {kwargs}"
            raise ValueError(msg)
    else:
        arglist = list(args)
        quat = kwargs.get("q",arglist.pop(0))

        if len(args) == 1 and len(qq:=args[0]) == 4:
            quat = qq
        elif len(args) == 1 and isinstance(qq:=args[0], QuatPose):
            quat = qq.q
        elif len(args) in (4,5):
            quat = np.asarray(args[:4])
            scalar_first = args[-1]
        else:
            msg = f"Unsupported type: {type(quat)}"
            raise ValueError(msg)

    if scalar_first:
        w, x, y, z = quat
    else:
        x, y, z, w = quat
    # Roll (x‑axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    # Pitch (y‑axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2_clamped = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2_clamped)

    # Yaw (z‑axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)

    return cast("array[sz[3], Float]", np.array([roll, pitch, yaw]))


def quat_mul(q1: array[sz[4], Float], q2: array[sz[4], Float]) -> array[sz[4], Float]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return ndarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
    )


def quat_conj(q: array[sz[4], Float]) -> array[sz[4], Float]:
    return q * np.array([1, -1, -1, -1])


def quat_apply(q: array[sz[4], Float], v: array[sz[3], Float]) -> array[sz[3], Float]:
    vq = cast("array[sz[4], Float]", np.concatenate([[0.0], v]))
    return quat_mul(quat_mul(q, vq), quat_conj(q))[1:]


def transform_pose(pose: QuatPose, frame: QuatPose) -> QuatPose:
    rel_t = quat_apply(quat_conj(frame.q), np.asarray(pose.t) - np.asarray(frame.t))
    rel_q = quat_mul(quat_conj(frame.q), pose.q)
    return QuatPose(rel_t, rel_q)


def compose_pose(a: QuatPose, b: QuatPose) -> QuatPose:
    t = a.t.view(np.ndarray) + quat_apply(a.q.view(np.ndarray), b.t.view(np.ndarray))
    q = quat_mul(a.q.view(np.ndarray), b.q.view(np.ndarray))
    return QuatPose(t, q)



@dataclass(frozen=True)
class Quaternion:
    w: float
    x: float
    y: float
    z: float

    def numpy(self) -> array[sz[4], Float]:
        return np.array([self.w, self.x, self.y, self.z])

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(*quat_mul(self.numpy(), other.numpy()))


    def __iter__(self) -> Iterator[Float]:
        return iter(self.numpy())
    @property
    def T(self) -> "Quaternion":
        return -self

    def __neg__(self) -> "Quaternion":
        return Quaternion(-self.w, -self.x, -self.y, -self.z)

    def __add__(self, other: "Quaternion|Float|array[sz[4],Float]") -> "Quaternion":
        if isinstance(other, Quaternion):
            return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)
        if isinstance(other, Float):
            return Quaternion(self.w + float(other), self.x + float(other), self.y + float(other), self.z + float(other))
        if isinstance(other, array):
            return Quaternion(self.w + other[0], self.x + other[1], self.y + other[2], self.z + other[3])
        msg = f"Unsupported type: {type(other)}"
        raise ValueError(msg)

    def __sub__(self, other: "Quaternion") -> "Quaternion":
        return self + (-other)

    def __truediv__(self, other: "Quaternion") -> "Quaternion":
        return self * other.T

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quaternion):
            return False
        return np.allclose(self.numpy(), other.numpy())

    @override
    def __ne__(self, other: object) -> bool:
        if not isinstance(other, Quaternion):
            return True
        return not self == other

    @override
    def __hash__(self) -> int:
        return hash(self.numpy().tobytes())

    def inverse(self) -> "Quaternion":
        """Return the multiplicative inverse of this quaternion.

        For unit quaternions (the usual case for rotations) this is just the
        conjugate.  For non-unit quaternions we divide by the squared norm.
        """
        arr = self.numpy()
        norm_sq = float(np.dot(arr, arr))
        if norm_sq == 0.0:
            msg = "Cannot invert a zero-norm quaternion"
            raise ZeroDivisionError(msg)
        inv_arr = np.asarray(quat_conj(arr)) / norm_sq
        return Quaternion(*inv_arr)


# ---------- Quaternion / Euler helpers ---------------------------------
def euler_to_quat(roll: Float, pitch: Float, yaw: Float) -> array[sz[4], Float]:
    """Convert ZYX Euler (roll=X, pitch=Y, yaw=Z) → quaternion [w,x,y,z]."""
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return cast("array[sz[4], Float]", np.array([w, x, y, z]))


def demo() -> None:
    # random original pose
    p = QuatPose(t=np.random.uniform(-10, 10, 3), q=euler_to_quat(*np.random.uniform(-np.pi, np.pi, 3)))

    # random frame (new origin/orientation)
    f = QuatPose(t=np.random.uniform(-5, 5, 3), q=euler_to_quat(*np.random.uniform(-np.pi, np.pi, 3)))

    p_in_f = transform_pose(p, f)
    p_back = compose_pose(f, p_in_f)

    # Check numeric equality
    np.allclose(p.t, p_back.t, atol=1e-8)
    np.allclose(p.q, p_back.q, atol=1e-8) or np.allclose(p.q, -p_back.q, atol=1e-8)

    quatpose_to_xyzrpy(p.t, p.q)
    quatpose_to_xyzrpy(p_in_f.t, p_in_f.q)



    # Reuse helper functions from previous cell (already defined in this notebook context)
    def random_xyzrpy():
        return np.concatenate(
            [
                np.random.uniform(-10, 10, 3),  # x,y,z
                np.random.uniform(-np.pi, np.pi, 3),  # roll,pitch,yaw
            ],
        )


    @overload
    def transform_xyzrpy(
        transform: array[sz[6], Float], frame: array[sz[6], Float], to: Literal["tuple"],
    ) -> tuple[array[sz[6], Float], tuple[array[sz[3], Float], array[sz[4], Float]]]: ...
    @overload
    def transform_xyzrpy(
        transform: array[sz[6], Float], frame: array[sz[6], Float], to: Literal["array"] = "array",
    ) -> array[sz[6], Float]: ...


    def transform_xyzrpy(transform: Any, frame: Any, to: Any = "array") -> Any:
        t_p, q_p = xyzrpy_to_quatpose(transform)
        t_f, q_f = xyzrpy_to_quatpose(frame)
        rel_t = quat_apply(quat_conj(q_f), t_p - t_f)
        rel_q = quat_mul(quat_conj(q_f), q_p)
        # back to xyzrpy
        roll, pitch, yaw = quat_to_euler(rel_q)
        if to == "array":
            return np.concatenate(np.array([rel_t, [roll, pitch, yaw]]))
        return np.concatenate(np.array([rel_t, [roll, pitch, yaw]])), (rel_t, rel_q)



    np.random.seed(7)
    N = 1000
    tol = 1e-5
    fail = 0
    for _ in range(N):
        pose_xyzrpy = random_xyzrpy()
        frame_xyzrpy = random_xyzrpy()

        transformed_xyzrpy, (rel_t, rel_q) = transform_xyzrpy(pose_xyzrpy, frame_xyzrpy, to="tuple")

        # Transform back: frame ∘ relative
        t_f, q_f = xyzrpy_to_quatpose(frame_xyzrpy)
        t_back = t_f + quat_apply(q_f, rel_t)
        q_back = quat_mul(q_f, rel_q)

        t_orig, q_orig = xyzrpy_to_quatpose(pose_xyzrpy)

        # compare translations
        t_ok = np.allclose(t_orig, t_back, atol=tol)
        # compare quaternions up to sign
        q_ok = np.allclose(q_orig, q_back, atol=tol) or np.allclose(q_orig, -q_back, atol=tol)

        if not (t_ok and q_ok):
            fail += 1


if __name__ == "__main__":
    demo()
