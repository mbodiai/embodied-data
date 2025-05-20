"""Fast **O(1)** transform look-ups.

The :class:`TransformLookup` registry stores the *absolute* pose of each
frame (as a :class:`~embdata.coordinate.Pose6D`) and pre-computes the
relative transform for every pair of frames.  Once initialised, queries
are answered in constant time::

    >>> from embdata.utils import TransformLookup
    >>> lookup = TransformLookup([camera_pose, object_pose])
    >>> pose_of_object_in_camera = lookup("camera", "object")  # O(1)

The class is optimised for *read* performance.  Registering additional
frames incurs an :math:`O(N)` update cost since the new frame's
relationship to every existing one has to be computed, but this happens
only once per new frame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

    from embdata.coordinate import Pose6D
    from embdata.sense.world_object import WorldObject


from typing_extensions import Self

from embdata.sense.world_object import WorldObject

_ = WorldObject.model_rebuild()
_empty_world_object =None

def get_empty_world_object() -> WorldObject:
    from embdata.coordinate import Pose6D
    from embdata.sense.scene import EmptyScene
    from embdata.sense.world_object import WorldObject
    _ =WorldObject.model_rebuild()
    global _empty_world_object
    if _empty_world_object is None:
        _empty_world_object = WorldObject(name="world",pose=Pose6D.zeros())
        empty_scene = EmptyScene()
        empty_scene.add_child(_empty_world_object)
    return _empty_world_object

def lookup_scene(scene: str) -> WorldObject:
    from embdata.sense.scene import EmptyScene

    empty_scene = EmptyScene()
    return empty_scene.nodes.get(scene, get_empty_world_object())

class TransformLookup:
    """Registry resolving relative transforms in constant time."""

    _singleton: TransformLookup | None = None
    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
        return cls._singleton

    def __init__(self, poses: Iterable[Pose6D] | None = None) -> None:

        self._abs: dict[str, Pose6D] = {}
        self._rel:dict[tuple[str, str], Pose6D] = {}

        if poses is not None:
            for p in poses:
                self.register(p)

    def register(self, pose: Pose6D) -> None:
        """Add *pose* to the registry and update the lookup table.

        If a frame of the same name already exists it is replaced with the
        new pose.
        """
        frame = pose.reference_frame()
        self._abs[frame] = pose
        # Drop cached transforms that involve *frame* as they are now stale.
        self._rel = {k: v for k, v in self._rel.items() if frame not in k}
        self._precompute_relations_for(frame)

    def __call__(self, source: str, target: str) -> Pose6D:
        """Return the pose of *target* expressed in *source*'s frame."""
        from embdata.coordinate import Pose6D
        if source == target:

            return Pose6D()
        try:
            return self._rel.get((source, target),Pose6D())
        except KeyError as exc:
            missing = [f for f in (source, target) if f not in self._abs]
            msg = f"Unknown frame(s): {', '.join(missing)}"
            raise KeyError(
                msg,
            ) from exc

    def _precompute_relations_for(self, new_frame: str) -> None:
        """Compute relative poses involving *new_frame* in both directions."""
        from embdata.coordinate import Pose6D
        new_abs = self._abs[new_frame]

        # Self-transform is the identity.
        self._rel[(new_frame, new_frame)] = Pose6D()

        for other_frame, other_abs in self._abs.items():
            if other_frame == new_frame:
                continue

            # Transform *other* expressed in *new* frame.
            self._rel[(new_frame, other_frame)] = other_abs.relative(new_abs)
            # And the inverse.
            self._rel[(other_frame, new_frame)] = new_abs.relative(other_abs)


def lookup_transform(from_frame: str, to_frame: str="world") -> Pose6D:
    """Lookup the transform between two frames."""
    return TransformLookup()(from_frame, to_frame)
