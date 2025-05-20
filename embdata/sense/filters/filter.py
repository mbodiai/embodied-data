from __future__ import annotations

from typing import Any, TypeVar, cast, get_type_hints
from weakref import ref

import open3d as o3d
from pydantic import Field

from embdata.actor import ActT as ActionType
from embdata.actor import ConfiguredActorProto
from embdata.actor import ObsT as ObservationType
from embdata.actor import RewardT as RewardType
from embdata.actor import StateT as StateType
from embdata.sample import Sample
from embdata.sense.filters.config import FilterConfig, Port
from embdata.utils.custom_logger import get_logger

o3dg = o3d.geometry
logger = get_logger(__name__)

FilterConfigT = TypeVar("FilterConfigT", bound=FilterConfig)
ParentStateType = TypeVar("ParentStateType", bound=Sample)


class FilterActor(ConfiguredActorProto[ObservationType|o3dg.PointCloud, ActionType|o3dg.PointCloud, StateType|Sample, RewardType,FilterConfigT]):
    """Abstract base class for filters that behave as Actors.

    Implements FilterActorProto and includes caching logic.
    Concrete filters must implement `_process_implementation` and
    provide their specific configuration model (subclass of BaseFilterConfig).
    Allows parameter overrides during the `act` call.
    """
    config: FilterConfigT
    parent: ref[ConfiguredActorProto[Any, Any, ParentStateType, Any, FilterConfigT]] | None = None
    depends: list[ConfiguredActorProto[Any, Any, ParentStateType, Any, FilterConfigT]] = Field(default_factory=list)


    @property
    def obs_type(self) -> type[ObservationType|o3dg.PointCloud]:
        hints = get_type_hints(self.__init__)
        if "observation" not in hints:
            msg = "observation not found in __init__"
            raise ValueError(msg)
        return hints["observation"]

    @property
    def act_type(self) -> type[ActionType|o3dg.PointCloud]:
        hints = get_type_hints(self.__init__)
        if "action" not in hints:
            msg = "action not found in __init__"
            raise ValueError(msg)
        return hints["action"]


    @property
    def state_type(self) -> type[StateType]:
        hints = get_type_hints(self.__init__)
        if "state" not in hints:
            msg = "state not found in __init__"
            raise ValueError(msg)
        return hints["state"]

    @property
    def reward_type(self) -> type[RewardType]:
        hints = get_type_hints(self.__init__)
        if "reward" not in hints:
            msg = "reward not found in __init__"
            raise ValueError(msg)
        return hints["reward"]

    @property
    def config_type(self) -> type[FilterConfigT]:
        hints = get_type_hints(self.__init__)
        if "config" not in hints:
            msg = "config not found in __init__"
            raise ValueError(msg)
        return hints["config"]


    def __init__(self, config: FilterConfigT, parent: ConfiguredActorProto | None = None):
        """Initializes the BaseFilterActor.

        Args:
            config: The configuration for this filter, including common fields
                    from BaseFilterConfig (name, cache_dir, enabled) and
                    specific fields from the ConfigType.
        """
        super().__init__(name=config.name)
        self.parent = ref(parent) if parent else None
        self.config = config


    def act(self, observation: ObservationType, **kwargs) -> ActionType:...

    def run_once(self, state: ParentStateType, config: FilterConfigT|None=None) -> Sample:
        """Run the filter once."""
        port = cast(Port, getattr(state, self.name))
        observation = port.observation
        config = getattr(port,"config", getattr(self, "config", Sample()))
        port.action = self.act(observation=observation, config=config)
        return state

