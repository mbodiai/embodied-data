"""The categories of the system."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Type, TypeVar

from embdata.sample import Sample

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

SampleState = TypeVar("SampleState", bound=Sample)

ObsT = TypeVar("ObsT", bound=Sample|object|None)
ActT = TypeVar("ActT", bound=Sample|object|None)
StateT = TypeVar("StateT", bound=Sample|None)
RewardT = TypeVar("RewardT", bound=Sample|float|object|None)

ObsT_co = TypeVar("ObsT_co", bound=Sample, covariant=True)
ActT_co = TypeVar("ActT_co", bound=Sample, covariant=True)
StateT_co = TypeVar("StateT_co", bound=Sample, covariant=True)
RewardT_co = TypeVar("RewardT_co", bound=Sample|float, covariant=True)
ConfigT = TypeVar("ConfigT", bound=Sample|object|None)


class BaseActor(Protocol[ObsT, ActT, StateT, RewardT]):
    name: str
    obs_type: Type[ObsT] | property
    act_type: Type[ActT] | property
    state_type: Type[StateT] | property
    reward_type: Type[RewardT] | property

    def __init__(self, name: str) -> None:
        self.name = name

class BaseConfiguredActor(BaseActor[ObsT, ActT, StateT, RewardT],Protocol[ObsT, ActT, StateT, RewardT,ConfigT]):
    obs_type: Type[ObsT] | property
    act_type: Type[ActT] | property
    state_type: Type[StateT] | property
    reward_type: Type[RewardT] | property
    config_type: Type[ConfigT] | property

class ConfiguredActorProto(BaseConfiguredActor[ObsT, ActT, StateT, RewardT, ConfigT]):

    def act(self: ConfiguredActorProto[ObsT, ActT, StateT, RewardT,ConfigT], observation: ObsT, state: StateT, **kwargs) -> ActT: ...

    def act_and_stream(
        self: ConfiguredActorProto[ObsT, ActT, StateT, RewardT,ConfigT], observation: ObsT, state: StateT | None = None,
    ) -> Generator[ActT, None, None]: ...

    def run(self, state: StateT|Sample) -> None:
        """Run the actor."""
        state = state or Sample(**{self.name: {"observation": self.obs_type(),"action": self.act_type()}})
        for obs in state[self.name]["observation"]:
            state[self.name]["action"] = self.act(obs, state)

    __init__ = BaseActor.__init__
    __call__ = act
    __iter__ = act_and_stream


class ActorProto(BaseActor[ObsT, ActT, StateT, RewardT]):

    def act(self: ActorProto[ObsT, ActT, StateT, RewardT], observation: ObsT, state: StateT, **kwargs) -> ActT: ...

    def act_and_stream(
        self: ActorProto[ObsT, ActT, StateT, RewardT], observation: ObsT, state: StateT | None = None,
    ) -> Generator[ActT, None, None]: ...

    def run(self, state: StateT|Sample) -> None:
        """Run the actor."""
        state = state or Sample(**{self.name: {"observation": self.obs_type(),"action": self.act_type()}})
        for obs in state[self.name]["observation"]:
            state[self.name]["action"] = self.act(obs, state)

    __init__ = BaseActor.__init__
    __call__ = act
    __iter__ = act_and_stream


class AsyncActorProto(BaseActor[ObsT, ActT, StateT, RewardT]):
    async def act(
        self: AsyncActorProto[ObsT, ActT, StateT, RewardT], observation: ObsT, state: StateT | None = None,
    ) -> ActT: ...
    async def act_and_stream(
        self: AsyncActorProto[ObsT, ActT, StateT, RewardT], observation: ObsT, state: StateT | None = None,
    ) -> AsyncGenerator[ActT, None]: ...


    __init__ = BaseActor.__init__
    async def arun(self, state: StateT|Sample) -> None:
        """Run the actor."""
        state = state or Sample(**{self.name: {"observation": self.obs_type(),"action": self.act_type()}})
        for obs in state[self.name]["observation"]:
            state[self.name]["action"] = await self.act(obs, state)

    __call__ = act
    __aiter__ = act_and_stream
