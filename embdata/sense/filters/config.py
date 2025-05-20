from typing import Any

from typing_extensions import TypedDict

from embdata.sample import Sample


class FilterConfig(Sample):
    @property
    def name(self) -> str:
        return type(self).__name__.lower().replace("config", "")

    enabled: bool = True

class FilterKwargs(TypedDict,total=False):
    enabled: bool


class Port(Sample):
    """Port for the filter."""
    observation: Any | None = None
    config: Any | None = None
    action: Any | None = None

