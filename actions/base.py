"""Action ABC — base class for packet action handlers."""

from abc import ABC, abstractmethod


class Action(ABC):
    """Base class for actions dispatched by RouterSink."""

    @abstractmethod
    def execute(self, packet: bytes, class_id: int) -> None:
        """Handle a classified packet."""
        ...

    def close(self) -> None:
        """Optional cleanup. Called when pipeline shuts down."""
        pass
