from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class ToolEvent:
    """Structured event for tool execution lifecycle."""
    type: str  # "start", "result"
    tool_name: str
    arguments: dict
    result: str | None = None
    is_error: bool = False


@dataclass
class Message:
    role: str  # "user", "assistant", "tool"
    content: str | list[Any]
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)


class BaseProvider(ABC):
    """Abstract base for all AI providers."""

    DEFAULT_MODELS: dict[str, str] = {}

    def __init__(self, api_key: str, model: str | None = None):
        self.api_key = api_key
        self.model = model or self._default_model()

    def _default_model(self) -> str:
        return list(self.DEFAULT_MODELS.values())[0]

    @abstractmethod
    def stream_response(
        self,
        messages: list[Message],
        tools: list[dict],
        system: str,
    ) -> Iterator[str | ToolCall]:
        """Yield text chunks and/or ToolCall objects."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def list_models(self) -> list[str]:
        return list(self.DEFAULT_MODELS.keys())

    @classmethod
    def fetch_available_models(cls, api_key: str) -> list[str]:
        """Fetch live model list from the provider API.

        Returns a list of model IDs. Subclasses override this to call
        the provider's real /models endpoint. Falls back to DEFAULT_MODELS.
        """
        return list(cls.DEFAULT_MODELS.keys())
