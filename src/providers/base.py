from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
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


@dataclass(frozen=True)
class ModelInfo:
    """Portable provider/model metadata for routing, UI display, and handoffs."""
    provider: str
    id: str
    name: str
    api: str
    input: list[str] = field(default_factory=lambda: ["text"])
    supports_tools: bool = True
    reasoning: bool = False
    context_window: int = 0
    max_output_tokens: int = 0
    base_url: str | None = None
    cost: dict[str, float] = field(
        default_factory=lambda: {
            "input": 0.0,
            "output": 0.0,
            "cache_read": 0.0,
            "cache_write": 0.0,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Message:
    role: str  # "user", "assistant", "tool"
    content: str | list[Any]
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    stop_reason: str = ""
    response_id: str = ""


class BaseProvider(ABC):
    """Abstract base for all AI providers."""

    DEFAULT_MODELS: dict[str, str] = {}
    SUPPORTS_TOOLS: bool = True
    SUPPORTS_IMAGES: bool = False
    API_NAME: str = "unknown"
    CONTEXT_WINDOW: int = 0
    MAX_OUTPUT_TOKENS: int = 0
    REASONING_MODEL_MARKERS: tuple[str, ...] = ()

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
        return list(self.DEFAULT_MODELS.values())

    @classmethod
    def model_info(cls, model_id: str) -> ModelInfo:
        provider_name = cls.name if isinstance(cls.name, str) else cls.__name__
        input_types = ["text", "image"] if getattr(cls, "SUPPORTS_IMAGES", False) else ["text"]
        markers = tuple(m.lower() for m in getattr(cls, "REASONING_MODEL_MARKERS", ()))
        lowered = (model_id or "").lower()
        return ModelInfo(
            provider=provider_name,
            id=model_id,
            name=_friendly_model_name(model_id),
            api=getattr(cls, "API_NAME", "unknown"),
            input=input_types,
            supports_tools=getattr(cls, "SUPPORTS_TOOLS", True),
            reasoning=bool(markers and any(marker in lowered for marker in markers)),
            context_window=getattr(cls, "CONTEXT_WINDOW", 0),
            max_output_tokens=getattr(cls, "MAX_OUTPUT_TOKENS", 0),
            base_url=getattr(cls, "BASE_URL", None),
        )

    @classmethod
    def list_model_infos(cls, models: list[str] | None = None) -> list[ModelInfo]:
        return [cls.model_info(model_id) for model_id in (models or list(cls.DEFAULT_MODELS.values()))]

    @classmethod
    def fetch_available_models(cls, api_key: str) -> list[str]:
        """Fetch live model list from the provider API.

        Returns a list of actual API model IDs. Subclasses override this
        to call the provider's real /models endpoint. Falls back to
        DEFAULT_MODELS values (the actual API IDs).
        """
        return list(cls.DEFAULT_MODELS.values())


def _friendly_model_name(model_id: str) -> str:
    tail = (model_id or "").split("/")[-1]
    return tail.replace("-", " ").replace("_", " ").title() or model_id
