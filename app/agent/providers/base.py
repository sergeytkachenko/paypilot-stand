""
from dataclasses import dataclass, field


@dataclass
class ModelResponse:
    text: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


class Provider:
    name = "base"

    def complete(self, system: str, messages: list[dict],
                 tools: list[dict]) -> ModelResponse:
        raise NotImplementedError


def get_provider():
    from app import config
    kind = config.LLM_PROVIDER
    if kind == "mock":
        from app.agent.providers.mock import MockProvider
        return MockProvider()
    if kind == "anthropic":
        from app.agent.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider()
    if kind == "openai":
        from app.agent.providers.openai_provider import OpenAIProvider
        return OpenAIProvider()
    if kind == "gemini":
        from app.agent.providers.gemini_provider import GeminiProvider
        return GeminiProvider()
    raise ValueError(f"Unknown LLM_PROVIDER={kind!r} "
        "(mock | anthropic | openai | gemini)")
