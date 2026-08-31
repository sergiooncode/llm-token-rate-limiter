import os

from app.providers.anthropic_provider import AnthropicProvider
from app.providers.base import Provider, UpstreamError

__all__ = ["Provider", "UpstreamError", "build_registry"]


def build_registry() -> dict[str, Provider]:
    """Only providers with credentials in the environment are routable."""
    registry: dict[str, Provider] = {}
    if os.getenv("ANTHROPIC_API_KEY"):
        registry["anthropic"] = AnthropicProvider()
    return registry
