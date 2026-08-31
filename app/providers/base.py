from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import contextmanager

import anthropic

from app.schemas import CompletionRequest, CompletionResponse, StreamChunk


class UpstreamError(Exception):
    """A provider call failed. `status_code` is what the caller gets back."""

    def __init__(self, provider: str, status_code: int, message: str) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.message = message


@contextmanager
def translate_errors(provider: str):
    try:
        yield
    except anthropic.APIStatusError as exc:
        raise UpstreamError(provider, exc.status_code, str(exc)) from exc
    except anthropic.APIConnectionError as exc:
        raise UpstreamError(provider, 502, f"{provider} unreachable: {exc}") from exc


class Provider(ABC):
    name: str

    @abstractmethod
    async def complete(self, request: CompletionRequest, model: str) -> CompletionResponse:
        """Run the request to completion. `model` is the provider-native id."""

    @abstractmethod
    def stream(self, request: CompletionRequest, model: str) -> AsyncIterator[StreamChunk]:
        """Yield `delta` chunks, then exactly one `done` chunk carrying usage."""
