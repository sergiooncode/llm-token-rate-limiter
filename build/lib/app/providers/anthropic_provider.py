from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic

from app.providers.base import Provider, translate_errors
from app.schemas import CompletionRequest, CompletionResponse, StreamChunk, Usage

# Sampling params (temperature/top_p/top_k) are rejected with a 400 on these
# models, so we drop `temperature` rather than forward a request we know fails.
NO_SAMPLING = {
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
}


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, client: AsyncAnthropic | None = None) -> None:
        self._client = client or AsyncAnthropic()

    def _params(self, request: CompletionRequest, model: str) -> dict[str, Any]:
        # `thinking` is deliberately unset: Claude Opus 5 runs adaptive thinking
        # by default, and older models need a different config shape.
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens,
            "messages": [message.model_dump() for message in request.messages],
        }
        if request.system:
            params["system"] = request.system
        if request.temperature is not None and model not in NO_SAMPLING:
            params["temperature"] = request.temperature
        return params

    async def complete(self, request: CompletionRequest, model: str) -> CompletionResponse:
        # Streaming under the hood even for non-streaming callers: it is what the
        # SDK wants for large max_tokens, and it avoids HTTP idle timeouts.
        with translate_errors(self.name):
            async with self._client.messages.stream(**self._params(request, model)) as stream:
                message = await stream.get_final_message()

        return CompletionResponse(
            id=message.id,
            provider=self.name,
            model=message.model,
            content="".join(block.text for block in message.content if block.type == "text"),
            finish_reason=message.stop_reason,
            usage=Usage(
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
            ),
        )

    async def stream(self, request: CompletionRequest, model: str) -> AsyncIterator[StreamChunk]:
        with translate_errors(self.name):
            async with self._client.messages.stream(**self._params(request, model)) as stream:
                async for event in stream:
                    if event.type == "content_block_delta" and event.delta.type == "text_delta":
                        yield StreamChunk(type="delta", text=event.delta.text)
                message = await stream.get_final_message()

        yield StreamChunk(
            type="done",
            finish_reason=message.stop_reason,
            usage=Usage(
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
            ),
        )
