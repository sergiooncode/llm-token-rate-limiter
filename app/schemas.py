from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class CompletionRequest(BaseModel):
    """Provider-neutral request. `model` is "<provider>/<model-id>"."""

    model: str = Field(examples=["anthropic/claude-opus-5", "openai/gpt-4.1"])
    messages: list[Message] = Field(min_length=1)
    system: str | None = None
    max_tokens: int = Field(default=4096, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    stream: bool = False


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class CompletionResponse(BaseModel):
    id: str
    provider: str
    model: str
    content: str
    finish_reason: str | None = None
    usage: Usage


class StreamChunk(BaseModel):
    """One SSE payload. `delta` carries text; `done` carries the final metadata."""

    type: Literal["delta", "done", "error"]
    text: str | None = None
    finish_reason: str | None = None
    usage: Usage | None = None
    message: str | None = None
