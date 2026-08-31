import hashlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.config import get_settings
from app.providers import Provider, UpstreamError, build_registry
from app.schemas import CompletionRequest, StreamChunk

logger = logging.getLogger(__name__)


async def connect_redis() -> Redis | None:
    """Return a Redis client, or None when no URL is configured.

    Fail-open: an unreachable Redis is logged, never fatal. The client is
    still returned so it can reconnect on its own once Redis comes back.
    Socket timeouts are short so a dead Redis cannot stall a completion.
    """
    url = get_settings().redis_url
    if not url:
        logger.warning("GATEWAY_REDIS_URL unset - rate limiting is disabled")
        return None

    client = Redis.from_url(
        url,
        decode_responses=True,
        socket_connect_timeout=1.0,
        socket_timeout=1.0,
    )
    try:
        await client.ping()
        logger.info("connected to redis")
    except RedisError as exc:
        logger.warning("redis unreachable at startup, continuing without it: %s", exc)
    return client


async def redis_status(client: Redis | None) -> str:
    if client is None:
        return "disabled"
    try:
        await client.ping()
    except RedisError:
        return "down"
    return "up"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.registry = build_registry()
    app.state.redis = await connect_redis()
    try:
        yield
    finally:
        if app.state.redis is not None:
            await app.state.redis.aclose()


app = FastAPI(title="LLM Gateway", version="0.1.0", lifespan=lifespan)


@app.exception_handler(UpstreamError)
async def handle_upstream_error(_: Request, exc: UpstreamError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message, "provider": exc.provider},
    )


async def caller_id(authorization: str | None = Header(default=None)) -> str:
    """Identify the calling service. Hashed so keys never reach logs."""
    keys = get_settings().allowed_keys
    if not keys:
        return "anonymous"
    token = (authorization or "").removeprefix("Bearer ").strip()
    if token not in keys:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
    return hashlib.sha256(token.encode()).hexdigest()[:12]


def resolve(registry: dict[str, Provider], model: str) -> tuple[Provider, str]:
    provider_name, _, model_id = model.partition("/")
    if not model_id:
        raise HTTPException(
            status_code=400,
            detail=f"model must be '<provider>/<model-id>', got {model!r}",
        )
    provider = registry.get(provider_name)
    if provider is None:
        available = sorted(registry) or ["none configured"]
        raise HTTPException(
            status_code=400,
            detail=f"unknown provider {provider_name!r}; available: {available}",
        )
    return provider, model_id


async def sse(provider: Provider, request: CompletionRequest, model: str) -> AsyncIterator[str]:
    """Server-sent events: one JSON payload per line, terminated by [DONE].

    The exception handler above cannot help here - the response has already
    started - so upstream failures are reported as a final `error` chunk.
    """
    try:
        async for chunk in provider.stream(request, model):
            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
    except UpstreamError as exc:
        error = StreamChunk(type="error", message=exc.message)
        yield f"data: {error.model_dump_json(exclude_none=True)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/completions", response_model=None)
async def completions(
    body: CompletionRequest,
    request: Request,
    caller: str = Depends(caller_id),
):
    provider, model = resolve(request.app.state.registry, body.model)

    # Token rate limiting goes here: reserve budget for `caller` before
    # dispatch, then reconcile against the usage the provider reports.
    if body.stream:
        return StreamingResponse(
            sse(provider, body, model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return await provider.complete(body, model)


@app.get("/healthz")
async def healthz(request: Request) -> dict[str, object]:
    # Stays "ok" when redis is down: the gateway still serves, unmetered.
    return {
        "status": "ok",
        "providers": sorted(request.app.state.registry),
        "redis": await redis_status(request.app.state.redis),
    }
