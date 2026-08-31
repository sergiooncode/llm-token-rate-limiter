"""Step 1: the Redis connection is fail-open at every stage."""

import pytest
from fakeredis.aioredis import FakeRedis
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError

import app.main as main
from app.config import get_settings

REDIS_URL = "redis://unused-because-the-client-is-faked:6379/0"


class BrokenRedis(FakeRedis):
    """Reachable object, unreachable server - what a Redis outage looks like."""

    async def ping(self):
        raise RedisConnectionError("connection refused")


def factory_for(client_class):
    class Factory:
        @staticmethod
        def from_url(url, **kwargs):
            return client_class()

    return Factory


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_healthz_reports_disabled_when_no_url_configured(monkeypatch):
    monkeypatch.setenv("GATEWAY_REDIS_URL", "")

    with TestClient(main.app) as client:
        body = client.get("/healthz").json()

    assert body["redis"] == "disabled"
    assert body["status"] == "ok"


def test_healthz_reports_up_when_redis_answers(monkeypatch):
    monkeypatch.setenv("GATEWAY_REDIS_URL", REDIS_URL)
    monkeypatch.setattr(main, "Redis", factory_for(FakeRedis))

    with TestClient(main.app) as client:
        body = client.get("/healthz").json()

    assert body["redis"] == "up"


def test_gateway_stays_healthy_while_redis_is_down(monkeypatch):
    """Fail-open: a Redis outage is reported, but the gateway keeps serving."""
    monkeypatch.setenv("GATEWAY_REDIS_URL", REDIS_URL)
    monkeypatch.setattr(main, "Redis", factory_for(BrokenRedis))

    with TestClient(main.app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "providers": [], "redis": "down"}


async def test_startup_survives_a_redis_that_never_answers(monkeypatch):
    """A failed ping must not raise, and must still hand back a client so it
    can reconnect once Redis returns."""
    monkeypatch.setenv("GATEWAY_REDIS_URL", REDIS_URL)
    monkeypatch.setattr(main, "Redis", factory_for(BrokenRedis))

    client = await main.connect_redis()

    assert client is not None


async def test_no_client_is_created_without_a_url(monkeypatch):
    monkeypatch.setenv("GATEWAY_REDIS_URL", "")

    assert await main.connect_redis() is None
