"""Token-bucket rate limiting over Redis, using optimistic concurrency.

The cost of an LLM call is not knowable before it runs - output tokens only
exist after generation - so the bucket is used in two phases:

    try_acquire()  gate on the current balance, deduct an *estimate*
    settle()       apply the difference once real usage is known

The gate admits any caller with at least one token rather than requiring the
full cost up front, and `settle` may push the balance negative. That bounded
debt is deliberate: it removes the need to over-reserve `max_tokens` for every
request, and refill pays the debt back down on its own.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

from redis.asyncio import Redis
from redis.exceptions import WatchError

from app.schemas import CompletionRequest, Usage

# Made configurable in step 6.
DEFAULT_CAPACITY = 100_000
DEFAULT_REFILL_PER_SECOND = 1_000.0

# Output tokens cost roughly 5x input on Claude, so a bucket that counted them
# equally would systematically under-charge the most expensive traffic.
OUTPUT_WEIGHT = 5

# Rough pre-flight guess, corrected by settle() as soon as real usage lands.
CHARS_PER_TOKEN = 4


class RateLimiterUnavailable(Exception):
    """The bucket could not be read or written. The caller decides the policy."""


@dataclass(frozen=True)
class Decision:
    allowed: bool
    remaining: float
    retry_after: float  # seconds until the bucket holds a usable token


def estimate_cost(request: CompletionRequest) -> int:
    """Pre-flight cost estimate. Input only - output is still unknown."""
    characters = len(request.system or "")
    characters += sum(len(message.content) for message in request.messages)
    return max(1, characters // CHARS_PER_TOKEN)


def actual_cost(usage: Usage) -> int:
    return usage.input_tokens + OUTPUT_WEIGHT * usage.output_tokens


class TokenBucket:
    def __init__(
        self,
        redis: Redis,
        capacity: int = DEFAULT_CAPACITY,
        refill_per_second: float = DEFAULT_REFILL_PER_SECOND,
        max_retries: int = 5,
    ) -> None:
        self._redis = redis
        self._capacity = float(capacity)
        self._refill_per_second = refill_per_second
        self._max_retries = max_retries
        # Long enough that a bucket empty at eviction time would have refilled
        # anyway, so dropping the key loses nothing.
        self._ttl_ms = int((capacity / refill_per_second + 60) * 1000)

    def _key(self, caller: str) -> str:
        return f"ratelimit:{caller}"

    def _refill(self, tokens_raw: str | None, ts_raw: str | None, now: float) -> float:
        if tokens_raw is None or ts_raw is None:
            return self._capacity  # an unseen caller starts with a full bucket

        # Wall clock, not monotonic: this timestamp is written by one process
        # and read by another, so it has to be comparable across machines.
        # Both clamps guard against NTP moving it - backwards steps must not
        # drain the bucket, forward jumps must not overfill it.
        elapsed = max(0.0, now - float(ts_raw))
        return min(self._capacity, float(tokens_raw) + elapsed * self._refill_per_second)

    async def _transact(self, key: str, mutate: Callable[[float], tuple[float, object]]):
        """Read-modify-write under WATCH, retrying if another writer wins.

        `mutate` receives the refilled balance and returns (new_balance, result).
        It runs between WATCH and EXEC, so it must stay pure and cheap.
        """
        async with self._redis.pipeline() as pipe:
            for _ in range(self._max_retries):
                try:
                    await pipe.watch(key)

                    # WATCH puts the pipeline in immediate mode: this read runs
                    # now instead of being queued. Buffering only starts at
                    # multi() below - the usual trap with redis-py.
                    tokens_raw, ts_raw = await pipe.hmget(key, "tokens", "ts")
                    now = time.time()
                    balance, result = mutate(self._refill(tokens_raw, ts_raw, now))

                    pipe.multi()
                    pipe.hset(key, mapping={"tokens": balance, "ts": now})
                    pipe.pexpire(key, self._ttl_ms)
                    await pipe.execute()
                    return result
                except WatchError:
                    # Someone else wrote this key between our read and EXEC.
                    continue

        raise RateLimiterUnavailable(
            f"gave up on {key} after {self._max_retries} contended attempts"
        )

    async def try_acquire(self, caller: str, estimated_cost: int) -> Decision:
        def mutate(balance: float) -> tuple[float, Decision]:
            if balance < 1:
                deficit = 1 - balance
                return balance, Decision(
                    allowed=False,
                    remaining=balance,
                    retry_after=deficit / self._refill_per_second,
                )
            remaining = balance - estimated_cost
            return remaining, Decision(allowed=True, remaining=remaining, retry_after=0.0)

        return await self._transact(self._key(caller), mutate)

    async def settle(self, caller: str, estimated_cost: int, usage: Usage) -> float:
        """Correct the estimate once the real usage is known.

        Applies only the difference, so nothing is charged twice. A caller that
        overran its estimate can end up with a negative balance.
        """
        delta = actual_cost(usage) - estimated_cost

        def mutate(balance: float) -> tuple[float, float]:
            corrected = balance - delta
            return corrected, corrected

        return await self._transact(self._key(caller), mutate)
