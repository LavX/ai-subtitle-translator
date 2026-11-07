"""Usage reported after every billed provider call, the attempt's cost budget, and the holds
that keep requests in flight from promising the same money twice."""

from collections.abc import Awaitable, Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from subtitle_translator.core.errors import ErrorKind, TranslationProviderError


@dataclass(frozen=True)
class Usage:
    """One provider call, and the running totals after it (an engine failover adds the earlier
    connections' spend to them). reservations names the
    stored holds this call settles (ids from the reserve sink; empty when nothing was held)."""

    tokens: int
    cost: float
    model: str
    total_tokens: int
    total_cost: float
    reservations: tuple[str, ...] = ()


@dataclass(frozen=True)
class Reservation:
    """What one request may cost at most, held from the budget until its call settles."""

    amount: float
    model: str
    id: str | None = None


UsageSink = Callable[[Usage], Awaitable[None]]
ReserveSink = Callable[[float, str], Awaitable[str | None]]


class BudgetExhaustedError(TranslationProviderError):
    """This request's worst-case charge does not fit what is left of the budget. Never fails over:
    the budget belongs to the job, not to a back end."""

    def __init__(self, message: str, provider: str = "unknown") -> None:
        super().__init__(message, provider, retryable=False, kind=ErrorKind.CANCELLED)
        self.reason = "budget_exhausted"


class UsageMeter:
    """Counts what an attempt has spent, and holds what its requests in flight may still cost.

    reserve() checks and holds in one step, with no await between the two, so batches running
    in parallel never each see the same remainder and together send more than it covers. Both
    sinks are awaited before the caller continues, so in Bazarr+ a hold is stored before its
    request is sent and a spend before the next request, and both survive a crash.
    """

    def __init__(
        self,
        budget: float | None = None,
        sink: UsageSink | None = None,
        reserve_sink: ReserveSink | None = None,
    ) -> None:
        self.budget = budget
        self._sink = sink
        self._reserve_sink = reserve_sink
        self.tokens = 0
        self.cost = 0.0
        self.reserved = 0.0
        self.calls = 0

    @property
    def remaining(self) -> float | None:
        return None if self.budget is None else max(0.0, self.budget - self.cost - self.reserved)

    def allows(self, estimate: float | None) -> bool:
        """A paid request fits when its estimate fits what is neither spent nor held.

        An unknown estimate is sent while any money is left; a free request always is.
        """
        if self.budget is None:
            return True
        if estimate is None:
            return self.cost + self.reserved < self.budget
        return estimate <= self.budget - self.cost - self.reserved

    async def reserve(
        self, estimate: float | None, model: str = "", provider: str = "unknown"
    ) -> Reservation:
        """Hold a request's worst case, or raise BudgetExhaustedError when it does not fit. An
        unknown estimate holds nothing: such a back end is held by reported cost only."""
        if not self.allows(estimate):
            left = self.remaining or 0.0
            wanted = "an unknown amount" if estimate is None else f"up to ${estimate:.4f}"
            raise BudgetExhaustedError(
                f"Cost budget exhausted: this request could cost {wanted} and ${left:.4f} is left",
                provider=provider,
            )
        amount = max(0.0, float(estimate or 0.0))
        self.reserved += amount
        reservation_id = None
        if amount and self._reserve_sink is not None:
            try:
                reservation_id = await self._reserve_sink(amount, model)
            except BaseException:
                self.reserved = max(0.0, self.reserved - amount)
                raise
        return Reservation(amount, model, reservation_id)

    async def record(
        self, tokens: int, cost: float, model: str, *, reservations: Iterable[Reservation] = ()
    ) -> None:
        """Settle a finished call: release its holds and count what it actually cost."""
        held = tuple(reservations)
        self.reserved = max(0.0, self.reserved - sum(r.amount for r in held))
        tokens = max(0, int(tokens or 0))
        cost = max(0.0, float(cost or 0.0))
        settled = tuple(r.id for r in held if r.id is not None)
        if not tokens and not cost and not settled:
            return
        self.tokens += tokens
        self.cost += cost
        if tokens or cost:
            self.calls += 1
        if self._sink is not None:
            await self._sink(Usage(tokens, cost, model, self.tokens, self.cost, settled))


# The holds of the provider call in progress. BatchProcessor._send opens one scope per call;
# every provider task has its own context, so parallel batches never mix.
_OPEN_CALL: ContextVar[list[Reservation] | None] = ContextVar(
    "subtitle_translator_open_call", default=None
)


@contextmanager
def call_scope() -> Iterator[list[Reservation]]:
    holds: list[Reservation] = []
    token = _OPEN_CALL.set(holds)
    try:
        yield holds
    finally:
        _OPEN_CALL.reset(token)


async def reserve_request(
    config: Any, worst_case: float | None, provider: str = "unknown", model: str = ""
) -> Reservation | None:
    """Called right before a request is sent. worst_case is the most the request can be
    charged: its input tokens plus the output-token limit it carries, at the model's rates;
    None when the back end cannot price it. Raises BudgetExhaustedError when it does not fit.
    Otherwise the hold joins the open call and is settled with that call's usage. Returns the
    hold, or None when the request carries no meter."""
    meter = getattr(config, "_usage_meter", None)
    if meter is None:
        return None
    reservation = await meter.reserve(worst_case, model, provider)
    holds = _OPEN_CALL.get()
    if holds is not None:
        holds.append(reservation)
    return reservation


async def settle_request(
    config: Any, reservation: Reservation | None, tokens: int = 0, cost: float = 0.0
) -> bool:
    """Close the hold of a request that is not the call's answer, with whatever it reported as
    charged, so a retry of the same call does not promise the same money twice and the charge
    is not lost. Only for a request whose billing is known: a rejected optional field, a
    refused key, a rate limit, a refused endpoint. A request that may have been billed without
    saying so keeps its hold instead. True when the charge was recorded on a meter."""
    meter = getattr(config, "_usage_meter", None)
    if meter is None or reservation is None:
        return False
    holds = _OPEN_CALL.get()
    if holds is not None:
        for position, held in enumerate(holds):
            if held is reservation:
                del holds[position]
                break
    await meter.record(tokens, cost, reservation.model, reservations=(reservation,))
    return True


async def release_request(config: Any, reservation: Reservation | None) -> None:
    """Give back the hold of a request the server rejected before it did any work."""
    await settle_request(config, reservation)

