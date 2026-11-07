"""The translation engine: one attempt against one back-end connection, or an ordered list
of connections with failover. Bazarr+ drives failover itself and calls attempt(); the
standalone service calls translate()."""

import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

from subtitle_translator.core.backends import (
    BackendContext,
    BackendRegistry,
    Connection,
    OpenedBackend,
)
from subtitle_translator.core.batch_processor import (
    BatchProcessor,
    BatchProgress,
    BatchResult,
    summarize_batch_failure,
)
from subtitle_translator.core.batch_sizing import BatchSizeResolver
from subtitle_translator.core.checkpoints import (
    Checkpoint,
    CheckpointLine,
    source_digest,
    usable_translations,
)
from subtitle_translator.core.context import Language, MediaContext
from subtitle_translator.core.errors import (
    FAILOVER_KINDS,
    ErrorKind,
    TranslationProviderError,
    error_kind,
    most_severe,
)
from subtitle_translator.core.models import MAX_LINE_CHARS
from subtitle_translator.core.settings import CoreSettings
from subtitle_translator.core.srt_parser import add_rtl_markers
from subtitle_translator.core.usage import ReserveSink, Usage, UsageMeter

logger = logging.getLogger(__name__)

# Every kind but partial ends an attempt with an error: partial results are only for per-batch
# content failures. An unexpected failure is internal and a back end that cannot take the
# request declined, and both keep their kind instead of passing as a partial translation with
# source lines in the gaps.
_ATTEMPT_FAILURE_KINDS = frozenset(ErrorKind) - {ErrorKind.PARTIAL}

# How long a connection sits out after a failover when the back end named no time.
DEFAULT_COOLDOWN: dict[ErrorKind, float] = {
    ErrorKind.RATE_LIMITED: 60.0,
    ErrorKind.DEPENDENCY_UNAVAILABLE: 60.0,
    ErrorKind.QUOTA_EXHAUSTED: 3600.0,
    ErrorKind.AUTH_REQUIRED: 3600.0,
}


@dataclass(frozen=True)
class Line:
    position: int
    text: str


@dataclass
class TranslationJob:
    source: Language
    target: Language
    lines: list[Line]
    media: MediaContext = field(default_factory=MediaContext)
    checkpoints: list[Checkpoint] = field(default_factory=list)
    budget: float | None = None
    add_rtl_marks: bool = True
    session_id: str | None = None


@dataclass(frozen=True)
class Progress:
    total_lines: int
    done_lines: int
    total_batches: int
    done_batches: int
    failed_batches: int
    tokens: int
    cost: float
    message: str
    connection_id: str


@dataclass(frozen=True)
class FailoverRecord:
    connection_id: str
    kind: ErrorKind
    message: str
    retry_after: float | None


# How many opened back ends an engine keeps. A request with its own key opens one per distinct
# key, so the cache is a small LRU: the least recently used one is retired and closed once no
# attempt runs on it, and its key leaves memory with it.
MAX_OPENED = 32


@dataclass
class EngineEvents:
    on_progress: Callable[[Progress], None] | None = None
    on_usage: Callable[[Usage], Awaitable[None]] | None = None
    on_checkpoint: Callable[[Checkpoint], Awaitable[None]] | None = None
    on_failover: Callable[[FailoverRecord], None] | None = None
    on_reserve: ReserveSink | None = None


def _carrying_spend(events: EngineEvents, tokens: int, cost: float) -> EngineEvents:
    """Events whose totals add what earlier attempts of the same job already spent. Each
    attempt meters only its own calls, so without this the totals a job reports (progress, the
    running totals of a usage record and of a checkpoint) would fall back to the last
    attempt's alone after a failover."""
    if not (tokens or cost):
        return events
    report, on_usage, on_checkpoint = events.on_progress, events.on_usage, events.on_checkpoint

    def carry(progress: Progress) -> None:
        report(replace(progress, tokens=progress.tokens + tokens, cost=progress.cost + cost))

    async def carry_usage(usage: Usage) -> None:
        await on_usage(
            replace(
                usage,
                total_tokens=usage.total_tokens + tokens,
                total_cost=usage.total_cost + cost,
            )
        )

    async def carry_checkpoint(checkpoint: Checkpoint) -> None:
        await on_checkpoint(
            replace(
                checkpoint,
                total_tokens=checkpoint.total_tokens + tokens,
                total_cost=checkpoint.total_cost + cost,
            )
        )

    return replace(
        events,
        on_progress=carry if report is not None else None,
        on_usage=carry_usage if on_usage is not None else None,
        on_checkpoint=carry_checkpoint if on_checkpoint is not None else None,
    )


@dataclass(frozen=True)
class BatchModel:
    """The lines one batch of this attempt translated, and the model that served it."""

    positions: tuple[int, ...]
    model: str


@dataclass
class TranslationOutcome:
    lines: list[Line]
    status: Literal["complete", "partial"]
    missing: list[int]
    backend: str
    connection_id: str
    model: str
    tokens: int
    cost: float
    summary: str = ""
    failovers: list[FailoverRecord] = field(default_factory=list)
    batches: list[BatchModel] = field(default_factory=list)


class EngineError(Exception):
    def __init__(
        self,
        kind: ErrorKind,
        message: str,
        *,
        connection_id: str = "",
        backend: str = "",
        model: str = "",
        retry_after: float | None = None,
        tokens: int = 0,
        cost: float = 0.0,
        checkpoints: Sequence[Checkpoint] = (),
        summary: str = "",
        field_errors: dict[str, str] | None = None,
        failovers: Sequence[FailoverRecord] = (),
        reason: str | None = None,
        reserved: float = 0.0,
    ):
        super().__init__(message)
        self.kind = kind
        self.reason = reason
        # The worst cases of calls that may have been billed without reporting: cut off before
        # they reported, or given up by the provider after the request may have reached the
        # server. They stay held against the job's budget.
        self.reserved = reserved
        self.message = message
        self.connection_id = connection_id
        self.backend = backend
        self.model = model
        self.retry_after = retry_after
        self.tokens = tokens
        self.cost = cost
        self.checkpoints = list(checkpoints)
        self.summary = summary
        self.field_errors = dict(field_errors or {})
        self.failovers = list(failovers)


@dataclass(eq=False)
class _Opened:
    opened: OpenedBackend
    resolver: BatchSizeResolver
    users: int = 0  # attempts running on it now


class TranslationEngine:
    """Long-lived: one per worker or service, so opened back ends (their key health, routing
    state and learned batch sizes) are shared by every job that uses the same connection."""

    def __init__(
        self,
        registry: BackendRegistry,
        base: CoreSettings | None = None,
        context_for: Callable[[str], BackendContext] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._registry = registry
        self._base = base or CoreSettings()
        self._context_for = context_for or (lambda type_id: BackendContext())
        self._opened: OrderedDict[str, _Opened] = OrderedDict()
        self._retired: list[_Opened] = []
        # connection id -> (fingerprint, when it may be tried again, whether the provider named
        # that wait, the tick it was set at). A wait the provider named is honoured; one of the
        # defaults is a guess. The tick lets an attempt that finishes late clear only a cooldown
        # that was set before it began.
        self._cooling: dict[str, tuple[str, float, bool, int]] = {}
        self._cooling_tick = 0
        self._clock = clock

    @property
    def registry(self) -> BackendRegistry:
        return self._registry

    def _open(self, connection: Connection) -> _Opened:
        # Two connections never share an opened back end, even with equal settings: each keeps
        # its own key health and learned sizes. A saved change opens a new one.
        key = f"{connection.id}:{connection.fingerprint()}"
        entry = self._opened.get(key)
        if entry is None:
            backend = self._registry.get(connection.backend_type)
            opened = backend.open(
                connection, self._base, self._context_for(connection.backend_type)
            )
            entry = _Opened(opened, BatchSizeResolver(opened.settings))
            self._opened[key] = entry
            while len(self._opened) > MAX_OPENED:
                _, oldest = self._opened.popitem(last=False)
                self._retired.append(oldest)
        else:
            self._opened.move_to_end(key)
        return entry

    def provider_for(self, connection: Connection):
        """The provider serving a connection (the standalone GUI reads its key health)."""
        return self._open(connection).opened.provider

    def context_for(self, type_id: str) -> BackendContext:
        """The per-type folders this engine gives a back end (the GUI reads sign-in state there)."""
        return self._context_for(type_id)

    async def reconfigure(
        self, base: CoreSettings, context_for: Callable[[str], BackendContext] | None = None
    ) -> None:
        """Change what every connection shares: timeouts, retries, the data folders. Each
        connection is opened again on its next attempt, so no back end keeps the old values.
        What was opened before is closed once no attempt uses it; attempts already running
        finish on it first. Cooldowns stay: they describe the remote service, not these
        settings."""
        self._base = base
        if context_for is not None:
            self._context_for = context_for
        retired, self._opened = list(self._opened.values()), OrderedDict()
        self._retired.extend(retired)
        for entry in retired:
            await self._release(entry)

    async def _release(self, entry: _Opened) -> None:
        """Close a retired back end once no attempt uses it. A provider that another opened
        connection still uses stays open, because the standalone shares one OpenRouter provider
        between connections; the built-in providers open a new HTTP client when they are used
        after close(), so a shared one that is handed out again still works."""
        if entry.users or entry not in self._retired:
            return
        self._retired.remove(entry)
        provider = entry.opened.provider
        if any(
            other.opened.provider is provider
            for other in (*self._opened.values(), *self._retired)
        ):
            return
        await provider.close()

    async def aclose(self) -> None:
        for entry in [*self._opened.values(), *self._retired]:
            await entry.opened.provider.close()
        self._opened.clear()
        self._retired.clear()

    async def attempt(
        self, job: TranslationJob, connection: Connection, events: EngineEvents | None = None
    ) -> TranslationOutcome:
        try:
            entry = self._open(connection)
        except TranslationProviderError as error:
            raise EngineError(
                error_kind(error),
                error.message,
                connection_id=connection.id,
                backend=connection.backend_type,
                field_errors=getattr(error, "field_errors", None),
            ) from error
        entry.users += 1
        try:
            # What the cache evicted to make room for this entry is closed once nothing uses it.
            for evicted in list(self._retired):
                await self._release(evicted)
            return await self._run(job, connection, entry, events or EngineEvents())
        finally:
            entry.users -= 1
            await self._release(entry)

    def _start_cooling(self, connection: Connection, until: float, named: bool) -> None:
        """Record a cooldown. A longer one still running for the same settings is kept, so a
        failure that reports a short wait never shortens a long one set by another job."""
        fingerprint = connection.fingerprint()
        self._cooling_tick += 1
        current = self._cooling.get(connection.id)
        if (
            current is not None
            and current[0] == fingerprint
            and current[1] > self._clock()
            and current[1] >= until
        ):
            until, named = current[1], current[2]
        self._cooling[connection.id] = (fingerprint, until, named, self._cooling_tick)

    def _cooling_until(self, connection: Connection) -> float:
        """When a connection that failed over may be tried again. The wait belongs to the
        settings and secrets that earned it: a connection saved with a new key or model under
        the same id is tried at once."""
        entry = self._cooling.get(connection.id)
        if entry is None or entry[0] != connection.fingerprint():
            return 0.0
        return entry[1]

    def _cooling_on_provider_wait(self, connection: Connection) -> bool:
        entry = self._cooling.get(connection.id)
        return entry is not None and entry[0] == connection.fingerprint() and entry[2]

    def _candidates(self, enabled: Sequence[Connection]) -> list[Connection]:
        """The connections a request walks. Those not cooling, in order. When every one is
        cooling, the cooldown is only a guess unless the provider named the wait, so the one
        whose guess ends soonest is tried: a request is never refused with nothing sent just
        because the only connection failed a minute ago (or ran out of credit before the
        operator topped it up). A wait the provider named, such as a Retry-After, is honoured."""
        now = self._clock()
        ready = [c for c in enabled if self._cooling_until(c) <= now]
        if ready:
            return ready
        guessed = [c for c in enabled if not self._cooling_on_provider_wait(c)]
        if guessed:
            return [min(guessed, key=self._cooling_until)]
        return []

    async def translate(
        self,
        job: TranslationJob,
        connections: Sequence[Connection],
        *,
        pinned: str | None = None,
        events: EngineEvents | None = None,
    ) -> TranslationOutcome:
        """Walk the ordered list. Fail over only on the four failover kinds, carrying the
        checkpoints and the spend forward; a pinned connection is tried alone."""
        events = events or EngineEvents()
        if pinned is not None:
            chosen = next((c for c in connections if c.id == pinned), None)
            if chosen is None:
                raise EngineError(
                    ErrorKind.INVALID_CONFIG,
                    f"No back-end connection is named {pinned!r}",
                    field_errors={"backend": "unknown connection"},
                )
            if not chosen.enabled:
                raise EngineError(ErrorKind.INVALID_CONFIG, f"The {pinned!r} connection is turned off")
            return await self.attempt(job, chosen, events)

        enabled = [c for c in connections if c.enabled]
        if not enabled:
            raise EngineError(ErrorKind.INVALID_CONFIG, "No back-end connection is turned on")

        checkpoints = list(job.checkpoints)
        tokens, cost, held = 0, 0.0, 0.0
        failovers: list[FailoverRecord] = []
        last: EngineError | None = None
        candidates = self._candidates(enabled)
        for position, connection in enumerate(candidates):
            # A call an earlier connection left unsettled may have been billed, so its worst
            # case stays held, as it would within one attempt.
            budget = None if job.budget is None else max(0.0, job.budget - cost - held)
            began = self._cooling_tick
            try:
                outcome = await self.attempt(
                    replace(job, checkpoints=checkpoints, budget=budget),
                    connection,
                    _carrying_spend(events, tokens, cost),
                )
            except EngineError as error:
                checkpoints.extend(error.checkpoints)
                tokens += error.tokens
                cost += error.cost
                held += error.reserved
                if error.kind not in FAILOVER_KINDS:
                    error.tokens, error.cost, error.reserved = tokens, cost, held
                    error.checkpoints, error.failovers = checkpoints, failovers
                    raise
                named = bool(error.retry_after)
                wait = error.retry_after if named else DEFAULT_COOLDOWN[error.kind]
                self._start_cooling(connection, self._clock() + wait, named)
                record = FailoverRecord(connection.id, error.kind, error.message, error.retry_after)
                failovers.append(record)
                # Ids and kinds only: an error message can carry a URL, and keys never belong here.
                following = candidates[position + 1] if position + 1 < len(candidates) else None
                logger.info(
                    "Failover: connection %s ended with %s, %s",
                    connection.id,
                    error.kind.value,
                    f"next is {following.id}" if following else "no connection left to try",
                )
                if events.on_failover is not None:
                    events.on_failover(record)
                last = error
                continue
            entry = self._cooling.get(connection.id)
            if entry is not None and entry[3] <= began:
                del self._cooling[connection.id]
            outcome.tokens += tokens
            outcome.cost += cost
            outcome.failovers = failovers
            return outcome

        if last is None:
            soonest = min(self._cooling_until(c) for c in enabled) - self._clock()
            raise EngineError(
                ErrorKind.RATE_LIMITED,
                "Every back-end connection is waiting out an earlier limit",
                retry_after=max(0.0, soonest),
                failovers=failovers,
            )
        last.tokens, last.cost, last.reserved = tokens, cost, held
        last.checkpoints, last.failovers = checkpoints, failovers
        raise last

    async def _run(
        self, job: TranslationJob, connection: Connection, entry: _Opened, events: EngineEvents
    ) -> TranslationOutcome:
        opened = entry.opened
        prior = usable_translations(job.lines, job.checkpoints)
        remaining = [line for line in job.lines if line.position not in prior]
        meter = UsageMeter(job.budget, events.on_usage, events.on_reserve)
        produced: list[Checkpoint] = []
        texts = {str(line.position): line.text for line in job.lines}

        async def on_root(result: BatchResult, requested: list[dict[str, str]]) -> None:
            asked = {str(line["index"]) for line in requested}
            kept = {
                int(t["index"]): CheckpointLine(
                    source_digest(texts[str(t["index"])]), t["content"]
                )
                for t in result.translations
                if str(t["index"]) in asked
                and t["content"].strip()
                and len(t["content"]) <= MAX_LINE_CHARS
            }
            if not kept:
                return
            checkpoint = Checkpoint(
                kept,
                result.tokens_used,
                result.cost,
                meter.tokens,
                meter.cost,
                connection.id,
                result.model_used or opened.model,
            )
            produced.append(checkpoint)
            if events.on_checkpoint is not None:
                await events.on_checkpoint(checkpoint)

        def on_progress(progress: BatchProgress) -> None:
            if events.on_progress is not None:
                events.on_progress(
                    Progress(
                        total_lines=len(job.lines),
                        done_lines=len(prior) + progress.completed_lines,
                        total_batches=progress.total_batches,
                        done_batches=progress.completed_batches,
                        failed_batches=progress.failed_batches,
                        tokens=meter.tokens,
                        cost=meter.cost,
                        message=progress.message,
                        connection_id=connection.id,
                    )
                )

        def fail(
            kind: ErrorKind,
            message: str,
            *,
            retry_after=None,
            summary="",
            reason=None,
        ) -> EngineError:
            if reason is None and kind is ErrorKind.CANCELLED:
                # The batch processor produces cancelled only when a request did not fit the budget.
                reason = "budget_exhausted"
            return EngineError(
                kind,
                message,
                reason=reason,
                connection_id=connection.id,
                backend=connection.backend_type,
                model=opened.model,
                retry_after=retry_after,
                tokens=meter.tokens,
                cost=meter.cost,
                checkpoints=produced,
                summary=summary,
                reserved=meter.reserved,
            )

        translations: dict[int, str] = {}
        summary = ""
        if remaining:
            processor = BatchProcessor(
                opened.provider, opened.settings, entry.resolver, meter=meter
            )
            config = opened.config.for_operation(job.session_id)
            try:
                result = await processor.process_all_batches(
                    lines=[
                        {"index": str(line.position), "content": line.text} for line in remaining
                    ],
                    source_language=job.source.code,
                    target_language=job.target.code,
                    context_title=job.media.title,
                    context_media_type=job.media.type,
                    batch_size=opened.batch_size,
                    progress_callback=on_progress,
                    config_override=config,
                    source=job.source,
                    target=job.target,
                    media=job.media,
                    on_root_result=on_root,
                )
            except asyncio.CancelledError:
                raise
            except TranslationProviderError as error:
                raise fail(
                    error_kind(error),
                    error.message,
                    retry_after=error.retry_after,
                    reason=getattr(error, "reason", None),
                ) from error
            except Exception as error:  # noqa: BLE001 - reported as internal, never swallowed
                raise fail(ErrorKind.INTERNAL, f"{type(error).__name__}: {error}") from error
            failed = [r for r in result.batch_results if not r.success]
            if failed:
                summary = summarize_batch_failure(result, result.progress.total_batches)
            stop = most_severe(
                r.error_kind for r in failed if r.error_kind in _ATTEMPT_FAILURE_KINDS
            )
            if stop is not None:
                of_kind = [r for r in failed if r.error_kind is stop]
                retry_after = max(
                    (r.retry_after for r in of_kind if r.retry_after is not None), default=None
                )
                raise fail(
                    stop,
                    of_kind[0].error or str(stop),
                    retry_after=retry_after,
                    summary=summary,
                    reason=next((r.reason for r in of_kind if r.reason), None),
                )
            for item in result.all_translations:
                try:
                    position = int(item["index"])
                except (KeyError, TypeError, ValueError):
                    continue
                content = item.get("content", "")
                if len(content) <= MAX_LINE_CHARS:
                    translations[position] = content

        merged = {**prior, **translations}
        out: list[Line] = []
        missing: list[int] = []
        for line in job.lines:
            if line.position in merged:
                text = merged[line.position]
                if job.add_rtl_marks and job.target.rtl and text.strip():
                    text = add_rtl_markers(text)
                if len(text) > MAX_LINE_CHARS:
                    # The marks pushed a translation past the longest line a caller can take.
                    # That position keeps its source text, as the cap already does for a
                    # translation that was too long without them.
                    missing.append(line.position)
                    out.append(line)
                    continue
                out.append(Line(line.position, text))
            else:
                missing.append(line.position)
                out.append(line)
        return TranslationOutcome(
            lines=out,
            status="partial" if missing else "complete",
            missing=missing,
            backend=connection.backend_type,
            connection_id=connection.id,
            model=opened.model,
            tokens=meter.tokens,
            cost=meter.cost,
            summary=summary,
            batches=[BatchModel(tuple(sorted(c.lines)), c.model) for c in produced],
        )
