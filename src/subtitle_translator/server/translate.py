"""Standalone requests and jobs on the engine, keeping the 2.1.1 HTTP contract."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from subtitle_translator.core.checkpoints import usable_translations
from subtitle_translator.core.context import Language, MediaContext
from subtitle_translator.core.engine import (
    EngineError,
    EngineEvents,
    Line,
    Progress,
    TranslationJob,
)
from subtitle_translator.core.models import TranslationConfig
from subtitle_translator.core.srt_parser import add_rtl_markers
from subtitle_translator.core.translator import SubtitleTranslator
from subtitle_translator.server.runtime import EngineTranslator, Runtime, single_provider_runtime


@dataclass
class RunResult:
    lines: list[Line]
    done: set[int]
    model_used: str
    tokens: int
    cost: float
    status: Literal["completed", "partial", "failed"]
    error: str | None = None

    @property
    def translated(self) -> int:
        return sum(1 for line in self.lines if line.position in self.done)


def language(value: str, runtime: Runtime) -> Language:
    code = (value or "").strip()
    if code.lower() == "auto":
        return Language.auto()
    return Language(code=code, name=code, rtl=runtime.core.is_rtl_language(code))


def runtime_for(translator) -> Runtime:
    if isinstance(translator, EngineTranslator):
        return translator.runtime
    cached = getattr(translator, "_engine_runtime", None)
    if not isinstance(cached, Runtime):
        cached = single_provider_runtime(translator.provider, translator.settings)
        if isinstance(translator, SubtitleTranslator):
            translator._engine_runtime = cached
    return cached


async def run_lines(
    runtime: Runtime,
    lines: list[Line],
    *,
    source: str,
    target: str,
    title: str | None = None,
    media_type: str | None = None,
    config: TranslationConfig | None = None,
    model: str | None = None,
    temperature: float | None = None,
    job_id: str | None = None,
    add_rtl_marks: bool = True,
    on_progress: Callable[[Progress], None] | None = None,
) -> RunResult:
    connections, pinned = runtime.plan(config, model=model, temperature=temperature)
    target_language = language(target, runtime)
    job = TranslationJob(
        source=language(source, runtime),
        target=target_language,
        lines=lines,
        media=MediaContext(type=media_type, title=title),
        add_rtl_marks=add_rtl_marks,
        session_id=job_id,
    )
    try:
        outcome = await runtime.engine.translate(job, connections, pinned=pinned, events=EngineEvents(on_progress=on_progress))
    except EngineError as error:
        done = usable_translations(lines, error.checkpoints)
        reason = error.summary or error.message
        model_used = error.model or runtime.planned_model(connections, pinned)
        if not done:
            return RunResult(list(lines), set(), model_used, error.tokens, error.cost, "failed", reason)
        mark = add_rtl_marks and target_language.rtl
        out = [
            Line(line.position, (add_rtl_markers(done[line.position]) if mark else done[line.position]))
            if line.position in done
            else line
            for line in lines
        ]
        return RunResult(out, set(done), model_used, error.tokens, error.cost, "partial", reason)
    missing = set(outcome.missing)
    done = {line.position for line in lines if line.position not in missing}
    # A cue with no text comes back as it went in: it is not a translation. Only the cues that
    # carry text decide whether anything was translated.
    worth_translating = {line.position for line in lines if line.text.strip()}
    if worth_translating and not (done & worth_translating):
        # Nothing came back: every attempt was billed, but no line is a translation.
        return RunResult(
            list(lines), set(), outcome.model, outcome.tokens, outcome.cost, "failed",
            outcome.summary or "Translation failed",
        )
    status = "completed" if outcome.status == "complete" else "partial"
    return RunResult(outcome.lines, done, outcome.model, outcome.tokens, outcome.cost, status, outcome.summary or None)
