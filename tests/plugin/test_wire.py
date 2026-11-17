# tests/plugin/test_wire.py
"""The dictionaries Bazarr+ sends become core types; malformed input is a named invalid_config."""

import pytest

from subtitle_translator.core.checkpoints import Checkpoint, CheckpointLine, source_digest
from subtitle_translator.core.engine import EngineError, Line, TranslationOutcome
from subtitle_translator.core.errors import ErrorKind
from subtitle_translator.plugin.wire import WireError, read_job, write_error, write_outcome

REQUEST = {
    "job_id": "job-7",
    "source": {"code": "en", "name": "English"},
    "target": {"code": "sr-Latn", "name": "Serbian", "script": "Latn", "rtl": False},
    "lines": [{"position": 0, "text": "<i>Hello</i>"}, {"position": 1, "text": "Bye"}],
    "media": {"type": "movie", "title": "Arrival", "year": 2016, "imdb_id": "tt2543164", "overview": "Linguist."},
    "config": {
        "settings": {"request_timeout": 300, "max_retries": 2},
        "connection": {"id": "c1", "type": "gemini", "enabled": True, "settings": {"model": "gemini-2.5-flash"}},
    },
    "secrets": {"api_keys": ["AIza-wire-secret"]},
    "checkpoints": [],
    "budget": 0.5,
}


def test_a_full_request_reads_into_core_types():
    job, connection, base = read_job(REQUEST)
    assert job.target.script == "Latn" and job.source.name == "English"
    assert job.lines == [Line(0, "<i>Hello</i>"), Line(1, "Bye")]
    assert job.media.imdb_id == "tt2543164" and job.budget == 0.5 and job.session_id == "job-7"
    assert job.add_rtl_marks is False
    assert connection.backend_type == "gemini" and connection.secrets == {"api_keys": ["AIza-wire-secret"]}
    assert "AIza-wire-secret" not in repr(connection)
    assert base.request_timeout == 300 and base.max_retries == 2


def test_auto_source_and_checkpoints():
    request = {**REQUEST, "source": {"code": "auto"}, "checkpoints": [Checkpoint(
        {0: CheckpointLine(source_digest("<i>Hello</i>"), "<i>Zdravo</i>")}, 5, 0.01, 5, 0.01, "c0", "m"
    ).to_dict()]}
    job, _, _ = read_job(request)
    assert job.source.is_auto and job.checkpoints[0].lines[0].text == "<i>Zdravo</i>"


@pytest.mark.parametrize(
    "patch,field",
    [
        ({"lines": [{"position": "zero", "text": "x"}]}, "lines[0].position"),
        ({"lines": [{"position": 0}]}, "lines[0].text"),
        ({"lines": "nope"}, "lines"),
        ({"target": "hu"}, "target"),
        ({"checkpoints": [{"lines": 3}]}, "checkpoints[0]"),
        ({"checkpoints": 7}, "checkpoints"),
        ({"config": {"connection": {"type": "gemini"}}}, "config.connection.id"),
        ({"budget": -1}, "budget"),
        ({"lines": [{"position": 0, "text": "x" * 2001}]}, "lines[0].text"),
        ({"lines": [None] * 10_001}, "lines"),                   # refused before any line is read
        ({"config": {**REQUEST["config"], "settings": {"max_retries": 1.5}}}, "config.settings.max_retries"),
        ({"config": {**REQUEST["config"], "settings": {"max_retries": 7}}}, "config.settings.max_retries"),
        ({"config": {**REQUEST["config"], "settings": {"request_timeout": 5}}}, "config.settings.request_timeout"),
    ],
)
def test_malformed_requests_are_invalid_config(patch, field):
    with pytest.raises(WireError) as caught:
        read_job({**REQUEST, **patch})
    assert caught.value.field == field


def test_outcome_and_error_shapes():
    outcome = TranslationOutcome([Line(0, "Zdravo")], "complete", [], "gemini", "c1", "gemini-2.5-flash", 9, 0.0)
    assert write_outcome(outcome) == {
        "lines": [{"position": 0, "text": "Zdravo"}], "status": "complete", "missing": [],
        "backend": "gemini", "connection": "c1", "model": "gemini-2.5-flash", "tokens": 9, "cost": 0.0,
        "batches": [],
    }
    error = EngineError(ErrorKind.QUOTA_EXHAUSTED, "daily", retry_after=3600.0)
    assert write_error(error) == {"code": "quota_exhausted", "message": "daily", "retry_after": 3600.0,
                                  "field_errors": {}, "reason": None}
    budget = EngineError(ErrorKind.CANCELLED, "Cost budget exhausted", reason="budget_exhausted")
    assert write_error(budget)["reason"] == "budget_exhausted"


def test_the_line_limit_is_the_one_describe_reports():
    from subtitle_translator.plugin import contract
    from subtitle_translator.plugin.wire import MAX_LINES

    assert contract.MAX_LINES is MAX_LINES
    job, _, _ = read_job({**REQUEST, "lines": [{"position": i, "text": "x"} for i in range(MAX_LINES)]})
    assert len(job.lines) == MAX_LINES
