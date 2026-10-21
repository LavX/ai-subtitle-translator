"""Checkpoints survive transport and never paste an old translation over edited dialogue."""

import pytest

from subtitle_translator.core.checkpoints import (
    Checkpoint,
    CheckpointLine,
    source_digest,
    usable_translations,
)
from subtitle_translator.core.engine import Line


def _checkpoint(lines, tokens=10, cost=0.01):
    return Checkpoint(
        lines={p: CheckpointLine(source_digest(src), text) for p, (src, text) in lines.items()},
        tokens=tokens, cost=cost, total_tokens=tokens, total_cost=cost, connection_id="c1", model="m",
    )


def test_round_trip_through_a_dict():
    original = _checkpoint({0: ("Hello", "Hola"), 1: ("Bye", "Adiós")})
    assert Checkpoint.from_dict(original.to_dict()) == original


@pytest.mark.parametrize("garbage", [{}, {"lines": "x"}, {"lines": {"a": {}}, "tokens": 1}, None])
def test_malformed_checkpoints_are_refused(garbage):
    with pytest.raises(ValueError):
        Checkpoint.from_dict(garbage)


def test_checkpoint_lines_whose_source_changed_are_translated_again():
    old = _checkpoint({0: ("Hello", "Hola"), 1: ("I will go.", "Iré.")})
    edited = [Line(0, "Hello"), Line(1, "I will not go.")]
    assert usable_translations(edited, [old]) == {0: "Hola"}


def test_a_later_checkpoint_wins():
    first = _checkpoint({0: ("Hello", "Hola")})
    second = _checkpoint({0: ("Hello", "Buenas")})
    assert usable_translations([Line(0, "Hello")], [first, second]) == {0: "Buenas"}
