"""The vendored srt module behaves like the PyPI one for what the parser uses."""

import pytest

from subtitle_translator._vendor import srt

SAMPLE = "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n2\n00:00:03,000 --> 00:00:04,500\nWorld\n\n"


def test_parse_and_compose_roundtrip():
    subs = list(srt.parse(SAMPLE))
    assert [s.content for s in subs] == ["Hello", "World"]
    assert subs[1].start.total_seconds() == 3.0
    assert srt.compose(subs) == SAMPLE


def test_parse_error_is_the_documented_exception():
    with pytest.raises(srt.SRTParseError):
        list(srt.parse("this is not srt at all\n"))


def test_srt_parser_uses_the_vendored_module():
    import subtitle_translator.core.srt_parser as parser_module

    assert parser_module.srt is srt
