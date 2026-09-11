"""Shared preview fixture has the same four captions in the backend parser."""

from pathlib import Path

from subtitle_translator.core.srt_parser import SRTParser


def test_blank_lines_around_timestamps_preserve_four_backend_captions():
    entries = SRTParser().parse((Path(__file__).parent / "ui" / "blank-lines.srt").read_text())
    assert [entry.content.strip() for entry in entries] == [
        "First cue",
        "<i>Second cue</i>\nSecond line",
        "Third cue <script>literal</script>",
        "Last cue",
    ]
    assert [entry.start.total_seconds() for entry in entries] == [1, 3, 3, 5]
    assert [entry.end.total_seconds() for entry in entries] == [2, 4, 4, 6]
