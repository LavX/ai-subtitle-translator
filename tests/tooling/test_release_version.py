"""A tag publishes only the version it names, and says whether it is a pre-release."""

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_release_version.py"


def run(tag, version):
    return subprocess.run(
        [sys.executable, str(SCRIPT), tag, "--version", version], capture_output=True, text=True
    )


def test_matching_final_and_prerelease_tags():
    assert run("v2.2.0", "2.2.0").stdout.strip() == "prerelease=false"
    assert run("v2.2.0rc1", "2.2.0rc1").stdout.strip() == "prerelease=true"
    assert run("v2.2.0.dev0", "2.2.0.dev0").stdout.strip() == "prerelease=true"


def test_a_mismatched_tag_is_refused():
    result = run("v2.2.0", "2.2.0rc1")
    assert result.returncode == 1 and "2.2.0rc1" in result.stderr


def test_the_publish_workflow_uses_trusted_publishing_and_stores_no_token():
    text = (SCRIPT.parents[1] / ".github" / "workflows" / "publish.yml").read_text()
    assert text.count("id-token: write") == 2
    assert "secrets." not in text and "password:" not in text
    assert "environment: pypi" in text and "environment: testpypi" in text


def _publish_workflow():
    return (SCRIPT.parents[1] / ".github" / "workflows" / "publish.yml").read_text()


def test_pypi_is_published_by_a_release_and_never_by_a_tag_push():
    text = _publish_workflow()
    triggers = text.split("\njobs:")[0]
    assert "release:\n    types: [published]" in triggers
    assert "push:" not in triggers and "tags:" not in triggers
    pypi = text.split("  publish-pypi:")[1].split("  publish-testpypi:")[0]
    testpypi = text.split("  publish-testpypi:")[1]
    assert "if: github.event_name == 'release'" in pypi
    assert "if: github.event_name == 'workflow_dispatch'" in testpypi
    assert "github.event.release.tag_name" in text and "GITHUB_REF" not in text


def test_every_action_in_the_publish_workflow_is_pinned_by_commit():
    import re

    text = _publish_workflow()
    uses = re.findall(r"uses:\s*(\S+)(.*)", text)
    assert uses
    for action, comment in uses:
        assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", action), action
        assert re.search(r"#\s*v\d", comment), action
