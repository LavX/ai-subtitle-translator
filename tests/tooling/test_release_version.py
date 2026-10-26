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
