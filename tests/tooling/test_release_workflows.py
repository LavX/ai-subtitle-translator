"""Image tags: a pre-release never moves latest."""

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "image_tags.sh"


def _tags(ref: str, version_input: str = "", push_latest: str = "false") -> dict[str, str]:
    env = dict(
        os.environ,
        REGISTRY="ghcr.io",
        IMAGE_NAME="lavx/ai-subtitle-translator",
        GITHUB_REF=ref,
        VERSION_INPUT=version_input,
        PUSH_LATEST=push_latest,
        SHORT_SHA="abc1234",
        PYTHON_VERSION="3.14",
        JIT_SUFFIX="-jit",
    )
    out = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, check=True)
    return dict(line.split("=", 1) for line in out.stdout.splitlines() if "=" in line)


def test_final_release_tag_moves_latest():
    result = _tags("refs/tags/v2.2.0")
    assert result["version"] == "2.2.0"
    assert "ghcr.io/lavx/ai-subtitle-translator:latest" in result["tags"].split(",")


def test_prerelease_tags_never_move_latest():
    for ref in ("refs/tags/v2.2.0rc1", "refs/tags/v2.2.0-rc.1", "refs/tags/v2.2.0b1", "refs/tags/v2.2.0.dev0"):
        result = _tags(ref)
        assert not any(tag.endswith(":latest") for tag in result["tags"].split(",")), ref


def test_manual_dispatch_moves_latest_only_when_asked():
    assert not any(t.endswith(":latest") for t in _tags("refs/heads/development", "v2.2.0rc1")["tags"].split(","))
    assert any(t.endswith(":latest") for t in _tags("refs/heads/main", "v2.2.0", "true")["tags"].split(","))
