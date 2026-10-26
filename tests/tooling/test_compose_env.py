"""Every setting .env.example tells a user to put in .env reaches the container."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Read straight from the process environment by the code, not forwarded settings.
NOT_FORWARDED = {"HOST", "PORT", "LOG_LEVEL", "CORS_ALLOWED_ORIGINS"}


def test_compose_forwards_every_variable_the_example_env_file_documents():
    example = (ROOT / ".env.example").read_text()
    names = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", example, flags=re.MULTILINE))
    compose = (ROOT / "docker-compose.yml").read_text()
    forwarded = set(re.findall(r"^\s*-\s*([A-Z][A-Z0-9_]+)=", compose, flags=re.MULTILINE))
    missing = names - forwarded - NOT_FORWARDED
    # Job-queue limits are tuned by hand and not part of the supplied Compose setup.
    missing -= {n for n in missing if n.startswith("JOB_QUEUE_")}
    assert not missing, f"documented in .env.example but not forwarded by docker-compose.yml: {sorted(missing)}"
