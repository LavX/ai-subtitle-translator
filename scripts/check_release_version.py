#!/usr/bin/env python3
"""Refuse to publish a tag that does not match the package version; report pre-releases."""

import argparse
import re
import sys
import tomllib
from pathlib import Path

PRE = re.compile(r"(a|b|rc|\.dev)\d+")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tag")
    parser.add_argument("--version", help="defaults to pyproject.toml's version")
    args = parser.parse_args()
    version = args.version or tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"]
    if args.tag != f"v{version}":
        print(f"tag {args.tag} does not match package version {version}", file=sys.stderr)
        return 1
    print(f"prerelease={'true' if PRE.search(version) else 'false'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
