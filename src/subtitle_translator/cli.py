"""CLI commands for AI Subtitle Translator."""

import argparse
import os
import sys

from subtitle_translator.config import get_settings
from subtitle_translator.crypto import generate_key


def regenerate_key():
    """Regenerate the encryption key file."""
    settings = get_settings()

    if settings.encryption_key:
        print("ERROR: ENCRYPTION_KEY env var is set. Remove it to use file-based keys.")
        sys.exit(1)

    if not settings.encryption_enabled:
        print("ERROR: Encryption is disabled (ENCRYPTION_ENABLED=false).")
        sys.exit(1)

    key_path = settings.encryption_key_file
    key = generate_key()

    os.makedirs(os.path.dirname(key_path) or ".", exist_ok=True)
    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(key.hex())

    print("New encryption key generated.")
    print(f"Saved to: {key_path}")
    print("Share this key with Bazarr for encrypted API key transport.")


def main():
    parser = argparse.ArgumentParser(description="AI Subtitle Translator CLI")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("regenerate-key", help="Generate a new encryption key")

    args = parser.parse_args()
    if args.command == "regenerate-key":
        regenerate_key()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
