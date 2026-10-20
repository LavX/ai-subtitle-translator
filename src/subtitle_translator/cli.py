"""Compatibility entry point: ``python -m subtitle_translator.cli`` keeps working."""

from subtitle_translator.server.cli import main, regenerate_key  # noqa: F401

if __name__ == "__main__":
    main()
