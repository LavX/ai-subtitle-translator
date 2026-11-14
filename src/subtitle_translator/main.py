"""Compatibility entry point.

``uvicorn subtitle_translator.main:app`` and ``python -m subtitle_translator.main`` keep working
after the service moved to ``subtitle_translator.server``. Needs the ``[server]`` extra.
"""

from subtitle_translator.server.main import app, create_app, lifespan  # noqa: F401

if __name__ == "__main__":
    from subtitle_translator.server.main import run

    run()
