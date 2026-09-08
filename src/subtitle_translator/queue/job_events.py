"""Coalesced in-process notifications for GUI sessions."""

import asyncio
from contextlib import contextmanager


class JobEvents:
    def __init__(self):
        self._subscribers: dict[str, set[asyncio.Event]] = {}

    @contextmanager
    def subscribe(self, owner: str):
        existing = self._subscribers.get(owner, set())
        if len(existing) >= 8 or sum(map(len, self._subscribers.values())) >= 128:
            raise RuntimeError("Too many GUI sessions")
        changed = asyncio.Event()
        self._subscribers.setdefault(owner, set()).add(changed)
        try:
            yield changed
        finally:
            self._subscribers[owner].discard(changed)
            if not self._subscribers[owner]:
                del self._subscribers[owner]

    def notify(self, owner: str | None):
        for changed in self._subscribers.get(owner, ()):
            changed.set()
