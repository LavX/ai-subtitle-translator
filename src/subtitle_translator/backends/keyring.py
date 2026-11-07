"""Round-robin over API keys, with health tracked per key and per project.

Gemini applies its limits per Google Cloud project and model, not per key, so keys the user
labels with the same project share every block. Health is reported by position only: no
character of a key ever leaves this class.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum


class KeyState(StrEnum):
    OK = "ok"
    COOLING = "cooling"
    EXHAUSTED = "exhausted"
    INVALID = "invalid"


@dataclass(frozen=True)
class KeyHealth:
    index: int
    group: str
    state: KeyState
    until: float | None
    reason: str | None


class KeyRing:
    def __init__(
        self,
        keys: Sequence[str],
        groups: Sequence[str | None] = (),
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._keys: list[str] = []
        self._groups: list[str] = []
        for position, raw in enumerate(keys):
            key = (raw or "").strip()
            if not key or key in self._keys:
                continue
            label = groups[position] if position < len(groups) else None
            self._keys.append(key)
            self._groups.append((label or "").strip() or f"key-{len(self._keys)}")
        self._clock = clock
        # A block remembers the tick it was set at, so a reply can be told apart from a block
        # that came after its request was sent.
        self._blocks: dict[tuple[str, str], tuple[float, KeyState, str, int]] = {}
        self._invalid: dict[str, str] = {}
        self._position = 0
        self._tick = 0

    def __len__(self) -> int:
        return len(self._keys)

    def _group(self, key: str) -> str:
        return self._groups[self._keys.index(key)]

    def _block(
        self, group: str, scope: str, now: float
    ) -> tuple[float, KeyState, str, int] | None:
        block = self._blocks.get((group, scope))
        return block if block is not None and block[0] > now else None

    def ticket(self) -> int:
        """A mark of this moment, taken when a request acquires its key and handed back to
        mark_success(), so that a reply clears only the blocks set before its request left."""
        return self._tick

    def _set_block(
        self, key: str, scope: str, until: float, state: KeyState, reason: str
    ) -> None:
        """Record a block. One still running that lasts longer is kept, so a short rate limit
        never shortens a quota block, but it takes the newer tick either way."""
        group = self._group(key)
        self._tick += 1
        current = self._block(group, scope, self._clock())
        if current is not None and current[0] >= until:
            until, state, reason = current[0], current[1], current[2]
        self._blocks[(group, scope)] = (until, state, reason, self._tick)

    def acquire(self, scope: str = "") -> str | None:
        now = self._clock()
        for offset in range(len(self._keys)):
            index = (self._position + offset) % len(self._keys)
            key = self._keys[index]
            if key in self._invalid or self._block(self._groups[index], scope, now):
                continue
            self._position = (index + 1) % len(self._keys)
            return key
        return None

    def mark_rate_limited(
        self, key: str, retry_after: float, scope: str = "", reason: str = "rate limited"
    ) -> None:
        self._set_block(
            key, scope, self._clock() + max(0.0, retry_after), KeyState.COOLING, reason
        )

    def mark_exhausted(
        self, key: str, until: float, scope: str = "", reason: str = "quota used up"
    ) -> None:
        self._set_block(key, scope, until, KeyState.EXHAUSTED, reason)

    def mark_invalid(self, key: str, reason: str = "rejected") -> None:
        self._invalid[key] = reason

    def mark_success(self, key: str, scope: str = "", *, since: int) -> None:
        """A request that got through proves the block it saw is over. A block set after that
        request acquired its key (since is its ticket) is newer than the proof and stays."""
        block = self._blocks.get((self._group(key), scope))
        if block is not None and block[3] <= since:
            del self._blocks[(self._group(key), scope)]

    def all_invalid(self) -> bool:
        return bool(self._keys) and all(key in self._invalid for key in self._keys)

    def wait(self, scope: str = "") -> tuple[KeyState, float] | None:
        now = self._clock()
        soonest: tuple[float, KeyState] | None = None
        for index, key in enumerate(self._keys):
            if key in self._invalid:
                continue
            block = self._block(self._groups[index], scope, now)
            if block is None:
                return None
            if soonest is None or block[0] < soonest[0]:
                soonest = (block[0], block[1])
        return None if soonest is None else (soonest[1], soonest[0] - now)

    def health(self, scope: str = "") -> list[KeyHealth]:
        now = self._clock()
        report = []
        for index, key in enumerate(self._keys):
            if key in self._invalid:
                report.append(
                    KeyHealth(
                        index + 1, self._groups[index], KeyState.INVALID, None, self._invalid[key]
                    )
                )
                continue
            block = self._block(self._groups[index], scope, now)
            if block is None:
                report.append(KeyHealth(index + 1, self._groups[index], KeyState.OK, None, None))
            else:
                report.append(
                    KeyHealth(index + 1, self._groups[index], block[1], block[0], block[2])
                )
        return report
