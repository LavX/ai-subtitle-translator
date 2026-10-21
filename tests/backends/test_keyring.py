"""Rotation across keys; limits shared by keys of one project; health without key material."""

from subtitle_translator.backends.keyring import KeyRing, KeyState


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


def test_blank_and_duplicate_keys_are_dropped_and_rotation_is_round_robin():
    ring = KeyRing([" a ", "", "b", "a", "c"])
    assert len(ring) == 3
    assert [ring.acquire() for _ in range(4)] == ["a", "b", "c", "a"]
    assert KeyRing([]).acquire() is None


def test_a_rate_limited_key_sits_out_its_wait_for_that_model_only():
    clock = Clock()
    ring = KeyRing(["a", "b"], clock=clock)
    ring.mark_rate_limited("a", 30, scope="flash")
    assert {ring.acquire("flash") for _ in range(3)} == {"b"}
    assert ring.acquire("pro") in {"a", "b"}
    clock.now += 31
    assert "a" in {ring.acquire("flash") for _ in range(2)}


def test_keys_of_one_project_share_its_limit():
    clock = Clock()
    ring = KeyRing(["a", "b", "c"], ["p1", "p1", None], clock=clock)
    ring.mark_exhausted("a", clock.now + 3600, scope="flash")
    assert {ring.acquire("flash") for _ in range(4)} == {"c"}  # b shares a's project


def test_wait_reports_the_soonest_block_and_its_state():
    clock = Clock()
    ring = KeyRing(["a", "b"], clock=clock)
    ring.mark_exhausted("a", clock.now + 500, scope="m")
    ring.mark_rate_limited("b", 40, scope="m")
    assert ring.acquire("m") is None
    assert ring.wait("m") == (KeyState.COOLING, 40)
    assert ring.wait("other") is None


def test_invalid_keys_never_come_back_and_all_invalid_is_detectable():
    ring = KeyRing(["a", "b"])
    ring.mark_invalid("a")
    assert {ring.acquire() for _ in range(3)} == {"b"}
    ring.mark_invalid("b")
    assert ring.acquire() is None and ring.all_invalid()


def test_health_names_positions_never_keys():
    clock = Clock()
    ring = KeyRing(["sk-secret-one", "sk-secret-two"], ["proj", None], clock=clock)
    ring.mark_rate_limited("sk-secret-two", 10, scope="m")
    health = ring.health("m")
    assert [(h.index, h.state) for h in health] == [(1, KeyState.OK), (2, KeyState.COOLING)]
    assert health[0].group == "proj"
    assert "sk-secret" not in repr(health)


def test_success_clears_a_block():
    clock = Clock()
    ring = KeyRing(["a"], clock=clock)
    ring.mark_rate_limited("a", 10, scope="m")
    ring.mark_success("a", scope="m")
    assert ring.acquire("m") == "a"
