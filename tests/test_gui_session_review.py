"""Independent GUI session lifecycle and privacy checks with synthetic data."""

import asyncio
import hashlib
import json
from contextlib import ExitStack

import pytest

from subtitle_translator import gui
from subtitle_translator.queue.job_events import JobEvents
from subtitle_translator.queue.job_manager import JobStatus, JobType
from tests import test_ui_api

FILE_REQUEST = test_ui_api.FILE_REQUEST
KEY_A = test_ui_api.KEY_A
KEY_B = test_ui_api.KEY_B
ui_environment = test_ui_api.ui_environment


class GuiSession:
    """Run the ASGI WebSocket boundary on the test's loop with bounded waits."""

    def __init__(self, app, origin="http://testserver", send_hook=None, scheme="ws", root_path=""):
        self.app = app
        self.origin = origin
        self.send_hook = send_hook
        self.scheme = scheme
        self.root_path = root_path
        self.incoming = asyncio.Queue(maxsize=32)
        self.outgoing = asyncio.Queue(maxsize=32)
        self.task = None

    async def __aenter__(self):
        headers = [(b"host", b"testserver")]
        if self.origin is not None:
            headers.append((b"origin", self.origin.encode()))
        scope = {
            "type": "websocket",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "scheme": self.scheme,
            "path": f"{self.root_path}/ui/session",
            "raw_path": f"{self.root_path}/ui/session".encode(),
            "query_string": b"",
            "root_path": self.root_path,
            "headers": headers,
            "client": ("127.0.0.1", 30000),
            "server": ("testserver", 80),
            "subprotocols": [],
            "state": {},
        }

        async def send(message):
            if self.send_hook:
                await self.send_hook(message)
            await self.outgoing.put(message)

        await self.incoming.put({"type": "websocket.connect"})
        self.task = asyncio.create_task(self.app(scope, self.incoming.get, send))
        return self

    async def __aexit__(self, *_):
        if not self.task.done():
            await self.incoming.put({"type": "websocket.disconnect", "code": 1000})
        try:
            await asyncio.wait_for(asyncio.shield(self.task), 1)
        except TimeoutError:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            raise AssertionError("GUI session did not finish disconnect cleanup") from None

    async def raw(self):
        return await asyncio.wait_for(self.outgoing.get(), 1)

    async def send_json(self, value):
        await self.incoming.put({"type": "websocket.receive", "text": json.dumps(value)})

    async def json(self):
        message = await self.raw()
        assert message["type"] == "websocket.send", message
        return json.loads(message.get("text") or message["bytes"])

    async def authenticate(self, key=KEY_A):
        assert (await self.raw())["type"] == "websocket.accept"
        await self.send_json({"type": "auth", "apiKey": key})
        ready = await self.json()
        assert ready["type"] == "ready", ready
        snapshot = await self.json()
        assert snapshot["type"] == "snapshot", snapshot
        return snapshot

    async def until(self, kind, request_id=None):
        for _ in range(20):
            frame = await self.json()
            if frame["type"] == kind and (request_id is None or frame.get("id") == request_id):
                return frame
        raise AssertionError(f"No {kind} message for request {request_id}")


def owner(key):
    return hashlib.sha256(b"subtitle-translator-ui\0" + key.encode()).hexdigest()


async def seeded_job(manager, key=KEY_A, **extra):
    return await manager.submit_job(
        request_data={**FILE_REQUEST, "_ui_owner": owner(key), **extra},
        job_type=JobType.TRANSLATE_FILE,
        api_key_override=key,
        metadata={"file_name": "synthetic.srt", "target_language": "hu"},
    )


def test_notifications_coalesce_without_waking_other_owners():
    events = JobEvents()
    with events.subscribe("owner-a") as first, events.subscribe("owner-a") as second:
        with events.subscribe("owner-b") as other:
            for _ in range(5000):
                events.notify("owner-a")
            assert first.is_set() and second.is_set() and not other.is_set()
            first.clear()
            events.notify("owner-b")
            assert not first.is_set() and second.is_set() and other.is_set()
    assert events._subscribers == {}


def test_subscription_capacity_releases_after_exception():
    events = JobEvents()
    with ExitStack() as stack:
        for _ in range(8):
            stack.enter_context(events.subscribe("owner-a"))
        with pytest.raises(RuntimeError):
            with events.subscribe("owner-a"):
                pytest.fail("Owner subscription limit was bypassed")
        with events.subscribe("owner-b"):
            pass
    assert events._subscribers == {}
    with pytest.raises(ValueError):
        with events.subscribe("owner-a"):
            raise ValueError("Synthetic subscriber failure")
    assert events._subscribers == {}


@pytest.mark.asyncio
async def test_store_failure_does_not_emit_success_invalidation(ui_environment, monkeypatch):
    _, manager, store, *_ = ui_environment
    job_id = await seeded_job(manager)

    def fail_save(_job):
        raise OSError("Synthetic store failure")

    with manager.events.subscribe(owner(KEY_A)) as changed:
        monkeypatch.setattr(store, "save_job", fail_save)
        with pytest.raises(OSError, match="Synthetic store failure"):
            manager.update_progress(job_id, 50, "Synthetic progress")
        assert not changed.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "origin", [None, "null", "http://testserver:81", "http://testserver.evil", "https://testserver"]
)
async def test_foreign_or_absent_origin_never_validates_a_key(ui_environment, origin):
    app, manager, _, _, network = ui_environment
    async with GuiSession(app, origin=origin) as session:
        response = await session.raw()
        assert response["type"] == "websocket.close", response
    assert network["calls"] == []
    assert manager.events._subscribers == {}


@pytest.mark.asyncio
async def test_rejected_authentication_send_has_a_deadline(ui_environment, monkeypatch):
    app, manager, _, _, network = ui_environment
    network["status"] = 401
    monkeypatch.setattr(gui, "SEND_TIMEOUT", 0.02)
    blocked = asyncio.Event()
    release = asyncio.Event()

    async def slow_error_reader(message):
        if message["type"] == "websocket.send":
            value = json.loads(message.get("text") or message["bytes"])
            if value.get("type") == "error":
                blocked.set()
                await release.wait()

    async with GuiSession(app, send_hook=slow_error_reader) as session:
        assert (await session.raw())["type"] == "websocket.accept"
        await session.send_json({"type": "auth", "apiKey": KEY_A})
        await asyncio.wait_for(blocked.wait(), 1)
        try:
            await asyncio.wait_for(asyncio.shield(session.task), 0.2)
        finally:
            release.set()
    assert manager.events._subscribers == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_status, expected", [(401, 401), (429, 429), (500, 503)])
async def test_idle_session_revalidates_and_stops_private_sends(
    ui_environment, monkeypatch, provider_status, expected
):
    app, manager, _, _, network = ui_environment
    monkeypatch.setattr(gui, "HEARTBEAT_SECONDS", 0.01)
    async with GuiSession(app) as session:
        await session.authenticate()
        assert len(network["calls"]) == 1
        network["now"] = 301
        network["status"] = provider_status
        failure = await session.until("error")
        assert failure["status"] == expected
        assert KEY_A not in json.dumps(failure)
        assert (await session.raw())["type"] == "websocket.close"
        assert len(network["calls"]) == 2
    assert manager.events._subscribers == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_kind", ["extra_request_field", "key_override", "unknown_command"])
async def test_command_validation_never_echoes_private_payload(ui_environment, invalid_kind):
    app, manager, *_ = ui_environment
    canary = "PRIVATE_COMMAND_CANARY_46a1"
    request = {**FILE_REQUEST, "content": canary}
    if invalid_kind == "extra_request_field":
        request["unexpected"] = KEY_A
    if invalid_kind == "key_override":
        request["config"] = {"apiKey": KEY_B}
    command = {
        "id": "invalid-submit",
        "type": "unknown" if invalid_kind == "unknown_command" else "submit",
        "payload": {"submissionId": "file-1", "request": request},
    }
    async with GuiSession(app) as session:
        await session.authenticate()
        await session.send_json(command)
        reply = await session.until("reply", command["id"])
        assert reply["error"]["status"] == 422
        for secret in (canary, KEY_A, KEY_B):
            assert secret not in json.dumps(reply)
        assert not manager.jobs
    assert manager.events._subscribers == {}


@pytest.mark.asyncio
async def test_disconnect_after_acceptance_keeps_job_and_reconnects_without_replay(ui_environment):
    app, manager, store, *_ = ui_environment
    reply_blocked = asyncio.Event()

    async def lose_submit_reply(message):
        if message["type"] == "websocket.send":
            value = json.loads(message.get("text") or message["bytes"])
            if value.get("type") == "reply" and value.get("id") == "submit-1":
                reply_blocked.set()
                await asyncio.Event().wait()

    async with GuiSession(app, send_hook=lose_submit_reply) as session:
        await session.authenticate()
        await session.send_json(
            {
                "id": "submit-1",
                "type": "submit",
                "payload": {"submissionId": "lost-ack-file", "request": FILE_REQUEST},
            }
        )
        await asyncio.wait_for(reply_blocked.wait(), 1)
        assert len(manager.jobs) == 1
        job_id = next(iter(manager.jobs))
    assert manager.events._subscribers == {}
    assert manager.get_job(job_id).status == JobStatus.QUEUED
    assert manager.queue.qsize() == 1
    assert store.load_job(job_id).request_data["_ui_submission"] == "lost-ack-file"
    async with GuiSession(app) as reconnected:
        snapshot = await reconnected.authenticate()
        assert [(job["jobId"], job["submissionId"]) for job in snapshot["jobs"]] == [
            (job_id, "lost-ack-file")
        ]
    assert len(manager.jobs) == manager.queue.qsize() == 1
    assert manager.events._subscribers == {}


@pytest.mark.asyncio
async def test_snapshot_excludes_source_result_key_and_provider_echoes(ui_environment):
    app, manager, *_ = ui_environment
    source = "PRIVATE_SOURCE_CANARY_8fb276"
    result = "PRIVATE_RESULT_CANARY_5d2e89"
    job_id = await seeded_job(manager, content=source, config={"apiKey": KEY_A})
    manager.set_job_partial(
        job_id, {"content": result}, f"Upstream echoed {KEY_A} {source} {result}"
    )
    async with GuiSession(app) as session:
        snapshot = await session.authenticate()
        assert snapshot["jobs"][0]["jobId"] == job_id
        serialized = json.dumps(snapshot)
        for forbidden in (KEY_A, source, result, "request_data", "api_key_override"):
            assert forbidden not in serialized
        assert "result" not in snapshot["jobs"][0]
        assert "content" not in snapshot["jobs"][0]
    assert manager.events._subscribers == {}


@pytest.mark.asyncio
async def test_other_owner_cannot_read_or_cancel_a_job(ui_environment):
    app, manager, *_ = ui_environment
    job_id = await seeded_job(manager)
    async with GuiSession(app) as session:
        snapshot = await session.authenticate(KEY_B)
        assert snapshot["jobs"] == []
        for command in ("job", "source", "cancel"):
            await session.send_json({"id": command, "type": command, "payload": {"jobId": job_id}})
            response = await session.until("reply", command)
            assert response["error"]["status"] == 404
            assert KEY_A not in json.dumps(response)
        assert manager.get_job(job_id).status == JobStatus.QUEUED
    assert manager.events._subscribers == {}


@pytest.mark.asyncio
async def test_secure_origin_and_proxy_root_path_work(ui_environment):
    app, manager, *_ = ui_environment
    async with GuiSession(
        app, origin="https://testserver", scheme="wss", root_path="/translator"
    ) as session:
        assert (await session.authenticate())["jobs"] == []
    assert manager.events._subscribers == {}


@pytest.mark.asyncio
async def test_explicit_owned_content_commands_do_not_leak_other_job_fields(ui_environment):
    app, manager, *_ = ui_environment
    source = "SOURCE_COMMAND_CANARY"
    translated = "RESULT_COMMAND_CANARY"
    job_id = await seeded_job(manager, content=source)
    manager.set_job_completed(job_id, {"content": translated, "privateExtra": KEY_A})
    async with GuiSession(app) as session:
        await session.authenticate()
        await session.send_json({"id": "source-1", "type": "source", "payload": {"jobId": job_id}})
        source_reply = await session.until("reply", "source-1")
        assert source_reply["value"] == {"content": source}
        await session.send_json({"id": "job-1", "type": "job", "payload": {"jobId": job_id}})
        result_reply = await session.until("reply", "job-1")
        assert result_reply["value"]["result"] == {"content": translated}
        for reply in (source_reply, result_reply):
            assert KEY_A not in json.dumps(reply)
            assert "privateExtra" not in json.dumps(reply)
    assert manager.events._subscribers == {}


@pytest.mark.asyncio
async def test_pending_command_limit_and_disconnect_cleanup(ui_environment, monkeypatch):
    app, manager, *_ = ui_environment
    active = set()

    async def slow_models(_identity):
        current = asyncio.current_task()
        active.add(current)
        try:
            await asyncio.Event().wait()
        finally:
            active.remove(current)

    monkeypatch.setattr(gui.ui_api, "models", slow_models)
    async with GuiSession(app) as session:
        await session.authenticate()
        for index in range(gui.MAX_COMMANDS + 1):
            await session.send_json({"id": f"models-{index}", "type": "models"})
        rejected = await session.until("reply", f"models-{gui.MAX_COMMANDS}")
        assert rejected["error"]["status"] == 429
        assert len(active) == gui.MAX_COMMANDS
    assert not active
    assert manager.events._subscribers == {}


@pytest.mark.asyncio
async def test_change_during_initial_snapshot_send_is_not_lost(ui_environment):
    app, manager, *_ = ui_environment
    job_id = await seeded_job(manager)
    blocked = asyncio.Event()
    release = asyncio.Event()
    first = True

    async def slow_initial_snapshot(message):
        nonlocal first
        if message["type"] == "websocket.send":
            value = json.loads(message.get("text") or message["bytes"])
            if value.get("type") == "snapshot" and first:
                first = False
                blocked.set()
                await release.wait()

    try:
        async with GuiSession(app, send_hook=slow_initial_snapshot) as session:
            assert (await session.raw())["type"] == "websocket.accept"
            await session.send_json({"type": "auth", "apiKey": KEY_A})
            assert (await session.json())["type"] == "ready"
            await asyncio.wait_for(blocked.wait(), 1)
            manager.update_progress(job_id, 73, "Synthetic latest activity")
            release.set()
            await session.until("snapshot")
            latest = await session.until("snapshot")
            assert latest["jobs"][0]["progress"] == 73
    finally:
        release.set()
    assert manager.events._subscribers == {}
