"""The browser GUI talks to one app-owned session with private job snapshots."""

import json

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tests import test_ui_api
from tests.test_ui_api import FILE_REQUEST, KEY_A, KEY_B

ui_environment = test_ui_api.ui_environment


def open_session(client, key=KEY_A):
    connection = client.websocket_connect("/ui/session", headers={"origin": "http://testserver"})
    ws = connection.__enter__()
    ws.send_json({"type": "auth", "apiKey": key})
    assert ws.receive_json()["type"] == "ready"
    assert ws.receive_json() == {"type": "snapshot", "jobs": []}
    return connection, ws


def until(ws, kind, request_id=None):
    for _ in range(20):
        value = ws.receive_json()
        if value["type"] == kind and (request_id is None or value.get("id") == request_id):
            return value
    raise AssertionError("Expected session message was not delivered")


def test_session_submission_snapshot_and_reconnect(ui_environment):
    app, manager, *_ = ui_environment
    client = TestClient(app)
    # The controller test uses the fixture manager without starting app workers.
    with client.websocket_connect("/ui/session", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "auth", "apiKey": KEY_A})
        assert ws.receive_json()["type"] == "ready"
        assert ws.receive_json() == {"type": "snapshot", "jobs": []}
        ws.send_json(
            {
                "id": "submit-1",
                "type": "submit",
                "payload": {"submissionId": "file-1", "request": FILE_REQUEST},
            }
        )
        reply = until(ws, "reply", "submit-1")
        job_id = reply["value"]["jobId"]
        assert len(manager.jobs) == 1
        ws.send_json({"id": "sync-1", "type": "restore"})
        snapshot = until(ws, "snapshot")
        assert snapshot["jobs"][0]["jobId"] == job_id
        assert snapshot["jobs"][0]["submissionId"] == "file-1"
        serialized = json.dumps(snapshot)
        assert KEY_A not in serialized and FILE_REQUEST["content"] not in serialized
        assert "request_data" not in serialized and "result" not in snapshot["jobs"][0]
    with client.websocket_connect("/ui/session", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "auth", "apiKey": KEY_A})
        assert ws.receive_json()["type"] == "ready"
        assert ws.receive_json()["jobs"][0]["jobId"] == job_id
    with client.websocket_connect("/ui/session", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "auth", "apiKey": KEY_B})
        assert ws.receive_json()["type"] == "ready"
        assert ws.receive_json()["jobs"] == []
        ws.send_json({"id": "other", "type": "job", "payload": {"jobId": job_id}})
        assert until(ws, "reply", "other")["error"]["status"] == 404


def test_wrong_origin_cannot_open_gui_session(ui_environment):
    client = TestClient(ui_environment[0])
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ui/session", headers={"origin": "http://other.example"}):
            pass
