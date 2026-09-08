"""Browser race checks through a real local GUI session and synthetic worker."""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import expect, sync_playwright
from ui_browser import SECOND_KEY, TOKEN, upload


class SessionProxy:
    def __init__(self, case):
        self.case = case
        self.connections = []
        self.submits = []
        self.commands = {}
        self.held = None
        self.held_cancel = None
        self.held_snapshot = None
        self.last_snapshot = None
        self.dropped = False
        self.job_replies = 0

    def install(self, page):
        page.add_init_script("window.reviewSockets = [];")
        page.route_web_socket("**/ui/session", self.handle)

    def handle(self, client):
        server = client.connect_to_server()
        self.connections.append(client)
        connection = len(self.connections)

        def from_client(raw):
            value = json.loads(raw)
            kind = value.get("type")
            if kind not in ("auth", "pong"):
                self.commands[(connection, value["id"])] = kind
            if kind == "submit":
                self.submits.append(
                    {
                        "connection": connection,
                        "id": value["id"],
                        "submissionId": value["payload"]["submissionId"],
                        "fileName": value["payload"]["request"]["fileName"],
                    }
                )
            if self.case == "cancel_race" and kind == "cancel":
                self.held_cancel = (server, raw)
                return
            server.send(raw)

        def from_server(raw):
            value = json.loads(raw)
            kind = value.get("type")
            command = self.commands.get((connection, value.get("id")))
            if kind == "snapshot":
                self.last_snapshot = value
                if self.case == "missing_membership" and any(
                    job.get("status") == "completed" for job in value.get("jobs", [])
                ):
                    self.held_snapshot = raw
                    return
            if kind == "reply" and command == "job":
                self.job_replies += 1
            should_hold = kind == "reply" and (
                command == "submit"
                and self.case in ("ack_race", "lost_ack")
                or command == "job"
                and self.case == "key_switch"
                and connection == 1
            )
            if should_hold and self.held is None:
                self.held = (client, raw)
                self.maybe_drop(client, server)
                return
            client.send(raw)
            self.maybe_drop(client, server)

        client.on_message(from_client)
        server.on_message(from_server)

    def maybe_drop(self, client, server):
        if self.case != "lost_ack" or self.dropped or self.held is None:
            return
        if not self.last_snapshot or not any(
            job.get("submissionId") == self.submits[0]["submissionId"]
            for job in self.last_snapshot.get("jobs", [])
        ):
            return
        self.dropped = True
        client.close(code=1012, reason="Synthetic reconnect")
        server.close(code=1012, reason="Synthetic reconnect")

    def release_reply(self):
        client, raw = self.held
        self.held = None
        client.send(raw)


def pump(page, predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Expected proxy boundary was not reached")
        page.wait_for_timeout(20)


def connect(page, base, key=TOKEN):
    page.goto(base + "/ui/")
    # Record the application's socket, not Playwright's native forwarding socket.
    page.evaluate("""async () => {
        const {GuiSession} = await import(new URL('./session.mjs', location.href).href);
        const open = GuiSession.prototype.open;
        GuiSession.prototype.open = function (...args) {
            const result = open.apply(this, args);
            window.reviewSockets.push(this.socket);
            return result;
        };
    }""")
    page.locator("#api-key").fill(key)
    page.locator("#connect").click()
    expect(page.locator("#connection-state")).to_have_text("Key accepted")
    expect(page.locator("#live-state")).to_have_text("Live updates")
    expect(page.locator('#models option[value="outside/curated-list:floor"]')).to_have_count(1)


def row(page, name):
    return page.locator("#files > li").filter(has=page.locator("h3", has_text=name))


def run_case(base, artifacts, case):
    expect.set_options(timeout=12000)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=os.environ.get("CHROMIUM_PATH")
            or shutil.which("chromium-browser")
            or shutil.which("chromium"),
            args=["--no-sandbox"],
        )
        page = browser.new_page(viewport={"width": 1280, "height": 960})
        errors, rest_calls = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "request",
            lambda request: rest_calls.append(request.url) if "/ui/api/" in request.url else None,
        )
        proxy = SessionProxy(case)
        proxy.install(page)
        try:
            connect(page, base)
            if case == "ack_race":
                upload(page, "Ack-race.srt", "Original before acknowledgement")
                page.locator("#translate").click()
                pump(page, lambda: proxy.held is not None)
                expect(row(page, "Ack-race.srt").locator(".state")).to_have_text("Completed")
                expect(page.locator("#preview-translated")).to_contain_text("HU:")
                proxy.release_reply()
                page.wait_for_timeout(80)
                expect(page.locator("#files > li")).to_have_count(1)
                expect(row(page, "Ack-race.srt").locator(".state")).to_have_text("Completed")
                expect(page.locator("#preview-source")).to_have_text(
                    "Original before acknowledgement"
                )
                assert len(proxy.submits) == 1

            elif case == "lost_ack":
                upload(page, "Accepted.srt", "Accepted source")
                upload(page, "Not-sent.srt", "Unsent source")
                page.locator("#translate").click()
                pump(page, lambda: proxy.dropped and len(proxy.connections) >= 2)
                expect(page.locator("#live-state")).to_have_text("Live updates")
                expect(row(page, "Accepted.srt").locator(".state")).to_have_text("Completed")
                expect(row(page, "Not-sent.srt").locator(".state")).to_have_text(
                    "Awaiting submission"
                )
                expect(page.locator("#files > li")).to_have_count(2)
                assert len(proxy.submits) == 1, proxy.submits

            elif case == "key_switch":
                upload(page, "Owner-A.srt", "Owner A original")
                page.locator("#translate").click()
                pump(page, lambda: proxy.held is not None)
                old_reply = json.loads(proxy.held[1])
                old_snapshot = proxy.last_snapshot
                page.evaluate("""() => {
                    window.reviewLateMessage = window.reviewSockets[0].onmessage;
                    window.reviewLateClose = window.reviewSockets[0].onclose;
                }""")
                page.locator("#disconnect").click()
                page.locator("#api-key").fill(SECOND_KEY)
                page.locator("#connect").click()
                expect(page.locator("#live-state")).to_have_text("Live updates")
                expect(page.locator("#files > li")).to_have_count(0)
                # Force callbacks queued by the old connection after the key switch.
                page.evaluate(
                    """frames => {
                    for (const frame of frames) window.reviewLateMessage({data: JSON.stringify(frame)});
                    window.reviewLateClose({code: 1006});
                }""",
                    [old_snapshot, old_reply],
                )
                expect(page.locator("#files > li")).to_have_count(0)
                upload(page, "Owner-B.srt", "Owner B original")
                page.locator("#translate").click()
                expect(row(page, "Owner-B.srt").locator(".state")).to_have_text("Completed")
                expect(page.locator("#preview-translated")).to_have_text("HU: Owner B original")
                expect(page.locator("body")).not_to_contain_text("Owner-A.srt")
                assert len(proxy.connections) == 2
                assert len(proxy.submits) == 2

            elif case == "hydration_forget":
                upload(page, "Preserved.srt", "Preserved original")
                page.locator("#translate").click()
                expect(page.locator("#preview-translated")).to_have_text("HU: Preserved original")
                before = proxy.job_replies
                for _ in range(3):
                    proxy.connections[-1].send(json.dumps(proxy.last_snapshot))
                    page.wait_for_timeout(30)
                expect(page.locator("#preview-translated")).to_have_text("HU: Preserved original")
                assert proxy.job_replies == before
                row(page, "Preserved.srt").get_by_role("button", name="Forget", exact=True).click()
                expect(page.locator("#files > li")).to_have_count(0)
                proxy.connections[-1].send(json.dumps(proxy.last_snapshot))
                page.wait_for_timeout(50)
                expect(page.locator("#files > li")).to_have_count(0)
                page.locator("#restore-jobs").click()
                expect(page.locator("#files > li")).to_have_count(1)
                expect(page.locator("#preview-source")).to_have_text("Preserved original")
                expect(page.locator("#preview-translated")).to_have_text("HU: Preserved original")
                with page.expect_download() as download:
                    row(page, "Preserved.srt").get_by_role(
                        "button", name="Download SRT", exact=True
                    ).click()
                assert "HU: Preserved original" in Path(download.value.path()).read_text()
                assert len(proxy.submits) == 1

            elif case == "cancel_race":
                upload(page, "Queue-holder.srt", "SLOW queue holder")
                upload(page, "Finished-before-cancel.srt", "Finished source")
                page.locator("#translate").click()
                pending = row(page, "Finished-before-cancel.srt")
                expect(pending.locator(".state")).to_have_text("Queued")
                pending.get_by_role("button", name="Cancel queued job").click()
                pump(page, lambda: proxy.held_cancel is not None)
                expect(pending.locator(".state")).to_have_text("Completed")
                server, message = proxy.held_cancel
                server.send(message)
                expect(page.locator("#notice")).to_contain_text("server record was preserved")
                expect(pending.locator(".state")).to_have_text("Completed")
                with page.expect_download() as download:
                    pending.get_by_role("button", name="Download SRT", exact=True).click()
                assert "HU: Finished source" in Path(download.value.path()).read_text()
                assert len(proxy.submits) == 2

            elif case == "missing_membership":
                upload(page, "Missing-window.srt", "SLOW completed outside loaded window")
                page.locator("#translate").click()
                pending = row(page, "Missing-window.srt")
                expect(pending.locator(".state")).to_have_text("Processing")
                pump(page, lambda: proxy.held_snapshot is not None, timeout=10)
                # Omit the completed job from the metadata window. The owned
                # status command still reads its current real fixture record.
                proxy.connections[-1].send(json.dumps({"type": "snapshot", "jobs": []}))
                expect(pending.locator(".state")).to_have_text("Completed")
                expect(
                    pending.get_by_role("button", name="Download SRT", exact=True)
                ).to_be_visible()
                assert len(proxy.submits) == 1

            assert not errors, errors
            assert not rest_calls, rest_calls
            page.screenshot(path=str(artifacts / "passed.png"), full_page=True)
            (artifacts / "receipt.json").write_text(
                json.dumps(
                    {
                        "case": case,
                        "result": "PASS",
                        "connections": len(proxy.connections),
                        "submissions": proxy.submits,
                        "browserRestCalls": rest_calls,
                        "pageErrors": errors,
                    },
                    indent=2,
                )
            )
        except Exception:
            page.screenshot(path=str(artifacts / "failed.png"), full_page=True)
            raise
        finally:
            browser.close()
    print(f"PASS: {case}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=[
            "ack_race",
            "lost_ack",
            "key_switch",
            "hydration_forget",
            "cancel_race",
            "missing_membership",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    cases = (
        [args.case]
        if args.case != "all"
        else [
            "ack_race",
            "lost_ack",
            "key_switch",
            "hydration_forget",
            "cancel_race",
            "missing_membership",
        ]
    )
    for case in cases:
        artifacts = args.artifacts / case
        artifacts.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="gui-session-review-") as data:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            base = f"http://127.0.0.1:{port}"
            with (artifacts / "fixture.log").open("w") as log:
                server = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("ui_fixture_server.py")),
                        "--port",
                        str(port),
                        "--data-dir",
                        data,
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                try:
                    deadline = time.monotonic() + 15
                    while time.monotonic() < deadline:
                        try:
                            with urllib.request.urlopen(base + "/health", timeout=1):
                                break
                        except OSError:
                            time.sleep(0.1)
                    else:
                        raise RuntimeError("Synthetic fixture did not start; see fixture.log")
                    run_case(base, artifacts, case)
                finally:
                    server.terminate()
                    server.wait(timeout=10)


if __name__ == "__main__":
    main()
