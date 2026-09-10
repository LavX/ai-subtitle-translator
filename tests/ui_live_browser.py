"""Browser checks for server-owned queue restoration and live status delivery."""

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
from ui_browser import HEADERS, TOKEN, upload


def run(base, artifacts, case):
    expect.set_options(timeout=12000)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=os.environ.get("CHROMIUM_PATH")
            or shutil.which("chromium")
            or shutil.which("chromium-browser"),
            args=["--no-sandbox"],
        )
        context = browser.new_context()
        page = context.new_page()
        errors, submissions, sockets = [], [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        rest_requests = []

        def track_socket(ws):
            sockets.append(ws.url)

            def sent(frame):
                value = json.loads(frame)
                if value.get("type") == "submit":
                    submissions.append(value)

            ws.on("framesent", sent)

        page.on("websocket", track_socket)
        page.on(
            "request",
            lambda request: (
                rest_requests.append(request.url) if "/ui/api/" in request.url else None
            ),
        )
        page.goto(base + "/ui/")
        page.locator("#remember-key").check()
        page.locator("#api-key").fill(TOKEN)
        page.locator("#connect").click()
        expect(page.locator("#connection-state")).to_have_text("Key accepted")
        upload(page, "Live queue.srt", "SLOW first", "SLOW second")
        page.locator("#translate").click()
        expect(page.locator("#files .state")).to_have_text("Processing")
        if case == "reload":
            before = page.request.get(base + "/ui/api/jobs", headers=HEADERS).json()["jobs"]
            page.reload()
            expect(page.locator("#connection-state")).to_have_text("Key accepted")
            expect(page.locator("#files h3")).to_have_text("Live queue.srt", timeout=2500)
            after = page.request.get(base + "/ui/api/jobs", headers=HEADERS).json()["jobs"]
            assert [job["jobId"] for job in before] == [job["jobId"] for job in after]
            assert len(submissions) == 1
        elif case == "status_failure":
            page.route(
                "**/ui/api/jobs/*",
                lambda route: (
                    route.fulfill(status=503, json={"detail": "fixture unavailable"})
                    if route.request.method == "GET"
                    else route.continue_()
                ),
            )
            expect(page.locator("#files .state")).to_have_text("Completed", timeout=18000)
            assert len(submissions) == 1
            assert sockets and all(TOKEN not in url for url in sockets)
        assert not rest_requests, rest_requests
        assert not errors, errors
        page.screenshot(path=str(artifacts / (case + ".png")), full_page=True)
        browser.close()
        print("PASS:", case)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["reload", "status_failure"], required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="translator-live-") as data:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        with (args.artifacts / "fixture.log").open("w") as log:
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
                for _ in range(150):
                    try:
                        with urllib.request.urlopen(base + "/health", timeout=1):
                            break
                    except OSError:
                        time.sleep(0.1)
                else:
                    raise RuntimeError("Fixture not ready")
                run(base, args.artifacts, args.case)
            finally:
                server.terminate()
                server.wait(timeout=10)


if __name__ == "__main__":
    main()
