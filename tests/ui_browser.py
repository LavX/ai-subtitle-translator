"""Optional browser acceptance checks. Requires Playwright and Chromium.

Run with PYTHONPATH=src python tests/ui_browser.py. A local fixture uses the real
API and queue with synthetic translations. No external model calls are made.
"""

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
import zipfile
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

TOKEN = "sk-or-v1-demo-not-a-real-key"
SECOND_KEY = "sk-or-v1-demo-second-key"
STORAGE_KEY = "subtitle-translator.openrouter-key.v1"
HEADERS = {"Authorization": "Bearer " + TOKEN}


def srt(*lines):
    return (
        "\n\n".join(
            f"{index}\n00:00:{index:02d},000 --> 00:00:{index:02d},900\n{line}"
            for index, line in enumerate(lines, 1)
        )
        + "\n"
    )


def upload(page, name, *lines):
    page.locator("#file-input").set_input_files(
        {"name": name, "mimeType": "application/x-subrip", "buffer": srt(*lines).encode()}
    )
    expect(page.locator("#files h3").filter(has_text=name)).to_be_visible()


def connect(page, base):
    page.goto(base + "/ui/")
    page.locator("#api-key").fill(TOKEN)
    page.locator("#connect").click()
    expect(page.locator("#connection-state")).to_have_text("Key accepted")


def run(base, artifacts):
    expect.set_options(timeout=15000)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=os.environ.get("CHROMIUM_PATH")
            or shutil.which("chromium-browser")
            or shutil.which("chromium"),
            args=["--no-sandbox"],
        )
        context = browser.new_context(viewport={"width": 1280, "height": 960})
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "console",
            lambda message: (
                errors.append(message.text) if "Content Security Policy" in message.text else None
            ),
        )
        submissions = []
        credentials = []
        rest = []

        def track(ws):
            assert TOKEN not in ws.url

            def sent(frame):
                value = json.loads(frame)
                if value.get("type") == "submit":
                    submissions.append(value["payload"]["request"])
                if value.get("type") == "auth":
                    credentials.append(value["apiKey"])

            ws.on("framesent", sent)

        page.on("websocket", track)
        page.on(
            "request",
            lambda request: rest.append(request.url) if "/ui/api/" in request.url else None,
        )
        page.goto(base + "/ui/")
        assert "Bazarr" not in page.locator("main").inner_text()
        assert page.locator('footer a[href="https://github.com/LavX/bazarr"]').count() >= 1
        expect(page.locator('input[type="password"]')).to_have_count(1)
        page.locator("#api-key").fill("incorrect-fixture-key")
        page.locator("#connect").click()
        expect(page.locator("#notice")).to_contain_text("rejected")
        expect(page.locator("#translate")).to_be_disabled()
        page.locator("#api-key").fill(TOKEN)
        page.locator("#connect").click()
        expect(page.locator("#connection-state")).to_have_text("Key accepted")
        expect(page.locator("#api-key")).to_have_value("")
        expect(page.locator("#model")).to_have_value("openai/gpt-5.6-luna:floor")
        expect(page.locator('#models option[value="outside/curated-list:floor"]')).to_have_count(1)
        assert page.locator(".workspace").bounding_box()["y"] < 280
        page.locator("#routing").select_option("nitro")
        expect(page.locator("#model")).to_have_value("openai/gpt-5.6-luna:nitro")
        page.locator("#routing").select_option("default")
        expect(page.locator("#model")).to_have_value("openai/gpt-5.6-luna")
        page.locator("#routing").select_option("floor")
        page.locator("#model").fill("fixture/model:nitro")
        upload(page, "Árvíz.srt", "Hello <script>alert(1)</script> & goodbye.")
        upload(page, "Second.srt", "Another line.")
        page.locator("#translate").click()
        expect(page.get_by_role("button", name="Download SRT", exact=True)).to_have_count(
            2, timeout=20000
        )
        assert len(submissions) == 2, submissions
        assert all(item["config"]["model"] == "fixture/model:floor" for item in submissions)
        assert all(item["config"]["provider"]["sort"] == "floor" for item in submissions)
        assert all(item["targetLanguage"] == "hu" for item in submissions)
        assert all("apiKey" not in item["config"] for item in submissions)
        assert all("x-auth-token" not in headers for headers in credentials)
        assert TOKEN in credentials
        assert not rest
        expect(page.locator("#preview-translated")).to_contain_text("<script>alert(1)</script>")
        assert page.locator("#caption-preview script").count() == 0
        with page.expect_download() as saved:
            page.locator("#files > li", has_text="Árvíz.srt").get_by_role(
                "button", name="Download SRT", exact=True
            ).click()
        single = saved.value
        assert single.suggested_filename == "Árvíz.hu.srt"
        assert "HU: Hello <script>" in Path(single.path()).read_text()
        with page.expect_download() as saved:
            page.locator("#download-all").click()
        with zipfile.ZipFile(saved.value.path()) as archive:
            assert archive.testzip() is None
            assert set(archive.namelist()) == {"Árvíz.hu.srt", "Second.hu.srt"}
            assert "HU: Hello" in archive.read("Árvíz.hu.srt").decode()
        assert page.evaluate("[localStorage.length, sessionStorage.length]") == [0, 0]
        page.screenshot(path=str(artifacts / "workspace-desktop.png"), full_page=True)
        page.set_viewport_size({"width": 375, "height": 812})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(path=str(artifacts / "workspace-mobile.png"), full_page=True)
        page.set_viewport_size({"width": 1280, "height": 960})
        print(
            "PASS: authentication, real batch jobs, floor routing, Unicode SRT/ZIP, mobile layout"
        )

        upload(page, "Partial.srt", "Good line", "FAIL")
        upload(page, "Failed.srt", "FAIL")
        page.locator("#translate").click()
        partial = page.locator("#files li").filter(
            has=page.get_by_role("heading", name="Partial.srt", exact=True)
        )
        failed = page.locator("#files li").filter(
            has=page.get_by_role("heading", name="Failed.srt", exact=True)
        )
        expect(partial.locator(".state")).to_contain_text("Partial", timeout=20000)
        expect(failed.locator(".state")).to_have_text("Failed", timeout=20000)
        with page.expect_download() as saved:
            partial.get_by_role("button", name="Download partial SRT").click()
        assert saved.value.suggested_filename == "Partial.partial.hu.srt"
        content = Path(saved.value.path()).read_text()
        assert "HU: Good line" in content and "FAIL" in content
        expect(failed.get_by_role("button", name="Download SRT")).to_have_count(0)
        print("PASS: real partial and failed jobs, preserved partial content and download marking")

        upload(page, "Slow.srt", "SLOW")
        upload(page, "Cancel.srt", "Should not translate")
        page.locator("#translate").click()
        queued = page.locator("#files li").filter(
            has=page.get_by_role("heading", name="Cancel.srt", exact=True)
        )
        queued.get_by_role("button", name="Cancel queued job").click()
        expect(queued.locator(".state")).to_have_text("Cancelled")
        page.locator("#disconnect").click()
        expect(page.locator("#api-key")).to_have_value("")
        page.locator("#api-key").fill(TOKEN)
        page.locator("#connect").click()
        expect(page.locator("#connection-state")).to_have_text("Key accepted")
        slow = page.locator("#files li").filter(
            has=page.get_by_role("heading", name="Slow.srt", exact=True)
        )
        expect(slow.locator(".state")).to_have_text("Completed", timeout=20000)
        print("PASS: queued cancellation and disconnect/reconnect resume")

        for routing, expected in [("nitro", "fixture/model:nitro"), ("default", "fixture/model")]:
            page.locator("#model").fill("fixture/model:floor")
            page.locator("#routing").select_option(routing)
            upload(page, routing + ".srt", "Routing check")
            page.locator("#translate").click()
            row = page.locator("#files li").filter(
                has=page.get_by_role("heading", name=routing + ".srt", exact=True)
            )
            expect(row.locator(".state")).to_have_text("Completed")
            assert submissions[-1]["config"]["model"] == expected
            assert submissions[-1]["config"]["provider"]["sort"] == routing
        assert not errors, errors
        assert not rest, rest
        print("PASS: routing commands, no job REST requests, no JavaScript or CSP errors")
        check_saved_keys(browser, base)
        browser.close()


def check_saved_keys(browser, base):
    context = browser.new_context()
    page = context.new_page()
    page.goto(base + "/ui/")
    page.locator("#api-key").fill(TOKEN)
    page.locator("#remember-key").check()
    page.locator("#connect").click()
    expect(page.locator("#connection-state")).to_have_text("Key accepted")
    assert page.evaluate("key => localStorage.getItem(key)", STORAGE_KEY) == TOKEN
    page.reload()
    expect(page.locator("#connection-state")).to_have_text("Key accepted")
    expect(page.locator("#api-key")).to_have_value("")
    page.locator("#restore-jobs").click()
    expect(page.get_by_role("button", name="Download SRT", exact=True).first).to_be_visible()
    page.locator("#remember-key").uncheck()
    assert page.evaluate("key => localStorage.getItem(key)", STORAGE_KEY) is None
    page.locator("#disconnect").click()
    page.reload()
    expect(page.locator("#connection-state")).to_have_text("Not connected")
    page.locator("#api-key").fill(TOKEN)
    page.locator("#remember-key").check()
    page.locator("#connect").click()
    expect(page.locator("#connection-state")).to_have_text("Key accepted")
    page.locator("#restore-jobs").click()
    expect(page.get_by_role("button", name="Download SRT", exact=True).first).to_be_visible()
    page.locator("#disconnect").click()
    page.locator("#api-key").fill(SECOND_KEY)
    assert page.evaluate("key => localStorage.getItem(key)", STORAGE_KEY) is None
    page.locator("#connect").click()
    expect(page.locator("#connection-state")).to_have_text("Key accepted")
    expect(page.locator("#files li")).to_have_count(0)
    page.evaluate("localStorage.setItem('unrelated-preference', 'kept')")
    page.locator("#forget-saved-key").click()
    expect(page.locator("#connection-state")).to_have_text("Not connected")
    assert page.evaluate("key => localStorage.getItem(key)", STORAGE_KEY) is None
    assert page.evaluate("localStorage.getItem('unrelated-preference')") == "kept"
    expect(page.locator("#api-key")).to_have_value("")

    page.locator("#api-key").fill("invalid-fixture-key")
    page.locator("#remember-key").check()
    page.locator("#connect").click()
    expect(page.locator("#notice")).to_contain_text("rejected")
    assert page.evaluate("key => localStorage.getItem(key)", STORAGE_KEY) is None

    page.locator("#api-key").fill(TOKEN)
    page.locator("#connect").click()
    expect(page.locator("#connection-state")).to_have_text("Key accepted")

    def unavailable(ws):
        def receive(_message):
            ws.send(
                json.dumps({"type": "error", "status": 503, "message": "Key validation failed"})
            )

        ws.on_message(receive)

    page.route_web_socket("**/ui/session", unavailable)
    page.reload()
    expect(page.locator("#notice")).to_contain_text("temporarily unavailable")
    assert page.evaluate("key => localStorage.getItem(key)", STORAGE_KEY) == TOKEN
    context.close()

    blocked = browser.new_context()
    blocked.add_init_script(
        """
      for (const method of ['getItem','setItem','removeItem']) {
        Storage.prototype[method] = () => { throw new DOMException('Blocked', 'SecurityError'); };
      }
    """
    )
    page = blocked.new_page()
    page.goto(base + "/ui/")
    page.locator("#api-key").fill(TOKEN)
    page.locator("#remember-key").check()
    page.locator("#connect").click()
    expect(page.locator("#connection-state")).to_have_text("Key accepted")
    expect(page.locator("#notice")).to_contain_text("could not save")
    blocked.close()
    print(
        "PASS: key-only login, opt-in saving, reload reuse, forget/uncheck, key switch isolation, invalid/outage/blocked storage"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url", help="Existing synthetic UI fixture, never a production service"
    )
    parser.add_argument("--artifacts", type=Path, default=Path("/tmp/translator-ui-browser"))
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    if args.base_url:
        run(args.base_url.rstrip("/"), args.artifacts)
        return
    with tempfile.TemporaryDirectory(prefix="translator-ui-fixture-") as data:
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
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    try:
                        with urllib.request.urlopen(base + "/health", timeout=1):
                            break
                    except OSError:
                        time.sleep(0.1)
                else:
                    raise RuntimeError("Fixture server did not start; see fixture.log")
                run(base, args.artifacts)
            finally:
                server.terminate()
                server.wait(timeout=10)


if __name__ == "__main__":
    main()
