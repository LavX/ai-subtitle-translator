"""Browser acceptance with the real API, queue, worker and provider HTTP path."""

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
from ui_browser import HEADERS, connect, upload


def run(base, artifacts, case):
    expect.set_options(timeout=18000)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=os.environ.get("CHROMIUM_PATH") or shutil.which("chromium"),
            args=["--no-sandbox"],
        )
        page = browser.new_page(viewport={"width": 1280, "height": 960})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        submissions = []

        def track(ws):
            def sent(frame):
                value = json.loads(frame)
                if value.get("type") == "submit":
                    submissions.append(value["payload"]["request"])

            ws.on("framesent", sent)

        page.on("websocket", track)
        connect(page, base)
        expect(page.locator('#models option[value="fixture/mandatory:floor"]')).to_have_count(1)

        if case == "rate_limit":
            upload(page, "Rate-limit.srt", *(f"RATE_LIMIT cue {i}" for i in range(10)))
            page.locator("#translate").click()
            expect(page.locator("#files .job-message")).to_contain_text(
                "rate limited; retry 1 after 2s backoff"
            )
            expect(page.locator("#files")).to_contain_text("0/1 batches")
            page.screenshot(path=str(artifacts / "rate-limited.png"), full_page=True)
            expect(page.get_by_role("button", name="Download SRT", exact=True)).to_have_count(1)
            trace = [
                json.loads(line)
                for line in (artifacts / "http-events.jsonl").read_text().splitlines()
            ]
            assert [entry["lineCount"] for entry in trace] == [10, 10]
            assert [entry["outcome"] for entry in trace] == ["rate-limited", "success"]
            expect(page.locator("#total-cost")).to_contain_text("$0.0011")
            assert len(submissions) == 1

        elif case == "tier":
            expect(page.locator("#service-tier")).to_have_value("default")
            expect(page.locator("#service-tier-hint")).to_contain_text("normal processing queue")
            page.locator(".request-options summary").click()
            expect(page.locator("#provider-only")).to_have_value("")
            page.locator("#provider-only").fill(" Azure ")
            upload(page, "Standard.srt", "Standard capacity cue")
            page.locator("#translate").click()
            expect(page.get_by_role("button", name="Download SRT", exact=True)).to_have_count(1)
            assert submissions[-1]["config"]["serviceTier"] == "default"
            assert submissions[-1]["config"]["provider"]["only"] == ["azure"]
            assert submissions[-1]["config"]["provider"]["allowFallbacks"] is False
            page.locator("#provider-only").fill("")
            page.locator("#service-tier").select_option("auto")
            expect(page.locator("#service-tier-hint")).to_contain_text("Flex")
            upload(page, "Routing.srt", "Routing capacity cue")
            page.locator("#translate").click()
            expect(page.get_by_role("button", name="Download SRT", exact=True)).to_have_count(2)
            assert "serviceTier" not in submissions[-1]["config"]
            assert "only" not in submissions[-1]["config"]["provider"]
            page.locator("#routing").select_option("nitro")
            expect(page.locator("#service-tier-hint")).to_contain_text("priority")
            page.locator("#service-tier").select_option("default")
            upload(page, "Standard-fast.srt", "Fast standard capacity cue")
            page.locator("#translate").click()
            expect(page.get_by_role("button", name="Download SRT", exact=True)).to_have_count(3)
            trace = [
                json.loads(line)
                for line in (artifacts / "http-events.jsonl").read_text().splitlines()
            ]
            assert [entry["serviceTier"] for entry in trace] == ["default", None, "default"]
            assert trace[0]["provider"]["only"] == ["azure"]
            assert trace[0]["provider"]["allow_fallbacks"] is False
            assert all("only" not in entry["provider"] for entry in trace[1:])
            assert [entry["model"] for entry in trace] == [
                "openai/gpt-5.6-luna:floor",
                "openai/gpt-5.6-luna:floor",
                "openai/gpt-5.6-luna:nitro",
            ]
            upload(page, "Recover-standard.srt", *(f"RECOVER tier cue {i}" for i in range(10)))
            page.locator("#provider-only").fill("azure")
            page.locator("#translate").click()
            expect(page.locator("#files .job-message").last).to_contain_text("request in progress")
            page.locator("#service-tier").select_option("auto")
            page.locator("#provider-only").fill("openai")
            page.clock.set_fixed_time(time.time() * 1000 + 130000)
            expect(page.locator("[data-waiting]")).to_contain_text("at least 2 minutes")
            expect(page.get_by_role("button", name="Download SRT", exact=True)).to_have_count(
                4, timeout=25000
            )
            trace = [
                json.loads(line)
                for line in (artifacts / "http-events.jsonl").read_text().splitlines()
            ]
            # Setup calls, then: error, timeout, one same-size retry, and the split.
            assert [entry["serviceTier"] for entry in trace[3:]] == ["default"] * 5
            assert all(entry["provider"]["only"] == ["azure"] for entry in trace[3:])
            assert [entry["outcome"] for entry in trace[3:]] == [
                "retryable-error",
                "timeout",
                "timeout",
                "success",
                "success",
            ]
            assert len(submissions) == 4
            page.set_viewport_size({"width": 375, "height": 812})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")

        elif case == "activity":
            upload(page, "Recovery.srt", *(f"RECOVER line {i}" for i in range(10)))
            page.locator("#translate").click()
            expect(page.locator("#files .job-message")).to_contain_text("request in progress")
            expect(page.locator("#files .job-message")).to_contain_text("retry 1", timeout=12000)
            expect(page.locator("#files")).to_contain_text("0/1 batches")
            expect(page.locator("#total-cost")).to_contain_text("$0.0000")
            snapshots = page.request.get(base + "/ui/api/jobs", headers=HEADERS).json()
            assert snapshots["jobs"][0]["completedLines"] == 0
            assert snapshots["jobs"][0]["tokensUsed"] in (None, 0)
            (artifacts / "retry-api.json").write_text(json.dumps(snapshots, indent=2))
            page.screenshot(path=str(artifacts / "retry.png"), full_page=True)
            expect(page.locator("#files .job-message")).to_contain_text("recovering after timeout")
            expect(page.locator("#files .job-message")).to_contain_text("5 lines")
            page.screenshot(path=str(artifacts / "timeout-recovery.png"), full_page=True)
            connect(page, base)
            page.locator("#restore-jobs").click()
            expect(page.locator("#files .job-message")).to_contain_text("recovering after timeout")
            expect(page.get_by_role("button", name="Download SRT", exact=True)).to_have_count(1)
            assert len(submissions) == 1
            trace = [
                json.loads(line)
                for line in (artifacts / "http-events.jsonl").read_text().splitlines()
            ]
            # One same-size retry follows the timeout before the split.
            assert [entry["outcome"] for entry in trace] == [
                "retryable-error",
                "timeout",
                "timeout",
                "success",
                "success",
            ]
            assert [entry["lineCount"] for entry in trace] == [10, 10, 10, 5, 5]

        elif case == "controls":
            chooser = []
            step = ["start"]
            page.on("filechooser", lambda dialog: chooser.append(step[0]))
            # Finished jobs list newest first.
            step[0] = "upload-older"
            upload(page, "Older.srt", "Older cue")
            page.locator("#translate").click()
            expect(page.get_by_role("button", name="Download SRT", exact=True)).to_have_count(1)
            step[0] = "upload-newer"
            upload(page, "Newer.srt", "Newer cue")
            page.locator("#translate").click()
            expect(page.get_by_role("button", name="Download SRT", exact=True)).to_have_count(2)
            expect(page.locator("#files h3").first).to_have_text("Newer.srt")
            # A running job sits above finished ones and can be cancelled mid-request.
            step[0] = "upload-running"
            upload(page, "Running.srt", *(f"RECOVER slow cue {i}" for i in range(5)))
            page.locator("#translate").click()
            expect(page.locator("#files .job-message").first).to_contain_text("request in progress")
            expect(page.locator("#files h3").first).to_have_text("Running.srt")
            step[0] = "cancel"
            page.get_by_role("button", name="Cancel", exact=True).click()
            expect(page.locator("#files .state").filter(has_text="Cancelled")).to_have_count(
                1, timeout=10000
            )
            jobs = page.request.get(base + "/ui/api/jobs", headers=HEADERS).json()["jobs"]
            running = [
                job
                for job in jobs
                if "Running" in (job.get("fileName") or job.get("jobName") or "")
            ]
            assert [job["status"] for job in running] == ["cancelled"], jobs
            # Forget removes the job from the service, not only from the page.
            page.locator("#files li").filter(has_text="Running.srt").get_by_role(
                "button", name="Forget", exact=True
            ).click()
            expect(page.locator("#files li").filter(has_text="Running.srt")).to_have_count(0)
            jobs = page.request.get(base + "/ui/api/jobs", headers=HEADERS).json()["jobs"]
            assert len(jobs) == 2 and all(
                "Running" not in (job.get("fileName") or job.get("jobName") or "") for job in jobs
            ), jobs
            step[0] = "reconnect-restore"
            connect(page, base)
            page.locator("#restore-jobs").click()
            expect(page.locator("#files li")).to_have_count(2)
            # Choosing a cue in the browser moves the preview and never opens the file picker.
            page.locator("#files li").filter(has_text="Newer.srt").get_by_role(
                "button", name="Preview"
            ).click()
            step[0] = "open-cue-browser"
            page.locator("#cue-browser summary").click()
            step[0] = "click-cue"
            page.locator("#cue-results button").first.click()
            expect(page.locator("#cue-position")).to_contain_text("Cue 1 of 1")
            assert not chooser, f"the file picker opened during: {chooser}"
            with page.expect_file_chooser():
                page.locator("#dropzone").click()

        elif case == "counts":
            upload(page, "Complete.srt", "Complete cue")
            expect(page.locator("#file-count")).to_contain_text("1 awaiting submission")
            page.locator("#translate").click()
            expect(page.get_by_role("button", name="Download SRT", exact=True)).to_have_count(1)
            upload(page, "Partial.srt", *(f"Good cue {i}" for i in range(10)), *["TIMEOUT"] * 5)
            expect(page.locator("#file-count")).to_contain_text("1 awaiting submission")
            expect(page.locator("#file-count")).to_contain_text("1 downloadable")
            page.locator("#translate").click()
            expect(page.get_by_role("button", name="Download partial SRT")).to_have_count(1)
            expect(page.locator("#file-count")).to_contain_text("0 awaiting submission")
            expect(page.locator("#file-count")).to_contain_text("2 downloadable (1 partial)")
            expect(page.locator("#files")).to_contain_text("10/15 cues translated")
            with page.expect_download() as saved:
                page.get_by_role("button", name="Download partial SRT").click()
            content = Path(saved.value.path()).read_text()
            assert "HU: Good cue 0" in content and "TIMEOUT" in content
            connect(page, base)
            page.locator("#restore-jobs").click()
            expect(page.locator("#file-count")).to_contain_text("2 downloadable (1 partial)")
            expect(page.locator("#files .job-error")).to_contain_text("Request timed out")
            expect(page.locator("#files .state").filter(has_text="Partial")).to_have_count(1)

        elif case == "terminal":
            page.locator("#model").fill("fixture/terminal")
            upload(page, "Stopped.srt", *(f"TIMEOUT line {i}" for i in range(25)))
            page.locator("#translate").click()
            expect(page.locator("#files .state")).to_have_text("Failed")
            expect(page.locator("#files .job-error")).to_contain_text("1 of 3 batches attempted")
            expect(page.locator("#files .job-error")).to_contain_text("2 not attempted")
            assert page.locator("#files").inner_text().count("2 not attempted") == 1
            expect(page.locator("#files .job-error")).to_contain_text("timed out")
            expect(page.locator("#file-count")).to_contain_text("0 downloadable")
            page.screenshot(path=str(artifacts / "stopped.png"), full_page=True)
            connect(page, base)
            page.locator("#restore-jobs").click()
            expect(page.locator("#files .job-error")).to_contain_text("2 not attempted")

        elif case == "cues":
            content = (Path(__file__).parent / "ui" / "blank-lines.srt").read_bytes()
            page.evaluate(
                """content => {
                const transfer = new DataTransfer();
                transfer.items.add(new File([content], 'Blank-lines.srt', {type: 'application/x-subrip'}));
                document.querySelector('#preview-source').dispatchEvent(new DragEvent('drop', {
                    bubbles: true, cancelable: true, dataTransfer: transfer,
                }));
            }""",
                content.decode(),
            )
            expect(page.locator("#cue-position")).to_have_text("Cue 1 of 4")
            assert page.locator("#preview-source").text_content() == "First cue"
            page.locator("#next-cue").click()
            expect(page.locator("#preview-source i")).to_have_text("Second cue")
            assert page.locator("#preview-source").text_content() == "Second cue\nSecond line"
            page.locator("#translate").click()
            expect(page.get_by_role("button", name="Download SRT", exact=True)).to_have_count(1)
            expect(page.locator("#preview-translated i")).to_have_text("Second cue")
            expect(page.locator("#preview-translated")).to_contain_text("HU:")
            assert (
                page.locator("#preview-translated").text_content()
                == "HU: \nSecond cue\nSecond line"
            )
            page.locator("#next-cue").click()
            expect(page.locator("#cue-position")).to_have_text("Cue 3 of 4")
            expect(page.locator("#preview-source")).to_have_text(
                "Third cue <script>literal</script>"
            )
            expect(page.locator("#preview-translated")).to_have_text(
                "HU: Third cue <script>literal</script>"
            )
            assert page.locator("#caption-preview script").count() == 0
            page.locator("#caption-preview").focus()
            page.keyboard.press("ArrowLeft")
            expect(page.locator("#cue-position")).to_have_text("Cue 2 of 4")

        elif case == "reasoning":
            page.locator(".request-options summary").click()
            expect(page.locator("#reasoning")).to_have_value("none")
            upload(page, "Off.srt", "Off cue")
            page.locator("#translate").click()
            expect(page.get_by_role("button", name="Download SRT", exact=True)).to_have_count(1)
            assert submissions[-1]["config"]["reasoning"] == {"effort": "none"}
            page.locator(".request-options summary").click()
            page.locator("#model").fill("fixture/mandatory:floor")
            upload(page, "Mandatory.srt", "Mandatory cue")
            expect(page.locator("#reasoning")).to_have_value("none")
            expect(page.locator("#reasoning-hint")).to_contain_text("requires reasoning")
            expect(page.locator("#reasoning-hint")).to_be_visible()
            expect(page.locator("#translate")).to_be_disabled()
            page.locator("#reasoning").select_option("default")
            page.locator("#translate").click()
            expect(page.get_by_role("button", name="Download SRT", exact=True)).to_have_count(2)
            assert "reasoning" not in submissions[-1]["config"]
            page.locator("#model").fill("anthropic/claude-sonnet-4.5:floor")
            page.locator("#reasoning").select_option("high")
            upload(page, "Effort.srt", "Effort cue")
            page.locator("#model").fill("fixture/low-only:floor")
            expect(page.locator("#reasoning")).to_have_value("high")
            expect(page.locator("#translate")).to_be_disabled()
            expect(page.locator("#reasoning-hint")).to_contain_text("not supported")
            page.locator("#model").fill("anthropic/claude-sonnet-4.5:floor")
            page.locator("#translate").click()
            expect(page.get_by_role("button", name="Download SRT", exact=True)).to_have_count(3)
            assert submissions[-1]["config"]["reasoning"] == {"effort": "high"}
            page.locator("#model").fill("future/freeform:floor")
            page.locator("#reasoning").select_option("none")
            upload(page, "Freeform.srt", "Freeform cue")
            page.locator("#translate").click()
            expect(page.get_by_role("button", name="Download SRT", exact=True)).to_have_count(4)
            trace = [
                json.loads(line)
                for line in (artifacts / "http-events.jsonl").read_text().splitlines()
            ]
            assert [entry["reasoning"] for entry in trace] == [
                {"effort": "none"},
                None,
                {"effort": "high"},
                {"effort": "none"},
            ]
            assert trace[0]["temperatureSent"] is False
            assert len(submissions) == 4
            page.set_viewport_size({"width": 375, "height": 812})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")

        assert not errors, errors
        page.screenshot(path=str(artifacts / "final.png"), full_page=True)
        browser.close()
        print(f"PASS: {case}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=[
            "activity",
            "counts",
            "terminal",
            "reasoning",
            "cues",
            "fallback",
            "tier",
            "rate_limit",
            "controls",
        ],
        required=True,
    )
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="translator-reliability-browser-") as data:
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
                    "--reliability-provider",
                    "--evidence",
                    str(args.artifacts / "http-events.jsonl"),
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
                if args.case == "fallback":
                    from ui_review_browser import run as run_review

                    run_review(base, args.artifacts)
                else:
                    run(base, args.artifacts, args.case)
            finally:
                server.terminate()
                server.wait(timeout=10)


if __name__ == "__main__":
    main()
