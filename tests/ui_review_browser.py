"""Browser regressions for subtitle review and independent live job updates."""

import json
import os
import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def run(base, artifacts):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=os.environ.get("CHROMIUM_PATH")
            or shutil.which("chromium-browser")
            or shutil.which("chromium"),
            args=["--no-sandbox"],
        )
        page = browser.new_page(viewport={"width": 1440, "height": 1050})
        errors = []

        page.on("pageerror", lambda error: errors.append(str(error)))
        content = "1\n00:00:01,000 --> 00:00:02,000\n<i>First line</i>\n\n2\n00:00:03,000 --> 00:00:04,000\nSecond line\n\n3\n00:00:05,000 --> 00:00:06,000\nLast line\n"
        result = {"content": content.replace("First", "Translated first")}
        jobs = [
            {
                "jobId": "slow",
                "fileName": "Slow.srt",
                "status": "processing",
                "progress": 0,
                "totalBatches": 6,
                "completedBatches": 0,
                "message": "Translated 0/10 lines (0/6 batches)",
                "startedAt": (datetime.now(UTC) - timedelta(minutes=5)).isoformat(),
            },
            {"jobId": "fast", "fileName": "Fast.srt", "status": "processing", "progress": 0},
            {
                "jobId": "partial",
                "fileName": "Partial.srt",
                "status": "partial",
                "progress": 100,
                "totalLines": 534,
                "completedLines": 100,
                "error": "134/534 lines translated. Request timed out after 120.0s",
                "result": result,
            },
        ]

        def respond(ws):
            def send(value):
                ws.send(json.dumps(value))

            def receive(raw):
                command = json.loads(raw)
                kind = command["type"]
                if kind == "auth":
                    send({"type": "ready", "ownerScope": "review-fixture"})
                    send({"type": "snapshot", "jobs": []})
                    return
                if kind == "pong":
                    return
                if kind == "models":
                    value = {"models": []}
                elif kind == "restore":
                    jobs[1].update(status="completed", progress=100, hasResult=True)
                    jobs[2]["hasResult"] = True
                    send(
                        {
                            "type": "snapshot",
                            "jobs": [
                                {k: v for k, v in job.items() if k != "result"} for job in jobs
                            ],
                        }
                    )
                    value = {"restored": True}
                elif kind == "source":
                    value = {"content": content}
                elif kind == "job":
                    value = {"result": result}
                else:
                    raise AssertionError(kind)
                send({"type": "reply", "id": command["id"], "value": value})

            ws.on_message(receive)

        page.route_web_socket("**/ui/session", respond)
        page.goto(base + "/ui/")
        page.locator("#api-key").fill("dummy-review-fixture")
        page.locator("#connect").click()
        expect(page.locator("#connection-state")).to_have_text("Key accepted")
        page.locator("#file-input").set_input_files(
            {"name": "Original.srt", "mimeType": "application/x-subrip", "buffer": content.encode()}
        )
        expect(page.locator("#preview-source i")).to_have_text("First line")
        page.locator("#cue-browser summary").click()
        page.locator("#cue-search").fill("Second")
        expect(page.locator("#cue-results li")).to_have_count(1)
        page.locator("#cue-results button").click()
        expect(page.locator("#cue-position")).to_have_text("Cue 2 of 3")
        page.locator("#cue-search").fill("no matches")
        expect(page.locator("#cue-search-status")).to_have_text("No matching cues.")
        page.locator("#cue-jump").fill("3")
        page.locator("#cue-jump").press("Enter")
        expect(page.locator("#preview-source")).to_have_text("Last line")
        page.locator("#caption-preview").focus()
        page.keyboard.press("ArrowLeft")
        expect(page.locator("#cue-position")).to_have_text("Cue 2 of 3")
        page.locator("#swap-languages").click()
        expect(page.locator("#source-language")).to_have_value("hu")
        expect(page.locator("#target-language")).to_have_value("en")
        page.locator("#restore-jobs").click()
        slow = page.locator("#files li").filter(
            has=page.get_by_role("heading", name="Slow.srt", exact=True)
        )
        expect(slow).to_contain_text("No batch has completed for at least 2 minutes")
        expect(slow).to_contain_text("The server has not reported request activity")
        expect(slow).not_to_contain_text("first batch is still running")
        partial = page.locator("#files li").filter(
            has=page.get_by_role("heading", name="Partial.srt", exact=True)
        )
        expect(partial).to_contain_text("134/534 cues translated (25%)")
        expect(partial).to_contain_text("Provider requests timed out")
        assert "100%" not in partial.inner_text()
        fast = page.locator("#files li").filter(
            has=page.get_by_role("heading", name="Fast.srt", exact=True)
        )
        expect(fast.get_by_role("button", name="Download SRT", exact=True)).to_be_visible(
            timeout=6000
        )
        page.locator("#cue-search").fill("")
        page.screenshot(path=str(artifacts / "review-desktop.png"), full_page=True)
        page.set_viewport_size({"width": 375, "height": 812})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(path=str(artifacts / "review-mobile.png"), full_page=True)
        assert not errors, errors
        page.unroute_all(behavior="ignoreErrors")
        browser.close()
        print(
            "PASS: safe formatting, cue search/jump/keyboard, language swap, accurate partial coverage and timeout reason, stalled job does not delay completed job"
        )


if __name__ == "__main__":
    output = Path(os.environ.get("REVIEW_ARTIFACTS", "/tmp/translator-ui-review-regressions"))
    output.mkdir(parents=True, exist_ok=True)
    run(sys.argv[1], output)
