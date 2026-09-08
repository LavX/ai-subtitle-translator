"""App-owned browser sessions with direct access to the translation queue."""

import asyncio
import json
import re
from contextlib import suppress

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from subtitle_translator import ui_api
from subtitle_translator.api.models import TranslateFileRequest
from subtitle_translator.queue.job_manager import Job, JobStatus

AUTH_TIMEOUT = 15
HEARTBEAT_SECONDS = 20
RECEIVE_TIMEOUT = 60
SEND_TIMEOUT = 10
MAX_FRAME_BYTES = 12_000_000
MAX_REPLY_BYTES = 16_000_000
MAX_COMMANDS = 4
_ID = r"^[a-zA-Z0-9_-]{1,64}$"
_ACTIVITY = re.compile(
    r"Batch \d+: (?:request in progress for \d+ lines \(attempt \d+\)|"
    r"using learned limit of \d+ lines per request|recovering with smaller \d+-line requests|"
    r"retry \d+ after (?:incomplete or invalid response|invalid response)|"
    r"(?:rate limited|provider error); retry \d+ after [\d.]+s backoff|"
    r"(?:rate-limit retry budget|timeout budget) exhausted(?:; stopping this batch)?|"
    r"rate-limit retries exhausted; stopping this batch|"
    r"request timed out(?:; retrying at the same size)?)(?: \(recovering after timeout\))?"
)


def metadata(job: Job) -> dict:
    """Project bounded status fields without provider bodies or subtitle content."""
    value = {
        "jobId": job.id,
        "status": job.status.value,
        "progress": job.progress,
        "submissionId": job.request_data.get("_ui_submission"),
        "hasResult": job.status in (JobStatus.COMPLETED, JobStatus.PARTIAL) and bool(job.result),
        "totalLines": job.total_lines,
        "completedLines": job.completed_lines,
        "totalBatches": job.total_batches,
        "completedBatches": job.completed_batches,
        "tokensUsed": job.tokens_used,
        "totalCost": job.total_cost or None,
        "startedAt": job.started_at.isoformat() if job.started_at else None,
        "createdAt": job.created_at.isoformat() if job.created_at else None,
        "completedAt": job.completed_at.isoformat() if job.completed_at else None,
    }
    for public, private in (
        ("fileName", "file_name"),
        ("jobName", "job_name"),
        ("targetLanguage", "target_language"),
        ("model", "model"),
    ):
        value[public] = str(getattr(job, private) or "")[:512]
    message = job.message or ""
    progress_message = re.fullmatch(
        r"Translated \d+/\d+ lines \(\d+/\d+ batches(?:, \d+ failed)?\)"
        r"|Job cancelled by user(?: after \d+/\d+ batches)?",
        message,
    )
    value["message"] = (
        message
        if len(message) <= 200 and (_ACTIVITY.fullmatch(message) or progress_message)
        else ""
    )
    value["error"] = None
    if job.status in (JobStatus.FAILED, JobStatus.PARTIAL):
        # Only extract service-generated counters. Never forward provider error text.
        coverage = re.match(r"\d+/\d+ lines translated\. ", job.error or "")
        error = (job.error or "")[len(coverage[0]) if coverage else 0 :]
        counters = re.match(
            r"(?:\d+ of \d+ batches attempted; \d+ failed; \d+ not attempted\.|\d+ of \d+ batches failed:|All \d+ batches failed:)",
            error,
        )
        reason = "Some provider requests did not finish successfully."
        if re.search(r"timed? out|timeout", error, re.IGNORECASE):
            reason = "Request timed out."
        elif re.search(r"429|rate.limit", error, re.IGNORECASE):
            reason = "Provider rate limit reached."
        elif re.search(r"402|insufficient credit", error, re.IGNORECASE):
            reason = "OpenRouter reported insufficient credit."
        value["error"] = (
            (coverage[0] if coverage else "") + (counters[0] + " " if counters else "") + reason
        )
    return value


class Command(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=_ID)
    type: str = Field(max_length=20)
    payload: dict = Field(default_factory=dict)


class Submission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    submissionId: str = Field(pattern=_ID)
    request: dict


class OwnedJob(BaseModel):
    model_config = ConfigDict(extra="forbid")
    jobId: str = Field(pattern=_ID)


class SessionEndedError(Exception):
    pass


gui_router = APIRouter(include_in_schema=False)


class GuiSession:
    def __init__(self, socket: WebSocket, identity: ui_api.UiIdentity):
        self.socket = socket
        self.identity = identity
        self.changed = asyncio.Event()
        self.stopped = asyncio.Event()
        self.commands: set[asyncio.Task] = set()
        self.send_lock = asyncio.Lock()
        self.auth_lock = asyncio.Lock()
        self.seen: set[str] = set()

    async def raw_send(self, value: dict) -> None:
        data = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if len(data.encode()) > MAX_REPLY_BYTES:
            raise SessionEndedError
        async with asyncio.timeout(SEND_TIMEOUT):
            async with self.send_lock:
                await self.socket.send_text(data)

    async def authorize(self) -> None:
        async with self.auth_lock:
            try:
                await ui_api._authenticate(f"Bearer {self.identity.api_key}")
            except HTTPException as exc:
                await self.raw_send(
                    {"type": "error", "status": exc.status_code, "message": "Key validation failed"}
                )
                self.stopped.set()
                raise SessionEndedError from None

    async def send(self, value: dict) -> None:
        await self.authorize()
        await self.raw_send(value)

    def snapshot(self) -> dict:
        jobs = [
            job
            for job in ui_api.job_manager.list_jobs(limit=len(ui_api.job_manager.jobs))
            if ui_api._belongs_to(job, self.identity)
        ]
        # Loaded history may be a restart-limited window. Absence is not deletion.
        active = [job for job in jobs if job.status in (JobStatus.QUEUED, JobStatus.PROCESSING)]
        history = [
            job for job in jobs if job.status not in (JobStatus.QUEUED, JobStatus.PROCESSING)
        ]
        snapshot = {"type": "snapshot", "jobs": [metadata(job) for job in active + history[:100]]}
        if len(history) > 100:
            snapshot["historyLimited"] = True
        return snapshot

    async def publish(self) -> None:
        while True:
            await self.changed.wait()
            self.changed.clear()
            await self.send(self.snapshot())

    async def heartbeat(self) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_SECONDS)
            await self.send({"type": "heartbeat"})

    async def execute(self, command: Command) -> None:
        try:
            await self.authorize()
            payload = command.payload
            if command.type == "submit":
                submission = Submission.model_validate(payload)
                allowed = {
                    field.alias or name for name, field in TranslateFileRequest.model_fields.items()
                }
                if set(submission.request) - allowed:
                    raise ValueError
                request = TranslateFileRequest.model_validate(submission.request)
                value = await ui_api.submit_gui_file(
                    request, self.identity, submission.submissionId
                )
                value = value.model_dump(mode="json", by_alias=True)
            elif command.type in ("job", "status", "source", "cancel", "forget"):
                job_id = OwnedJob.model_validate(payload).jobId
                job = ui_api._owned_job(job_id, self.identity)
                if command.type == "job":
                    value = metadata(job)
                    value["result"] = (
                        {"content": job.result.get("content", "")} if value["hasResult"] else None
                    )
                elif command.type == "status":
                    value = metadata(job)
                elif command.type == "source":
                    value = {"content": job.request_data.get("content", "")}
                elif command.type == "forget":
                    value = await ui_api.forget_job(job_id, self.identity)
                else:
                    value = (await ui_api.cancel_job(job_id, self.identity)).model_dump(
                        mode="json", by_alias=True
                    )
            elif command.type == "models" and not payload:
                value = (await ui_api.models(self.identity)).model_dump(mode="json", by_alias=True)
            elif command.type == "restore" and not payload:
                self.changed.set()
                value = {"restored": True}
            else:
                raise ValueError
            await self.send({"type": "reply", "id": command.id, "value": value})
        except (ValidationError, ValueError):
            await self.send(
                {
                    "type": "reply",
                    "id": command.id,
                    "error": {"status": 422, "message": "Invalid command"},
                }
            )
        except HTTPException as exc:
            await self.send(
                {
                    "type": "reply",
                    "id": command.id,
                    "error": {
                        "status": exc.status_code,
                        "message": "Command could not be completed",
                    },
                }
            )
        except (SessionEndedError, WebSocketDisconnect, TimeoutError, OSError, RuntimeError):
            self.stopped.set()
        except Exception:
            # Do not expose submitted data or raw provider exceptions in error frames.
            await self.send(
                {
                    "type": "reply",
                    "id": command.id,
                    "error": {"status": 500, "message": "Command could not be completed"},
                }
            )

    def command_done(self, task: asyncio.Task) -> None:
        self.commands.discard(task)
        if not task.cancelled() and task.exception() is not None:
            self.stopped.set()

    async def receive(self) -> None:
        window, count = asyncio.get_running_loop().time(), 0
        while True:
            async with asyncio.timeout(RECEIVE_TIMEOUT):
                value = await receive_frame(self.socket, MAX_FRAME_BYTES)
            now = asyncio.get_running_loop().time()
            if now - window >= 1:
                window, count = now, 0
            count += 1
            if count > 40:
                raise SessionEndedError
            if value == {"type": "pong"}:
                continue
            try:
                command = Command.model_validate(value)
            except ValidationError:
                await self.send({"type": "error", "status": 422, "message": "Invalid command"})
                continue
            if command.id in self.seen or len(self.seen) >= 4096:
                raise SessionEndedError
            self.seen.add(command.id)
            if len(self.commands) >= MAX_COMMANDS:
                await self.send(
                    {
                        "type": "reply",
                        "id": command.id,
                        "error": {"status": 429, "message": "Too many pending commands"},
                    }
                )
                continue
            task = asyncio.create_task(self.execute(command))
            self.commands.add(task)
            task.add_done_callback(self.command_done)

    async def run(self) -> None:
        with ui_api.job_manager.events.subscribe(self.identity.owner) as self.changed:
            await self.send({"type": "ready", "ownerScope": self.identity.owner})
            self.changed.set()
            tasks = {
                asyncio.create_task(self.receive()),
                asyncio.create_task(self.publish()),
                asyncio.create_task(self.heartbeat()),
                asyncio.create_task(self.stopped.wait()),
            }
            try:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            finally:
                tasks.update(self.commands)
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)


async def receive_frame(socket: WebSocket, max_bytes: int) -> dict:
    frame = await socket.receive()
    if frame["type"] == "websocket.disconnect":
        raise WebSocketDisconnect(frame.get("code", 1000))
    data = frame.get("text")
    if not isinstance(data, str) or len(data.encode()) > max_bytes:
        raise SessionEndedError
    try:
        value = json.loads(data)
    except (ValueError, RecursionError):
        raise SessionEndedError from None
    if not isinstance(value, dict):
        raise SessionEndedError
    return value


@gui_router.websocket("/ui/session")
async def gui_session(socket: WebSocket) -> None:
    host = socket.headers.get("host", "")
    # The page must come from this host. Either scheme is accepted, because a
    # reverse proxy that terminates TLS hands the service a plain ws scope while
    # the browser's Origin says https.
    if not host or socket.headers.get("origin") not in {f"http://{host}", f"https://{host}"}:
        await socket.close(code=1008)
        return
    await socket.accept()
    try:
        async with asyncio.timeout(AUTH_TIMEOUT):
            frame = await receive_frame(socket, 1024)
            if (
                set(frame) != {"type", "apiKey"}
                or frame["type"] != "auth"
                or not isinstance(frame["apiKey"], str)
            ):
                raise SessionEndedError
            identity = await ui_api._authenticate(f'Bearer {frame["apiKey"]}')
        await GuiSession(socket, identity).run()
    except HTTPException as exc:
        with suppress(TimeoutError, WebSocketDisconnect, RuntimeError, OSError):
            async with asyncio.timeout(SEND_TIMEOUT):
                await socket.send_json(
                    {"type": "error", "status": exc.status_code, "message": "Key validation failed"}
                )
    except (SessionEndedError, WebSocketDisconnect, TimeoutError, OSError, RuntimeError):
        pass
    finally:
        with suppress(WebSocketDisconnect, RuntimeError, OSError, TimeoutError):
            async with asyncio.timeout(SEND_TIMEOUT):
                await socket.close(code=1000)
