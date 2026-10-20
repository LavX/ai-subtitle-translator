"""Serve the optional browser workspace without exposing application files."""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response

STATIC_DIR = Path(__file__).resolve().parent / "static"
ASSET_TYPES = {
    "index.html": "text/html",
    "app.css": "text/css",
    "app.js": "text/javascript",
    "session.mjs": "text/javascript",
    "archive.mjs": "text/javascript",
    "outfit.woff2": "font/woff2",
    "outfit-OFL.txt": "text/plain",
}
UI_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
        "img-src 'self' data:; font-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}

ui_router = APIRouter(include_in_schema=False)


def _serve_asset(filename: str) -> Response:
    if filename not in ASSET_TYPES:
        raise HTTPException(status_code=404, detail="Not Found", headers=UI_HEADERS)
    path = STATIC_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Not Found", headers=UI_HEADERS)
    if filename == "index.html":
        return HTMLResponse(path.read_text(encoding="utf-8"), headers=UI_HEADERS)
    return FileResponse(path, media_type=ASSET_TYPES[filename], headers=UI_HEADERS)


@ui_router.get("/ui")
def ui_redirect(request: Request) -> RedirectResponse:
    return RedirectResponse(request.url_for("ui_index"), headers=UI_HEADERS)


@ui_router.get("/ui/", response_class=HTMLResponse)
def ui_index() -> Response:
    return _serve_asset("index.html")


@ui_router.get("/ui/{filename:path}")
def ui_asset(filename: str) -> Response:
    return _serve_asset(filename)
