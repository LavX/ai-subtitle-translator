"""Optional static UI hosting and the boundary with the existing API."""

import httpx
import pytest

from subtitle_translator import main, web
from subtitle_translator.api import routes
from subtitle_translator.config import Settings


@pytest.fixture
def static_assets(tmp_path, monkeypatch):
    """Use known files without depending on the frontend implementation."""
    directory = tmp_path / "static"
    directory.mkdir()
    (directory / "index.html").write_text(
        '<!doctype html><html><head><link rel="stylesheet" href="app.css"></head>'
        '<body><main>Subtitle workspace</main><script type="module" src="app.js">'
        "</script></body></html>",
        encoding="utf-8",
    )
    (directory / "app.css").write_text("body { color: #123; }", encoding="utf-8")
    (directory / "app.js").write_text("import './archive.mjs';", encoding="utf-8")
    (directory / "archive.mjs").write_text("export const archive = true;", encoding="utf-8")
    (directory / "outfit.woff2").write_bytes(b"test font")
    (directory / "outfit-OFL.txt").write_text("Test font license", encoding="utf-8")
    for parent in (tmp_path, directory):
        for name in (".env", "encryption.key", "main.py", "config.py", "bootstrap.json"):
            (parent / name).write_text("private-file-sentinel", encoding="utf-8")

    monkeypatch.setattr(web, "STATIC_DIR", directory)
    return directory


@pytest.fixture
def app_factory(monkeypatch, static_assets):
    monkeypatch.delenv("UI_ENABLED", raising=False)

    def create(settings=None, **overrides):
        if settings is None:
            settings = Settings(_env_file=None, **overrides)
        monkeypatch.setattr(main, "get_settings", lambda: settings)
        return main.create_app()

    return create


def client_for(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")


async def test_real_settings_default_keeps_ui_and_assets_disabled(app_factory):
    settings = Settings(_env_file=None)
    assert settings.ui_enabled is False
    async with client_for(app_factory(settings)) as client:
        for path in ("/ui", "/ui/", "/ui/index.html", "/ui/app.js", "/ui/app.css"):
            assert (await client.get(path)).status_code == 404


@pytest.mark.parametrize("env_value, expected_status", [("true", 200), ("false", 404)])
async def test_environment_flag_controls_ui_registration(
    app_factory, monkeypatch, env_value, expected_status
):
    monkeypatch.setenv("UI_ENABLED", env_value)
    async with client_for(app_factory(Settings(_env_file=None))) as client:
        assert (await client.get("/ui/")).status_code == expected_status


async def test_ui_redirect_preserves_mount_prefix(app_factory):
    transport = httpx.ASGITransport(app=app_factory(ui_enabled=True), root_path="/translator")
    async with httpx.AsyncClient(transport=transport, base_url="https://example.test") as client:
        response = await client.get("/translator/ui", follow_redirects=False)
    assert response.status_code in (307, 308)
    assert response.headers["location"] == "https://example.test/translator/ui/"


@pytest.mark.parametrize(
    "path, content_type, content",
    [
        ("/ui/", "text/html", "<main>Subtitle workspace</main>"),
        ("/ui/index.html", "text/html", "<main>Subtitle workspace</main>"),
        ("/ui/app.css", "text/css", "body { color: #123; }"),
        ("/ui/app.js", "text/javascript", "import './archive.mjs';"),
        ("/ui/archive.mjs", "text/javascript", "export const archive = true;"),
        ("/ui/outfit.woff2", "font/woff2", "test font"),
        ("/ui/outfit-OFL.txt", "text/plain", "Test font license"),
    ],
)
async def test_enabled_ui_serves_only_packaged_asset_types(
    app_factory, monkeypatch, tmp_path, path, content_type, content
):
    monkeypatch.chdir(tmp_path.parent)
    async with client_for(app_factory(ui_enabled=True)) as client:
        response = await client.get(path)
    assert response.status_code == 200
    assert response.headers["content-type"].split(";")[0] == content_type
    assert content in response.text


@pytest.mark.parametrize("path", ["/ui/", "/ui/index.html", "/ui/app.js", "/ui/app.css"])
async def test_ui_response_has_self_contained_security_policy(app_factory, path):
    async with client_for(app_factory(ui_enabled=True)) as client:
        response = await client.get(path)
    assert response.status_code == 200
    policy = {
        directive.split()[0]: directive.split()[1:]
        for directive in response.headers["content-security-policy"].split(";")
        if directive.strip()
    }
    assert policy == {
        "default-src": ["'none'"],
        "script-src": ["'self'"],
        "style-src": ["'self'"],
        "connect-src": ["'self'"],
        "img-src": ["'self'", "data:"],
        "font-src": ["'self'"],
        "base-uri": ["'none'"],
        "frame-ancestors": ["'none'"],
        "form-action": ["'self'"],
    }
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    if path in ("/ui/", "/ui/index.html"):
        assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "path",
    [
        "/ui/.env",
        "/ui/encryption.key",
        "/ui/main.py",
        "/ui/config.py",
        "/ui/bootstrap.json",
        "/ui/config",
        "/ui/key",
        "/ui/missing.js",
        "/ui/static/",
        "/ui/%2e%2e/.env",
        "/ui/%2e%2e/encryption.key",
        "/ui/%2e%2e/main.py",
        "/ui/%2e%2e%2fconfig.py",
        "/ui/%252e%252e%252fencryption.key",
        "/ui/%2e%2e%5cencryption.key",
        "/ui/app.js/.env",
    ],
)
async def test_ui_does_not_expose_private_or_unlisted_files(app_factory, path):
    async with client_for(app_factory(ui_enabled=True)) as client:
        response = await client.get(path)
    assert response.status_code == 404
    assert "private-file-sentinel" not in response.text


async def test_public_shell_does_not_bypass_existing_header_auth(app_factory, monkeypatch):
    token = "test-shared-auth-token"
    monkeypatch.setattr(routes, "_auth_token", token)
    async with client_for(
        app_factory(ui_enabled=True, openrouter_api_key="test-provider-key")
    ) as client:
        shell = await client.get("/ui/")
        assert shell.status_code == 200
        assert token not in shell.text
        assert "test-provider-key" not in shell.text
        assert (await client.get("/api/v1/jobs/ui-unknown-job")).status_code == 401
        assert (await client.get(f"/api/v1/jobs/ui-unknown-job?token={token}")).status_code == 401
        authorized = await client.get(
            "/api/v1/jobs/ui-unknown-job", headers={"X-Auth-Token": token}
        )
    assert authorized.status_code == 404
    assert authorized.json()["detail"]["error"] == "job_not_found"


async def test_ui_keeps_api_schema_and_documentation_unchanged(app_factory):
    disabled_schema = app_factory().openapi()
    enabled_app = app_factory(ui_enabled=True)
    assert enabled_app.openapi() == disabled_schema
    async with client_for(enabled_app) as client:
        for path in ("/docs", "/redoc", "/openapi.json"):
            response = await client.get(path)
            assert response.status_code == 200
            assert "content-security-policy" not in response.headers
