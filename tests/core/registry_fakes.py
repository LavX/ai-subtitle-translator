"""Fake entry points and back-end types for the registry tests. No network, no real packages."""

from types import SimpleNamespace

from subtitle_translator.core.backends import (
    BackendType,
    FieldType,
    OpenedBackend,
    SettingField,
    Validation,
)
from subtitle_translator.core.models import TranslationConfig

LOADED: list[str] = []


class EchoBackend(BackendType):
    type_id = "echo"
    display_name = "Echo"

    def fields(self):
        return [SettingField("model", FieldType.TEXT, "Model", default="echo-1")]

    def validate(self, settings, secrets):
        return Validation(patch={"model": settings.get("model") or "echo-1"})

    def open(self, connection, base, context):
        from tests.core.engine_fakes import ScriptedProvider

        return OpenedBackend(
            ScriptedProvider([]),
            base,
            TranslationConfig(),
            connection.settings.get("model", "echo-1"),
        )


class SignInBackend(EchoBackend):
    type_id = "signin"
    display_name = "Sign-in harness"
    uses_sign_in = True
    login_argv = ("vendor-cli", "login", "--device-auth")

    async def sign_in_status(self, connection, context):
        """Signed in when the vendor's login file is in this connection's sign-in folder."""
        from subtitle_translator.core.backends import SignInStatus

        folder = context.sign_in_dir(connection.id)
        if folder is None:
            return SignInStatus(False, "No data folder")
        return SignInStatus((folder / "signed-in").exists())


class Misnamed(EchoBackend):
    type_id = "not-what-the-entry-point-says"


class FakeEntryPoint:
    def __init__(self, name, target, dist=None, version="1.0.0", explode=False):
        self.name = name
        self.value = f"tests.core.registry_fakes:{name}"
        self.group = "subtitle_translator.backends"
        self.dist = SimpleNamespace(name=dist, version=version) if dist else None
        self._target = target
        self._explode = explode

    def load(self):
        LOADED.append(self.name)
        if self._explode:
            raise ImportError("vendor library missing")
        return self._target


def entry_points_from(*points):
    def entry_points(group):
        assert group == "subtitle_translator.backends"
        return list(points)

    return entry_points
