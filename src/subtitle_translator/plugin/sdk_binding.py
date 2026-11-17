"""The bazarr.translator.v1 classes on bazarr-plugin-sdk. The only module that imports the SDK.

SDK names used (bazarr-plugin-sdk 0.1.0):

| Name | Where |
| --- | --- |
| ``TranslatorPlugin``, ``KIND_API`` | ``bazarr_plugin_sdk.translator.v1`` |
| ``PluginContext`` (``progress``, ``heartbeat``, ``reserve``, ``usage``, ``checkpoint``, ``cancelled``, ``budget``, ``data_dir``, ``temp_dir``, ``artifacts_dir``, ``heartbeat_interval``) | ``bazarr_plugin_sdk`` |
| ``PluginError(code, message, *, retry_after, field_errors, reason)`` | ``bazarr_plugin_sdk`` |
| ``ERROR_CODES`` | ``bazarr_plugin_sdk`` |
| ``run_conformance`` | ``bazarr_plugin_sdk.testing`` |

The host carries its own copy of the SDK and puts it on the worker's path, so the package does
not depend on it at run time.
"""

from typing import ClassVar

from bazarr_plugin_sdk import ERROR_CODES, PluginContext, PluginError
from bazarr_plugin_sdk.translator.v1 import TranslatorPlugin

from subtitle_translator.core.backends import BackendContext
from subtitle_translator.plugin.contract import ContractError, TranslatorContract
from subtitle_translator.plugin.scope import PluginScope


class _SdkSink:
    def __init__(self, ctx: PluginContext) -> None:
        self._ctx = ctx

    def progress(self, done: int, total: int, message: str) -> None:
        self._ctx.progress(done, total, message)

    def heartbeat(self) -> None:
        self._ctx.heartbeat()

    async def reserve(self, amount: float, model: str) -> str | None:
        return await self._ctx.reserve(amount, model)

    async def usage(self, report: dict) -> None:
        await self._ctx.usage(
            report["tokens"], report["cost"], report["model"], report["reservations"]
        )

    async def checkpoint(self, data: dict) -> None:
        await self._ctx.checkpoint(data)


def _context(ctx: PluginContext | None) -> BackendContext:
    if ctx is None:
        return BackendContext()
    return BackendContext(
        data_dir=ctx.data_dir, artifacts_dir=ctx.artifacts_dir, temp_dir=ctx.temp_dir
    )


def _sdk_error(error: ContractError) -> PluginError:
    code = error.code if error.code in ERROR_CODES else "internal"
    return PluginError(
        code,
        error.message,
        retry_after=error.retry_after,
        field_errors=error.field_errors or None,
        reason=error.reason,
    )


class _Plugin(TranslatorPlugin):
    def _scope(self) -> PluginScope:
        raise NotImplementedError

    def __init__(self) -> None:
        super().__init__()
        self._contract = TranslatorContract(self._scope())

    def describe(self) -> dict:
        return self._contract.describe()

    def validate(self, config: dict) -> dict:
        try:
            return self._contract.validate(config)
        except ContractError as error:
            raise _sdk_error(error) from None

    def migrate(self, from_version: int, settings: dict, connections: list[dict]) -> dict:
        try:
            return self._contract.migrate(from_version, settings, connections)
        except ContractError as error:
            raise _sdk_error(error) from None

    async def options(
        self, field: str, config: dict, secrets: dict, ctx: PluginContext | None = None
    ) -> list[dict]:
        try:
            return await self._contract.options(field, config, secrets, _context(ctx))
        except ContractError as error:
            raise _sdk_error(error) from None

    async def test(
        self, config: dict, secrets: dict, ctx: PluginContext | None = None
    ) -> dict:
        try:
            return await self._contract.test(
                config,
                secrets,
                _context(ctx),
                sink=_SdkSink(ctx) if ctx is not None else None,
                budget=ctx.budget if ctx is not None else None,
            )
        except ContractError as error:
            raise _sdk_error(error) from None

    async def translate(self, request: dict, ctx: PluginContext) -> dict:
        try:
            return await self._contract.translate(
                request,
                _SdkSink(ctx),
                heartbeat_interval=ctx.heartbeat_interval,
                context=_context(ctx),
                cancelled=ctx.cancelled,
            )
        except ContractError as error:
            raise _sdk_error(error) from None
        finally:
            # The SDK runs every call on its own event loop, and an opened back end holds an HTTP
            # client tied to the loop it was first used on, so nothing opened here may outlive
            # the call.
            await self._contract.release()


class BuiltinTranslatorPlugin(_Plugin):
    """The ai_translator plugin: OpenRouter, Gemini and the OpenAI-compatible back end, never
    extension code."""

    def _scope(self) -> PluginScope:
        return PluginScope.builtin()


class ExtensionTranslatorPlugin(_Plugin):
    """Subclass once per extension package, naming its PyPI distribution."""

    distribution: ClassVar[str] = ""

    def _scope(self) -> PluginScope:
        if not self.distribution:
            raise TypeError("an extension plugin names its distribution")
        return PluginScope.extension(self.distribution)
