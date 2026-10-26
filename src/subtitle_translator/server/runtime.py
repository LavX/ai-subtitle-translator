"""The standalone service's engine, its connection list and the translator the routes use."""

import asyncio
from dataclasses import dataclass, replace
from pathlib import Path

from subtitle_translator import __version__
from subtitle_translator.backends.openrouter import OpenRouterProvider
from subtitle_translator.backends.openrouter_backend import (
    OpenRouterBackendType,
    settings_from_config,
)
from subtitle_translator.core.backends import (
    BUILTIN_DISTRIBUTION,
    BackendContext,
    BackendRegistry,
    BackendScope,
    Connection,
    RegisteredBackend,
)
from subtitle_translator.core.engine import TranslationEngine
from subtitle_translator.core.models import TranslationConfig
from subtitle_translator.core.provider import TranslationProvider
from subtitle_translator.core.settings import CoreSettings
from subtitle_translator.server.config import Settings, get_settings
from subtitle_translator.server.connections import build_connections

REQUEST_CONNECTION = "request"


class _SharedOpenRouter:
    """One OpenRouter provider per settings snapshot, shared by every key, as in 2.1.1: the
    key travels with each request, and the connection pool and SmartFast state are shared."""

    def __init__(self) -> None:
        self._providers: dict[str, OpenRouterProvider] = {}

    def __call__(self, core: CoreSettings) -> OpenRouterProvider:
        snapshot = core.model_copy(update={"openrouter_api_key": ""}).model_dump_json()
        provider = self._providers.get(snapshot)
        if provider is None:
            provider = self._providers[snapshot] = OpenRouterProvider(core.model_copy(update={"openrouter_api_key": ""}))
        return provider


@dataclass
class Runtime:
    settings: object
    core: CoreSettings
    registry: BackendRegistry
    engine: TranslationEngine
    connections: list[Connection]
    openrouter: TranslationProvider

    def plan(
        self,
        config: TranslationConfig | None,
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> tuple[list[Connection], str | None]:
        """The connections a request may use, and the one it pins.

        A request that brings its own OpenRouter key (a Bazarr+ 2.7 sidecar request, or a GUI
        user) runs only on that key, with every option it sent, and never fails over to the
        server's own back ends. The server's current default model, temperature and output
        budget fill in what it leaves out, and they travel in its connection as they do in the
        server's own OpenRouter connection, so a change of them reaches it without reopening
        the other back ends. config.backend pins a configured connection. Otherwise a
        request's model, temperature and routing options apply to OpenRouter connections only.
        """
        if config is not None and config.api_key:
            wanted = {k: v for k, v in settings_from_config(config).items() if v is not None}
            if not wanted.get("model"):
                wanted["model"] = model or getattr(self.settings, "openrouter_default_model", self.core.openrouter_default_model)
            if temperature is not None:
                wanted.setdefault("temperature", temperature)
            wanted.setdefault(
                "temperature", getattr(self.settings, "openrouter_temperature", self.core.openrouter_temperature)
            )
            wanted.setdefault(
                "max_tokens", getattr(self.settings, "openrouter_max_tokens", self.core.openrouter_max_tokens)
            )
            pinned = Connection(REQUEST_CONNECTION, "openrouter", settings=wanted, secrets={"api_key": config.api_key})
            return [pinned], REQUEST_CONNECTION
        if config is not None and config.backend:
            return self.connections, config.backend
        overlay = {k: v for k, v in (settings_from_config(config).items() if config else []) if v is not None}
        if model:
            overlay.setdefault("model", model)
        if temperature is not None:
            overlay.setdefault("temperature", temperature)
        if not overlay:
            return self.connections, None
        return [
            replace(c, settings={**c.settings, **overlay}) if c.backend_type == "openrouter" else c
            for c in self.connections
        ], None

    def planned_model(self, connections: list[Connection], pinned: str | None) -> str:
        chosen = next((c for c in connections if c.id == pinned), None) if pinned else None
        chosen = chosen or next((c for c in connections if c.enabled), None)
        return str(chosen.settings.get("model", "")) if chosen else ""


def _registry(openrouter_factory) -> BackendRegistry:
    registry = BackendRegistry.discover(BackendScope.everything())
    registry.override(OpenRouterBackendType(provider_factory=openrouter_factory))
    return registry


def _data_root(settings: object) -> Path:
    return Path(getattr(settings, "backends_data_dir", "/app/data/backends"))


# Settings that every OpenRouter connection carries itself: its key, model, temperature and
# output budget. The server's connection takes them from build_connections, and a request
# with its own key takes its key from the request and the other three from plan() when the
# request does not set them. A change of one therefore reaches OpenRouter through the
# connections alone, and the engine's base value of it is never read.
_PER_CONNECTION = frozenset(
    {
        "openrouter_api_key",
        "openrouter_default_model",
        "openrouter_temperature",
        "openrouter_max_tokens",
    }
)


def _shared_by_every_connection(core: CoreSettings, settings: object) -> tuple[dict, Path]:
    return core.model_dump(exclude=_PER_CONNECTION), _data_root(settings)


def _context_for(settings: object):
    root = _data_root(settings)

    def context(type_id: str) -> BackendContext:
        return BackendContext(data_dir=root / type_id)

    return context


def single_provider_runtime(provider: TranslationProvider, settings: object) -> Runtime:
    """A runtime whose OpenRouter connection is served by an existing provider (tests, embedding)."""
    if isinstance(settings, Settings):
        core = settings.core()
    elif isinstance(settings, CoreSettings):
        core = settings
    else:
        core = CoreSettings()
    registry = BackendRegistry([RegisteredBackend("openrouter", None, BUILTIN_DISTRIBUTION, __version__, "placeholder")])
    registry.override(OpenRouterBackendType(provider_factory=lambda _core: provider))
    connection = Connection(
        "openrouter",
        "openrouter",
        enabled=True,
        settings={"model": core.openrouter_default_model},
        secrets={"api_key": core.openrouter_api_key},
    )
    engine = TranslationEngine(registry, core, _context_for(settings))
    return Runtime(settings, core, registry, engine, [connection], provider)


_runtime: Runtime | None = None
_runtime_lock = asyncio.Lock()


async def get_runtime() -> Runtime:
    """The shared runtime. A settings change rebuilds the connection list. When it also changes
    what every connection shares (timeouts, retries, batch sizes, the data folders), the engine
    opens each connection again with the new values and closes what it had once the jobs
    running on it finish. A new OpenRouter key, model, temperature or output budget changes only
    the OpenRouter connections: the server's own, rebuilt here, and for the model, temperature
    and output budget also the ones Runtime.plan builds for requests with their own key, which
    read the current values. The other back ends keep their key health and learned batch
    sizes."""
    global _runtime
    async with _runtime_lock:
        settings = get_settings()
        if _runtime is None:
            shared = _SharedOpenRouter()
            core = settings.core()
            registry = _registry(shared)
            engine = TranslationEngine(registry, core, _context_for(settings))
            _runtime = Runtime(settings, core, registry, engine, build_connections(settings, registry), shared(core))
        elif _runtime.settings is not settings:
            core = settings.core()
            if _shared_by_every_connection(core, settings) != _shared_by_every_connection(
                _runtime.core, _runtime.settings
            ):
                await _runtime.engine.reconfigure(core, _context_for(settings))
            _runtime = replace(
                _runtime,
                settings=settings,
                core=core,
                connections=build_connections(settings, _runtime.registry),
            )
    return _runtime


async def close_runtime() -> None:
    global _runtime
    if _runtime is not None:
        await _runtime.engine.aclose()
        await _runtime.openrouter.close()
        _runtime = None


class EngineTranslator:
    """The translator surface the 2.1.1 routes use, served by the engine."""

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        self.settings = runtime.core
        self.provider = runtime.openrouter

    async def translate_content(self, request, config_override=None):
        raise NotImplementedError("the routes do not run on the engine yet")

    async def translate_file(self, content, source_language, target_language, title=None, model=None,
                             temperature=None, config_override=None):
        raise NotImplementedError("the routes do not run on the engine yet")

    def _server_openrouter(self) -> TranslationProvider:
        """The provider for the health probe and the model list. It is the one the engine opened
        for the server's OpenRouter connection, so it has the current timeouts and the engine
        closes it, run with the server's key and default model on that connection pool: the
        shared provider holds no key, and 2.1.1's probe reports unhealthy without one. Without
        an OpenRouter connection it is the runtime's own provider."""
        connection = next((c for c in self.runtime.connections if c.id == "openrouter"), None)
        provider = self.runtime.engine.provider_for(connection) if connection else self.provider
        if isinstance(provider, OpenRouterProvider):
            provider = provider.with_settings(self.runtime.core)
        return provider

    async def health_check(self) -> bool:
        return await self._server_openrouter().health_check()

    async def get_available_models(self) -> list[dict]:
        return await self._server_openrouter().get_available_models()

    async def close(self) -> None:
        await close_runtime()


async def get_translator() -> EngineTranslator:
    return EngineTranslator(await get_runtime())


async def close_translator() -> None:
    await close_runtime()
