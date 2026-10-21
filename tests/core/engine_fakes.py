"""A scripted provider and back-end type for engine tests. Deterministic, no network."""

from subtitle_translator.core.backends import (
    BackendRegistry,
    BackendType,
    FieldType,
    OpenedBackend,
    RegisteredBackend,
    SettingField,
    Validation,
)
from subtitle_translator.core.models import TranslationConfig
from subtitle_translator.core.provider import TranslationProvider, TranslationResult
from subtitle_translator.core.usage import reserve_request


class ScriptedProvider(TranslationProvider):
    """Each call consumes one script step: an exception to raise, a callable(batch) returning a
    TranslationResult, or "ok" (translate every line as T<text>). An empty script means "ok"."""

    def __init__(self, script=(), tokens=10, cost=0.01, price=None, name="scripted"):
        self.script = list(script)
        self.batches = []
        self.tokens = tokens
        self.cost = cost
        self.price = price  # the worst-case charge of each request, if priced
        self.prices_requests = price is not None
        self.name = name
        self.closed = False

    @property
    def provider_name(self):
        return self.name

    async def translate_batch(self, batch, model=None, temperature=None, config_override=None):
        if self.price is not None:
            await reserve_request(config_override, self.price, self.name)
        self.batches.append(batch)
        step = self.script.pop(0) if self.script else "ok"
        if isinstance(step, BaseException):
            raise step
        if callable(step):
            return step(batch)
        return TranslationResult(
            translations=[{"index": line["index"], "content": f"T{line['content']}"} for line in batch.lines],
            model_used="scripted-model",
            total_tokens=self.tokens,
            cost=self.cost,
        )

    async def health_check(self):
        return True

    async def close(self):
        self.closed = True

    async def get_available_models(self):
        return []


class ScriptedBackend(BackendType):
    """A back-end type whose connections are served by providers the test hands in, by id."""

    type_id = "scripted"
    display_name = "Scripted"

    def __init__(self, providers: dict[str, ScriptedProvider], batch_size=5, parallel=1):
        self.providers = providers
        self.batch_size = batch_size
        self.parallel = parallel
        self.opened = []
        self.bases = []

    def fields(self):
        return [SettingField("model", FieldType.TEXT, "Model", default="scripted-model")]

    def validate(self, settings, secrets):
        return Validation(patch=dict(settings))

    def open(self, connection, base, context):
        self.opened.append(connection.id)
        self.bases.append(base)
        config = TranslationConfig(model="scripted-model", parallelBatches=self.parallel)
        return OpenedBackend(
            self.providers[connection.id], base, config, "scripted-model", self.batch_size
        )


def registry_with(backend: BackendType) -> BackendRegistry:
    return BackendRegistry(
        [RegisteredBackend(backend.type_id, backend, "lavx-ai-subtitle-translator", "0", None)]
    )
