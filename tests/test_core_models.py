"""The request and config types live in the core; the API and SmartFast re-export them."""

from subtitle_translator.backends import smartfast
from subtitle_translator.core import models as core_models
from subtitle_translator.server.api import models as api_models


def test_core_models_are_the_api_models():
    for name in (
        "SubtitleLine",
        "ReasoningConfig",
        "ProviderConfig",
        "TranslationConfig",
        "TranslateContentRequest",
    ):
        assert getattr(api_models, name) is getattr(core_models, name), name
    assert api_models.PROVIDER_SORT_VALUES is core_models.PROVIDER_SORT_VALUES


def test_smartfast_policy_lives_in_the_core():
    assert smartfast.SmartFastPolicy is core_models.SmartFastPolicy


def test_line_limit_is_one_constant():
    field = core_models.SubtitleLine.model_fields["line"]
    assert any(getattr(m, "max_length", None) == core_models.MAX_LINE_CHARS for m in field.metadata)
    assert core_models.MAX_LINE_CHARS == 2000


def test_translation_config_for_operation_binds_session_id():
    bound = core_models.TranslationConfig(model="x").for_operation(job_id="job-1")
    assert bound._smartfast_session_id is not None
    assert bound.model == "x"


def test_sanitize_strips_control_characters():
    assert core_models._sanitize_text("a\x00b\x1b[31mc") == "abc"
