"""Every failure maps onto one kind of the shared vocabulary."""

import asyncio

import pytest

from subtitle_translator.core import provider as provider_module
from subtitle_translator.core.errors import (
    FAILOVER_KINDS,
    STOP_KINDS,
    AuthenticationError,
    DeclinedError,
    DependencyUnavailableError,
    ErrorKind,
    InvalidConfigError,
    InvalidResponseError,
    ProviderTimeoutError,
    QuotaExhaustedError,
    RateLimitError,
    TranslationProviderError,
    error_kind,
    most_severe,
)


def test_vocabulary_is_exactly_the_spec():
    assert {k.value for k in ErrorKind} == {
        "auth_required", "invalid_config", "rate_limited", "quota_exhausted",
        "dependency_unavailable", "declined", "cancelled", "partial", "internal",
    }
    assert FAILOVER_KINDS == {
        ErrorKind.RATE_LIMITED, ErrorKind.QUOTA_EXHAUSTED,
        ErrorKind.AUTH_REQUIRED, ErrorKind.DEPENDENCY_UNAVAILABLE,
    }
    assert ErrorKind.PARTIAL not in FAILOVER_KINDS and ErrorKind.INVALID_CONFIG not in FAILOVER_KINDS
    assert ErrorKind.INVALID_CONFIG in STOP_KINDS and ErrorKind.DECLINED in STOP_KINDS
    # An unreachable back end is worth a retry first; the batch processor stops new batches
    # once one batch has used its retries on it.
    assert ErrorKind.DEPENDENCY_UNAVAILABLE not in STOP_KINDS


@pytest.mark.parametrize(
    "error,kind",
    [
        (AuthenticationError("bad key"), ErrorKind.AUTH_REQUIRED),
        (RateLimitError("slow", retry_after=3), ErrorKind.RATE_LIMITED),
        (ProviderTimeoutError("late"), ErrorKind.DEPENDENCY_UNAVAILABLE),
        (InvalidResponseError("garbled"), ErrorKind.PARTIAL),
        (QuotaExhaustedError("spent"), ErrorKind.QUOTA_EXHAUSTED),
        (InvalidConfigError("no model", field_errors={"model": "required"}), ErrorKind.INVALID_CONFIG),
        (DependencyUnavailableError("down"), ErrorKind.DEPENDENCY_UNAVAILABLE),
        (DeclinedError("no such pair"), ErrorKind.DECLINED),
        (TranslationProviderError("x", status_code=401), ErrorKind.AUTH_REQUIRED),
        (TranslationProviderError("x", status_code=402), ErrorKind.QUOTA_EXHAUSTED),
        (TranslationProviderError("x", status_code=404), ErrorKind.INVALID_CONFIG),
        (TranslationProviderError("x", status_code=429), ErrorKind.RATE_LIMITED),
        (TranslationProviderError("x", retryable=True, status_code=503), ErrorKind.DEPENDENCY_UNAVAILABLE),
        (TranslationProviderError("Network error: refused", retryable=True), ErrorKind.DEPENDENCY_UNAVAILABLE),
        (TranslationProviderError("x", status_code=400), ErrorKind.INTERNAL),
        (TranslationProviderError("x", status_code=403), ErrorKind.INTERNAL),
        (TranslationProviderError("x", kind=ErrorKind.INVALID_CONFIG), ErrorKind.INVALID_CONFIG),
        (asyncio.CancelledError(), ErrorKind.CANCELLED),
        (ValueError("bug"), ErrorKind.INTERNAL),
    ],
)
def test_error_kind(error, kind):
    assert error_kind(error) is kind


def test_quota_is_never_retryable_and_keeps_its_wait():
    error = QuotaExhaustedError("daily limit", retry_after=3600.0)
    assert error.retryable is False and error.retry_after == 3600.0


def test_a_call_that_may_have_reached_the_server_says_its_billing_is_unknown():
    assert ProviderTimeoutError("late").billing_unknown is True
    assert DependencyUnavailableError("reset", billing_unknown=True).billing_unknown is True
    assert DependencyUnavailableError("down").billing_unknown is False
    assert TranslationProviderError("x", retryable=True, status_code=503).billing_unknown is False


def test_most_severe_prefers_quota_then_auth():
    assert most_severe([ErrorKind.PARTIAL, ErrorKind.RATE_LIMITED, None]) is ErrorKind.RATE_LIMITED
    assert most_severe([ErrorKind.RATE_LIMITED, ErrorKind.QUOTA_EXHAUSTED]) is ErrorKind.QUOTA_EXHAUSTED
    assert most_severe([ErrorKind.DEPENDENCY_UNAVAILABLE, ErrorKind.AUTH_REQUIRED]) is ErrorKind.AUTH_REQUIRED
    assert most_severe([None]) is None


def test_provider_module_still_exports_the_exceptions():
    assert provider_module.RateLimitError is RateLimitError
    assert provider_module.TranslationProviderError is TranslationProviderError
