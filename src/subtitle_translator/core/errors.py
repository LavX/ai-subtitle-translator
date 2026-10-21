"""The shared error vocabulary and the provider exceptions that carry it.

The kinds are the ones every Bazarr+ plugin kind answers with. Four of them move a job to
the next back-end connection; content and validation failures never do.
"""

import asyncio
from collections.abc import Iterable
from enum import StrEnum


class ErrorKind(StrEnum):
    AUTH_REQUIRED = "auth_required"
    INVALID_CONFIG = "invalid_config"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXHAUSTED = "quota_exhausted"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    DECLINED = "declined"
    CANCELLED = "cancelled"
    PARTIAL = "partial"
    INTERNAL = "internal"


FAILOVER_KINDS = frozenset(
    {
        ErrorKind.RATE_LIMITED,
        ErrorKind.QUOTA_EXHAUSTED,
        ErrorKind.AUTH_REQUIRED,
        ErrorKind.DEPENDENCY_UNAVAILABLE,
    }
)

# Conditions of the connection or of the whole request, not of one batch: the batch that
# meets one stops at once, without a retry, and no new batch is started. An unreachable
# back end is not here, because a transport error or a 5xx is worth a retry first; the
# batch processor stops new batches after such a batch has used its retries.
STOP_KINDS = frozenset(
    {
        ErrorKind.AUTH_REQUIRED,
        ErrorKind.QUOTA_EXHAUSTED,
        ErrorKind.RATE_LIMITED,
        ErrorKind.INVALID_CONFIG,
        ErrorKind.DECLINED,
        ErrorKind.CANCELLED,
    }
)

_SEVERITY = (
    ErrorKind.QUOTA_EXHAUSTED,
    ErrorKind.AUTH_REQUIRED,
    ErrorKind.INVALID_CONFIG,
    ErrorKind.CANCELLED,
    ErrorKind.RATE_LIMITED,
    ErrorKind.DEPENDENCY_UNAVAILABLE,
    ErrorKind.DECLINED,
    ErrorKind.INTERNAL,
    ErrorKind.PARTIAL,
)


def most_severe(kinds: Iterable[ErrorKind | None]) -> ErrorKind | None:
    """The kind an attempt reports when its batches met several."""
    present = {kind for kind in kinds if kind is not None}
    return next((kind for kind in _SEVERITY if kind in present), None)


class TranslationProviderError(Exception):
    """Base exception for translation provider errors."""

    def __init__(
        self,
        message: str,
        provider: str = "unknown",
        retryable: bool = False,
        status_code: int | None = None,
        tokens_used: int = 0,
        cost: float = 0.0,
        retry_after: float | None = None,
        *,
        kind: ErrorKind | None = None,
        billing_unknown: bool = False,
    ):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.retryable = retryable
        self.status_code = status_code
        self.tokens_used = tokens_used
        self.cost = cost
        self.retry_after = retry_after
        self.kind = kind
        # The request may have reached the server and been billed although no usage came back:
        # the provider stopped waiting for the answer, or the connection broke after it was sent.
        self.billing_unknown = billing_unknown
        self.routing_diagnostics: dict | None = None


class ProviderTimeoutError(TranslationProviderError):
    """A provider request exceeded its transport timeout. The server may still have answered
    and billed it."""

    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message, provider, retryable=True, billing_unknown=True)


class RateLimitError(TranslationProviderError):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str, provider: str = "unknown", retry_after: float | None = None):
        super().__init__(message, provider, retryable=True, status_code=429)
        self.retry_after = retry_after


class AuthenticationError(TranslationProviderError):
    """Raised when authentication fails."""

    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message, provider, retryable=False, status_code=401)


class InvalidResponseError(TranslationProviderError):
    """Raised when the provider returns an invalid response."""

    def __init__(
        self,
        message: str,
        provider: str = "unknown",
        raw_response: str | None = None,
        tokens_used: int = 0,
        cost: float = 0.0,
    ):
        super().__init__(message, provider, retryable=True)
        self.raw_response = raw_response
        self.tokens_used = tokens_used
        self.cost = cost


class QuotaExhaustedError(TranslationProviderError):
    """The account or project has no quota or credit left until retry_after (if known)."""

    def __init__(
        self,
        message: str,
        provider: str = "unknown",
        retry_after: float | None = None,
        status_code: int | None = None,
    ):
        super().__init__(
            message,
            provider,
            retryable=False,
            status_code=status_code,
            retry_after=retry_after,
            kind=ErrorKind.QUOTA_EXHAUSTED,
        )


class InvalidConfigError(TranslationProviderError):
    """The connection's settings cannot work; field_errors names the fields at fault."""

    def __init__(
        self,
        message: str,
        provider: str = "unknown",
        field_errors: dict[str, str] | None = None,
        status_code: int | None = None,
    ):
        super().__init__(
            message, provider, retryable=False, status_code=status_code, kind=ErrorKind.INVALID_CONFIG
        )
        self.field_errors = dict(field_errors or {})


class DependencyUnavailableError(TranslationProviderError):
    """Something the back end needs (its server, a binary) cannot be reached now."""

    def __init__(
        self,
        message: str,
        provider: str = "unknown",
        retry_after: float | None = None,
        status_code: int | None = None,
        *,
        billing_unknown: bool = False,
    ):
        super().__init__(
            message,
            provider,
            retryable=True,
            status_code=status_code,
            retry_after=retry_after,
            kind=ErrorKind.DEPENDENCY_UNAVAILABLE,
            billing_unknown=billing_unknown,
        )


class DeclinedError(TranslationProviderError):
    """The engine cannot take this request at all: language_unsupported or server_unreachable."""

    def __init__(self, message: str, provider: str = "unknown", reason: str = "language_unsupported"):
        super().__init__(message, provider, retryable=False, kind=ErrorKind.DECLINED)
        self.reason = reason


def error_kind(error: BaseException) -> ErrorKind:
    """Classify any failure onto the shared vocabulary."""
    if isinstance(error, asyncio.CancelledError):
        return ErrorKind.CANCELLED
    if not isinstance(error, TranslationProviderError):
        return ErrorKind.INTERNAL
    if error.kind is not None:
        return error.kind
    if isinstance(error, AuthenticationError):
        return ErrorKind.AUTH_REQUIRED
    if isinstance(error, RateLimitError):
        return ErrorKind.RATE_LIMITED
    if isinstance(error, ProviderTimeoutError):
        return ErrorKind.DEPENDENCY_UNAVAILABLE
    if isinstance(error, InvalidResponseError):
        return ErrorKind.PARTIAL
    code = error.status_code
    if code == 401:
        return ErrorKind.AUTH_REQUIRED
    if code == 402:
        return ErrorKind.QUOTA_EXHAUSTED
    if code == 429:
        return ErrorKind.RATE_LIMITED
    if code == 404:
        return ErrorKind.INVALID_CONFIG
    if code == 408 or (code is not None and code >= 500):
        return ErrorKind.DEPENDENCY_UNAVAILABLE
    if code is None and error.retryable:
        return ErrorKind.DEPENDENCY_UNAVAILABLE
    return ErrorKind.INTERNAL
