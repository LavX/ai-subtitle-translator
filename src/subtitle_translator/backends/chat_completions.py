"""Shared response handling for back ends that speak the chat-completions protocol."""

import json
import logging
import math
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from subtitle_translator.core.provider import (
    AuthenticationError,
    InvalidResponseError,
    ProviderTimeoutError,
    RateLimitError,
    TranslationProvider,
    TranslationProviderError,
    TranslationResult,
)

logger = logging.getLogger(__name__)


def _strip_trailing_commas(text: str) -> str:
    """Drop a comma that directly precedes a closing bracket, outside strings only.

    A blanket substitution would also rewrite a comma inside a translated line
    that happens to end in "," before a bracket, and then accept the altered text
    as a successful translation.
    """
    out: list[str] = []
    in_string = False
    escaped = False
    length = len(text)
    for position, char in enumerate(text):
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == ",":
            following = position + 1
            while following < length and text[following] in " \t\r\n":
                following += 1
            if following < length and text[following] in "]}":
                continue
        out.append(char)
    return "".join(out)


def _retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=UTC)
            seconds = max(0.0, (date - datetime.now(UTC)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


class ChatCompletionsProvider(TranslationProvider):
    """A back end whose replies are OpenAI-style chat completions.

    Subclasses build and send the request; this class turns the response into a
    TranslationResult, classifies HTTP and envelope errors, and repairs the JSON models
    tend to produce."""

    error_label: str = "Provider"
    debug_logger: logging.Logger = logging.getLogger(f"{__name__}.debug")

    def _output_ceiling(self, model_used: str) -> int | None:
        return None

    def _length_remedy(self, model_used: str, max_tokens: int | None) -> str:
        return "Lower the batch size or raise the output budget."

    async def _process_response(
        self, response: httpx.Response, model_used: str, max_tokens: int | None = None
    ) -> TranslationResult:
        """
        Process the chat-completions API response.

        Args:
            response: HTTP response from the provider
            model_used: Model that was used for translation
            max_tokens: Output budget sent with the request, for truncation diagnostics

        Returns:
            TranslationResult with parsed translations

        Raises:
            Various TranslationProviderError subclasses on failure
        """
        try:
            data = response.json()
        except ValueError:
            data = None

        usage = data.get("usage") if isinstance(data, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        if not isinstance(total_tokens, int) or isinstance(total_tokens, bool) or total_tokens < 0:
            total_tokens = None
        try:
            cost = float(usage["cost"])
            if not math.isfinite(cost) or cost < 0:
                cost = None
        except (KeyError, ValueError, TypeError):
            cost = None

        def fail(detail: Any) -> None:
            code = response.status_code if response.status_code >= 400 else None
            if code is None and isinstance(detail, dict):
                raw_code = detail.get("code")
                if isinstance(raw_code, str) and raw_code.isascii() and raw_code.isdigit():
                    raw_code = int(raw_code)
                if (
                    isinstance(raw_code, int)
                    and not isinstance(raw_code, bool)
                    and 400 <= raw_code <= 599
                ):
                    code = raw_code
            if code == 401:
                error = AuthenticationError(
                    f"Invalid {self.error_label} API key", provider=self.provider_name
                )
            elif code == 429:
                error = RateLimitError(
                    f"{self.error_label} rate limit exceeded", provider=self.provider_name
                )
            elif code == 408:
                # A provider-side timeout is a stall like any other and takes the
                # same-size retry and recovery path, not a one-shot failure.
                error = ProviderTimeoutError(
                    f"{self.error_label} reported a request timeout (408)",
                    provider=self.provider_name,
                )
                error.status_code = code
            elif code is not None and code >= 500:
                error = TranslationProviderError(
                    f"{self.error_label} server error: {code}",
                    provider=self.provider_name,
                    retryable=True,
                    status_code=code,
                )
            else:
                message = detail.get("message") if isinstance(detail, dict) else None
                message = message if isinstance(message, str) else "Provider returned an error"
                # The status travels in the message: it is all the persisted job
                # error keeps, and the GUI's guidance is keyed on it.
                labelled = (
                    f"{self.error_label} API error {code}"
                    if code is not None
                    else f"{self.error_label} API error"
                )
                error = TranslationProviderError(
                    f"{labelled}: {message[:500]}",
                    provider=self.provider_name,
                    retryable=code is None,
                    status_code=code,
                )
            error.tokens_used = total_tokens or 0
            error.cost = cost or 0.0
            error.retry_after = _retry_after_seconds(response.headers.get("retry-after"))
            raise error

        # The server may commit HTTP 200 before an upstream error arrives.
        # Classify its envelope before checking for translated choices, otherwise
        # rate limits incorrectly trigger adaptive batch shrinking.
        if response.status_code >= 400:
            fail(data.get("error") if isinstance(data, dict) else None)
        if not isinstance(data, dict):
            raise InvalidResponseError(
                "Invalid JSON response: expected an object",
                provider=self.provider_name,
                raw_response=response.text[:1000],
                billing_unknown=cost is None,
            )
        if data.get("error") is not None:
            fail(data["error"])

        # Extract translated content
        choices = data.get("choices", [])
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise InvalidResponseError(
                "No valid choices in response",
                provider=self.provider_name,
                raw_response=str(data)[:1000],
                tokens_used=total_tokens or 0,
                cost=cost or 0.0,
                billing_unknown=cost is None,
            )

        if isinstance(choices[0], dict) and choices[0].get("finish_reason") == "error":
            fail(choices[0].get("error"))

        if choices[0].get("finish_reason") == "length":
            limit = f"{max_tokens}-token output budget" if max_tokens else "own output ceiling"
            logger.warning(
                f"{model_used} stopped at its {limit} and the reply may be cut short. "
                f"{self._length_remedy(model_used, max_tokens)}"
            )

        message = choices[0].get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content:
            raise InvalidResponseError(
                "Empty content in response",
                provider=self.provider_name,
                raw_response=str(data)[:1000],
                tokens_used=total_tokens or 0,
                cost=cost or 0.0,
                billing_unknown=cost is None,
            )

        # Debug logging: Log the raw response content
        self.debug_logger.debug("=== RECEIVED FROM OPENROUTER ===")
        self.debug_logger.debug(f"Model used: {model_used}")
        self.debug_logger.debug(
            f"Token usage - prompt: {prompt_tokens}, completion: {completion_tokens}, total: {total_tokens}"
        )
        self.debug_logger.debug(f"Raw content: {content}")

        # Parse the JSON array from content
        try:
            translations, note = self._parse_translations_with_note(content)
        except InvalidResponseError as error:
            error.tokens_used = total_tokens or 0
            error.cost = cost or 0.0
            # A reply that states no charge may still have been billed, whereas an explicit
            # zero is a stated charge. Only the first keeps its hold.
            error.billing_unknown = cost is None
            raise

        # Debug logging: Log parsed translations
        self.debug_logger.debug("=== PARSED TRANSLATIONS ===")
        self.debug_logger.debug(f"Translations count: {len(translations)}")
        self.debug_logger.debug(
            f"Translations: {json.dumps(translations, ensure_ascii=False, indent=2)}"
        )

        return TranslationResult(
            translations=translations,
            model_used=model_used,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
            note=note,
            raw_response=data,
        )

    def _validate_and_warn_unchanged(
        self,
        original_lines: list[dict[str, str]],
        translations: list[dict[str, str]],
    ) -> None:
        """
        Validate that translations are actually different from the original text.

        This is a safety check to detect when the model returns the original text
        instead of actually translating it. Logs warnings but does not fail.

        Args:
            original_lines: Original lines sent for translation
            translations: Translations received from the model
        """
        # Build lookup maps
        original_map: dict[str, str] = {
            str(line["index"]): line["content"] for line in original_lines
        }

        unchanged_indices: list[int] = []
        for trans in translations:
            idx = str(trans["index"])
            translated_content = trans["content"]
            original_content = original_map.get(idx, "")

            # Check if the content is exactly the same (potential untranslated)
            if translated_content.strip() == original_content.strip():
                try:
                    unchanged_indices.append(int(idx))
                except ValueError:
                    # If index is not an integer, just log it
                    pass
                self.debug_logger.warning(
                    f"⚠️ INDEX {idx}: Translation identical to original! "
                    f"Original: '{original_content[:50]}...' -> "
                    f"Translated: '{translated_content[:50]}...'"
                )

        # Calculate the percentage of unchanged translations
        if len(translations) > 0:
            unchanged_pct = (len(unchanged_indices) / len(translations)) * 100
            if unchanged_pct >= 50:
                logger.warning(
                    f"⚠️ SAFETY CHECK: {unchanged_pct:.1f}% of translations are identical to original! "
                    f"This may indicate the model is not translating properly. "
                    f"Unchanged indices: {unchanged_indices[:10]}{'...' if len(unchanged_indices) > 10 else ''}"
                )
            elif unchanged_pct >= 20:
                logger.info(
                    f"ℹ️ {unchanged_pct:.1f}% of translations are identical to original (may be intentional for names/numbers). "
                    f"Count: {len(unchanged_indices)}"
                )

    def _parse_translations(self, content: str) -> list[dict[str, str]]:
        """Parse translation JSON from LLM response; see _parse_translations_with_note."""
        return self._parse_translations_with_note(content)[0]

    def _parse_translations_with_note(
        self, content: str
    ) -> tuple[list[dict[str, str]], str | None]:
        """
        Parse translation JSON from LLM response.

        Args:
            content: Raw content string from LLM response

        Returns:
            List of translation dictionaries, and a note when the reply had to be
            repaired or cut down to its usable part

        Raises:
            InvalidResponseError: If JSON parsing fails
        """
        original_content = content
        repair_note = None
        try:
            # Models produce backslashes JSON cannot take: a \u escape without four hex
            # digits (or truncated), and escaped characters JSON does not allow escaping,
            # most often an apostrophe as \'. One pass handles both so that an escaped
            # backslash pair is consumed first and its second half is never read as the
            # start of another escape: the pair is kept, a broken Unicode escape is
            # dropped, and the backslash of an invalid escape is dropped.
            import re

            def _sanitize_escape(match):
                return match.group(0) if match.group(0) == "\\\\" else ""

            content = re.sub(
                r"\\\\|\\u(?![0-9a-fA-F]{4})[0-9a-fA-F]{0,3}|\\(?![\"\\/bfnrtu])",
                _sanitize_escape,
                content,
            )

            # Parse JSON, detecting duplicate keys that json.loads() would silently deduplicate.
            # Some models return {"index":"0","content":"...","index":"1","content":"..."} which is
            # valid JSON but json.loads() keeps only the last value per key, losing translations.
            def _handle_duplicate_keys(pairs: list[tuple[str, str]]) -> dict | list[dict]:
                nonlocal repair_note
                keys = [k for k, _ in pairs]
                if len(keys) == len(set(keys)):
                    return dict(pairs)
                repair_note = "reply contained duplicate JSON keys"
                # Recover the distinct objects represented by repeated keys.
                logger.warning(
                    f"Detected duplicate JSON keys in LLM response, "
                    f"recovering {len(pairs) // 2} translations via object_pairs_hook"
                )
                result, current, seen = [], {}, set()
                for k, v in pairs:
                    if k in seen:
                        result.append(current)
                        current, seen = {}, set()
                    current[k] = v
                    seen.add(k)
                if current:
                    result.append(current)
                return result

            # Strict mode rejects valid cues if models use raw newlines/tabs in multi-line
            # strings. Only those (and a carriage return) are worth keeping: any other raw
            # control character would otherwise ride through into the SRT output.
            content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", content)
            parsed = json.loads(content, strict=False, object_pairs_hook=_handle_duplicate_keys)

            # JSON mode only permits an object at the top level, so the reply is an
            # object wrapping the array. The prompt asks for "translations"; a model
            # that picks its own wrapper name still puts the array under one key.
            if isinstance(parsed, dict):
                list_keys = [key for key, value in parsed.items() if isinstance(value, list)]
                known = [
                    key
                    for key in ["translations", "results", "data", "lines", "subtitles"]
                    if key in list_keys
                ]
                if known:
                    parsed = parsed[known[0]]
                elif "index" in parsed and "content" in parsed:
                    # A single translation object; the caller notices when more were asked for
                    parsed = [parsed]
                elif len(list_keys) == 1:
                    parsed = parsed[list_keys[0]]
                else:
                    raise InvalidResponseError(
                        f"Unexpected response structure: {list(parsed.keys())}",
                        provider=self.provider_name,
                        raw_response=content[:1000],
                    )

            if not isinstance(parsed, list):
                raise InvalidResponseError(
                    f"Expected list, got {type(parsed).__name__}",
                    provider=self.provider_name,
                    raw_response=content[:1000],
                )

            # Validate and normalize each translation entry
            translations = []
            for item in parsed:
                if not isinstance(item, dict):
                    logger.warning(f"Skipping non-dict item in translations: {item}")
                    continue

                # Handle various key names
                index = str(item.get("index", item.get("idx", item.get("position", ""))))
                text = item.get("content", item.get("text", item.get("translation", "")))

                if index and text is not None:
                    translations.append({"index": index, "content": str(text)})

            return translations, repair_note or (
                "reply required JSON escape repair" if content != original_content else None
            )

        except json.JSONDecodeError as e:
            # Try to extract JSON from markdown code blocks
            import re

            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
            if json_match:
                translations, note = self._parse_translations_with_note(json_match.group(1))
                return translations, note or "reply required JSON extraction"

            # Try to find JSON array directly
            array_match = re.search(r"\[[\s\S]*\]", content)
            if array_match and array_match.group(0) != content:
                try:
                    translations, note = self._parse_translations_with_note(array_match.group(0))
                    return translations, note or "reply required JSON extraction"
                except Exception:
                    pass

            # Small models are loose with JSON: a trailing comma before a closing
            # bracket, or an object cut off by the token limit. A trailing comma is
            # repaired. A reply whose tail is broken keeps the complete objects before
            # the break; the caller sees a short reply and re-requests the rest.
            repaired = _strip_trailing_commas(content)
            if repaired != content:
                try:
                    translations, note = self._parse_translations_with_note(repaired)
                    return translations, note or "reply contained trailing JSON commas"
                except InvalidResponseError:
                    pass
            excerpt = content[max(0, e.pos - 30) : e.pos + 30].replace("\n", " ")
            salvaged = self._salvage_complete_objects(content)
            if salvaged:
                note = f"reply was not valid JSON: {e.msg} near {excerpt!r}"
                logger.warning(
                    f"{note}; kept {len(salvaged)} complete translations before the break"
                )
                return salvaged, note

            raise InvalidResponseError(
                f"Failed to parse JSON: {str(e)} near {excerpt!r}",
                provider=self.provider_name,
                raw_response=content[:1000],
            ) from e

    def _salvage_complete_objects(self, content: str) -> list[dict[str, str]]:
        """Return the translation objects that decode cleanly before a broken tail."""
        decoder = json.JSONDecoder(strict=False)
        start = content.find("[")
        if start < 0:
            return []
        position = start + 1
        found: list[dict[str, str]] = []
        while True:
            while position < len(content) and content[position] in " \t\r\n,":
                position += 1
            if position >= len(content) or content[position] != "{":
                break
            try:
                item, end = decoder.raw_decode(content, position)
            except json.JSONDecodeError:
                break
            if not isinstance(item, dict):
                break
            index = str(item.get("index", item.get("idx", item.get("position", ""))))
            text = item.get("content", item.get("text", item.get("translation", "")))
            if index and text is not None:
                found.append({"index": index, "content": str(text)})
            position = end
        return found
