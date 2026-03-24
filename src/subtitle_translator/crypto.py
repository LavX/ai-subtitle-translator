"""AES-256-GCM encryption/decryption and key management for API keys."""

import base64
import logging
import os
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

_NONCE_SIZE = 12
_KEY_SIZE = 32
_HEX_KEY_LEN = _KEY_SIZE * 2  # 64 hex chars


def generate_key() -> bytes:
    """Return 32 cryptographically random bytes suitable for AES-256."""
    return os.urandom(_KEY_SIZE)


def encrypt(plaintext: str, key: bytes) -> str:
    """Encrypt plaintext with AES-256-GCM.

    Returns a string of the form ``enc:<base64(nonce + ciphertext + tag)>``.
    A fresh 12-byte nonce is generated for every call, so the same plaintext
    produces different ciphertexts on each invocation.
    """
    nonce = os.urandom(_NONCE_SIZE)
    aesgcm = AESGCM(key)
    # AESGCM.encrypt returns ciphertext with the 16-byte GCM tag appended.
    ciphertext_and_tag = aesgcm.encrypt(nonce, plaintext.encode(), None)
    payload = base64.b64encode(nonce + ciphertext_and_tag).decode()
    return f"enc:{payload}"


def decrypt(ciphertext: str, key: bytes) -> str:
    """Decrypt a value produced by :func:`encrypt`.

    Raises:
        ValueError: if the input is not in the expected format, the key is
            wrong, or the data has been tampered with.
    """
    if not ciphertext.startswith("enc:"):
        raise ValueError("ciphertext is missing the 'enc:' prefix")

    payload_b64 = ciphertext[4:]
    if not payload_b64:
        raise ValueError("ciphertext payload is empty")

    try:
        raw = base64.b64decode(payload_b64)
    except Exception as exc:
        raise ValueError(f"ciphertext is not valid base64: {exc}") from exc

    if len(raw) < _NONCE_SIZE:
        raise ValueError("ciphertext is too short to contain a nonce")

    nonce = raw[:_NONCE_SIZE]
    ciphertext_and_tag = raw[_NONCE_SIZE:]

    try:
        aesgcm = AESGCM(key)
        plaintext_bytes = aesgcm.decrypt(nonce, ciphertext_and_tag, None)
    except Exception as exc:
        raise ValueError(f"decryption failed (wrong key or tampered data): {exc}") from exc

    return plaintext_bytes.decode()


def load_or_generate_key(env_key: str, key_file_path: str) -> bytes:
    """Load an AES-256 key from an env variable, a file, or generate a new one.

    Resolution order:
    1. If the environment variable *env_key* is set and non-empty, validate
       that it is a 64-character hex string and return the decoded bytes.
    2. Elif *key_file_path* exists and contains a valid 64-char hex string,
       load and return the decoded bytes.
    3. Otherwise generate a new key, write its hex representation to
       *key_file_path* (creating parent directories as needed, chmod 0o600),
       and return the key.

    Raises:
        ValueError: if the env variable or file contains an invalid key.
    """
    env_value = os.environ.get(env_key, "")
    if env_value:
        return _parse_hex_key(env_value, source=f"environment variable {env_key!r}")

    key_path = Path(key_file_path)
    if key_path.exists():
        file_value = key_path.read_text().strip()
        return _parse_hex_key(file_value, source=f"key file {key_file_path!r}")

    logger.info(
        "No existing key found; generating a new AES-256 key and saving to %s", key_file_path
    )
    key = generate_key()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(key.hex())
    os.chmod(key_path, 0o600)
    return key


def _parse_hex_key(value: str, source: str) -> bytes:
    """Validate and decode a 64-char hex key string.

    Raises:
        ValueError: if the string is not a valid 64-char hex key.
    """
    if len(value) != _HEX_KEY_LEN:
        raise ValueError(
            f"Key from {source} must be exactly {_HEX_KEY_LEN} hex characters (got {len(value)})"
        )
    try:
        return bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"Key from {source} is not valid hex: {exc}") from exc
