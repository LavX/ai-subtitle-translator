"""Tests for the crypto module (AES-256-GCM encryption and key management)."""

import os
import stat
from pathlib import Path

import pytest

from subtitle_translator.crypto import decrypt, encrypt, generate_key, load_or_generate_key


class TestGenerateKey:
    def test_returns_32_bytes(self) -> None:
        key = generate_key()
        assert isinstance(key, bytes)
        assert len(key) == 32

    def test_unique_each_call(self) -> None:
        key1 = generate_key()
        key2 = generate_key()
        assert key1 != key2


class TestEncryptDecrypt:
    def test_roundtrip(self) -> None:
        key = generate_key()
        plaintext = "sk-or-v1-supersecretapikey"
        ciphertext = encrypt(plaintext, key)
        assert decrypt(ciphertext, key) == plaintext

    def test_different_ciphertexts_each_time(self) -> None:
        key = generate_key()
        plaintext = "same plaintext"
        ct1 = encrypt(plaintext, key)
        ct2 = encrypt(plaintext, key)
        # Different nonces produce different ciphertexts
        assert ct1 != ct2

    def test_ciphertext_has_enc_prefix(self) -> None:
        key = generate_key()
        ct = encrypt("hello", key)
        assert ct.startswith("enc:")

    def test_decrypt_wrong_key_raises(self) -> None:
        key1 = generate_key()
        key2 = generate_key()
        ct = encrypt("secret", key1)
        with pytest.raises(ValueError):
            decrypt(ct, key2)

    def test_decrypt_tampered_data_raises(self) -> None:
        key = generate_key()
        ct = encrypt("secret", key)
        # Flip a character in the base64 payload
        prefix, payload = ct.split(":", 1)
        tampered = prefix + ":" + payload[:-4] + "AAAA"
        with pytest.raises(ValueError):
            decrypt(tampered, key)

    def test_decrypt_without_enc_prefix_raises(self) -> None:
        key = generate_key()
        with pytest.raises(ValueError):
            decrypt("notencrypted", key)

    def test_decrypt_empty_after_prefix_raises(self) -> None:
        key = generate_key()
        with pytest.raises(ValueError):
            decrypt("enc:", key)


class TestLoadOrGenerateKey:
    def test_load_from_env_hex(self, monkeypatch: pytest.MonkeyPatch) -> None:
        key = generate_key()
        hex_key = key.hex()
        monkeypatch.setenv("TEST_CRYPTO_KEY", hex_key)
        result = load_or_generate_key("TEST_CRYPTO_KEY", "/nonexistent/path/key.hex")
        assert result == key

    def test_invalid_env_hex_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_CRYPTO_KEY", "not-valid-hex-string-of-64-chars!!")
        with pytest.raises(ValueError):
            load_or_generate_key("TEST_CRYPTO_KEY", "/nonexistent/path/key.hex")

    def test_env_hex_wrong_length_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_CRYPTO_KEY", "deadbeef")  # too short
        with pytest.raises(ValueError):
            load_or_generate_key("TEST_CRYPTO_KEY", "/nonexistent/path/key.hex")

    def test_load_from_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("TEST_CRYPTO_KEY", raising=False)
        key = generate_key()
        key_file = tmp_path / "key.hex"
        key_file.write_text(key.hex())
        result = load_or_generate_key("TEST_CRYPTO_KEY", str(key_file))
        assert result == key

    def test_generates_and_saves_when_no_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("TEST_CRYPTO_KEY", raising=False)
        key_file = tmp_path / "subdir" / "key.hex"
        result = load_or_generate_key("TEST_CRYPTO_KEY", str(key_file))
        assert isinstance(result, bytes)
        assert len(result) == 32
        assert key_file.exists()
        saved_hex = key_file.read_text().strip()
        assert len(saved_hex) == 64
        assert bytes.fromhex(saved_hex) == result

    def test_generated_file_has_restricted_permissions(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("TEST_CRYPTO_KEY", raising=False)
        key_file = tmp_path / "key.hex"
        load_or_generate_key("TEST_CRYPTO_KEY", str(key_file))
        file_mode = stat.S_IMODE(os.stat(key_file).st_mode)
        assert file_mode == 0o600
