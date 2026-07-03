import base64

import pytest

pytest.importorskip("cryptography")

from akquant.gateway.brokers.qmf.crypto import (
    QMFCryptoError,
    decrypt_password,
    encrypt_password,
)

_KEY = base64.b64encode(b"0" * 32).decode("ascii")


def test_encrypt_decrypt_roundtrip() -> None:
    """encrypt_password/decrypt_password should round-trip the plaintext."""
    token = encrypt_password("s3cret", _KEY)
    assert token and token != "s3cret"
    assert decrypt_password(token, _KEY) == "s3cret"


def test_wire_format_length() -> None:
    """Wire format should be base64(nonce(12B) || ciphertext || tag(16B))."""
    raw = base64.b64decode(encrypt_password("x", _KEY), validate=True)
    # nonce(12) + tag(16) + 至少 1 字节密文
    assert len(raw) > 12 + 16


def test_bad_key_raises() -> None:
    """Invalid base64 or wrong-length keys should raise QMFCryptoError."""
    with pytest.raises(QMFCryptoError):
        encrypt_password("x", "not-base64!!")
    with pytest.raises(QMFCryptoError):
        encrypt_password("x", base64.b64encode(b"short").decode("ascii"))
