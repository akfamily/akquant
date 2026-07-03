"""密码字段 AES-256-GCM 加密，与 chibi_quant common/password_crypto.py 互解.

线格式: base64(nonce(12B) || ciphertext || tag(16B))，密钥 base64(32B)。
"""

from __future__ import annotations

import base64
import os

_KEY_LEN = 32
_NONCE_LEN = 12


class QMFCryptoError(RuntimeError):
    """密钥无效或加解密失败."""


def _load_key(key_b64: str) -> bytes:
    try:
        key = base64.b64decode(key_b64.encode("ascii"), validate=True)
    except (ValueError, base64.binascii.Error) as exc:  # type: ignore[attr-defined]
        raise QMFCryptoError(f"密钥必须是 base64(32B): {exc}") from exc
    if len(key) != _KEY_LEN:
        raise QMFCryptoError(f"密钥解码后须为 {_KEY_LEN} 字节，实际 {len(key)}")
    return key


def encrypt_password(plaintext: str, key_b64: str) -> str:
    """AES-256-GCM 加密并返回 base64(nonce||ciphertext||tag)."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key = _load_key(key_b64)
    nonce = os.urandom(_NONCE_LEN)
    encrypted = AESGCM(key).encrypt(nonce, plaintext.encode("utf-8"), None)
    return base64.b64encode(nonce + encrypted).decode("ascii")


def decrypt_password(ciphertext: str, key_b64: str) -> str:
    """解密 base64(nonce||ciphertext||tag)."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key = _load_key(key_b64)
    try:
        raw = base64.b64decode(ciphertext.encode("ascii"), validate=True)
    except (ValueError, base64.binascii.Error) as exc:  # type: ignore[attr-defined]
        raise QMFCryptoError(f"base64 解码失败: {exc}") from exc
    if len(raw) <= _NONCE_LEN + 16:
        raise QMFCryptoError("密文长度不足")
    nonce, encrypted = raw[:_NONCE_LEN], raw[_NONCE_LEN:]
    try:
        decrypted: bytes = AESGCM(key).decrypt(nonce, encrypted, None)
    except Exception as exc:  # noqa: BLE001
        raise QMFCryptoError("AES-GCM 认证/解密失败") from exc
    return decrypted.decode("utf-8")
