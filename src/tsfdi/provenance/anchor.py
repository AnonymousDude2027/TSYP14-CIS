"""The trust anchor.

We claim a **software trust anchor** and nothing more.  ``TrustAnchor`` is a
small interface with exactly the operations a hardware anchor (TPM 2.0 NV index,
ATECC608 secure element, ARM TrustZone) would provide:

    sign(payload) -> signature
    public_key_bytes() -> bytes
    attest() -> dict          # self-description of the anchor

``SoftwareAnchor`` keeps the Ed25519 private key in process memory.  That is
weak: an attacker with code execution on the host can steal it.  The specification
allows an emulated anchor and no hardware purchase is required, so the honest
presentation is: L0 raises the bar from "anyone who can reach the network" to
"anyone who can execute code on the sensor host", and the architecture is
designed so that the second case is covered by L1-L5 rather than pretended away.

Keys are stored in ``data/keys`` with owner-only permissions and are *not*
committed (``.gitignore``); ``make keys`` regenerates them deterministically from
a documented seed for the reproducibility runs, and the key fingerprint is
written into every result file so a reader can tell which anchor produced them.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


class TrustAnchor:
    """Interface a hardware anchor would implement."""

    def sign(self, payload: bytes) -> bytes:  # pragma: no cover - interface
        raise NotImplementedError

    def public_key_bytes(self) -> bytes:  # pragma: no cover - interface
        raise NotImplementedError

    def attest(self) -> dict:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class SoftwareAnchor(TrustAnchor):
    private_key: Ed25519PrivateKey
    label: str = "software"

    # -- construction ------------------------------------------------------
    @classmethod
    def generate(cls, label: str = "software") -> "SoftwareAnchor":
        return cls(Ed25519PrivateKey.generate(), label)

    @classmethod
    def from_seed(cls, seed: bytes, label: str = "software-deterministic") -> "SoftwareAnchor":
        if len(seed) != 32:
            raise ValueError("Ed25519 seeds are 32 bytes")
        return cls(Ed25519PrivateKey.from_private_bytes(seed), label)

    @classmethod
    def load_or_create(cls, path: Path | None = None, seed: bytes | None = None) -> "SoftwareAnchor":
        path = path or Path("data/keys/anchor.pem")
        if path.is_file():
            key = serialization.load_pem_private_key(path.read_bytes(), password=None)
            return cls(key)
        anchor = cls.from_seed(seed) if seed else cls.generate()
        path.parent.mkdir(parents=True, exist_ok=True)
        pem = anchor.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        path.write_bytes(pem)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        return anchor

    # -- interface ---------------------------------------------------------
    def sign(self, payload: bytes) -> bytes:
        return self.private_key.sign(payload)

    def public_key(self) -> Ed25519PublicKey:
        return self.private_key.public_key()

    def public_key_bytes(self) -> bytes:
        return self.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def fingerprint(self) -> str:
        import hashlib

        return hashlib.sha256(self.public_key_bytes()).hexdigest()[:32]

    def attest(self) -> dict:
        return {
            "type": "software",
            "algorithm": "Ed25519",
            "label": self.label,
            "public_key_hex": self.public_key_bytes().hex(),
            "fingerprint": self.fingerprint(),
            "hardware_backed": False,
            "limitations": [
                "private key held in process memory",
                "an attacker with code execution on the host can sign arbitrary data",
                "states L1-L5 are required for that case (adversary A3)",
            ],
        }

    def save_attestation(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.attest(), indent=2), encoding="utf-8")
