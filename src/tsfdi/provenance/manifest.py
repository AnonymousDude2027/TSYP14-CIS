"""Signed data manifest: the reference series is pinned by hash.

A reference that can be edited defeats every comparison in the pipeline, so the
bundle manifest is signed by the anchor and the pipeline refuses to start if any
hash differs (``io.verify_bundle(strict=True)``).  This is also the mechanism
that makes the jury run reproducible offline: the same manifest and the same
anchored key produce the same refusal or the same acceptance, on any machine.

Signature scheme: Ed25519 over the manifest text, with the signature stored
next to it in ``MANIFEST.sig`` together with the signer fingerprint.
"""
from __future__ import annotations

import json
from pathlib import Path

from .anchor import SoftwareAnchor
from .record import GENESIS


def manifest_text(hashes: dict[str, str]) -> bytes:
    lines = ["# TSFDI data bundle manifest v1"]
    for name in sorted(hashes):
        lines.append(f"{hashes[name]}  {name}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def sign_manifest(anchor: SoftwareAnchor, hashes: dict[str, str], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    text = manifest_text(hashes)
    (directory / "MANIFEST.sha256").write_bytes(text)
    sig = anchor.sign(text)
    payload = {
        "algorithm": "Ed25519",
        "public_key_hex": anchor.public_key_bytes().hex(),
        "fingerprint": anchor.fingerprint(),
        "signature_hex": sig.hex(),
        "n_files": len(hashes),
        "prev_hash": GENESIS,
    }
    p = directory / "MANIFEST.sig"
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def verify_manifest(directory: Path) -> tuple[bool, list[str]]:
    """Verify the signature over the manifest and every file hash."""
    import hashlib

    man = directory / "MANIFEST.sha256"
    sigp = directory / "MANIFEST.sig"
    if not man.is_file():
        return False, ["MANIFEST.sha256 missing"]
    problems: list[str] = []
    if sigp.is_file():
        payload = json.loads(sigp.read_text(encoding="utf-8"))
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

            pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(payload["public_key_hex"]))
            pub.verify(bytes.fromhex(payload["signature_hex"]), man.read_bytes())
        except Exception as exc:  # noqa: BLE001
            problems.append(f"manifest signature invalid: {exc}")
    else:
        problems.append("MANIFEST.sig missing")
    for line in man.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, name = line.split(None, 1)
        f = directory / name.strip()
        if not f.is_file():
            problems.append(f"missing file {name}")
            continue
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        if h != digest:
            problems.append(f"hash mismatch {name}")
    return (not problems), problems
