"""Canonical record encoding, hash chaining and Ed25519 signatures.

Canonicalisation is the whole game in a provenance layer: a signature is only
meaningful if every implementation agrees on the exact bytes that were signed.
We therefore do not sign JSON (key order, whitespace, float formatting and
Unicode escaping are all implementation-defined).  Instead:

    canonical_i = "v1|" sensor_id "|" seq "|" ts "|" variable "|" value "|" prev_hash

with

* ``seq`` as a decimal integer, no padding;
* ``ts`` as ISO-8601 in UTC with a trailing ``Z`` and second resolution;
* ``value`` as a fixed-point decimal string with exactly three decimals, so
  that 12.5, 12.50 and 1.25e1 cannot produce three different signatures for the
  same measurement;
* ``prev_hash`` as lowercase hex, 64 characters;
* ``|`` as separator (no field may contain it -- ``variable`` is validated).

Then

    hash_i = SHA-256( hash_{i-1} || canonical_i )
    sig_i  = Ed25519( sk, canonical_i || hash_i )

Including ``prev_hash`` in the canonical string and the resulting hash in the
signature makes the chain both tamper-evident (any change to an earlier record
breaks every later hash) and truncation-evident (a deleted record shows up as a
sequence gap or a chain break).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

ALLOWED_VAR = re.compile(r"^[A-Z0-9_]{1,16}$")
GENESIS = "0" * 64


def _check(value: str) -> str:
    if "|" in value:
        raise ValueError("field may not contain the separator '|'")
    return value


def fixed3(x: float) -> str:
    """Fixed-point decimal string with three decimals (canonical for a value)."""
    return f"{float(x):.3f}"


@dataclass
class Record:
    sensor_id: str
    seq: int
    ts: str
    variable: str
    value: str
    prev_hash: str = GENESIS
    sig: str = ""
    hash: str = field(default="", compare=False)

    def __post_init__(self):
        if not ALLOWED_VAR.match(self.variable):
            raise ValueError(f"invalid variable name {self.variable!r}")

    # -- canonical form ---------------------------------------------------
    def canonical(self) -> bytes:
        parts = [
            "v1",
            _check(self.sensor_id),
            str(int(self.seq)),
            _check(self.ts),
            self.variable,
            _check(self.value),
            _check(self.prev_hash),
        ]
        return "|".join(parts).encode("utf-8")

    def compute_hash(self) -> str:
        h = hashlib.sha256()
        h.update(bytes.fromhex(self.prev_hash) if self.prev_hash != GENESIS else b"\x00" * 32)
        h.update(self.canonical())
        return h.hexdigest()

    def signed_payload(self) -> bytes:
        """canonical_i || hash_i -- what the signature covers."""
        return self.canonical() + self.compute_hash().encode("ascii")

    def as_dict(self) -> dict:
        return {
            "sensor_id": self.sensor_id, "seq": int(self.seq), "ts": self.ts,
            "variable": self.variable, "value": self.value,
            "prev_hash": self.prev_hash, "hash": self.compute_hash(),
            "sig": self.sig,
        }


def canonical_bytes(rec: Record) -> bytes:
    return rec.canonical()


def hash_record(rec: Record) -> str:
    h = rec.compute_hash()
    rec.hash = h
    return h


def sign_record(rec: Record, private_key) -> Record:
    """Sign in place and return the record (Ed25519 over canonical||hash)."""
    rec.hash = rec.compute_hash()
    rec.sig = private_key.sign(rec.signed_payload()).hex()
    return rec


def verify_record(rec: Record, public_key, prev_hash: str | None = None) -> bool:
    """Verify signature and (optionally) that the record chains onto ``prev_hash``."""
    from cryptography.exceptions import InvalidSignature

    if prev_hash is not None and rec.prev_hash != prev_hash:
        return False
    rec.hash = rec.compute_hash()
    try:
        public_key.verify(bytes.fromhex(rec.sig), rec.signed_payload())
        return True
    except (InvalidSignature, ValueError):
        return False


def from_values(sensor_id: str, seq: int, timestamp, variable: str, value: float,
                prev_hash: str = GENESIS) -> Record:
    """Build a record from raw values using the canonical formatting."""
    import pandas as pd

    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        iso = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        iso = ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    return Record(sensor_id=sensor_id, seq=int(seq), ts=iso, variable=variable,
                  value=fixed3(value), prev_hash=prev_hash)
