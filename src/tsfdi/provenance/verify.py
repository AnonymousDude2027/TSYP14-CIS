"""The streaming verifier (L0) and its typed errors.

Verifier rules, in the order they are applied:

1. **signature** valid for the record's canonical bytes and hash;
2. **sequence**: ``seq`` is exactly the previous accepted ``seq`` + 1 -- so both
   reordering and deletion are caught;
3. **timestamp**: non-decreasing, within a plausible reporting gap of the
   previous accepted record, and not in the future (a stale-but-signed record is
   a replay, even if its signature is perfect, and is caught by the sequence rule);
4. **chain**: ``prev_hash`` equals the hash of the last accepted record.

Each violation raises a *typed* error.  The type matters downstream: a
``BadSignature`` is an outsider, ``ReplayOrReorder`` and ``Gap`` are outsiders
or a broken link, and any of them implies ``SECURITY_INCIDENT`` in the response
ladder (an honest sensor cannot produce them).  None of them implies anything
about the *values*: that is L1-L5's job.

The verifier is streaming: it accepts one record at a time and keeps O(1) state,
because the pipeline must run day by day (spec section 13).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from .record import Record, verify_record


def _as_utc(ts) -> pd.Timestamp:
    """Parse a timestamp and normalise it to tz-aware UTC (canonical form is UTC)."""
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


class ProvenanceViolation(Exception):
    """Base class for L0 violations."""


class BadSignature(ProvenanceViolation):
    pass


class ReplayOrReorder(ProvenanceViolation):
    pass


class Gap(ProvenanceViolation):
    pass


class ChainBreak(ProvenanceViolation):
    pass


class TimestampSkew(ProvenanceViolation):
    pass


@dataclass
class Verdict:
    ok: bool
    records: int = 0
    errors: list[dict] = field(default_factory=list)
    last_seq: int = -1
    last_hash: str = ""
    last_ts: str | None = None

    @property
    def violated(self) -> bool:
        return bool(self.errors)

    def summary(self) -> dict:
        kinds: dict[str, int] = {}
        for e in self.errors:
            kinds[e["type"]] = kinds.get(e["type"], 0) + 1
        return {"ok": self.ok, "records": self.records, "violations": kinds}


class ProvenanceVerifier:
    """Accept or reject sensor records, one at a time."""

    def __init__(self, public_key, skew_seconds: int = 900, expected_start_seq: int = 0,
                 max_gap_seconds: int = 40 * 86400, now=None):
        self.public_key = public_key
        self.skew_seconds = int(skew_seconds)      # tolerance of the wall clock
        self.max_gap_seconds = int(max_gap_seconds)  # deepest silence still treated as a cadence gap
        self._now = now                            # injectable clock (determinism in tests)
        self.expected_start_seq = int(expected_start_seq)
        self.last_seq = self.expected_start_seq - 1
        self.last_hash = None          # None => accept the genesis prev_hash
        self.last_ts: pd.Timestamp | None = None
        self.errors: list[dict] = []
        self.accepted = 0

    # ------------------------------------------------------------------ core
    def check(self, rec: Record) -> bool:
        """Return True if the record is accepted; record the error otherwise."""
        try:
            self._check(rec)
        except ProvenanceViolation as exc:
            self.errors.append({"type": type(exc).__name__, "seq": int(rec.seq),
                                "ts": rec.ts, "detail": str(exc)})
            return False
        self.last_seq = int(rec.seq)
        self.last_hash = rec.compute_hash()
        self.last_ts = _as_utc(rec.ts)
        self.accepted += 1
        return True

    def _check(self, rec: Record) -> None:
        if not verify_record(rec, self.public_key):
            raise BadSignature(f"signature does not verify for seq {rec.seq}")
        if int(rec.seq) <= self.last_seq:
            raise ReplayOrReorder(f"seq {rec.seq} <= last accepted {self.last_seq}")
        if int(rec.seq) > self.last_seq + 1:
            raise Gap(f"seq jump {self.last_seq} -> {rec.seq} ({int(rec.seq) - self.last_seq - 1} missing)")
        if self.last_hash is not None and rec.prev_hash != self.last_hash:
            raise ChainBreak(f"prev_hash {rec.prev_hash[:16]}... != {self.last_hash[:16]}...")
        ts = _as_utc(rec.ts)
        if self.last_ts is not None:
            if ts < self.last_ts:
                raise TimestampSkew(f"timestamp goes backwards: {rec.ts} < {self.last_ts}")
            # A forward jump is only suspicious if it is *not* a plausible reporting
            # gap.  Daily series legitimately skip days (a sensor outage), so the
            # bound is max_gap_seconds rather than the clock skew; a much larger
            # jump means a record was pushed far out of order.
            if (ts - self.last_ts).total_seconds() > self.max_gap_seconds:
                raise TimestampSkew(
                    f"timestamp jumps forward by {(ts - self.last_ts).total_seconds():.0f} s"
                    f" (> max_gap_seconds)")
        else:
            now = pd.Timestamp(self._now) if self._now is not None else pd.Timestamp.now(tz="UTC")
            now = _as_utc(now)
            if ts > now + pd.Timedelta(seconds=self.skew_seconds):
                raise TimestampSkew(f"record dated in the future: {rec.ts}")

    # -------------------------------------------------------------- bulk use
    def run(self, records) -> Verdict:
        t0 = time.perf_counter()
        n = 0
        for rec in records:
            n += 1
            self.check(rec)
        _ = time.perf_counter() - t0
        return Verdict(ok=not self.errors, records=n, errors=list(self.errors),
                       last_seq=self.last_seq, last_hash=self.last_hash or "")
