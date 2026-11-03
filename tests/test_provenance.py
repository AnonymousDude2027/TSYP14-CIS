"""L0: the provenance layer must reject every outsider attack in the spec."""
import numpy as np
import pandas as pd
import pytest

from tsfdi.provenance import (BadSignature, ChainBreak, Gap, ProvenanceVerifier,
                             ReplayOrReorder, SoftwareAnchor, TimestampSkew,
                             sign_record, verify_record)
from tsfdi.provenance.record import Record, fixed3, from_values
from tsfdi.provenance.manifest import sign_manifest, verify_manifest


@pytest.fixture(scope="module")
def anchor():
    return SoftwareAnchor.from_seed(b"0123456789abcdef0123456789abcdef")


@pytest.fixture(scope="module")
def chain(anchor):
    prev = "0" * 64
    recs = []
    for i in range(50):
        r = from_values("SITE0000001", i, pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                        "TMAX", 20.0 + i * 0.1, prev_hash=prev)
        sign_record(r, anchor.private_key)
        prev = r.hash
        recs.append(r)
    return recs


def test_valid_chain_verifies(anchor, chain):
    v = ProvenanceVerifier(anchor.public_key(), skew_seconds=36 * 3600)
    assert all(v.check(r) for r in chain)
    assert not v.errors


def test_tampered_value_is_rejected(anchor, chain):
    v = ProvenanceVerifier(anchor.public_key(), skew_seconds=36 * 3600)
    for r in chain[:10]:
        v.check(r)
    bad = chain[10]
    bad.value = fixed3(float(bad.value) + 5.0)         # tamper
    assert not v.check(bad)
    assert v.errors[-1]["type"] == "BadSignature"


def test_outsider_cannot_sign(anchor, chain):
    rogue = SoftwareAnchor.generate()
    v = ProvenanceVerifier(anchor.public_key(), skew_seconds=36 * 3600)
    for r in chain[:5]:
        v.check(r)
    forged = from_values("SITE0000001", 5, pd.Timestamp("2020-01-06"), "TMAX", 99.0,
                         prev_hash=chain[4].hash)
    sign_record(forged, rogue.private_key)
    assert not v.check(forged)
    assert v.errors[-1]["type"] == "BadSignature"


def test_replay_of_a_signed_record_is_rejected(anchor, chain):
    v = ProvenanceVerifier(anchor.public_key(), skew_seconds=36 * 3600)
    for r in chain[:5]:
        v.check(r)
    assert not v.check(chain[1])            # perfectly signed, but already seen
    assert v.errors[-1]["type"] == "ReplayOrReorder"


def test_reordering_is_rejected(anchor, chain):
    """A record that arrives too early is refused but does not poison the stream."""
    v = ProvenanceVerifier(anchor.public_key(), skew_seconds=36 * 3600)
    for r in chain[:3]:                       # seq 0,1,2 accepted
        v.check(r)
    assert not v.check(chain[4])              # seq 4 arrives before seq 3
    assert v.errors[-1]["type"] == "Gap"
    assert v.check(chain[3])                  # the missing record arrives: stream recovers
    assert v.check(chain[4])                  # and the buffered one is now in order
    assert not v.check(chain[1])              # a late duplicate
    assert v.errors[-1]["type"] == "ReplayOrReorder"


def test_deleted_record_is_detected_as_a_gap(anchor, chain):
    v = ProvenanceVerifier(anchor.public_key(), skew_seconds=36 * 3600)
    for r in chain[:5]:
        v.check(r)
    assert not v.check(chain[6])           # record 5 dropped
    assert v.errors[-1]["type"] == "Gap"


def test_chain_break_is_detected(anchor, chain):
    v = ProvenanceVerifier(anchor.public_key(), skew_seconds=36 * 3600)
    for r in chain[:5]:
        v.check(r)
    spliced = from_values("SITE0000001", 5, pd.Timestamp("2020-01-06"), "TMAX", 21.0,
                          prev_hash="f" * 64)
    sign_record(spliced, anchor.private_key)
    assert not v.check(spliced)
    assert v.errors[-1]["type"] in ("ChainBreak",)


def test_old_record_pushed_later_is_a_timestamp_or_sequence_error(anchor, chain):
    v = ProvenanceVerifier(anchor.public_key(), skew_seconds=3600)
    for r in chain[:20]:
        v.check(r)
    assert not v.check(chain[21])          # gap first
    assert v.errors[-1]["type"] == "Gap"


def test_stale_timestamp_is_rejected(anchor):
    """A later record with an older timestamp must be refused."""
    v = ProvenanceVerifier(anchor.public_key(), skew_seconds=3600)
    r0 = from_values("S", 0, pd.Timestamp("2021-06-01T00:00:00Z"), "TMAX", 20.0)
    sign_record(r0, anchor.private_key)
    assert v.check(r0)
    r1 = from_values("S", 1, pd.Timestamp("2021-05-01T00:00:00Z"), "TMAX", 21.0, prev_hash=r0.hash)
    sign_record(r1, anchor.private_key)
    assert not v.check(r1)
    assert v.errors[-1]["type"] == "TimestampSkew"


def test_canonical_encoding_quantises_to_one_millidegree():
    """Three decimals are the documented resolution: below it, two representations
    of the same measurement must produce identical signed bytes."""
    a = from_values("S", 1, pd.Timestamp("2020-01-01T00:00:00Z"), "TMAX", 12.5)
    b = from_values("S", 1, pd.Timestamp("2020-01-01T00:00:00Z"), "TMAX", 12.5000000001)
    assert a.canonical() == b.canonical()
    c = from_values("S", 1, pd.Timestamp("2020-01-01T00:00:00Z"), "TMAX", 12.51)
    assert a.canonical() != c.canonical()


def test_manifest_signature_and_hashes(tmp_path, anchor):
    (tmp_path / "a.csv").write_text("x,y\n1,2\n")
    (tmp_path / "b.csv").write_text("x,y\n3,4\n")
    import hashlib

    hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in tmp_path.glob("*.csv")}
    sign_manifest(anchor, hashes, tmp_path)
    ok, problems = verify_manifest(tmp_path)
    assert ok, problems
    (tmp_path / "a.csv").write_text("x,y\n1,3\n")     # poison the reference
    ok, problems = verify_manifest(tmp_path)
    assert not ok
    assert any("hash mismatch" in p for p in problems)
