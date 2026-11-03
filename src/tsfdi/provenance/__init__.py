"""L0: provenance -- signed, hash-chained records (spec section 10).

    record.py    canonical bytes, hash chain, Ed25519 signature
    anchor.py    SoftwareAnchor: the documented trust-anchor interface
    verify.py    the streaming verifier and its typed errors
    manifest.py  signed data manifest (the reference is pinned by hash)

Stated limit, repeated here because it drives the whole architecture:
provenance proves *who sent what, in which order, unchanged*.  It proves nothing
about whether the sensor measured truthfully.  An attacker who holds the signing
key (A3) passes L0 completely -- which is exactly why L1-L5 exist.
"""
from .record import Record, canonical_bytes, hash_record, sign_record, verify_record  # noqa: F401
from .anchor import SoftwareAnchor, TrustAnchor  # noqa: F401
from .verify import (BadSignature, ChainBreak, Gap, ProvenanceVerifier,  # noqa: F401
                     ProvenanceViolation, ReplayOrReorder, TimestampSkew)
from .manifest import sign_manifest, verify_manifest  # noqa: F401
