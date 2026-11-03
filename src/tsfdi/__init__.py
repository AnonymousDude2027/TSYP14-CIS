"""TSFDI: tamper-evident sensing against false-data injection in in-situ feeds.

Layered pipeline (see docs/report.md, section 3):

    L0 provenance -> L1 signal processing (SP1-SP3) -> L2 detectors (D1-D4)
    -> L3 consistency and fingerprint (D6, SP4-SP6) + availability (D7) -> L4 diagnosis
    -> L5 response and impact.  L6 is the offline adversary lab.

The package never reads the network: all data comes from ``data/bundle``
(see ``scripts/fetch_data.py``).
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
