# Threat model

This document states what the system defends, against whom, and — with equal
care — what it does not defend. It is the contract that the code is written
against; every claim below is traceable to a module and to a test.

## 1. Assets and what "silent failure" means

| Asset | Why it matters | Where it lives |
| --- | --- | --- |
| The reported value of a surface observation on a given day | It enters reanalysis, agronomic advisories, frost/heat warnings and, indirectly, insurance and water management | `data/bundle/*.csv` (GHCN-Daily station series) |
| The trusted reference series | Every comparison in the pipeline is relative to it; poisoning it defeats detection without touching the sensor | `data/bundle/*.csv` + signed `MANIFEST.sha256` |
| The calibration model (expectation, whitening, thresholds) | The false-alarm budget is only meaningful if the model was fitted on clean data | `results/cache/models/*.pkl`, fitted on 2015-2019 only |
| The provenance key | If stolen, an outsider becomes an insider (A3) | `data/keys/anchor.pem` (never committed) |

**Silent failure** is the central danger of this challenge and it has a precise
meaning here: *the system reports a value that was never measured, the value
looks plausible, and nobody notices.* It is silent in three distinct ways, and
the architecture treats them separately:

1. **Silent for the analyst** — no exception, no missing-data flag, no
   out-of-range value. This is what L0 (provenance) and L1-L3 (statistics)
   address.
2. **Silent for the statistician** — the corrupted series still passes the
   ordinary quality-control checks (range, monotonicity of TMAX/TMIN,
   distribution of the month), because the attacker simply copies a plausible
   number. This is why the design compares the sensor against an *independent
   reference* rather than against the sensor's own history (SP2, D1, D2).
3. **Silent in the evaluation** — a detector that alarms on everything, or a
   metric that rewards the detector for alarming early, hides the failure. This
   is why thresholds are calibrated to a false-alarm budget and why the metrics
   are per-attack-type with detection delay and false alarms reported together
   (section 12 of the specification), never point-adjusted F1.

## 2. System boundary and trust assumptions

```
                 untrusted                              trusted (hash-pinned)
   +-------------------------------+        +--------------------------------------+
   | station / sensor / logger     |        | NASA POWER reanalysis reference      |
   | signs each reading (Ed25519)  |        | (independent instrument + model)     |
   +---------------+---------------+        +------------------+-------------------+
                   |                                            |
              L0 provenance (signature, sequence, chain)        |
                   |                                            |
                   +----------------> L1 SP1-SP3 ---------------+
                                          |
                                   L2 detectors (D1-D6)
                                          |
                                   L3 consistency / fingerprint
                                          |
                                   L4 diagnosis (rule table + logistic regression)
                                          |
                                   L5 action ladder + decision impact
```

Trusted: the reanalysis reference (pinned by hash), the calibration-period
model, the verification key, and the clock. Untrusted: everything arriving from
the station, including the *timestamps* (bounded by a skew check, not trusted
blindly) and the fact that a reading was taken at all.

## 3. Adversary hierarchy

The hierarchy follows section 9 of the specification. For each level the table
gives the capability assumed, what it defeats, and which layer is expected to
catch it. The last column is the honest answer, not an aspiration.

| Level | Capability | What it defeats | Detected by | Honest statement |
| --- | --- | --- | --- | --- |
| **A0 naive** | Can change values in transit or in the logger; no knowledge of the system | Range and format checks | L2 statistics (D1-D4) | Easy: the bias is large and inconsistent with the reference. |
| **A1a reference copy** | Knows the reference is public; replaces the reading with a copy of it (through the calibration regression) | The residual becomes ~0, so any residual test sees nothing | D6 fingerprint ("too clean"), D3 if the copy is exact, and the residual variance ratio | Hard for the mean-shift detectors by construction: the *point* of A1a is to make the residual small. The fingerprint is the countermeasure, which is why SP6 exists. |
| **A1b reference copy + white noise** | As A1a, with noise matched to the calibration residual variance | The variance ratio, if the noise is matched well | D6 (band powers and ACF of the noise vs the real residual), D3 | The residual scale is right but the *spectral shape* is not: weather residuals are red, white noise is flat. |
| **A1c reference copy + AR-matched noise** | Matches the residual ACF as well | Band powers and ACF of the residual | D6 coherence and cross-band structure; D1 only if the bias is large enough to matter | **Expected to be hard.** An adversary who reproduces the sensor's own noise model from the public reference is, statistically, indistinguishable from the sensor on the residual channel. We report what remains (coherence with the reference at the low band, temporal persistence of a copied day) and we do not claim more. |
| **A2 detector-aware, white-box** | Reads the detector thresholds and optimises a bias path that keeps every statistic below its threshold | Everything, in the limit | Nothing, in the limit: this is a game against the budget | The correct response is to measure the *damage at a fixed false-alarm budget* (Urbina-style metric, section 12) instead of claiming detection. If a small bias is undetectable, the report must say at what magnitude detection begins (the detectability frontier, Figure 1). |
| **A2 detector-aware, black-box** | Sees only alarm / no alarm, adapts by bisection | Same as A2 white-box, with a slower ramp | The *rate* of change: a slow ramp is the hardest case for CUSUM (D2 exists for this) | Reported as damage-vs-budget curve. |
| **A3 compromised signer** | Holds the signing key: valid signatures on invented values | L0 entirely | L1-L5 only | Provenance raises the bar from "anyone on the network" to "anyone with code execution on the host". The architecture must therefore never *depend* on L0 for value integrity, and it does not: L0 is orthogonal to L1-L5. |
| **Outsider** (no key) | Can replay a signed record, tamper, delete, reorder, or push a stale record | Naive hash-only schemes | L0: `BadSignature`, `ReplayOrReorder`, `Gap`, `ChainBreak` | Fully solved, and tested in `tests/test_provenance.py`. |

A0 is the standard profile (12 % step, 15 % ramp, L = 7 replay). A1a/b/c and A2
are implemented in `attacks/adaptive.py` and evaluated in Table 2 and Figure 2.

## 4. Attacker knowledge and the calibration/test split

The attacker is assumed to know: the public reference, the general structure of
the system (a residual test against a reference), and the standard profile. The
attacker is *not* assumed to know the calibration coefficients, the thresholds,
or the fitted AR parameters unless the A2 level is being evaluated — in which
case it is explicitly granted white-box access to the detector statistics.

Defence against overfitting: calibration uses 2015-2019 only, model selection
never sees 2020-2024, two sites (MIAMI\_INTL and NAVACERRADA) are held out from
every tuning decision, and the attack families `scaling`, `noise_shaped` and A2
are held out from all tuning and appear only in the evaluation.

## 5. What this system does **not** claim

These are stated up front because a threat model that only lists strengths is
marketing:

1. **No physical-truth verification.** Without an independent reference there is
   no detection at all; a site with a bad reference (low coherence, see
   `results/gonogo.md`) gets correspondingly weaker claims. This is measured per
   site, not assumed.
2. **No protection against A3 with a slow, small bias.** If the signer is
   compromised and the bias is below the minimum detectable magnitude for that
   (site, variable) at the operating budget, the pipeline will not detect it. The
   minimum detectable magnitude is reported per pair.
3. **No protection against an adversary who also corrupts the reference.** The
   reference is pinned by hash, so *editing* it is detected; *choosing a
   different* reference is a governance problem, out of scope.
4. **Precipitation is harder than temperature**, for a documented reason: the
   reference product correlates weakly with a point gauge on daily rain, the
   variable is zero-inflated, and the residual distribution is not Gaussian in
   any simple coordinate. The pipeline uses a rank-Gauss transform (SP3) for
   precipitation and calibrates its false-alarm budget separately
   (section "per-variable budgets" in `docs/design.md`). Claims for PRCP are
   restricted accordingly.
5. **The rule table can output "undetermined".** A stuck-at fault and a
   constant replay are identical in the data; the system says so rather than
   guessing, and the confusion matrix reports that class.
6. **No LLM in the loop.** Explanations are templated from the evidence
   dictionary, so the same alarm always produces the same text (auditable).
