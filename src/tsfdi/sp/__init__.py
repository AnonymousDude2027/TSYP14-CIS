"""Signal-processing stages SP1-SP6 (spec section 7).

SP1 conditioning      -> ``conditioning.py``  (units, calendar, lag, gap flags)
SP2 expectation       -> ``expectation.py``   (robust harmonic + reference regression)
SP3 whitening         -> ``whitening.py``     (AR(p) innovations, seasonal scale)
SP4 multi-scale       -> ``wavelet.py``       (DWT of z to levels 1..5)
SP5 matched filtering -> ``matched.py``       (self-similarity / replay search)
SP6 spectral          -> ``fingerprint.py``   (Welch PSD, coherence, Mahalanobis)

The chain SP1-SP3 is *not* cosmetic: it is what makes the false-alarm
calibration in ``detect/thresholds.py`` hold on real data, because CUSUM and
EWMA assume independent, homoscedastic innovations.  The ablation
``results/tables/table_ablation.csv`` measures this claim.
"""
from .conditioning import Conditioning  # noqa: F401
from .expectation import HarmonicsReferenceModel, GradientBoostingReferenceModel, fit_expectation  # noqa: F401
from .whitening import Whitener  # noqa: F401
