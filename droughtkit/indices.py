"""Climatological drought indices (SPI, SPEI) used as context for the satellite indicators.

All functions take a *monthly* ``pandas.Series`` with a ``DatetimeIndex`` and
return a series on the same index. Each calendar month is fitted separately
(January accumulations are only compared with other Januaries), which removes
the seasonal cycle and makes values comparable across months and climates.

References
----------
McKee et al. (1993) SPI; Vicente-Serrano et al. (2010) SPEI;
Stagge et al. (2015) distribution choice; Thornthwaite (1948) PET.
For production work see e.g. the ``climate-indices`` package (referenced, not required).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import special, stats

#: Standardised indices are clipped here. Beyond
#: +/-3.09 (probability ~0.001) the fitted tail is not trustworthy.
Z_LIMIT = 3.09

#: Default WMO reference period.
DEFAULT_CALIBRATION = (1991, 2020)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _check_monthly(series: pd.Series) -> pd.Series:
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Expected a pandas Series with a DatetimeIndex.")
    full = pd.date_range(series.index.min(), series.index.max(), freq="MS")
    if len(full) != len(series) or not series.index.equals(full):
        # Reindex silently would hide gaps; make them explicit NaN instead.
        series = series.reindex(full)
    return series.astype(float)


def accumulate(series: pd.Series, scale: int) -> pd.Series:
    """Rolling sum over ``scale`` months, ending at each month.

    SPI-3 for May uses March + April + May. The first ``scale - 1`` months
    are NaN because the window is incomplete.
    """
    series = _check_monthly(series)
    return series.rolling(scale, min_periods=scale).sum()


def _calibration_mask(index: pd.DatetimeIndex, calibration) -> np.ndarray:
    if calibration is None:
        return np.ones(len(index), dtype=bool)
    start, end = calibration
    return (index.year >= start) & (index.year <= end)


def _empirical_z(values: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Gringorten plotting-position z-scores of ``values`` within the sample ``ref``."""
    ref = np.sort(ref)
    n = len(ref)
    below = np.searchsorted(ref, values, side="left")
    ties = np.searchsorted(ref, values, side="right") - below
    rank = below + 0.5 * ties + 0.5
    return _to_z((rank - 0.44) / (n + 0.12))


def _to_z(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(prob, 1e-12, 1 - 1e-12)
    return np.clip(stats.norm.ppf(prob), -Z_LIMIT, Z_LIMIT)


# ---------------------------------------------------------------------------
# distribution fits
# ---------------------------------------------------------------------------

def _gamma_thom(x: np.ndarray) -> tuple[float, float]:
    """Gamma shape/scale using Thom's (1958) maximum-likelihood approximation."""
    x = x[x > 0]
    mean = x.mean()
    a = np.log(mean) - np.log(x).mean()
    if a <= 0:  # all values identical
        return np.nan, np.nan
    shape = (1 + np.sqrt(1 + 4 * a / 3)) / (4 * a)
    return shape, mean / shape


def _loglogistic_pwm(x: np.ndarray) -> tuple[float, float, float]:
    """Three-parameter log-logistic fit with unbiased probability-weighted moments.

    This is the estimator used in the original SPEI paper (Vicente-Serrano
    et al., 2010). Returns (alpha = scale, beta = shape, gamma = location).
    """
    x = np.sort(x)
    n = len(x)
    i = np.arange(1, n + 1)
    # unbiased PWMs  w_s = 1/n * sum( C(n-i, s) / C(n-1, s) * x_i )
    w0 = x.mean()
    w1 = np.sum((n - i) / (n - 1) * x) / n
    w2 = np.sum((n - i) * (n - i - 1) / ((n - 1) * (n - 2)) * x) / n
    beta = (2 * w1 - w0) / (6 * w1 - w0 - 6 * w2)
    if not np.isfinite(beta) or beta <= 1:
        return np.nan, np.nan, np.nan
    g = special.gamma(1 + 1 / beta) * special.gamma(1 - 1 / beta)
    alpha = (w0 - 2 * w1) * beta / g
    gamma_loc = w0 - alpha * g
    return alpha, beta, gamma_loc


def _loglogistic_cdf(x, alpha, beta, gamma_loc):
    with np.errstate(divide="ignore", invalid="ignore"):
        cdf = 1.0 / (1.0 + (alpha / (x - gamma_loc)) ** beta)
    return np.where(x <= gamma_loc, 1e-12, cdf)


# ---------------------------------------------------------------------------
# indices
# ---------------------------------------------------------------------------

def spi(
    precip: pd.Series,
    scale: int = 3,
    calibration: tuple[int, int] | None = DEFAULT_CALIBRATION,
    min_years: int = 20,
) -> pd.Series:
    """Standardized Precipitation Index (McKee et al., 1993).

    1. accumulate precipitation over ``scale`` months;
    2. for each calendar month, fit a two-parameter gamma distribution to the
       non-zero accumulations in the calibration period;
    3. account for zeros with a mixed distribution H(x) = q + (1 - q) G(x);
    4. transform the cumulative probability to a standard normal deviate.
    """
    acc = accumulate(precip, scale)
    cal = _calibration_mask(acc.index, calibration)
    out = pd.Series(np.nan, index=acc.index, name=f"SPI-{scale}")
    for month in range(1, 13):
        sel = acc.index.month == month
        ref = acc[sel & cal].dropna().to_numpy()
        if len(ref) < min_years:
            continue
        q = np.mean(ref == 0)
        shape, scl = _gamma_thom(ref)
        if not np.isfinite(shape):
            continue
        values = acc[sel].to_numpy()
        prob = q + (1 - q) * stats.gamma.cdf(values, shape, scale=scl)
        out[sel] = np.where(np.isnan(values), np.nan, _to_z(prob))
    return out


def spei(
    precip: pd.Series,
    pet: pd.Series,
    scale: int = 3,
    calibration: tuple[int, int] | None = DEFAULT_CALIBRATION,
    min_years: int = 20,
) -> pd.Series:
    """Standardized Precipitation-Evapotranspiration Index (Vicente-Serrano et al., 2010).

    Same idea as SPI but applied to the climatic water balance D = P - PET,
    which can be negative, so a three-parameter log-logistic distribution is
    fitted instead of a gamma distribution.

    The PWM log-logistic fit fails for strongly negatively skewed samples
    (shape parameter <= 1). For those calendar months an empirical
    (Gringorten) standardisation is used instead and a note is stored in
    ``result.attrs["empirical_months"]``.
    """
    balance = _check_monthly(precip) - _check_monthly(pet)
    acc = accumulate(balance, scale)
    cal = _calibration_mask(acc.index, calibration)
    out = pd.Series(np.nan, index=acc.index, name=f"SPEI-{scale}")
    fallback = []
    for month in range(1, 13):
        sel = acc.index.month == month
        ref = acc[sel & cal].dropna().to_numpy()
        if len(ref) < min_years:
            continue
        values = acc[sel].to_numpy()
        params = _loglogistic_pwm(ref)
        if np.all(np.isfinite(params)):
            z = _to_z(_loglogistic_cdf(values, *params))
        else:
            z = _empirical_z(values, ref)
            fallback.append(month)
        out[sel] = np.where(np.isnan(values), np.nan, z)
    out.attrs["empirical_months"] = fallback
    return out


def standardized_empirical(
    series: pd.Series,
    scale: int = 1,
    calibration: tuple[int, int] | None = DEFAULT_CALIBRATION,
    how: str = "sum",
    higher_is_wetter: bool = True,
) -> pd.Series:
    """Non-parametric standardised index (e.g. SSMI for soil moisture, SVI for NDVI).

    Each value is ranked against the calibration sample for the same calendar
    month using the Gringorten plotting position and converted to a z-score.
    No distribution is assumed, which suits bounded variables such as soil
    moisture or NDVI. Set ``how='mean'`` for state variables.
    """
    s = _check_monthly(series)
    agg = s.rolling(scale, min_periods=scale)
    acc = agg.sum() if how == "sum" else agg.mean()
    if not higher_is_wetter:
        acc = -acc
    cal = _calibration_mask(acc.index, calibration)
    out = pd.Series(np.nan, index=acc.index)
    for month in range(1, 13):
        sel = acc.index.month == month
        ref = np.sort(acc[sel & cal].dropna().to_numpy())
        if len(ref) < 10:
            continue
        n = len(ref)
        values = acc[sel].to_numpy()
        # number of reference values below each value (ties count half)
        below = np.searchsorted(ref, values, side="left")
        ties = np.searchsorted(ref, values, side="right") - below
        rank = below + 0.5 * ties + 0.5  # 1-based fractional rank
        prob = (rank - 0.44) / (n + 0.12)  # Gringorten
        out[sel] = np.where(np.isnan(values), np.nan, _to_z(prob))
    return out


# ---------------------------------------------------------------------------
# classification and PET
# ---------------------------------------------------------------------------

SPI_CLASSES = [
    (-np.inf, -2.0, "extremely dry"),
    (-2.0, -1.5, "severely dry"),
    (-1.5, -1.0, "moderately dry"),
    (-1.0, 1.0, "near normal"),
    (1.0, 1.5, "moderately wet"),
    (1.5, 2.0, "very wet"),
    (2.0, np.inf, "extremely wet"),
]


def classify_standardized(z: pd.Series) -> pd.Series:
    """McKee et al. (1993) / WMO (2012) categories for SPI-like indices."""
    bins = [c[0] for c in SPI_CLASSES] + [np.inf]
    labels = [c[2] for c in SPI_CLASSES]
    return pd.cut(z, bins=bins, labels=labels, right=False)


def _daylength_hours(lat_deg: float, doy: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    decl = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(lat) * np.tan(decl), -1, 1))
    return 24 / np.pi * ws


def thornthwaite_pet(temp_c: pd.Series, latitude: float) -> pd.Series:
    """Monthly potential evapotranspiration (mm) after Thornthwaite (1948).

    Needs only mean temperature, which makes it a robust teaching default
    when radiation and humidity are unavailable. Prefer FAO-56
    Penman-Monteith ET0 when the inputs exist (e.g. ERA5 via Open-Meteo).
    The heat index is computed from the long-term monthly climatology.
    """
    t = _check_monthly(temp_c)
    clim = t.groupby(t.index.month).mean().clip(lower=0)
    heat_index = np.sum((clim / 5) ** 1.514)
    a = 6.75e-7 * heat_index**3 - 7.71e-5 * heat_index**2 + 1.792e-2 * heat_index + 0.49239
    mid = t.index + pd.Timedelta(days=14)
    days = t.index.days_in_month.to_numpy()
    daylen = _daylength_hours(latitude, mid.dayofyear.to_numpy())
    tpos = t.clip(lower=0).to_numpy()
    pet = 16 * (daylen / 12) * (days / 30) * (10 * tpos / heat_index) ** a
    return pd.Series(pet, index=t.index, name="PET")
