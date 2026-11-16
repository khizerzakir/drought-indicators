"""ERA5 climate context for the case study, with an offline fallback.

:func:`load_climate` returns monthly ERA5 / ERA5-Land for one point (Open-Meteo
archive API: free, no key, CC BY 4.0), cached as CSV. If the API is unreachable
it falls back to :func:`offline_sample`, a reproducible *synthetic* record that
mimics the Marchfeld. ``.attrs["source"]`` always says which one was used.

Austrian alternatives worth knowing (not used by default): GeoSphere Austria's
SPARTACUS (1 km daily precipitation/temperature since 1961) and WINFORE (1 km
ET0 and SPEI) on the GeoSphere Data Hub, https://data.hub.geosphere.at.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = ["precipitation_sum", "et0_fao_evapotranspiration", "temperature_2m_mean"]
SOIL_VAR = "soil_moisture_7_to_28cm"  # ERA5-Land layer 2 (hourly in the API)

#: Default site: Marchfeld, Lower Austria (see :data:`droughtkit.maps.MARCHFELD`).
SITE = {"lat": 48.25, "lon": 16.70}

#: Synthetic drought years built into the offline sample (month ranges).
SAMPLE_DROUGHTS = {
    1971: (4, 8), 1976: (3, 7), 1983: (6, 9), 1992: (6, 8), 2000: (4, 6), 2003: (2, 9),
    2011: (3, 5), 2015: (6, 9), 2018: (4, 10), 2022: (3, 8), 2025: (10, 12), 2026: (1, 8),
}


def _get_json(url: str, params: dict, timeout: int = 120, retries: int = 2):
    query = urllib.parse.urlencode(params)
    last = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(f"{url}?{query}", timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as exc:  # noqa: BLE001 - network errors vary by platform
            last = exc
            time.sleep(2 * (attempt + 1))
    raise ConnectionError(f"Open-Meteo request failed: {last}")


def _daily_to_monthly(d: pd.DataFrame) -> pd.DataFrame:
    d = d.assign(time=lambda x: pd.to_datetime(x["time"])).set_index("time")
    out = pd.DataFrame({
        "precip_mm": d["precipitation_sum"].resample("MS").sum(min_count=25),
        "et0_mm": d["et0_fao_evapotranspiration"].resample("MS").sum(min_count=25),
    })
    if "temperature_2m_mean" in d:
        out["tmean_c"] = d["temperature_2m_mean"].resample("MS").mean()
    return out


# ---------------------------------------------------------------------------
# point time series
# ---------------------------------------------------------------------------

def fetch_open_meteo(lat: float, lon: float, start: str = "1961-01-01",
                     end: str = "2026-10-31", soil_moisture: bool = True) -> pd.DataFrame:
    """Monthly P (mm), ET0 (mm), mean T (degC) and optional soil moisture (m3/m3)."""
    daily = _get_json(OPEN_METEO_ARCHIVE, {
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "daily": ",".join(DAILY_VARS), "timezone": "UTC",
    })
    monthly = _daily_to_monthly(pd.DataFrame(daily["daily"]))
    if soil_moisture:
        try:
            parts = []
            for y0 in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1, 10):
                s = max(pd.Timestamp(start), pd.Timestamp(f"{y0}-01-01"))
                e = min(pd.Timestamp(end), pd.Timestamp(f"{y0 + 9}-12-31"))
                h = _get_json(OPEN_METEO_ARCHIVE, {
                    "latitude": lat, "longitude": lon,
                    "start_date": s.date().isoformat(), "end_date": e.date().isoformat(),
                    "hourly": SOIL_VAR, "timezone": "UTC", "models": "era5_land",
                })
                parts.append(pd.Series(h["hourly"][SOIL_VAR],
                                       index=pd.to_datetime(h["hourly"]["time"])))
            monthly["soil_moisture"] = pd.concat(parts).astype(float).resample("MS").mean()
        except Exception as exc:  # noqa: BLE001
            print(f"Soil moisture not retrieved ({exc}); continuing without it.")
    monthly.index.name = "time"
    monthly.attrs["source"] = f"ERA5 via Open-Meteo ({lat:.2f} N, {lon:.2f} E)"
    return monthly


def load_climate(lat: float = SITE["lat"], lon: float = SITE["lon"],
                 start: str = "1961-01-01", end: str = "2026-10-31",
                 cache_dir: str | Path = "data", offline: bool = False,
                 soil_moisture: bool = True) -> pd.DataFrame:
    """Monthly climate for a point: cache -> Open-Meteo -> offline sample."""
    if offline:
        return offline_sample(start[:4], end[:4])
    cache = Path(cache_dir) / f"openmeteo_{lat:.3f}_{lon:.3f}_{start}_{end}.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col="time", parse_dates=True).asfreq("MS")
        df.attrs["source"] = f"ERA5 via Open-Meteo ({lat:.2f} N, {lon:.2f} E), cached"
        return df
    try:
        df = fetch_open_meteo(lat, lon, start, end, soil_moisture=soil_moisture)
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache)
        return df
    except ConnectionError as exc:
        print(f"{exc}\n-> falling back to the offline synthetic sample.")
        return offline_sample(start[:4], end[:4])


# ---------------------------------------------------------------------------
# offline synthetic data
# ---------------------------------------------------------------------------

def offline_sample(start: str | int = 1961, end: str | int = 2026, seed: int = 7) -> pd.DataFrame:
    """Reproducible synthetic monthly climate + land-surface record (NOT real data).

    Mimics a dry continental lowland such as the Marchfeld (~550 mm/yr with a
    summer maximum, ~750 mm reference ET0) with a warming trend and the drought
    years in :data:`SAMPLE_DROUGHTS`, so that every method in the course gives
    interpretable results offline.
    Columns: precip_mm, et0_mm, tmean_c, soil_moisture, ndvi, lst_c.
    """
    rng = np.random.default_rng(seed)
    t = pd.date_range(f"{start}-01-01", f"{end}-12-01", freq="MS")
    n = len(t)
    m = t.month.to_numpy()
    yr = t.year.to_numpy()
    phase = 2 * np.pi * (m - 1) / 12

    tmean = 10.3 - 10.5 * np.cos(phase - 0.35) + 0.03 * (yr - 1990) + rng.normal(0, 1.2, n)
    p_mean = 46 - 20 * np.cos(phase - 0.2)
    precip = rng.gamma(4.0, p_mean / 4.0)

    heat = np.zeros(n)
    for y, (m0, m1) in SAMPLE_DROUGHTS.items():
        sel = (yr == y) & (m >= m0) & (m <= m1)
        precip[sel] *= rng.uniform(0.12, 0.45, sel.sum())
        heat[sel] = rng.uniform(1.5, 3.0, sel.sum())
    tmean = tmean + heat

    rad = 0.5 - 0.42 * np.cos(phase - 0.25)
    et0 = np.clip(98 * rad * (tmean + 17.8) / 30, 4, None) * rng.normal(1, 0.05, n)

    smax, store = 160.0, 110.0
    sm = np.empty(n)
    for i in range(n):
        aet = 0.8 * et0[i] * store / smax
        store = float(np.clip(store + precip[i] - aet, 10, smax))
        sm[i] = 0.10 + 0.28 * store / smax
    sm += rng.normal(0, 0.006, n)

    sm_anom = sm - pd.Series(sm).groupby(m).transform("mean").to_numpy()
    phen = 0.40 + 0.38 * np.clip(np.sin(np.pi * (m - 3) / 8), 0, None)
    lag = np.roll(sm_anom, 1)
    lag[0] = 0
    ndvi = phen + 2.2 * lag * (m >= 4) * (m <= 10) + rng.normal(0, 0.015, n)
    lst = tmean + 5 * rad - 70 * sm_anom + rng.normal(0, 0.8, n)

    df = pd.DataFrame({
        "precip_mm": precip.round(1), "et0_mm": et0.round(1), "tmean_c": tmean.round(2),
        "soil_moisture": sm.round(4), "ndvi": np.clip(ndvi, 0, 1).round(4),
        "lst_c": lst.round(2),
    }, index=pd.DatetimeIndex(t, name="time", freq="MS"))
    df.attrs["source"] = "offline synthetic sample (not real observations)"
    return df
