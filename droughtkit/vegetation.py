"""Spectral indices and drought indicators from satellite composites.

Two steps turn reflectance into a drought signal:

1. **Spectral indices** describe the surface in one composite: greenness
   (NDVI, EVI), canopy water (NDMI, MSI), open water (NDWI), heat (LST).
2. **Drought indicators** compare an index with its own history *for the
   same calendar month*: anomaly, standardised anomaly (z-score), and the
   condition indices VCI, TCI and VHI (Kogan 1995).

Every function accepts ``xarray`` objects with a ``time`` coordinate (a full
raster cube or an AOI series) and, where noted, ``pandas.Series``.
"""

from __future__ import annotations

import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# spectral indices (common band names, reflectance 0-1)
# ---------------------------------------------------------------------------

INDEX_INFO = {
    "NDVI": ("(nir - red) / (nir + red)", "greenness, green biomass", "falls with stress, harvest, senescence"),
    "EVI": ("2.5 (nir - red) / (nir + 6 red - 7.5 blue + 1)", "greenness, less saturation over dense crops", "as NDVI"),
    "NDMI": ("(nir - swir1) / (nir + swir1)", "canopy water content", "falls, often before NDVI"),
    "MSI": ("swir1 / nir", "moisture stress", "rises with stress"),
    "NDWI": ("(green - nir) / (green + nir)", "open water, wet surfaces", "not a vegetation index"),
}


def _nd(a, b):
    den = a + b
    return ((a - b) / den.where(den != 0)).clip(-1, 1)


def spectral_indices(ds: xr.Dataset) -> xr.Dataset:
    """NDVI, EVI, NDMI, MSI and NDWI from a dataset with common band names.

    LST is carried over if present (Landsat).
    """
    out = xr.Dataset({
        "NDVI": _nd(ds.nir, ds.red),
        "EVI": (2.5 * (ds.nir - ds.red) / (ds.nir + 6 * ds.red - 7.5 * ds.blue + 1)).clip(-1, 1),
        "NDMI": _nd(ds.nir, ds.swir1),
        "MSI": (ds.swir1 / ds.nir.where(ds.nir > 0.02)).clip(0, 3),
        "NDWI": _nd(ds.green, ds.nir),
    })
    if "lst" in ds:
        out["LST"] = ds.lst
    if "n_obs" in ds:
        out["n_obs"] = ds.n_obs
    return out


# ---------------------------------------------------------------------------
# baselines and drought indicators
# ---------------------------------------------------------------------------

def _as_da(obj):
    if isinstance(obj, pd.Series):
        return xr.DataArray(obj.to_numpy(), coords={"time": obj.index}, dims="time", name=obj.name), True
    return obj, False


def _back(da, was_series, name):
    da = da.rename(name).drop_vars("month", errors="ignore")
    return da.to_series() if was_series else da


def _reference(da, baseline):
    if baseline is None:
        return da
    yrs = da["time"].dt.year
    return da.where((yrs >= baseline[0]) & (yrs <= baseline[1]), drop=True)


def monthly_climatology(obj, baseline: tuple[int, int] | None = None) -> xr.Dataset:
    """Per-calendar-month median, std, min, max and number of valid years."""
    da, _ = _as_da(obj)
    g = _reference(da, baseline).groupby("time.month")
    return xr.Dataset({"median": g.median("time"), "std": g.std("time"), "min": g.min("time"),
                       "max": g.max("time"), "n_years": g.count("time")})


def anomaly(obj, baseline=None, standardize: bool = False, min_years: int = 3):
    """Departure from the monthly median; ``standardize=True`` divides by the monthly std (z-score)."""
    da, s = _as_da(obj)
    clim = monthly_climatology(da, baseline).sel(month=da["time"].dt.month)
    out = da - clim["median"]
    if standardize:
        out = out / clim["std"].where(clim["std"] > 0)
    out = out.where(clim["n_years"] >= min_years)
    return _back(out, s, f"{da.name or 'value'}_{'z' if standardize else 'anom'}")


def condition_index(obj, baseline=None, invert: bool = False, min_years: int = 5):
    """Position between the historical monthly minimum (0) and maximum (100).

    This is Kogan's VCI for NDVI. With NDMI it is often called a moisture
    condition index; with ``invert=True`` and LST it is the TCI.
    """
    da, s = _as_da(obj)
    clim = monthly_climatology(da, baseline).sel(month=da["time"].dt.month)
    span = (clim["max"] - clim["min"]).where(lambda x: x > 0)
    num = (clim["max"] - da) if invert else (da - clim["min"])
    out = (100 * num / span).where(clim["n_years"] >= min_years).clip(0, 100)
    return _back(out, s, "condition")


def vci(ndvi, baseline=None, min_years: int = 5):
    """Vegetation Condition Index (Kogan 1995): 0 = lowest NDVI on record for that month."""
    return condition_index(ndvi, baseline, invert=False, min_years=min_years).rename("VCI")


def tci(lst, baseline=None, min_years: int = 5):
    """Temperature Condition Index (Kogan 1995): 0 = hottest on record for that month."""
    return condition_index(lst, baseline, invert=True, min_years=min_years).rename("TCI")


def vhi(vci_values, tci_values, alpha: float = 0.5):
    """Vegetation Health Index = alpha * VCI + (1 - alpha) * TCI (alpha = 0.5 by convention)."""
    return (alpha * vci_values + (1 - alpha) * tci_values).rename("VHI")


#: Common VHI reading (after Kogan). Thresholds are a convention, not physics.
VHI_CLASSES = [(0, 10, "extreme"), (10, 20, "severe"), (20, 30, "moderate"),
               (30, 40, "mild"), (40, 100.01, "no drought")]


def classify_vhi(values: pd.Series) -> pd.Series:
    bins = [c[0] for c in VHI_CLASSES] + [VHI_CLASSES[-1][1]]
    return pd.cut(values, bins=bins, labels=[c[2] for c in VHI_CLASSES], right=False)


def area_below(da: xr.DataArray, threshold: float) -> pd.Series:
    """Share of valid AOI pixels below ``threshold`` per time step (e.g. VCI < 35)."""
    spatial = [d for d in da.dims if d != "time"]
    valid = da.notnull().sum(spatial)
    return (100 * (da < threshold).sum(spatial) / valid.where(valid > 0)).to_series()
