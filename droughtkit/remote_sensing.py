"""Sentinel-2, Landsat and MODIS from Microsoft Planetary Computer, as monthly composites.

No account is needed: ``planetary_computer.sign_inplace`` adds short-lived read
tokens. Data are loaded lazily with ``odc-stac`` and dask, so only the pixels
inside the area of interest (AOI) are read.

Sentinel-2 and Landsat bands are renamed to **common names** (blue, green, red,
nir, swir1, swir2) and scaled to surface reflectance (0-1), so that the same
index and composite code works for both sensors.

Collections
-----------
* ``sentinel-2-l2a``  10-20 m optical, 5-day revisit, 2015-
* ``landsat-c2-l2``   30 m optical + thermal (LST), 8-16 days, Landsat 8/9 2013-
* ``modis-13A1-061``  500 m 16-day NDVI (``13Q1`` = 250 m), 2000-
* ``modis-11A2-061``  1 km 8-day land-surface temperature, 2000-

Asset names occasionally change: print ``list(items[0].assets)`` if a band is
not found.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COMMON = ["blue", "green", "red", "nir", "swir1", "swir2"]

#: Sentinel-2 L2A band -> common name
S2_BANDS = {"B02": "blue", "B03": "green", "B04": "red", "B08": "nir", "B11": "swir1", "B12": "swir2"}
#: Sentinel-2 Scene Classification classes kept as clear: vegetation, bare soil, water.
#: Removed: saturated (1), dark/shadow (2, 3), unclassified (7), cloud (8, 9), cirrus (10), snow (11).
S2_CLEAR = (4, 5, 6)
#: Landsat Collection 2 asset -> common name
LS_BANDS = {"blue": "blue", "green": "green", "red": "red", "nir08": "nir",
            "swir16": "swir1", "swir22": "swir2"}


def utm_epsg(bbox) -> int:
    """EPSG code of the UTM zone at the centre of a lon/lat bbox."""
    lon = (bbox[0] + bbox[2]) / 2
    lat = (bbox[1] + bbox[3]) / 2
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def open_catalog():
    import planetary_computer
    from pystac_client import Client

    return Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def search(collection: str, bbox, start: str, end: str, query: dict | None = None,
           catalog=None, months: tuple[int, ...] | None = None):
    catalog = catalog or open_catalog()
    items = list(catalog.search(collections=[collection], bbox=bbox,
                                datetime=f"{start}/{end}", query=query).items())
    if months:
        def item_month(item):
            dt = item.datetime
            if dt is None:
                dt = pd.Timestamp(item.properties["start_datetime"]).to_pydatetime()
            return dt.month

        items = [it for it in items if item_month(it) in months]
    if not items:
        raise RuntimeError(f"No {collection} items for {bbox} {start}/{end}.")
    return items


# ---------------------------------------------------------------------------
# Sentinel-2
# ---------------------------------------------------------------------------

def load_sentinel2(bbox, start: str, end: str, max_cloud: int = 70, resolution: int = 20,
                   months: tuple[int, ...] | None = None, catalog=None) -> xr.Dataset:
    """Cloud-masked, harmonised Sentinel-2 L2A surface reflectance (common band names).

    * pixel mask from the Scene Classification Layer (SCL), keeping classes 4-6;
    * harmonisation: from processing baseline 04.00 (25 January 2022) digital
      numbers carry a +1000 offset, which is removed per acquisition;
    * reflectance = DN / 10 000, reprojected to the local UTM zone.
    """
    from odc.stac import load

    items = search("sentinel-2-l2a", bbox, start, end, {"eo:cloud_cover": {"lt": max_cloud}},
                   catalog, months)
    cube = load(items, bands=[*S2_BANDS, "SCL"], bbox=bbox, resolution=resolution,
                crs=f"EPSG:{utm_epsg(bbox)}", groupby="solar_day", chunks={"x": 1024, "y": 1024})

    offset_by_day = {}
    for it in items:
        base = float(it.properties.get("s2:processing_baseline", "0") or 0)
        offset_by_day[pd.Timestamp(it.datetime.date())] = 1000 if base >= 4.0 else 0
    offset = xr.DataArray([offset_by_day.get(pd.Timestamp(t).normalize(), 0) for t in cube.time.values],
                          coords={"time": cube.time}, dims="time")

    clear = cube.SCL.isin(S2_CLEAR)
    refl = ((cube[list(S2_BANDS)] - offset) / 10_000).rename(S2_BANDS)
    refl = refl.where(clear & (refl.red > 0) & (refl.red < 1.0))
    refl.attrs["sensor"] = "Sentinel-2 L2A"
    return refl


# ---------------------------------------------------------------------------
# Landsat 8/9
# ---------------------------------------------------------------------------

def load_landsat(bbox, start: str, end: str, max_cloud: int = 70,
                 months: tuple[int, ...] | None = None, catalog=None) -> xr.Dataset:
    """Landsat 8/9 Collection 2 Level-2: reflectance (common names) and LST in degC.

    * scaling: reflectance = DN * 0.0000275 - 0.2; LST = DN * 0.00341802 + 149 K;
    * pixel mask from QA_PIXEL bits 0-5 (fill, dilated cloud, cirrus, cloud, shadow, snow).
    """
    from odc.stac import load

    items = search("landsat-c2-l2", bbox, start, end,
                   {"eo:cloud_cover": {"lt": max_cloud},
                    "platform": {"in": ["landsat-8", "landsat-9"]}}, catalog, months)
    ds = load(items, bands=[*LS_BANDS, "lwir11", "qa_pixel"], bbox=bbox, resolution=30,
              crs=f"EPSG:{utm_epsg(bbox)}", groupby="solar_day", chunks={"x": 1024, "y": 1024})
    qa = ds.qa_pixel.astype("uint16")
    clear = (qa & 0b111111) == 0
    refl = (ds[list(LS_BANDS)].where(ds.red > 0) * 0.0000275 - 0.2).rename(LS_BANDS)
    refl = refl.where(clear & (refl.red > 0) & (refl.red < 1.0))
    refl["lst"] = (ds.lwir11.where(ds.lwir11 > 0) * 0.00341802 + 149.0 - 273.15).where(clear)
    refl.attrs["sensor"] = "Landsat 8/9 C2 L2"
    return refl


# ---------------------------------------------------------------------------
# MODIS
# ---------------------------------------------------------------------------

def load_modis_ndvi(bbox, start: str, end: str, catalog=None, product: str = "13A1",
                    crs: str | None = None, months: tuple[int, ...] | None = None) -> xr.DataArray:
    """MODIS Terra 16-day NDVI (MOD13A1 500 m or MOD13Q1 250 m), good/marginal pixels only."""
    from odc.stac import load

    res = {"13Q1": 250, "13A1": 500}[product]
    items = search(f"modis-{product}-061", bbox, start, end, catalog=catalog, months=months)
    items = [it for it in items if "MOD" in it.id.upper()] or items   # Terra only
    pre = f"{res}m_16_days"
    ds = load(items, bands=[f"{pre}_NDVI", f"{pre}_pixel_reliability"], bbox=bbox,
              resolution=res, crs=crs or f"EPSG:{utm_epsg(bbox)}", groupby="solar_day",
              chunks={}, dtype="float32")
    ndvi = ds[f"{pre}_NDVI"].where(ds[f"{pre}_NDVI"] > -2000) * 1e-4
    return ndvi.where(ds[f"{pre}_pixel_reliability"].isin([0, 1])).rename("ndvi")


def load_modis_lst(bbox, start: str, end: str, catalog=None, crs: str | None = None,
                   months: tuple[int, ...] | None = None) -> xr.DataArray:
    """MODIS Terra 8-day daytime LST (MOD11A2, 1 km) in degC; QC bits 0-1 <= 1."""
    from odc.stac import load

    items = search("modis-11A2-061", bbox, start, end, catalog=catalog, months=months)
    items = [it for it in items if "MOD" in it.id.upper()] or items
    ds = load(items, bands=["LST_Day_1km", "QC_Day"], bbox=bbox, resolution=1000,
              crs=crs or f"EPSG:{utm_epsg(bbox)}", groupby="solar_day",
              chunks={}, dtype="float32")
    lst = ds["LST_Day_1km"].where(ds["LST_Day_1km"] > 0) * 0.02 - 273.15
    return lst.where((ds["QC_Day"].astype("uint8") & 0b11) <= 1).rename("lst")


# ---------------------------------------------------------------------------
# monthly composites
# ---------------------------------------------------------------------------

def monthly_composite(data, months: tuple[int, ...] | None = None):
    """Median of all clear observations in each calendar month, plus an observation count.

    Masking happens before this step, so clouds never enter the median. Months
    with no clear observation stay NaN, and ``n_obs`` (Dataset input) records
    how many clear observations each pixel had, so coverage can be reported.
    """
    comp = data.resample(time="MS").median()
    if isinstance(data, xr.Dataset):
        ref = next(iter(data.data_vars))
        comp["n_obs"] = data[ref].notnull().resample(time="MS").sum().astype("int16")
    if months:
        comp = comp.sel(time=comp.time.dt.month.isin(months))
    return comp


def valid_fraction(data: xr.DataArray) -> pd.Series:
    """Share of AOI pixels with a valid value, per time step."""
    spatial = [d for d in data.dims if d != "time"]
    return data.notnull().mean(spatial).to_series().rename("valid_fraction")


def aoi_median(data: xr.DataArray, min_valid: float = 0.3) -> pd.Series:
    """Spatial median per time step; NaN where less than ``min_valid`` of the AOI is valid."""
    spatial = [d for d in data.dims if d != "time"]
    frac = data.notnull().mean(spatial)
    return data.median(spatial, skipna=True).where(frac >= min_valid).to_series()


def rgb(ds: xr.Dataset, gain: float = 3.2) -> np.ndarray:
    """True-colour image array (y, x, 3) for plotting with imshow."""
    img = np.stack([ds.red.values, ds.green.values, ds.blue.values], axis=-1) * gain
    img = np.clip(img, 0, 1) ** 0.85
    return np.nan_to_num(img, nan=0.86)          # masked / empty pixels in light grey


def monthly_cube(sensor: str, bbox, years, months=range(4, 10), cache_dir="data/composites",
                 catalog=None, **kwargs) -> xr.Dataset:
    """Multi-year monthly median composites for ``sensor`` ('sentinel2' or 'landsat').

    One year is loaded, composited and cached as NetCDF at a time, so a long
    record can be built step by step and re-runs are instant.
    """
    from pathlib import Path

    loader = {"sentinel2": load_sentinel2, "landsat": load_landsat}[sensor]
    months = tuple(months)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    parts = []
    for y in years:
        f = cache / f"{sensor}_{y}_{min(months):02d}-{max(months):02d}.nc"
        if f.exists():
            parts.append(xr.open_dataset(f))
            continue
        end = (pd.Timestamp(y, max(months), 1) + pd.offsets.MonthEnd(0)).date().isoformat()
        try:
            raw = loader(bbox, f"{y}-{min(months):02d}-01", end, months=months, catalog=catalog,
                         **kwargs)
        except RuntimeError as err:          # no scenes in that period
            print(y, err)
            continue
        comp = monthly_composite(raw, months).compute()
        comp = comp.drop_vars("spatial_ref", errors="ignore")
        comp.to_netcdf(f)
        parts.append(comp)
        print(f"{sensor} {y}: {comp.sizes['time']} monthly composites")
    cube = xr.concat(parts, "time")
    cube.attrs["source"] = {"sentinel2": "Copernicus Sentinel-2 L2A", "landsat": "USGS Landsat 8/9 C2 L2"}[sensor] \
        + " via Microsoft Planetary Computer, monthly median composites"
    return cube
