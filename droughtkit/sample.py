"""Synthetic satellite data for offline teaching (NOT real observations).

When Planetary Computer cannot be reached, the notebooks use these
stand-ins so that every step (masking, compositing, indices, anomalies,
condition indices, maps) can still be practised and its result interpreted.

* :func:`sample_composites` - Sentinel-2/Landsat-like monthly median composites
  for a Marchfeld-like block of fields at 20 m: rectangular parcels of winter
  wheat, maize, sugar beet, irrigated vegetables and grassland, a riparian
  forest strip and a village. Rainfed crops respond to the drought years of
  :func:`droughtkit.data.offline_sample`; irrigated fields barely do.
* :func:`sample_scenes` - individual cloudy acquisitions for one month, to
  demonstrate masking and median compositing.
* :func:`sample_modis` - MODIS-like 1 km NDVI and LST for eastern Austria.

Every object carries ``attrs["source"]`` stating that it is synthetic.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import xarray as xr

from .data import offline_sample

SYNTH = "synthetic sample (not real observations)"
CROPS = ["winter wheat", "maize", "sugar beet", "vegetables", "grassland", "forest", "built-up"]
#: fractional green cover by month (Apr-Sep) for each class
PHENOLOGY = {
    "winter wheat": [0.60, 0.85, 0.70, 0.15, 0.10, 0.10],
    "maize": [0.05, 0.20, 0.60, 0.90, 0.90, 0.60],
    "sugar beet": [0.05, 0.30, 0.70, 0.90, 0.90, 0.80],
    "vegetables": [0.10, 0.45, 0.85, 0.85, 0.50, 0.25],
    "grassland": [0.60, 0.75, 0.70, 0.60, 0.55, 0.60],
    "forest": [0.72, 0.82, 0.85, 0.85, 0.83, 0.78],
    "built-up": [0.10, 0.12, 0.12, 0.12, 0.12, 0.10],
}
#: how strongly each class responds to a soil-water deficit
SENSITIVITY = {"winter wheat": 0.7, "maize": 1.0, "sugar beet": 0.9, "vegetables": 0.9,
               "grassland": 0.8, "forest": 0.25, "built-up": 0.0}
AIR_T = {4: 11.0, 5: 16.0, 6: 20.0, 7: 22.5, 8: 22.0, 9: 17.0}

SOIL = np.array([0.08, 0.11, 0.14, 0.20, 0.30, 0.26])      # blue green red nir swir1 swir2
VEG = np.array([0.03, 0.07, 0.03, 0.42, 0.19, 0.08])


def _utm_origin():
    from pyproj import Transformer

    from .maps import MARCHFELD
    b = MARCHFELD["bbox"]
    x, y = Transformer.from_crs(4326, 32633, always_xy=True).transform((b[0] + b[2]) / 2,
                                                                       (b[1] + b[3]) / 2)
    return float(x), float(y)


@lru_cache(maxsize=4)
def sample_parcels(ny: int = 220, nx: int = 300, res: int = 20, seed: int = 3):
    """Field map: crop class and irrigation flag per pixel (synthetic)."""
    rng = np.random.default_rng(seed)
    crop = np.zeros((ny, nx), dtype="int8")
    irr = np.zeros((ny, nx), dtype=bool)
    field = np.zeros((ny, nx), dtype="int32")
    fid = 0
    probs = np.array([0.36, 0.23, 0.13, 0.14, 0.14])
    irr_p = {0: 0.05, 1: 0.5, 2: 0.45, 3: 0.95, 4: 0.0}
    x = 0
    while x < nx:
        w = int(rng.integers(8, 28))
        y = 0
        while y < ny:
            h = int(rng.integers(5, 18))
            c = int(rng.choice(5, p=probs))
            crop[y:y + h, x:x + w] = c
            irr[y:y + h, x:x + w] = rng.random() < irr_p[c]
            field[y:y + h, x:x + w] = fid
            fid += 1
            y += h
        x += w
    # riparian forest along the south (Danube floodplain) and a village block
    yy, xx = np.mgrid[0:ny, 0:nx]
    edge = ny - 22 - 6 * np.sin(xx / 23.0)
    crop[yy > edge] = 5
    irr[yy > edge] = False
    field[yy > edge] = fid
    crop[40:72, 190:236] = 6
    irr[40:72, 190:236] = False
    field[40:72, 190:236] = fid + 1
    x0, y0 = _utm_origin()
    xs = x0 + (np.arange(nx) - nx / 2 + 0.5) * res
    ys = y0 + (ny / 2 - np.arange(ny) - 0.5) * res
    return xr.Dataset({"crop": (("y", "x"), crop), "irrigated": (("y", "x"), irr),
                       "field": (("y", "x"), field)},
                      coords={"y": ys, "x": xs}, attrs={"crs": "EPSG:32633", "source": SYNTH})


def _stress_table(years, months):
    """Soil-water deficit (0-1) per (year, month) from the offline climate sample."""
    clim = offline_sample(1961, max(years))
    sm = clim.soil_moisture
    z = (sm - sm.groupby(sm.index.month).transform("mean")) / sm.groupby(sm.index.month).transform("std")
    return {(y, m): float(np.clip(-z.get(pd.Timestamp(y, m, 1), 0) / 2.2, 0, 1))
            for y in years for m in months}, clim


def sample_composites(years=range(2017, 2027), months=range(4, 10), seed: int = 5,
                      gaps: bool = True) -> xr.Dataset:
    """Monthly median composites with common band names, LST (degC) and n_obs (synthetic)."""
    rng = np.random.default_rng(seed)
    par = sample_parcels()
    crop = par.crop.values
    irr = par.irrigated.values
    field = par.field.values
    n_fields = int(field.max()) + 1
    ny, nx = crop.shape
    years, months = list(years), list(months)
    stress, clim = _stress_table(years, months)
    t_anom = (clim.tmean_c - clim.tmean_c.groupby(clim.index.month).transform("mean"))

    # per-parcel crop rotation: each year reshuffles which class a parcel holds
    times, bands, lst, nobs = [], [], [], []
    for y in years:
        rot = crop      # same crop every year here; real crop rotation adds noise to anomalies
        vigour = (1 + 0.08 * rng.standard_normal(n_fields))[field]   # field-to-field, year-to-year
        for k, m in enumerate(months):
            fc = np.zeros((ny, nx))
            sens = np.zeros((ny, nx))
            for ci, name in enumerate(CROPS):
                sel = rot == ci
                fc[sel] = PHENOLOGY[name][m - 4]
                sens[sel] = SENSITIVITY[name]
            s = stress[(y, m)] * sens * np.where(irr & (rot < 5), 0.12, 1.0)
            fc = np.clip(fc * vigour * (1 - 0.55 * s) + rng.normal(0, 0.02, fc.shape), 0, 0.97)
            water = np.clip(1 - 0.9 * s + rng.normal(0, 0.04, fc.shape), 0, 1)
            veg = VEG[:, None, None] + np.array([0, 0, 0.03, -0.06, 0.13, 0.09])[:, None, None] * (1 - water)
            refl = SOIL[:, None, None] * (1 - fc) + veg * fc
            refl *= rng.normal(1, 0.02, refl.shape)
            bands.append(refl.astype("float32"))
            ta = AIR_T[m] + float(t_anom.get(pd.Timestamp(y, m, 1), 0))
            lst.append((ta + 11 * (1 - fc) + 7 * (1 - water) * fc - 1.5 * (irr & (rot < 5))
                        + rng.normal(0, 0.6, fc.shape)).astype("float32"))
            # clear looks per pixel: month-to-month weather plus smooth cloud patterns
            from scipy.ndimage import gaussian_filter
            blob = gaussian_filter(rng.standard_normal((ny, nx)), 18)
            blob /= blob.std()
            n = np.clip(np.round(rng.uniform(0.5, 5) + 1.6 * blob), 0, 6).astype("int16")
            nobs.append(n)
            times.append(pd.Timestamp(y, m, 1))
    b = np.stack(bands)  # time, band, y, x
    ds = xr.Dataset({name: (("time", "y", "x"), b[:, i]) for i, name in
                     enumerate(["blue", "green", "red", "nir", "swir1", "swir2"])},
                    coords={"time": times, "y": par.y, "x": par.x})
    ds["lst"] = (("time", "y", "x"), np.stack(lst))
    ds["n_obs"] = (("time", "y", "x"), np.stack(nobs))
    gap = (ds.n_obs == 0) & gaps              # no clear look: the composite stays empty
    for v in ["blue", "green", "red", "nir", "swir1", "swir2", "lst"]:
        ds[v] = ds[v].where(~gap)
    ds.attrs.update(source=f"Sentinel-2/Landsat-like {SYNTH}", crs="EPSG:32633")
    return ds


def sample_scenes(year: int = 2026, month: int = 7, n: int = 6, seed: int = 9) -> xr.Dataset:
    """Individual acquisitions with clouds and shadows plus an SCL-like class layer (synthetic)."""
    from scipy.ndimage import gaussian_filter, shift

    rng = np.random.default_rng(seed)
    comp = sample_composites(years=[year], months=[month], gaps=False).isel(time=0)
    ny, nx = comp.red.shape
    days = np.sort(rng.choice(np.arange(2, 29), n, replace=False))
    scenes, scl = [], []
    for i in range(n):
        field = gaussian_filter(rng.standard_normal((ny, nx)), 14)
        field /= field.std()
        cover = rng.uniform(0.05, 0.6)
        cloud = field > np.quantile(field, 1 - cover)
        shadow = shift(cloud.astype(float), (9, -12), order=0) > 0.5
        shadow &= ~cloud
        bands = []
        for b in ["blue", "green", "red", "nir", "swir1", "swir2"]:
            v = comp[b].values * rng.normal(1, 0.03, (ny, nx))
            v = np.where(cloud, rng.uniform(0.45, 0.7) + 0.05 * field, v)
            v = np.where(shadow, v * 0.35, v)
            bands.append(v)
        scenes.append(np.stack(bands))
        cls = np.where(np.isin(sample_parcels().crop.values, [6]), 5, 4)
        cls = np.where(cloud, 9, np.where(shadow, 3, cls))
        scl.append(cls)
    s = np.stack(scenes)
    ds = xr.Dataset({b: (("time", "y", "x"), s[:, i].astype("float32")) for i, b in
                     enumerate(["blue", "green", "red", "nir", "swir1", "swir2"])},
                    coords={"time": [pd.Timestamp(year, month, d) for d in days],
                            "y": comp.y, "x": comp.x})
    ds["SCL"] = (("time", "y", "x"), np.stack(scl).astype("uint8"))
    ds.attrs.update(source=f"Sentinel-2-like scenes, {SYNTH}", crs="EPSG:32633")
    return ds


def sample_modis(years=range(2001, 2027), months=range(4, 10), res: int = 1000,
                 seed: int = 13) -> xr.Dataset:
    """MODIS-like monthly NDVI and daytime LST (degC) for eastern Austria in EPSG:3416 (synthetic)."""
    from scipy.ndimage import gaussian_filter
    from shapely import contains_xy

    from .maps import EAST_AUSTRIA_BBOX, _box, austria

    rng = np.random.default_rng(seed)
    b = _box(EAST_AUSTRIA_BBOX).total_bounds
    xs = np.arange(b[0] + res / 2, b[2], res)
    ys = np.arange(b[3] - res / 2, b[1], -res)
    X, Y = np.meshgrid(xs, ys)
    inside = contains_xy(austria().geometry.iloc[0], X, Y)
    # broad land-cover pattern: Pannonian arable east/south-east, forested hills west and north-west
    east = np.clip((X - 560_000) / 180_000, 0, 1)
    south_alps = np.clip((380_000 - Y) / 60_000, 0, 1) * np.clip((640_000 - X) / 100_000, 0, 1)
    texture = gaussian_filter(rng.standard_normal(X.shape), 3)
    texture /= texture.std()
    forest = np.clip(0.55 - 0.7 * east + 0.4 * south_alps + 0.25 * texture, 0, 1)
    years, months = list(years), list(months)
    stress, clim = _stress_table(years, months)
    t_anom = (clim.tmean_c - clim.tmean_c.groupby(clim.index.month).transform("mean"))
    arable_phen = {4: 0.45, 5: 0.62, 6: 0.66, 7: 0.55, 8: 0.50, 9: 0.42}
    forest_phen = {4: 0.62, 5: 0.80, 6: 0.85, 7: 0.85, 8: 0.83, 9: 0.76}
    nd, lt, times = [], [], []
    for y in years:
        foot = gaussian_filter(rng.standard_normal(X.shape), 18)
        foot = foot / foot.std()
        weight = np.clip(0.25 + 0.75 * east + 0.35 * foot, 0, 1.2)
        for m in months:
            s = stress[(y, m)] * weight
            base = arable_phen[m] * (1 - forest) + forest_phen[m] * forest
            wobble = gaussian_filter(rng.standard_normal(X.shape), 4)
            wobble /= wobble.std()
            ndvi = (base * (1 - 0.45 * s * (1 - 0.6 * forest)) * (1 + 0.035 * wobble)
                    + rng.normal(0, 0.006, X.shape))
            lst = (AIR_T[m] + 2 + float(t_anom.get(pd.Timestamp(y, m, 1), 0)) + 6 * (1 - forest)
                   + 7 * s + 0.9 * wobble + rng.normal(0, 0.25, X.shape))
            nd.append(np.where(inside, ndvi, np.nan).astype("float32"))
            lt.append(np.where(inside, lst, np.nan).astype("float32"))
            times.append(pd.Timestamp(y, m, 1))
    ds = xr.Dataset({"ndvi": (("time", "y", "x"), np.stack(nd)),
                     "lst": (("time", "y", "x"), np.stack(lt))},
                    coords={"time": times, "y": ys, "x": xs})
    ds.attrs.update(source=f"MODIS-like {SYNTH}", crs="EPSG:3416")
    return ds


def sample_landsat(s2: xr.Dataset | None = None, seed: int = 21) -> xr.Dataset:
    """Landsat-like monthly composites: 30 m grid, slight spectral offset, fewer clear months (synthetic)."""
    rng = np.random.default_rng(seed)
    s2 = sample_composites() if s2 is None else s2
    x = np.arange(float(s2.x.min()), float(s2.x.max()), 30.0)
    y = np.arange(float(s2.y.max()), float(s2.y.min()), -30.0)
    ls = s2.drop_vars("n_obs").interp(x=x, y=y, method="linear")
    ls["nir"] = ls.nir * 0.97 + rng.normal(0, 0.006, ls.nir.shape)
    ls["red"] = ls.red * 1.03 + rng.normal(0, 0.003, ls.red.shape)
    missing = rng.random(ls.sizes["time"]) < 0.15     # whole months without a clear Landsat scene
    ls = ls.where(xr.DataArray(~missing, coords={"time": ls.time}, dims="time"))
    ls.attrs.update(source=f"Landsat-like {SYNTH}", crs="EPSG:32633")
    return ls
