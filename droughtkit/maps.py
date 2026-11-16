"""Maps for Austria and the Marchfeld case study.

Vector context ships with the package (``droughtkit/geodata``):

* country outlines, rivers, lakes and places: Natural Earth 1:10m (public domain)
* Austrian federal states and districts (2021): Statistik Austria via
  github.com/ginseng666/GeoJSON-TopoJSON-Austria (CC BY 4.0)

Maps are drawn in ETRS89 / Austria Lambert (EPSG:3416), the projection used by
GeoSphere Austria's gridded datasets.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import plotting as dkp

GEODATA = Path(__file__).parent / "geodata"
AT_CRS = "EPSG:3416"

#: Case-study site: the Marchfeld, Lower Austria. Austria's main arable plain
#: between Vienna, the Danube and the March, roughly 500-550 mm rain per year,
#: heavily dependent on groundwater irrigation.
MARCHFELD = {
    "name": "Marchfeld, Lower Austria",
    "lat": 48.25,
    "lon": 16.70,
    # Sentinel-2 / Landsat AOI around Lassee-Obersiebenbrunn (lon/lat)
    "bbox": (16.62, 48.20, 16.82, 48.30),
    # wider Marchfeld extent for regional maps
    "region": (16.30, 48.08, 17.00, 48.42),
}
#: Bounding box for Austria-wide grids (lon/lat).
AUSTRIA_BBOX = (9.5, 46.25, 17.25, 49.0)
#: Eastern Austria (Vienna, Lower Austria, Burgenland) for MODIS maps.
EAST_AUSTRIA_BBOX = (14.4, 47.3, 17.2, 49.05)


@lru_cache(maxsize=None)
def layer(name: str):
    """Load a packaged vector layer as a GeoDataFrame in EPSG:4326."""
    import geopandas as gpd

    return gpd.read_file(GEODATA / f"{name}.geojson").set_crs(4326, allow_override=True)


def austria(crs: str = AT_CRS):
    return layer("countries").query("iso3 == 'AUT'").to_crs(crs)


def _box(bbox, crs=AT_CRS):
    import geopandas as gpd
    from shapely.geometry import box

    return gpd.GeoSeries([box(*bbox)], crs=4326).to_crs(crs)


MAIN_RIVERS = ["Danube", "Donau", "Inn", "Salzach", "Enns", "Traun", "Mur", "Drau", "Morava",
               "Dyje", "Raab", "Rhine", "Rhein", "Isar", "Vltava", "Váh", "Sava"]


@lru_cache(maxsize=None)
def relief(crs: str = AT_CRS, res: float = 2500.0):
    """Low-resolution shaded relief (Natural Earth, via basemap-data) resampled to ``crs``.

    Returns (image, extent) for ``imshow``.
    """
    from PIL import Image
    from pyproj import Transformer

    f = next(GEODATA.glob("relief_*.png"))
    w, s_, e, n = (float(v) for v in f.stem.split("_")[1:])
    img = np.asarray(Image.open(f), dtype=float)
    ny, nx = img.shape
    b = _box((w, s_, e, n), crs).total_bounds
    xs = np.arange(b[0], b[2], res)
    ys = np.arange(b[3], b[1], -res)
    X, Y = np.meshgrid(xs, ys)
    lon, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform(X, Y)
    col = ((lon - w) / (e - w) * nx).astype(int)
    row = ((n - lat) / (n - s_) * ny).astype(int)
    ok = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
    out = np.full(X.shape, np.nan)
    out[ok] = img[row[ok], col[ok]]
    return out, [b[0], b[2], b[1], b[3]]


def context(ax, extent=None, crs: str = AT_CRS, states: bool = True, districts: bool = False,
            rivers: bool | str = "main", places: int = 6, neighbours: bool = True,
            labels: bool = True, river_width: float = 1.0, shade: bool = False,
            fill_austria: bool = True):
    """Draw Austrian map context on ``ax``. ``extent`` is a lon/lat bbox.

    ``rivers='main'`` draws only the main rivers, ``True`` all Natural Earth rivers.
    ``shade=True`` adds a subtle shaded-relief background.
    """
    if neighbours:
        layer("countries").to_crs(crs).plot(ax=ax, color="#ece4d2", edgecolor="#cfc5ae", lw=0.6,
                                            zorder=0)
    at = austria(crs)
    if fill_austria:
        at.plot(ax=ax, color=dkp.PAPER, edgecolor="none", zorder=0.5)
    if shade:
        img, ext = relief(crs)
        ax.imshow(img, extent=ext, cmap="Greys_r", alpha=0.28, zorder=0.6, interpolation="bilinear")
    at.plot(ax=ax, facecolor="none", edgecolor=dkp.INK, lw=1.1, zorder=5)
    if districts:
        layer("austria_districts").to_crs(crs).boundary.plot(ax=ax, color="#b7ad97", lw=0.35, zorder=4)
    if states:
        layer("austria_states").to_crs(crs).boundary.plot(ax=ax, color="#8f8771", lw=0.6, zorder=4)
    if rivers:
        rv = layer("rivers")
        if rivers == "main":
            rv = rv[rv["name"].isin(MAIN_RIVERS)]
        rv = rv.to_crs(crs)
        rv.plot(ax=ax, color="#5b9aa0", lw=river_width, zorder=3)
    layer("lakes").to_crs(crs).plot(ax=ax, color="#bcd9d6", edgecolor="#5b9aa0", lw=0.4, zorder=3)
    if places:
        pl = layer("places").query("adm0_a3 == 'AUT'").nlargest(places, "pop_max").to_crs(crs)
        ax.scatter(pl.geometry.x, pl.geometry.y, s=10, color=dkp.INK, zorder=6)
        if labels:
            for _, p in pl.iterrows():
                ax.annotate(p["name"], (p.geometry.x, p.geometry.y), xytext=(3, 3),
                            textcoords="offset points", fontsize=8, color=dkp.INK, zorder=7)
    if extent is not None:
        b = _box(extent, crs).total_bounds
        ax.set_xlim(b[0], b[2])
        ax.set_ylim(b[1], b[3])
    ax.set_aspect("equal")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def scalebar(ax, km: float = 50, loc=(0.05, 0.06), color=dkp.INK):
    """Simple scale bar for a metric (projected) axis."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x0 + loc[0] * (x1 - x0)
    y = y0 + loc[1] * (y1 - y0)
    ax.plot([x, x + km * 1000], [y, y], color=color, lw=2.5, solid_capstyle="butt", zorder=8)
    ax.text(x + km * 500, y + 0.015 * (y1 - y0), f"{km:g} km", ha="center", va="bottom",
            fontsize=8, color=color, zorder=8)


def study_area_map(figsize=(13, 4.4)):
    """Two-panel locator: Austria with the Marchfeld, and the datasets' footprints."""
    import geopandas as gpd
    from shapely.geometry import box

    fig, (a0, a1) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.45, 1]})

    # --- overview -----------------------------------------------------------
    context(a0, extent=(9.3, 46.25, 17.4, 49.15), places=8, shade=True)
    _box(MARCHFELD["region"]).plot(ax=a0, facecolor=dkp.DRY, alpha=0.25, edgecolor=dkp.DRY,
                                   lw=1.5, zorder=6)
    gx = layer("austria_districts").query("name == 'Gänserndorf'").to_crs(AT_CRS)
    gx.boundary.plot(ax=a0, color=dkp.DRY, lw=0.8, zorder=6)
    rb = _box(MARCHFELD["region"]).total_bounds
    a0.annotate("Marchfeld", (rb[2], rb[3]), xytext=(18, 26), textcoords="offset points",
                fontsize=11, fontweight="bold", color=dkp.DRY,
                arrowprops=dict(arrowstyle="-", color=dkp.DRY, lw=0.8))
    scalebar(a0, 100)
    a0.set_title("Austria: federal states, rivers and the case-study region")

    # --- zoom: what each dataset sees ----------------------------------------
    context(a1, extent=MARCHFELD["region"], states=False, districts=True, places=2,
            river_width=1.8, neighbours=True, rivers=True)
    # ERA5 0.25 degree cells (the grid Open-Meteo samples)
    lon0, lat0, lon1, lat1 = MARCHFELD["region"]
    cells = [box(x, y, x + 0.25, y + 0.25)
             for x in np.arange(np.floor(lon0 * 4) / 4, lon1, 0.25)
             for y in np.arange(np.floor(lat0 * 4) / 4, lat1, 0.25)]
    gpd.GeoSeries(cells, crs=4326).to_crs(AT_CRS).boundary.plot(ax=a1, color=dkp.WET, lw=0.9,
                                                                linestyle="--", zorder=6)
    _box(MARCHFELD["bbox"]).plot(ax=a1, facecolor=dkp.OCHRE, alpha=0.25, edgecolor=dkp.OCHRE,
                                 lw=1.6, zorder=6)
    pt = gpd.GeoSeries.from_xy([MARCHFELD["lon"]], [MARCHFELD["lat"]], crs=4326).to_crs(AT_CRS)
    a1.scatter(pt.x, pt.y, marker="*", s=140, color=dkp.DRY, edgecolor="white", zorder=9)
    sb = _box(MARCHFELD["bbox"]).total_bounds
    a1.text(sb[0], sb[3] + 600, "Sentinel-2 / Landsat AOI", color="#8a5d12", fontsize=9,
            fontweight="bold", zorder=9)
    a1.text(pt.x.iloc[0] + 800, pt.y.iloc[0] - 2500, "ERA5 point", color=dkp.DRY, fontsize=9,
            fontweight="bold", zorder=9)
    a1.text(0.02, 0.97, "dashed: ERA5 0.25° cells (~19 × 28 km)\n"
            "MODIS pixels 250 m–1 km · Landsat 30 m · Sentinel-2 10–20 m",
            transform=a1.transAxes, fontsize=8, va="top", color=dkp.INK,
            bbox=dict(facecolor=dkp.PAPER, edgecolor="none", alpha=0.85), zorder=10)
    for name, xy in (("Danube", (16.52, 48.14)), ("March", (16.93, 48.33))):
        p = gpd.GeoSeries.from_xy([xy[0]], [xy[1]], crs=4326).to_crs(AT_CRS)
        a1.text(p.x.iloc[0], p.y.iloc[0], name, color="#3f7f86", fontsize=9, style="italic",
                rotation=-8 if name == "Danube" else 70, zorder=9)
    scalebar(a1, 10)
    a1.set_title("Marchfeld: one ERA5 cell, thousands of satellite pixels")
    for a in (a0, a1):
        a.set_xlabel("")
        a.set_ylabel("")
    fig.tight_layout()
    return fig
