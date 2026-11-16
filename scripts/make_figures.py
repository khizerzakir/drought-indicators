"""Regenerate the figures used in the slides and the book (images/*.png).

    python scripts/make_figures.py

The study-area map always uses real boundaries. Satellite and climate figures
use the cached results in ``data/`` written by the notebooks when they were run
online; otherwise they use the offline synthetic stand-ins and say so on the
figure.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

import droughtkit as dk  # noqa: E402
from droughtkit import maps as dkm  # noqa: E402
from droughtkit import plotting as dkp  # noqa: E402
from droughtkit import remote_sensing as rs  # noqa: E402
from droughtkit import sample as sm  # noqa: E402

OUT = ROOT / "images"
OUT.mkdir(exist_ok=True)
DATA = ROOT / "data"
TARGET = "2026-07-01"
S2_BASE = (2017, 2025)
MODIS_BASE = (2001, 2020)

dkp.use_style()
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "font.family": ["Inter", "DejaVu Sans"]})


def save(fig, name, note=None):
    if note:
        dkp.source_note(fig, note)
    fig.savefig(OUT / name, dpi=170)
    plt.close(fig)
    print("wrote", OUT / name)


def bare(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel(""); ax.grid(False)
    ax.set_aspect("equal")


# ---- inputs -------------------------------------------------------------------
s2 = xr.open_dataset(DATA / "s2_marchfeld_monthly.nc").load() if (DATA / "s2_marchfeld_monthly.nc").exists() \
    else sm.sample_composites()
ls = xr.open_dataset(DATA / "landsat_marchfeld_monthly.nc").load() if (DATA / "landsat_marchfeld_monthly.nc").exists() \
    else sm.sample_landsat(s2)
modis = xr.open_dataset(DATA / "modis_east_austria_monthly.nc").load() \
    if (DATA / "modis_east_austria_monthly.nc").exists() else sm.sample_modis()
S2SRC, LSSRC, MSRC = (d.attrs.get("source", "") for d in (s2, ls, modis))
SYN_S2 = "synthetic" in S2SRC
print(S2SRC, "|", MSRC)

idx = dk.spectral_indices(s2)
extent = [float(s2.x.min()), float(s2.x.max()), float(s2.y.min()), float(s2.y.max())]

# ---- study area -----------------------------------------------------------------
save(dkm.study_area_map(), "study_area.png",
     "Boundaries: Natural Earth; Statistik Austria (CC BY 4.0). Relief: Natural Earth.")

fig, ax = plt.subplots(figsize=(10, 4.6))
dkm.context(ax, extent=(9.3, 46.25, 17.4, 49.15), places=0, shade=True, labels=False)
dkm._box(dkm.MARCHFELD["region"]).plot(ax=ax, facecolor=dkp.DRY, alpha=0.35, edgecolor=dkp.DRY,
                                       lw=1.5, zorder=6)
ax.set_xlabel(""); ax.set_ylabel("")
fig.patch.set_alpha(0); ax.set_facecolor("none")
fig.savefig(OUT / "austria_hero.png", dpi=170, transparent=True)
plt.close(fig)

# ---- compositing, step by step ----------------------------------------------------
if SYN_S2:
    scenes = sm.sample_scenes(2026, 7)
    clear = scenes.SCL.isin(rs.S2_CLEAR)
    masked = scenes[rs.COMMON].where(clear)
    comp = rs.monthly_composite(masked).isel(time=0)
    n = 4
    fig, ax = plt.subplots(2, n + 1, figsize=(16, 5.6))
    for i in range(n):
        ax[0, i].imshow(rs.rgb(scenes.isel(time=i))); ax[0, i].set_title(str(scenes.time.values[i])[:10], fontsize=11)
        ax[1, i].imshow(rs.rgb(masked.isel(time=i)))
        ax[1, i].set_title(f"after masking: {float(clear.isel(time=i).mean()):.0%} clear", fontsize=11)
    ax[0, n].imshow(rs.rgb(comp)); ax[0, n].set_title("July median", fontsize=11, fontweight="bold")
    m = ax[1, n].imshow(comp.n_obs, cmap=dkp.SAND_TEAL, vmin=0, vmax=6)
    ax[1, n].set_title("clear looks per pixel", fontsize=11)
    plt.colorbar(m, ax=ax[1, n], shrink=0.8)
    for a in ax.ravel():
        bare(a)
    save(fig, "compositing.png", scenes.attrs["source"])

# ---- the 2026 season in true colour ----------------------------------------------
year = s2.sel(time=s2.time.dt.year == 2026)
fig, ax = plt.subplots(1, year.sizes["time"], figsize=(3 * year.sizes["time"], 2.9))
for a, t in zip(np.atleast_1d(ax), year.time.values):
    a.imshow(rs.rgb(year.sel(time=t))); a.set_title(pd.Timestamp(t).strftime("%B"), fontsize=12); bare(a)
save(fig, "season_rgb.png", f"2026 monthly median composites · {S2SRC}")

# ---- indices for July -------------------------------------------------------------------
mon = idx.sel(time=TARGET)
fig, ax = plt.subplots(1, 4, figsize=(17, 3.9))
ax[0].imshow(rs.rgb(s2.sel(time=TARGET)), extent=extent); ax[0].set_title("true colour")
for a, (name, cmap, lo, hi) in zip(ax[1:], [("NDVI", dkp.SAND_TEAL, 0, 0.9),
                                           ("NDMI", dkp.DRY_WET, -0.3, 0.4),
                                           ("MSI", dkp.DRY_WET.reversed(), 0.3, 1.6)]):
    m = a.imshow(mon[name].values, extent=extent, cmap=cmap, vmin=lo, vmax=hi)
    a.set_title(name); plt.colorbar(m, ax=a, shrink=0.75)
for a in ax:
    bare(a)
save(fig, "s2_indices.png", f"July 2026 · {S2SRC}")

# ---- anomalies ------------------------------------------------------------------------
nd_z = dk.anomaly(idx.NDVI, S2_BASE, standardize=True)
nm_a = dk.anomaly(idx.NDMI, S2_BASE)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.1))
ax[0].imshow(rs.rgb(s2.sel(time=TARGET)), extent=extent); ax[0].set_title("true colour, July 2026")
m = ax[1].imshow(nd_z.sel(time=TARGET).values, extent=extent, cmap=dkp.DRY_WET, vmin=-3, vmax=3)
ax[1].set_title("NDVI z-score vs July 2017–2025"); plt.colorbar(m, ax=ax[1], shrink=0.75)
m = ax[2].imshow(nm_a.sel(time=TARGET).values, extent=extent, cmap=dkp.DRY_WET, vmin=-0.2, vmax=0.2)
ax[2].set_title("NDMI anomaly (canopy water)"); plt.colorbar(m, ax=ax[2], shrink=0.75)
for a in ax:
    bare(a)
save(fig, "s2_anomalies.png", S2SRC)

# ---- area under stress -------------------------------------------------------------------
share = dk.area_below(nd_z, -1.0)
tab = share.groupby([share.index.year, share.index.month]).first().unstack()
fig, ax = plt.subplots(figsize=(11, 3.8))
w = 0.13
for k, mth in enumerate(tab.columns):
    ax.bar(tab.index + (k - 2.5) * w, tab[mth], width=w, color=dkp.DRY_WET(0.08 + 0.14 * k),
           label=pd.Timestamp(2000, mth, 1).strftime("%b"))
ax.set(ylabel="% of area", title="Share of the study area with NDVI z-score below −1")
ax.legend(ncol=6, fontsize=9, loc="upper left")
save(fig, "area_stress.png", S2SRC)

# ---- seasonal cycle ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 3.8))
series = rs.aoi_median(idx.NDVI)
table = series.groupby([series.index.year, series.index.month]).first().unstack()
ref = table.loc[S2_BASE[0]:S2_BASE[1]]
mm = table.columns
ax.fill_between(mm, ref.min(), ref.max(), color=dkp.WET, alpha=0.15, label="2017–2025 range")
ax.plot(mm, ref.median(), color=dkp.WET, lw=1.6, label="reference median")
ax.plot(mm, table.loc[2026], "o-", color=dkp.DRY, lw=2.2, label="2026")
ax.set_xticks(mm, [pd.Timestamp(2000, k, 1).strftime("%b") for k in mm])
ax.set(title="Study-area median NDVI by month", ylabel="NDVI"); ax.legend(fontsize=9)
save(fig, "seasonal_cycle.png", S2SRC)

# ---- MODIS VCI / TCI / VHI ----------------------------------------------------------------
vci = dk.vci(modis.ndvi, MODIS_BASE)
tci = dk.tci(modis.lst, MODIS_BASE)
vhi = dk.vhi(vci, tci)
gx = dkm.layer("austria_districts").query("name == 'Gänserndorf'").to_crs(dkm.AT_CRS)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
for a, (name, da) in zip(ax, [("VCI", vci), ("TCI", tci), ("VHI", vhi)]):
    d = da.sel(time=TARGET)
    m = a.pcolormesh(d.x, d.y, d, cmap=dkp.DRY_WET, vmin=0, vmax=100, shading="nearest", rasterized=True)
    dkm.context(a, extent=dk.EAST_AUSTRIA_BBOX, places=4, fill_austria=False)
    gx.boundary.plot(ax=a, color=dkp.INK, lw=1.3, zorder=7)
    a.set_title(f"{name}, July 2026"); a.set_xlabel(""); a.set_ylabel("")
cb = fig.colorbar(m, ax=ax, shrink=0.7, pad=0.01); cb.outline.set_visible(False)
save(fig, "modis_vhi.png", f"{MSRC} · baseline 2001–2020 · outline: Gänserndorf district")

july = vhi.sel(time=vhi.time.dt.month == 7)
years = [y for y in [2003, 2015, 2017, 2018, 2022, 2026] if y in july.time.dt.year]
fig, ax = plt.subplots(1, len(years), figsize=(3.2 * len(years), 3))
for a, y in zip(ax, years):
    d = july.sel(time=f"{y}-07-01")
    a.pcolormesh(d.x, d.y, d, cmap=dkp.DRY_WET, vmin=0, vmax=100, shading="nearest", rasterized=True)
    dkm.context(a, extent=dk.EAST_AUSTRIA_BBOX, places=0, neighbours=False, fill_austria=False, rivers=False)
    a.set_title(f"July {y}", fontsize=12); a.set_xlabel(""); a.set_ylabel("")
save(fig, "modis_years.png", f"MODIS VHI · {MSRC}")

# ---- climate context ---------------------------------------------------------------------
clim = dk.load_climate(cache_dir=DATA)
spi3 = dk.spi(clim.precip_mm, 3, calibration=(1991, 2020))
spei3 = dk.spei(clim.precip_mm, clim.et0_mm, 3, calibration=(1991, 2020))
fig, ax = plt.subplots(2, 1, figsize=(11, 4.4), sharex=True)
dkp.index_bars(spi3["2001":], ax[0], title="SPI-3: rainfall")
dkp.index_bars(spei3["2001":], ax[1], title="SPEI-3: rainfall minus evaporative demand")
for a in ax:
    a.set_yticks([-2, 0, 2])
save(fig, "spi_spei.png", f"Marchfeld · {clim.attrs['source']}")

# ---- the single-season baseline mistake ----------------------------------------------------
from shapely import contains_xy  # noqa: E402

in_gx = xr.DataArray(contains_xy(gx.geometry.iloc[0], *np.meshgrid(modis.x, modis.y)),
                     coords={"y": modis.y, "x": modis.x}, dims=("y", "x"))
nd = modis.ndvi.where(in_gx).median(["x", "y"]).to_series()
yr = nd["2015"]
naive = 100 * (yr - yr.min()) / (yr.max() - yr.min())
proper = dk.vci(nd, MODIS_BASE)["2015"]
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.axhspan(0, 35, color=dkp.DRY, alpha=0.08)
ax.plot(naive.index, naive, "o-", color="#9aa19f", lw=2, label="min/max of one season")
ax.plot(proper.index, proper, "o-", color=dkp.WET, lw=3, label="min/max of the same month, 2001–2020")
ax.set(ylim=(-3, 103), ylabel="VCI", title="Marchfeld, 2015: two ways to scale the same NDVI")
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b"))
ax.legend(loc="upper right", fontsize=9)
save(fig, "vci_baseline.png", MSRC)
