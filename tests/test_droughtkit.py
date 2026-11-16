"""Sanity checks for droughtkit. Run with:  pytest -q"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import droughtkit as dk
from droughtkit import remote_sensing as rs
from droughtkit import sample as sm

CAL = (1971, 2020)


@pytest.fixture(scope="module")
def clim():
    return dk.offline_sample()


@pytest.fixture(scope="module")
def comp():
    return sm.sample_composites()


# ---- satellite --------------------------------------------------------------

def test_spectral_indices_formulas():
    ds = xr.Dataset({b: ("x", np.array([v])) for b, v in
                     dict(blue=0.03, green=0.07, red=0.03, nir=0.42, swir1=0.19, swir2=0.08).items()})
    ix = dk.spectral_indices(ds).isel(x=0)
    assert float(ix.NDVI) == pytest.approx((0.42 - 0.03) / 0.45)
    assert float(ix.NDMI) == pytest.approx((0.42 - 0.19) / 0.61)
    assert float(ix.MSI) == pytest.approx(0.19 / 0.42)
    assert float(ix.EVI) == pytest.approx(2.5 * 0.39 / (0.42 + 0.18 - 0.225 + 1))


def test_monthly_composite_ignores_masked_pixels():
    t = pd.to_datetime(["2026-07-03", "2026-07-13", "2026-07-23", "2026-08-02"])
    red = xr.DataArray([[0.05], [np.nan], [0.07], [0.04]], dims=("time", "x"), coords={"time": t})
    comp = rs.monthly_composite(xr.Dataset({"red": red})).isel(x=0)
    assert float(comp.red.sel(time="2026-07-01")) == pytest.approx(0.06)
    assert int(comp.n_obs.sel(time="2026-07-01")) == 2


def test_composite_scenes_remove_clouds():
    scenes = sm.sample_scenes()
    clear = scenes[["red", "nir"]].where(scenes.SCL.isin(rs.S2_CLEAR))
    med = clear.median("time")
    assert float(med.red.max()) < 0.3          # no bright cloud left in the median
    assert float(scenes.red.max()) > 0.4       # but individual scenes had clouds


def test_drought_signal_rainfed_vs_irrigated(comp):
    ix = dk.spectral_indices(comp)
    z = dk.anomaly(ix.NDVI, baseline=(2017, 2025), standardize=True).sel(time="2026-07-01")
    par = sm.sample_parcels()
    arable = par.crop < 5
    rainfed = float(z.where(arable & ~par.irrigated).median())
    irrigated = float(z.where(arable & par.irrigated).median())
    assert rainfed < -1
    assert irrigated > rainfed + 0.5


def test_vci_uses_monthly_baseline():
    idx = pd.date_range("2001-01-01", "2010-12-01", freq="MS")
    season = 0.3 + 0.4 * np.sin(np.pi * (idx.month - 1) / 11)
    ndvi = pd.Series(season, index=idx) + np.linspace(0, 0.05, len(idx))
    v = dk.vci(ndvi)
    assert v.groupby(v.index.month).min().max() == pytest.approx(0)
    assert v.groupby(v.index.month).max().min() == pytest.approx(100)


def test_modis_vhi_detects_2026(clim):
    m = sm.sample_modis(years=range(2001, 2027), months=[7])
    v = dk.vhi(dk.vci(m.ndvi, (2001, 2020)), dk.tci(m.lst, (2001, 2020)))
    assert float(v.sel(time="2026-07-01").mean()) < 40
    share = dk.area_below(v, 40)
    assert share["2026-07-01"] > 50


# ---- climate context ----------------------------------------------------------

def test_spi_is_standard_normal_in_calibration(clim):
    s = dk.spi(clim.precip_mm, 3, calibration=CAL)
    ref = s[(s.index.year >= CAL[0]) & (s.index.year <= CAL[1])].dropna()
    assert abs(ref.mean()) < 0.1
    assert 0.9 < ref.std() < 1.1


def test_spei_known_drought(clim):
    s = dk.spei(clim.precip_mm, clim.et0_mm, 3, calibration=CAL)
    assert s.notna().sum() > 700
    assert s["2026-07-01"] < -1.5


def test_study_area_map(tmp_path):
    pytest.importorskip("geopandas")
    from droughtkit import maps
    maps.study_area_map().savefig(tmp_path / "map.png")
