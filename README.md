# Drought Using Remote Sensing & Open Data

A reproducible tutorial on satellite-based drought monitoring. Sentinel-2 and Landsat **monthly median
composites** are turned into spectral indices and drought indicators; and we use **MODIS** that provides the long record for
condition indices; **ERA5** SPI and SPEI give the climate context. The case study is the **Marchfeld**,
Lower Austria.

- Tutorial: <https://khizerzakir.github.io/drought-indicators/> 

- Slides: <https://khizerzakir.github.io/drought-indicators/presentation.html>

| Step | Data | Result |
|---|---|---|
| Monthly median composites | Sentinel-2 L2A (20 m), Landsat 8/9 C2 L2 (30 m) | one clean image per month + clear-observation counts |
| Spectral indices | composites | NDVI, EVI, NDMI, MSI, NDWI, LST |
| Drought indicators | indices vs the same month in reference years | anomalies, z-scores, VCI, area under stress |
| Long-term condition | MODIS MOD13A1 / MOD11A2, 2001– | VCI, TCI, VHI for eastern Austria |
| Climate context | ERA5 via Open-Meteo | SPI, SPEI |

## Contents

```
index.qmd                 overview, Austrian open data, inspiration
tutorial.qmd              methods guide
presentation.qmd          lecture slides (theme in theme/slides.scss)
notebooks/
  01_monthly_composites.ipynb        cloudy scenes -> masked -> monthly median cube (Sentinel-2, Landsat)
  02_indices_from_composites.ipynb   indices, anomalies, z-scores, VCI/TCI/VHI, area under stress, maps
  03_modis_condition_indices.ipynb   MODIS VCI/TCI/VHI for eastern Austria + ERA5 SPI/SPEI context - currently under development
droughtkit/               the teaching library used everywhere
  remote_sensing.py       STAC loading, masking, harmonisation, monthly composites
  vegetation.py           spectral indices and drought indicators
  maps.py, geodata/       Austria / Marchfeld map context
  sample.py               synthetic offline stand-ins
  data.py, indices.py     ERA5 point series, SPI and SPEI
tests/                    pytest checks
scripts/make_figures.py   regenerates images/ (uses cached real data if present)
```

## Quick start

Requires Python 3.11+ and, to build the site, [Quarto](https://quarto.org/docs/get-started/).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q tests
jupyter lab notebooks/
```

```python
import droughtkit as dk
from droughtkit import remote_sensing as rs

s2 = rs.monthly_cube("sentinel2", dk.MARCHFELD["bbox"], years=range(2017, 2027), months=range(4, 10))
idx = dk.spectral_indices(s2)                                    # NDVI, EVI, NDMI, MSI, NDWI
ndvi_z = dk.anomaly(idx.NDVI, baseline=(2017, 2025), standardize=True)
stressed = dk.area_below(ndvi_z, -1)                             # % of the area per month
```

### Offline mode

Without internet access the notebooks use `droughtkit.sample`: a **synthetic** Marchfeld-like block of
fields (rainfed and irrigated crops) and a MODIS-like record for eastern Austria. Every figure states its
source. The outputs committed in `notebooks/` were produced in offline mode; run the notebooks online to
replace them with real Sentinel-2, Landsat, MODIS and ERA5 results, then run `python scripts/make_figures.py`.

## Build the site

```bash
quarto render                                   # book -> docs/
quarto render presentation.qmd --no-execute     # slides -> presentation.html
```

Pushing to `main` runs `.github/workflows/publish-quarto.yml`: install, run the tests, render the tutorial and
slides, and deploy to GitHub Pages. Notebooks are rendered from their stored outputs.

## Inspiration (references only, not dependencies)

### Github Repos:

- monocongo/climate_indices 
- TerziTB/PyDRGHT 
- bennyistanto/eddi 
- mintproject/droughtIndices
- Laura-Guillory/DroughtMonitor 

### Papers and others:

- EUMETSAT Drought & Vegetation Data Cube ·

- Di Paola et al. (2025), *Building a framework for a synoptic overview of drought*, Sci. Total Environ. 958, 177949.

## Data and licences

- Sentinel-2 (Copernicus), Landsat (USGS) and MODIS (NASA) via Microsoft Planetary Computer; 
- ERA5 (Copernicus C3S) via Open-Meteo (CC BY 4.0). 
- Map layers: Natural Earth (public domain);
- Austrian state and district boundaries from Statistik Austria via ginseng666/GeoJSON-TopoJSON-Austria (CC BY 4.0).
