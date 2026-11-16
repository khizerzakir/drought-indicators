"""droughtkit: small, readable helpers for satellite-based drought monitoring.

Main path (satellite):
    remote_sensing  Sentinel-2, Landsat and MODIS via STAC -> monthly median composites
    vegetation      spectral indices (NDVI, EVI, NDMI, MSI, NDWI) and drought indicators
                    (anomalies, z-scores, VCI, TCI, VHI, area under stress)
    maps            Austria / Marchfeld map context and study-area map
    sample          synthetic offline stand-ins for all satellite inputs

Climate context:
    data, indices   ERA5 point series and SPI / SPEI
"""

from .data import load_climate, offline_sample
from .indices import accumulate, classify_standardized, spei, spi, standardized_empirical
from .maps import EAST_AUSTRIA_BBOX, MARCHFELD
from .vegetation import (anomaly, area_below, classify_vhi, condition_index, monthly_climatology,
                         spectral_indices, tci, vci, vhi)

__all__ = [
    "load_climate", "offline_sample",
    "accumulate", "classify_standardized", "spei", "spi", "standardized_empirical",
    "EAST_AUSTRIA_BBOX", "MARCHFELD",
    "anomaly", "area_below", "classify_vhi", "condition_index", "monthly_climatology",
    "spectral_indices", "tci", "vci", "vhi",
]

__version__ = "0.4.0"
