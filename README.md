# Drought indicators from remote sensing

This repository contains a Quarto-based tutorial for deriving drought indicators from
publicly available Sentinel-2 imagery.

## Prerequisites

- [Quarto](https://quarto.org/docs/get-started/)
- Python 3.10 or newer

Create an environment and install the Python dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Render the tutorial:

```bash
quarto render
```

The notebook queries the public Microsoft Planetary Computer STAC API. The first render
requires an internet connection and downloads the imagery needed for the example AOI.
