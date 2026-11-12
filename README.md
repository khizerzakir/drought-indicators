# Drought indicators from remote sensing

This repository contains a Quarto-based tutorial for deriving drought indicators from
publicly available Sentinel-2 imagery.

It also includes two executable notebooks:

- `notebooks/01_sentinel2_drought_indices.ipynb` reproduces the remote-sensing workflow.
- `notebooks/02_drought_indicators_comparison.ipynb` compares VCI, TCI, and VHI using a
  reproducible synthetic example.

The Quarto book starts with a conceptual overview in `index.qmd` and continues with
the hands-on workflow in `tutorial.qmd`. The rendered site is configured for GitHub
Pages and is published automatically by `.github/workflows/publish-quarto.yml` when
changes are pushed to `main`.

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

The output is written to `docs/`, which is the artifact deployed to GitHub Pages. The
first render requires an internet connection and downloads the imagery needed for the
example AOI.

After enabling GitHub Pages for **GitHub Actions** in the repository settings, pushes
to `main` publish the book at:

<https://khizerzakir.github.io/drought-indicators/>
