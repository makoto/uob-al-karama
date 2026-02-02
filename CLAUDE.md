# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Al Karama Urban Digital Twin — a research/analysis project combining street-level imagery (Mapillary) and satellite data (Google Earth Engine) to assess urban climate, vegetation, building morphology, and sustainability metrics for the Al Karama district in Dubai. Results are served as static HTML viewers.

## Git Operations

- Do NOT run `git commit` or `git push` automatically
- Always let the user review changes and commit/push manually
- It's okay to stage files with `git add` and show `git status` or `git diff`

## Environment Setup

- Python environment: conda environment named `zensvi`
- Activate: `conda activate zensvi`
- Python path: `/opt/anaconda3/envs/zensvi/bin/python`
- No `requirements.txt` exists; dependencies are installed manually via pip
- External API keys: Mapillary token in `.env`, Google Earth Engine project `uobdubai`

## Running Scripts

All analysis scripts are in `scripts/` and run individually:

```bash
python scripts/<script_name>.py
```

There is no unified pipeline or build system. Scripts are run manually in a logical order based on data dependencies.

### Data Acquisition (run first)
- `download_al_karama_svi.py` — Download street-level imagery from Mapillary
- `download_satellite_images.py` — Download Sentinel-2/Landsat from Google Earth Engine

### Core Analysis
- `calculate_gvi.py` — Green View Index from street imagery
- `calculate_svf.py` — Sky View Factor
- `satellite_analysis.py` — LST, NDVI, built-up indices from satellite data
- `satellite_full_area.py` — Full area satellite processing
- `combined_analysis.py` — Overlay street-level + satellite metrics

### Specialized Analysis
- `shade_analysis.py` — Shadow/shade modeling
- `urban_climate_analysis.py` — Temperature patterns
- `network_analysis.py` (v2, v3 variants) — Urban network connectivity
- `walking_route_analysis.py` — Pedestrian routing
- `gee_building_heights.py`, `estimate_building_heights.py` — 3D building estimation
- `heat_mitigation_priorities.py` — Climate adaptation planning

### Data Quality
- `baseline_snapshot.py --tag before` — Capture data snapshot before changes
- `baseline_snapshot.py --tag after --compare` — Compare after changes

### Static Site (docs/)
```bash
cd docs && python -m http.server 8000
```
- `viewer.html` — 3D Digital Twin (Three.js)
- `viewer_2d.html` — 2D Printable maps
- `gvi_point_map.html` — Interactive GVI heatmap
- `docs/scripts/export_vector_maps.py` — Regenerate SVG/PDF map exports

## Architecture

### Data Flow
```
Mapillary API / Google Earth Engine / OSM
         ↓
   scripts/*.py  (Python analysis)
         ↓
   output/ & docs/data/  (GeoJSON, CSV, PNG, PLY)
         ↓
   docs/*.html  (Leaflet 2D maps, Three.js 3D viewer)
```

### Key Dependencies
- **Geospatial**: `geopandas`, `shapely`, `osmnx`, `momepy`
- **Remote sensing**: `ee` (Google Earth Engine)
- **Street-level imagery**: `zensvi` (custom library for GVI/SVF/segmentation)
- **Visualization**: `folium` (Leaflet maps), `matplotlib`
- **Data**: `pandas`, `numpy`, `scipy`, `networkx`, `sklearn`

### Input Data
- `input/Al_Karama.geojson` — Study area boundary
- `data/svi_images/` — Downloaded street view imagery
- `.env` — Mapillary API token (not committed)

### Output Data
- `output/` — Generated analysis results (not committed)
- `docs/data/` — Processed data served by web viewers
- `baselines/` — Data quality snapshots for comparison
