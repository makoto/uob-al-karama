# Al Karama Urban Digital Twin

Interactive visualizations and analysis layers for the Al Karama district.

## Setup

This is a static site — no build step or dependencies required.

### Local development server

```bash
cd docs
python -m http.server 8000
```

Then open http://localhost:8000.

### Pages

- `/` — Landing page
- `/viewer.html` — 3D Digital Twin viewer
- `/viewer_2d.html` — 2D Printable Viewer
- `/gvi_point_map.html` — GVI point map
- `/segmentation_gallery.html` — Segmentation gallery
- `/learning_report.html` — Learning report

## Exporting vector maps

To regenerate the individual SVG/PDF exports (with basemap tiles):

```bash
cd docs
python scripts/export_vector_maps.py
```

Output files are written to `docs/data/al_karama/exports/`.
