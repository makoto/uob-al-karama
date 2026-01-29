#!/usr/bin/env python3
"""
Export viewer data: sets up the data directory structure for the unified 3D viewer.

Creates docs/data/al_karama/ with all required GeoJSON, JSON, and CSV files
converted from the various analysis output directories.

Run from the project root directory:
    python scripts/export_viewer_data.py
"""

import csv
import json
import os
import shutil
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def round_value(value, decimals=6):
    """Try to parse a value as a number and round it; return as-is if not numeric."""
    if isinstance(value, float):
        return round(value, decimals)
    if isinstance(value, str):
        try:
            f = float(value)
            # Preserve ints that were stored as strings (e.g. "3" -> 3)
            if f == int(f) and "." not in value and "e" not in value.lower():
                return int(f)
            return round(f, decimals)
        except (ValueError, OverflowError):
            return value
    return value


def parse_row(row, keep_fields=None):
    """Convert a CSV DictReader row to a dict with numeric parsing.

    Parameters
    ----------
    row : dict
        A single row from csv.DictReader.
    keep_fields : list[str] | None
        If provided, only these fields are included in the output.
    """
    out = {}
    for key, val in row.items():
        if keep_fields is not None and key not in keep_fields:
            continue
        out[key] = round_value(val)
    return out


def csv_to_json(src, dst, keep_fields=None):
    """Read a CSV file and write it as a JSON array of objects.

    Parameters
    ----------
    src : Path
        Source CSV path.
    dst : Path
        Destination JSON path.
    keep_fields : list[str] | None
        If provided, only these columns are included.
    """
    if not src.exists():
        print(f"  WARNING: source not found, skipping: {src}")
        return
    rows = []
    with open(src, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(parse_row(row, keep_fields=keep_fields))
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False)
    print(f"  Converted {src} -> {dst}  ({len(rows)} rows)")


def copy_file(src, dst):
    """Copy a file using shutil.copy2, with a warning if the source is missing."""
    if not src.exists():
        print(f"  WARNING: source not found, skipping: {src}")
        return
    shutil.copy2(src, dst)
    print(f"  Copied {src} -> {dst}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    root = Path(os.getcwd())
    docs = root / "docs"
    data_dir = docs / "data" / "al_karama"
    shadows_dir = data_dir / "shadows"

    # ------------------------------------------------------------------
    # 1. Create directory structure
    # ------------------------------------------------------------------
    print("Step 1: Creating directory structure ...")
    data_dir.mkdir(parents=True, exist_ok=True)
    shadows_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Created {data_dir}")
    print(f"  Created {shadows_dir}")

    # ------------------------------------------------------------------
    # 2. Copy existing files
    # ------------------------------------------------------------------
    print("\nStep 2: Copying existing files ...")

    copy_file(
        docs / "osm_3d" / "al_karama_buildings.geojson",
        data_dir / "buildings.geojson",
    )
    copy_file(
        docs / "shade_analysis" / "canopy_polygons_with_height.geojson",
        data_dir / "canopy.geojson",
    )
    copy_file(
        docs / "network_analysis" / "street_network.geojson",
        data_dir / "streets.geojson",
    )
    copy_file(
        docs / "shade_analysis" / "pois_cache.json",
        data_dir / "pois.json",
    )
    copy_file(
        docs / "shade_analysis" / "street_shade.csv",
        data_dir / "street_shade.csv",
    )

    # Shadow GeoJSON files for hours 06 through 18 (even hours)
    for hour in range(6, 19, 2):
        tag = f"{hour:02d}"
        copy_file(
            docs / "shade_analysis" / f"shadows_{tag}.geojson",
            shadows_dir / f"shadows_{tag}.geojson",
        )

    # ------------------------------------------------------------------
    # 3. Convert CSV files to JSON
    # ------------------------------------------------------------------
    print("\nStep 3: Converting CSV files to JSON ...")

    # 3a) street_metrics
    csv_to_json(
        docs / "network_analysis" / "street_metrics.csv",
        data_dir / "street_metrics.json",
    )

    # 3b) segment_comfort
    csv_to_json(
        docs / "walking_routes" / "segment_comfort.csv",
        data_dir / "segment_comfort.json",
    )

    # 3c) priority_points (subset of columns)
    csv_to_json(
        docs / "heat_mitigation" / "priority_scores.csv",
        data_dir / "priority_points.json",
        keep_fields=[
            "lon", "lat", "svf", "gvi", "lst", "ndvi", "ndbi",
            "priority_score", "priority_level",
        ],
    )

    # 3d) clusters
    csv_to_json(
        docs / "quick_analysis" / "clusters.csv",
        data_dir / "clusters.json",
    )

    # 3e) distance_to_green
    csv_to_json(
        docs / "quick_analysis" / "distance_to_green.csv",
        data_dir / "distance_to_green.json",
    )

    # 3f) combined_svi (subset of columns)
    csv_to_json(
        docs / "combined_analysis" / "combined_data.csv",
        data_dir / "combined_svi.json",
        keep_fields=["lon", "lat", "svf", "gvi", "lst", "ndvi_satellite"],
    )

    # ------------------------------------------------------------------
    # 4. Generate area.json manifest
    # ------------------------------------------------------------------
    print("\nStep 4: Generating area.json manifest ...")

    area_manifest = {
        "id": "al_karama",
        "name": "Al Karama",
        "city": "Dubai, UAE",
        "center": [55.3025, 25.2485],
        "zoom": 15,
        "analysis_date": "2024-07-15",
        "timezone_offset": 4,
        "times": [6, 8, 10, 12, 14, 16, 18],
        "time_labels": [
            "6:00 AM", "8:00 AM", "10:00 AM", "12:00 PM",
            "2:00 PM", "4:00 PM", "6:00 PM",
        ],
        "layers": {
            "buildings": "buildings.geojson",
            "canopy": "canopy.geojson",
            "streets": "streets.geojson",
            "pois": "pois.json",
            "street_shade": "street_shade.csv",
            "street_metrics": "street_metrics.json",
            "segment_comfort": "segment_comfort.json",
            "priority": "priority_points.json",
            "clusters": "clusters.json",
            "distance_to_green": "distance_to_green.json",
            "combined_svi": "combined_svi.json",
            "shadows": {
                "06": "shadows/shadows_06.geojson",
                "08": "shadows/shadows_08.geojson",
                "10": "shadows/shadows_10.geojson",
                "12": "shadows/shadows_12.geojson",
                "14": "shadows/shadows_14.geojson",
                "16": "shadows/shadows_16.geojson",
                "18": "shadows/shadows_18.geojson",
            },
        },
        "sun_info": {
            "6":  {"altitude": 3.7,  "azimuth": 68.0,  "shadow_factor": 15.32},
            "8":  {"altitude": 29.7, "azimuth": 78.7,  "shadow_factor": 1.75},
            "10": {"altitude": 56.6, "azimuth": 88.9,  "shadow_factor": 0.66},
            "12": {"altitude": 83.1, "azimuth": 122.6, "shadow_factor": 0.12},
            "14": {"altitude": 67.9, "azimuth": 264.9, "shadow_factor": 0.41},
            "16": {"altitude": 40.8, "azimuth": 277.2, "shadow_factor": 1.16},
            "18": {"altitude": 14.3, "azimuth": 287.2, "shadow_factor": 3.93},
        },
        "utc_timestamps": {
            "6":  "2024-07-15T02:00:00Z",
            "8":  "2024-07-15T04:00:00Z",
            "10": "2024-07-15T06:00:00Z",
            "12": "2024-07-15T08:00:00Z",
            "14": "2024-07-15T10:00:00Z",
            "16": "2024-07-15T12:00:00Z",
            "18": "2024-07-15T14:00:00Z",
        },
        "stats": {
            "buildings": 3243,
            "trees": 1033,
            "streets": 5656,
            "pois": 1290,
            "avg_shade": 35.8,
            "avg_lst": 48.6,
        },
    }

    area_path = data_dir / "area.json"
    with open(area_path, "w", encoding="utf-8") as fh:
        json.dump(area_manifest, fh, indent=2, ensure_ascii=False)
    print(f"  Written {area_path}")

    # ------------------------------------------------------------------
    # 5. Generate areas.json (top-level index)
    # ------------------------------------------------------------------
    print("\nStep 5: Generating areas.json ...")

    areas_index = [
        {
            "id": "al_karama",
            "name": "Al Karama",
            "city": "Dubai, UAE",
            "center": [55.3025, 25.2485],
        }
    ]

    areas_path = docs / "data" / "areas.json"
    with open(areas_path, "w", encoding="utf-8") as fh:
        json.dump(areas_index, fh, indent=2, ensure_ascii=False)
    print(f"  Written {areas_path}")

    print("\nDone. Data directory is ready at:", data_dir)


if __name__ == "__main__":
    main()
