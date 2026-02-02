#!/usr/bin/env python3
"""
Export viewer data: sets up the data directory structure for the unified 3D viewer.

Creates docs/data/al_karama/ with all required GeoJSON, JSON, and CSV files
converted from the various analysis output directories.

Supports seasonal data layout:
    python scripts/export_viewer_data.py --season summer_2025
    python scripts/export_viewer_data.py --season winter_2025
    python scripts/export_viewer_data.py --season all

Run from the project root directory.
"""

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path

from seasons_config import SEASONS, DEFAULT_SEASON, get_season_config


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
            if f == int(f) and "." not in value and "e" not in value.lower():
                return int(f)
            return round(f, decimals)
        except (ValueError, OverflowError):
            return value
    return value


def parse_row(row, keep_fields=None):
    """Convert a CSV DictReader row to a dict with numeric parsing."""
    out = {}
    for key, val in row.items():
        if keep_fields is not None and key not in keep_fields:
            continue
        out[key] = round_value(val)
    return out


def csv_to_json(src, dst, keep_fields=None):
    """Read a CSV file and write it as a JSON array of objects."""
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
# Solar position (duplicated from shade_analysis to avoid heavy imports)
# ---------------------------------------------------------------------------

def solar_position(year, month, day, hour_utc, lat, lon):
    """NOAA solar position algorithm — returns (altitude, azimuth) in degrees."""
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    jdn = day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    jd = jdn + (hour_utc - 12.0) / 24.0
    jc = (jd - 2451545.0) / 36525.0
    L0 = (280.46646 + jc * (36000.76983 + 0.0003032 * jc)) % 360
    M = (357.52911 + jc * (35999.05029 - 0.0001537 * jc)) % 360
    M_rad = math.radians(M)
    C = (math.sin(M_rad) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
         + math.sin(2 * M_rad) * (0.019993 - 0.000101 * jc)
         + math.sin(3 * M_rad) * 0.000289)
    sun_lon = L0 + C
    omega = 125.04 - 1934.136 * jc
    sun_app_lon = sun_lon - 0.00569 - 0.00478 * math.sin(math.radians(omega))
    obliq_mean = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    obliq_corr = obliq_mean + 0.00256 * math.cos(math.radians(omega))
    obliq_rad = math.radians(obliq_corr)
    sin_dec = math.sin(obliq_rad) * math.sin(math.radians(sun_app_lon))
    dec = math.asin(sin_dec)
    tan_half_obliq = math.tan(obliq_rad / 2)
    y_eq = tan_half_obliq ** 2
    L0_rad = math.radians(L0)
    eqt = 4 * math.degrees(
        y_eq * math.sin(2 * L0_rad)
        - 2 * 0.016708634 * math.sin(M_rad)
        + 4 * 0.016708634 * y_eq * math.sin(M_rad) * math.cos(2 * L0_rad)
        - 0.5 * y_eq * y_eq * math.sin(4 * L0_rad)
        - 1.25 * 0.016708634 ** 2 * math.sin(2 * M_rad)
    )
    time_offset = eqt + 4 * lon
    tst = hour_utc * 60 + time_offset
    ha = (tst / 4) - 180
    ha_rad = math.radians(ha)
    lat_rad = math.radians(lat)
    sin_alt = (math.sin(lat_rad) * math.sin(dec)
               + math.cos(lat_rad) * math.cos(dec) * math.cos(ha_rad))
    altitude = math.degrees(math.asin(max(-1, min(1, sin_alt))))
    cos_az = (math.sin(dec) - math.sin(lat_rad) * sin_alt) / (
        math.cos(lat_rad) * math.cos(math.radians(altitude)))
    cos_az = max(-1, min(1, cos_az))
    azimuth = math.degrees(math.acos(cos_az))
    if ha > 0:
        azimuth = 360 - azimuth
    return altitude, azimuth


def compute_sun_info(season_cfg, lat=25.2485, lon=55.3025, tz_offset=4):
    """Compute sun_info and utc_timestamps dicts for a season."""
    year, month, day = season_cfg["solar_date"]
    times = [6, 8, 10, 12, 14, 16, 18]
    sun_info = {}
    utc_timestamps = {}
    for h in times:
        hour_utc = h - tz_offset
        alt, az = solar_position(year, month, day, hour_utc, lat, lon)
        if alt > 0:
            sf = round(1.0 / math.tan(math.radians(alt)), 2)
        else:
            sf = None  # sun below horizon
        sun_info[str(h)] = {
            "altitude": round(alt, 1),
            "azimuth": round(az, 1),
            "shadow_factor": sf,
        }
        utc_timestamps[str(h)] = f"{year:04d}-{month:02d}-{day:02d}T{hour_utc:02d}:00:00Z"
    return sun_info, utc_timestamps


# ---------------------------------------------------------------------------
# Season-independent data export
# ---------------------------------------------------------------------------

def export_common(docs, data_dir):
    """Copy season-independent files to data_dir root."""
    print("\n  Copying season-independent files ...")

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

    # Street metrics & segment comfort (season-independent network data)
    csv_to_json(
        docs / "network_analysis" / "street_metrics.csv",
        data_dir / "street_metrics.json",
    )
    csv_to_json(
        docs / "walking_routes" / "segment_comfort.csv",
        data_dir / "segment_comfort.json",
    )


# ---------------------------------------------------------------------------
# Season-specific data export
# ---------------------------------------------------------------------------

def export_season(docs, data_dir, season_id):
    """Copy/convert season-specific files into data_dir/<season_id>/."""
    season_cfg = get_season_config(season_id)
    season_dir = data_dir / season_id
    shadows_dir = season_dir / "shadows"

    print(f"\n  Exporting season: {season_cfg['label']} -> {season_dir}")
    season_dir.mkdir(parents=True, exist_ok=True)
    shadows_dir.mkdir(parents=True, exist_ok=True)

    # Shade analysis outputs (season-specific output dir)
    shade_src = docs / "shade_analysis" / season_id
    copy_file(shade_src / "street_shade.csv", season_dir / "street_shade.csv")

    for hour in range(6, 19, 2):
        tag = f"{hour:02d}"
        copy_file(
            shade_src / f"shadows_{tag}.geojson",
            shadows_dir / f"shadows_{tag}.geojson",
        )

    # Combined SVI + satellite — check multiple source locations
    root = Path(os.getcwd())
    combined_candidates = [
        root / "output" / "combined_analysis" / season_id / "combined_data.csv",
        docs / "combined_analysis" / season_id / "combined_data.csv",
        root / "output" / "combined_analysis" / "combined_data.csv",
        docs / "combined_analysis" / "combined_data.csv",
    ]
    combined_src = next((p for p in combined_candidates if p.exists()), None)
    if combined_src:
        csv_to_json(
            combined_src,
            season_dir / "combined_svi.json",
            keep_fields=["lon", "lat", "svf", "gvi", "lst", "ndvi_satellite"],
        )
    else:
        print(f"  WARNING: combined_data.csv not found for {season_id}")

    # Satellite grid (full area)
    root = Path(os.getcwd())
    full_csv = root / "output" / "satellite_full" / season_id / "full_area_data.csv"
    if full_csv.exists():
        csv_to_json(full_csv, season_dir / "satellite_grid.json")

    # Priority points
    csv_to_json(
        docs / "heat_mitigation" / "priority_scores.csv",
        season_dir / "priority_points.json",
        keep_fields=[
            "lon", "lat", "svf", "gvi", "lst", "ndvi", "ndbi",
            "priority_score", "priority_level",
        ],
    )

    # Clusters
    csv_to_json(
        docs / "quick_analysis" / "clusters.csv",
        season_dir / "clusters.json",
    )

    # Distance to green
    csv_to_json(
        docs / "quick_analysis" / "distance_to_green.csv",
        season_dir / "distance_to_green.json",
    )


# ---------------------------------------------------------------------------
# Area manifest generation
# ---------------------------------------------------------------------------

def generate_area_json(data_dir, season_ids):
    """Generate the enhanced area.json with seasons metadata."""
    print("\n  Generating area.json manifest ...")

    # Season-independent layer paths (relative to data_dir)
    common_layers = {
        "boundary": "boundary.geojson",
        "buildings": "buildings.geojson",
        "canopy": "canopy.geojson",
        "streets": "streets.geojson",
        "pois": "pois.json",
        "street_metrics": "street_metrics.json",
        "segment_comfort": "segment_comfort.json",
    }

    # Season-dependent layer keys — paths resolved by prepending season data_path
    season_layer_keys = {
        "street_shade": "street_shade.csv",
        "combined_svi": "combined_svi.json",
        "satellite_grid": "satellite_grid.json",
        "priority": "priority_points.json",
        "clusters": "clusters.json",
        "distance_to_green": "distance_to_green.json",
        "shadows": {
            "06": "shadows/shadows_06.geojson",
            "08": "shadows/shadows_08.geojson",
            "10": "shadows/shadows_10.geojson",
            "12": "shadows/shadows_12.geojson",
            "14": "shadows/shadows_14.geojson",
            "16": "shadows/shadows_16.geojson",
            "18": "shadows/shadows_18.geojson",
        },
    }

    # Build seasons object
    seasons = {}
    for sid in season_ids:
        cfg = get_season_config(sid)
        sun_info, utc_timestamps = compute_sun_info(cfg)
        seasons[sid] = {
            "label": cfg["label"],
            "analysis_date": "{:04d}-{:02d}-{:02d}".format(*cfg["solar_date"]),
            "data_path": sid + "/",
            "sun_info": sun_info,
            "utc_timestamps": utc_timestamps,
            "layers": season_layer_keys,
            "stats": {},  # will be populated when data exists
        }

    area_manifest = {
        "id": "al_karama",
        "name": "Al Karama",
        "city": "Dubai, UAE",
        "center": [55.3025, 25.2485],
        "zoom": 15,
        "timezone_offset": 4,
        "times": [6, 8, 10, 12, 14, 16, 18],
        "time_labels": [
            "6:00 AM", "8:00 AM", "10:00 AM", "12:00 PM",
            "2:00 PM", "4:00 PM", "6:00 PM",
        ],
        "available_seasons": list(season_ids),
        "default_season": DEFAULT_SEASON,
        "layers": common_layers,
        "seasons": seasons,
    }

    area_path = data_dir / "area.json"
    with open(area_path, "w", encoding="utf-8") as fh:
        json.dump(area_manifest, fh, indent=2, ensure_ascii=False)
    print(f"  Written {area_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export viewer data")
    parser.add_argument(
        "--season", default="all",
        help="Season id to export (e.g. summer_2025, winter_2025, or 'all')",
    )
    args = parser.parse_args()

    if args.season == "all":
        season_ids = list(SEASONS.keys())
    else:
        season_ids = [args.season]
        # Validate
        for sid in season_ids:
            get_season_config(sid)

    root = Path(os.getcwd())
    docs = root / "docs"
    data_dir = docs / "data" / "al_karama"

    # ------------------------------------------------------------------
    # 1. Create directory structure
    # ------------------------------------------------------------------
    print("Step 1: Creating directory structure ...")
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Created {data_dir}")

    # ------------------------------------------------------------------
    # 2. Season-independent files
    # ------------------------------------------------------------------
    print("\nStep 2: Season-independent files ...")
    export_common(docs, data_dir)

    # ------------------------------------------------------------------
    # 3. Season-specific files
    # ------------------------------------------------------------------
    print("\nStep 3: Season-specific files ...")
    for sid in season_ids:
        export_season(docs, data_dir, sid)

    # ------------------------------------------------------------------
    # 4. Generate area.json manifest
    # ------------------------------------------------------------------
    print("\nStep 4: Generating area.json manifest ...")
    generate_area_json(data_dir, season_ids)

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
