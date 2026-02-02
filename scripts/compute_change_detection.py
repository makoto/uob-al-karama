#!/usr/bin/env python3
"""Compute 5-year change detection between Summer 2020 and Summer 2025 satellite grids.

Reads the 30 m satellite grids for both years, validates alignment, computes
per-point deltas (LST, NDVI, NDBI, NDWI), and writes a combined JSON file
for use in the web viewers.
"""

import json
import os
import statistics

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "docs", "data", "al_karama")

INPUT_2020 = os.path.join(DATA_DIR, "summer_2020", "satellite_grid.json")
INPUT_2025 = os.path.join(DATA_DIR, "summer_2025", "satellite_grid.json")
OUTPUT = os.path.join(DATA_DIR, "change_summer_5yr.json")


def main():
    # Load data
    print(f"Loading {INPUT_2020} ...")
    with open(INPUT_2020) as f:
        grid_2020 = json.load(f)
    print(f"  → {len(grid_2020):,} points")

    print(f"Loading {INPUT_2025} ...")
    with open(INPUT_2025) as f:
        grid_2025 = json.load(f)
    print(f"  → {len(grid_2025):,} points")

    # Validate alignment
    assert len(grid_2020) == len(grid_2025), (
        f"Grid lengths differ: {len(grid_2020)} vs {len(grid_2025)}"
    )
    # Spot-check first and last coordinates
    for idx in [0, -1]:
        assert grid_2020[idx]["lat"] == grid_2025[idx]["lat"], (
            f"Lat mismatch at index {idx}"
        )
        assert grid_2020[idx]["lon"] == grid_2025[idx]["lon"], (
            f"Lon mismatch at index {idx}"
        )
    print("Grid alignment validated ✓")

    # Compute deltas
    metrics = ["lst", "ndvi", "ndbi", "ndwi"]
    result = []
    deltas = {m: [] for m in metrics}

    for p20, p25 in zip(grid_2020, grid_2025):
        point = {
            "lat": round(p20["lat"], 6),
            "lon": round(p20["lon"], 6),
        }
        for m in metrics:
            v20 = p20.get(m, 0) or 0
            v25 = p25.get(m, 0) or 0
            d = v25 - v20
            point[f"{m}_2020"] = round(v20, 4)
            point[f"{m}_2025"] = round(v25, 4)
            point[f"d_{m}"] = round(d, 4)
            deltas[m].append(d)
        result.append(point)

    # Write output
    print(f"\nWriting {OUTPUT} ...")
    with open(OUTPUT, "w") as f:
        json.dump(result, f, separators=(",", ":"))
    size_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
    print(f"  → {len(result):,} points, {size_mb:.1f} MB")

    # Summary statistics
    print("\n── Summary Statistics ──")
    for m in metrics:
        vals = deltas[m]
        mn = min(vals)
        mx = max(vals)
        avg = statistics.mean(vals)
        med = statistics.median(vals)
        sd = statistics.stdev(vals)
        print(f"  Δ{m.upper():4s}  min={mn:+.4f}  max={mx:+.4f}  "
              f"mean={avg:+.4f}  median={med:+.4f}  sd={sd:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
