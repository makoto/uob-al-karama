#!/usr/bin/env python3
"""
Generate season-specific Heat Mitigation Priority data.

Uses seasonal satellite LST data to compute priority scores that reflect
actual thermal conditions in summer vs winter.

Output: docs/data/al_karama/{season}/priority_points.json for each season
"""

import json
import os
import numpy as np
from scipy.spatial import cKDTree

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'docs', 'data', 'al_karama')

SEASONS = ['summer_2020', 'winter_2020', 'summer_2025', 'winter_2025']

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, separators=(',', ':'))

def compute_priority_scores(svi_data, sat_data):
    """Compute priority scores using seasonal satellite data.

    Args:
        svi_data: List of dicts with lat, lon, gvi, svf
        sat_data: List of dicts with lat, lon, lst, ndvi, ndbi

    Returns:
        List of dicts with priority scores added
    """
    # Build KD-tree for satellite data
    sat_coords = np.array([[d['lat'], d['lon']] for d in sat_data])
    sat_tree = cKDTree(sat_coords)

    # Extract satellite values as arrays for fast lookup
    sat_lst = np.array([d.get('lst', 0) or 0 for d in sat_data])
    sat_ndvi = np.array([d.get('ndvi', 0) or 0 for d in sat_data])
    sat_ndbi = np.array([d.get('ndbi', 0) or 0 for d in sat_data])

    # Find nearest satellite cell for each SVI point
    svi_coords = np.array([[d['lat'], d['lon']] for d in svi_data])
    _, indices = sat_tree.query(svi_coords, k=1)

    # Get values
    lst_vals = sat_lst[indices]
    ndvi_vals = sat_ndvi[indices]
    gvi_vals = np.array([d.get('gvi', 0) or 0 for d in svi_data])
    svf_vals = np.array([d.get('svf', 0) or 0 for d in svi_data])

    # Normalize to 0-1 scale
    def normalize(arr):
        min_val, max_val = arr.min(), arr.max()
        if max_val == min_val:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)

    lst_norm = normalize(lst_vals)        # Higher temp = higher priority
    gvi_norm = 1 - normalize(gvi_vals)    # Lower vegetation = higher priority
    ndvi_norm = 1 - normalize(ndvi_vals)  # Lower vegetation = higher priority
    svf_norm = normalize(svf_vals)        # Higher sky view = higher priority

    # Combined priority score (weighted average)
    # Weights: LST (40%), GVI (25%), NDVI (20%), SVF (15%)
    priority_scores = (
        0.40 * lst_norm +
        0.25 * gvi_norm +
        0.20 * ndvi_norm +
        0.15 * svf_norm
    )

    # Classify priority levels
    def get_priority_level(score):
        if score >= 0.75:
            return 'Critical'
        elif score >= 0.6:
            return 'High'
        elif score >= 0.4:
            return 'Medium'
        else:
            return 'Low'

    # Build output
    result = []
    for i, d in enumerate(svi_data):
        result.append({
            'lat': d['lat'],
            'lon': d['lon'],
            'lst': float(lst_vals[i]),
            'gvi': float(gvi_vals[i]),
            'ndvi': float(ndvi_vals[i]),
            'svf': float(svf_vals[i]),
            'priority_score': float(priority_scores[i]),
            'priority_level': get_priority_level(priority_scores[i])
        })

    return result

def main():
    print("=" * 60)
    print("GENERATING SEASONAL PRIORITY DATA")
    print("=" * 60)

    # Load SVI data (same for all seasons - it's street-level imagery)
    # Use combined_svi from any season as base (GVI, SVF don't change seasonally)
    svi_path = os.path.join(DATA_DIR, 'summer_2025', 'combined_svi.json')
    print(f"\nLoading SVI data from {svi_path}...")
    svi_data = load_json(svi_path)
    print(f"  Loaded {len(svi_data)} SVI points")

    for season in SEASONS:
        print(f"\n{'=' * 40}")
        print(f"Processing {season}")
        print('=' * 40)

        # Load seasonal satellite data
        sat_path = os.path.join(DATA_DIR, season, 'satellite_grid.json')
        if not os.path.exists(sat_path):
            print(f"  Warning: {sat_path} not found, skipping")
            continue

        sat_data = load_json(sat_path)
        print(f"  Loaded {len(sat_data)} satellite points")

        # Compute LST stats
        lsts = [d['lst'] for d in sat_data if d.get('lst')]
        print(f"  LST range: {min(lsts):.1f}°C - {max(lsts):.1f}°C (mean: {sum(lsts)/len(lsts):.1f}°C)")

        # Compute priority scores
        print("  Computing priority scores...")
        priority_data = compute_priority_scores(svi_data, sat_data)

        # Stats
        levels = {}
        for d in priority_data:
            level = d['priority_level']
            levels[level] = levels.get(level, 0) + 1

        print("  Priority distribution:")
        for level in ['Critical', 'High', 'Medium', 'Low']:
            count = levels.get(level, 0)
            pct = count / len(priority_data) * 100
            print(f"    {level}: {count} ({pct:.1f}%)")

        # Save
        out_path = os.path.join(DATA_DIR, season, 'priority_points.json')
        save_json(priority_data, out_path)
        print(f"  Saved: {out_path}")
        print(f"  Size: {os.path.getsize(out_path) / 1024:.0f} KB")

    print("\n" + "=" * 60)
    print("DONE - Priority data regenerated for all seasons")
    print("=" * 60)

if __name__ == '__main__':
    main()
