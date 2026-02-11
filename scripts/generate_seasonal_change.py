#!/usr/bin/env python3
"""
Generate Seasonal Change data (Summer 2025 → Winter 2025).

Compares satellite metrics (LST, NDVI, NDBI) and priority scores between
summer and winter of the same year to show seasonal variation.

Output: docs/data/al_karama/change_seasonal.json
"""

import json
import os
import numpy as np
from scipy.spatial import cKDTree

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'docs', 'data', 'al_karama')

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, separators=(',', ':'))

def main():
    print("=" * 60)
    print("GENERATING SEASONAL CHANGE DATA")
    print("Summer 2025 → Winter 2025")
    print("=" * 60)

    # Load summer 2025 satellite data
    summer_sat_path = os.path.join(DATA_DIR, 'summer_2025', 'satellite_grid.json')
    print(f"\nLoading summer satellite data...")
    summer_sat = load_json(summer_sat_path)
    print(f"  Loaded {len(summer_sat)} points")

    # Load winter 2025 satellite data
    winter_sat_path = os.path.join(DATA_DIR, 'winter_2025', 'satellite_grid.json')
    print(f"Loading winter satellite data...")
    winter_sat = load_json(winter_sat_path)
    print(f"  Loaded {len(winter_sat)} points")

    # Load priority data
    summer_pri_path = os.path.join(DATA_DIR, 'summer_2025', 'priority_points.json')
    winter_pri_path = os.path.join(DATA_DIR, 'winter_2025', 'priority_points.json')
    print(f"Loading priority data...")
    summer_pri = load_json(summer_pri_path)
    winter_pri = load_json(winter_pri_path)
    print(f"  Summer priority: {len(summer_pri)} points")
    print(f"  Winter priority: {len(winter_pri)} points")

    # Build KD-tree for winter satellite data to match with summer
    winter_coords = np.array([[d['lat'], d['lon']] for d in winter_sat])
    winter_tree = cKDTree(winter_coords)

    # Build KD-tree for priority data
    summer_pri_coords = np.array([[d['lat'], d['lon']] for d in summer_pri])
    winter_pri_coords = np.array([[d['lat'], d['lon']] for d in winter_pri])
    winter_pri_tree = cKDTree(winter_pri_coords)

    # Extract winter values as arrays
    winter_lst = np.array([d.get('lst') or 0 for d in winter_sat])
    winter_ndvi = np.array([d.get('ndvi') or 0 for d in winter_sat])
    winter_ndbi = np.array([d.get('ndbi') or 0 for d in winter_sat])

    winter_priority = np.array([d.get('priority_score') or 0 for d in winter_pri])

    print("\nComputing seasonal changes...")

    result = []
    stats = {
        'd_lst': [], 'd_ndvi': [], 'd_ndbi': [], 'd_priority': []
    }

    for i, d in enumerate(summer_sat):
        lat, lon = d['lat'], d['lon']
        summer_lst = d.get('lst')
        summer_ndvi = d.get('ndvi')
        summer_ndbi = d.get('ndbi')

        # Find matching winter point
        _, idx = winter_tree.query([lat, lon])

        # Compute deltas (winter - summer)
        d_lst = None
        d_ndvi = None
        d_ndbi = None

        if summer_lst is not None and winter_lst[idx] != 0:
            d_lst = float(winter_lst[idx] - summer_lst)
            stats['d_lst'].append(d_lst)

        if summer_ndvi is not None and winter_ndvi[idx] != 0:
            d_ndvi = float(winter_ndvi[idx] - summer_ndvi)
            stats['d_ndvi'].append(d_ndvi)

        if summer_ndbi is not None and winter_ndbi[idx] != 0:
            d_ndbi = float(winter_ndbi[idx] - summer_ndbi)
            stats['d_ndbi'].append(d_ndbi)

        # Store satellite change point
        result.append({
            'lat': lat,
            'lon': lon,
            'lst_summer': summer_lst,
            'lst_winter': float(winter_lst[idx]) if winter_lst[idx] != 0 else None,
            'd_lst': round(d_lst, 2) if d_lst is not None else None,
            'ndvi_summer': summer_ndvi,
            'ndvi_winter': float(winter_ndvi[idx]) if winter_ndvi[idx] != 0 else None,
            'd_ndvi': round(d_ndvi, 4) if d_ndvi is not None else None,
            'ndbi_summer': summer_ndbi,
            'ndbi_winter': float(winter_ndbi[idx]) if winter_ndbi[idx] != 0 else None,
            'd_ndbi': round(d_ndbi, 4) if d_ndbi is not None else None,
        })

    # Now add priority changes (different grid - priority is at SVI point locations)
    # We'll create a separate section or merge by finding nearest satellite point
    print("Adding priority score changes...")

    # Build tree for result points
    result_coords = np.array([[d['lat'], d['lon']] for d in result])
    result_tree = cKDTree(result_coords)

    # For each priority point, find nearest result point and add priority delta
    priority_deltas = {}
    for i, sp in enumerate(summer_pri):
        lat, lon = sp['lat'], sp['lon']
        summer_score = sp.get('priority_score', 0)

        # Find matching winter priority point
        _, widx = winter_pri_tree.query([lat, lon])
        winter_score = winter_pri[widx].get('priority_score', 0)

        d_priority = winter_score - summer_score
        stats['d_priority'].append(d_priority)

        # Find nearest satellite grid point to attach this data
        _, ridx = result_tree.query([lat, lon])
        if ridx not in priority_deltas:
            priority_deltas[ridx] = []
        priority_deltas[ridx].append(d_priority)

    # Average priority deltas for each grid cell
    for ridx, deltas in priority_deltas.items():
        result[ridx]['d_priority'] = round(np.mean(deltas), 4)

    # Fill in None for points without priority data
    for d in result:
        if 'd_priority' not in d:
            d['d_priority'] = None

    # Print statistics
    print("\n" + "-" * 40)
    print("SEASONAL CHANGE STATISTICS")
    print("-" * 40)

    for metric, values in stats.items():
        if values:
            arr = np.array(values)
            print(f"\n{metric}:")
            print(f"  Range: {arr.min():.3f} to {arr.max():.3f}")
            print(f"  Mean:  {arr.mean():.3f}")
            print(f"  Std:   {arr.std():.3f}")

    # Save output
    out_path = os.path.join(DATA_DIR, 'change_seasonal.json')
    save_json(result, out_path)
    print(f"\nSaved: {out_path}")
    print(f"  Points: {len(result)}")
    print(f"  Size: {os.path.getsize(out_path) / (1024*1024):.1f} MB")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

if __name__ == '__main__':
    main()
