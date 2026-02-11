#!/usr/bin/env python3
"""
Generate season-specific Climate Cluster data.

Uses seasonal satellite LST data combined with SVI metrics (GVI, SVF) to
perform K-means clustering that reflects actual thermal conditions per season.

Output: docs/data/al_karama/{season}/clusters.json for each season
"""

import json
import os
import numpy as np
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'docs', 'data', 'al_karama')

SEASONS = ['summer_2020', 'winter_2020', 'summer_2025', 'winter_2025']

# Cluster labels based on characteristics
CLUSTER_LABELS = {
    0: 'Hot & Barren',
    1: 'Shaded Urban',
    2: 'Warm Urban',
    3: 'Cool & Green'
}

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, separators=(',', ':'))

def compute_clusters(svi_data, sat_data, n_clusters=4):
    """Compute climate clusters using seasonal satellite data.

    Args:
        svi_data: List of dicts with lat, lon, gvi, svf
        sat_data: List of dicts with lat, lon, lst, ndvi

    Returns:
        List of dicts with cluster assignments
    """
    # Build KD-tree for satellite data
    sat_coords = np.array([[d['lat'], d['lon']] for d in sat_data])
    sat_tree = cKDTree(sat_coords)

    # Extract satellite values
    sat_lst = np.array([d.get('lst', 0) or 0 for d in sat_data])
    sat_ndvi = np.array([d.get('ndvi', 0) or 0 for d in sat_data])

    # Find nearest satellite cell for each SVI point
    svi_coords = np.array([[d['lat'], d['lon']] for d in svi_data])
    _, indices = sat_tree.query(svi_coords, k=1)

    # Build feature matrix: GVI, SVF, LST, NDVI
    gvi_vals = np.array([d.get('gvi', 0) or 0 for d in svi_data])
    svf_vals = np.array([d.get('svf', 0) or 0 for d in svi_data])
    lst_vals = sat_lst[indices]
    ndvi_vals = sat_ndvi[indices]

    features = np.column_stack([gvi_vals, svf_vals, lst_vals, ndvi_vals])

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_scaled)

    # Analyze cluster characteristics to assign labels
    cluster_means = {}
    for c in range(n_clusters):
        mask = clusters == c
        cluster_means[c] = {
            'gvi': gvi_vals[mask].mean(),
            'svf': svf_vals[mask].mean(),
            'lst': lst_vals[mask].mean(),
            'ndvi': ndvi_vals[mask].mean(),
            'count': mask.sum()
        }

    # Sort clusters by LST (descending) to assign consistent labels
    sorted_clusters = sorted(cluster_means.keys(),
                            key=lambda c: cluster_means[c]['lst'],
                            reverse=True)

    # Map: original cluster ID -> label based on LST ranking
    # Highest LST = Hot & Barren (0), Lowest LST = Cool & Green (3)
    cluster_label_map = {}
    for rank, orig_cluster in enumerate(sorted_clusters):
        cluster_label_map[orig_cluster] = CLUSTER_LABELS[rank]

    # Build output
    result = []
    for i, d in enumerate(svi_data):
        orig_cluster = clusters[i]
        result.append({
            'lat': d['lat'],
            'lon': d['lon'],
            'gvi': float(gvi_vals[i]),
            'svf': float(svf_vals[i]),
            'lst': float(lst_vals[i]),
            'ndvi': float(ndvi_vals[i]),
            'cluster': int(orig_cluster),
            'cluster_label': cluster_label_map[orig_cluster]
        })

    return result, cluster_means, cluster_label_map

def main():
    print("=" * 60)
    print("GENERATING SEASONAL CLUSTER DATA")
    print("=" * 60)

    # Load SVI data (same for all seasons)
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

        # Compute clusters
        print("  Computing K-means clusters...")
        cluster_data, cluster_means, label_map = compute_clusters(svi_data, sat_data)

        # Stats
        print("  Cluster characteristics:")
        for orig_cluster, label in label_map.items():
            stats = cluster_means[orig_cluster]
            print(f"    {label}: {stats['count']} pts, "
                  f"LST={stats['lst']:.1f}°C, GVI={stats['gvi']:.3f}, "
                  f"SVF={stats['svf']:.2f}, NDVI={stats['ndvi']:.3f}")

        # Save
        out_path = os.path.join(DATA_DIR, season, 'clusters.json')
        save_json(cluster_data, out_path)
        print(f"  Saved: {out_path}")
        print(f"  Size: {os.path.getsize(out_path) / 1024:.0f} KB")

    print("\n" + "=" * 60)
    print("DONE - Cluster data regenerated for all seasons")
    print("=" * 60)

if __name__ == '__main__':
    main()
