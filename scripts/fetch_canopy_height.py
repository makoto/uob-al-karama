#!/usr/bin/env python3
"""
Fetch Meta 1m Global Canopy Height Map for Al Karama, Dubai
============================================================
Downloads canopy height data from Google Earth Engine and produces:
1. A GeoTIFF raster of canopy heights
2. A GeoJSON of tree canopy polygons (thresholded > 2m)
3. Summary statistics and a quick preview map
"""

import ee
import os
import sys
import json
import time
import numpy as np
import geopandas as gpd
from shapely.geometry import shape

# ── Configuration ──
BBOX = [55.290, 25.230, 55.320, 25.255]  # Al Karama bounding box
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'shade_analysis')
CANOPY_MIN_HEIGHT = 2.0  # meters — threshold to consider as "tree"

# Meta 1m canopy height dataset on GEE
META_CANOPY_DATASET = 'projects/meta-forest-monitoring-okw37/assets/v1/canopy_height'


def initialize_ee():
    """Initialize Earth Engine."""
    try:
        ee.Initialize(project='uobdubai')
        print("  Initialized GEE with project 'uobdubai'")
        return
    except Exception:
        pass
    try:
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
        print("  Initialized GEE with high-volume endpoint")
        return
    except Exception:
        pass
    try:
        ee.Initialize()
        print("  Initialized GEE with defaults")
        return
    except Exception as e:
        print(f"  Failed to initialize GEE: {e}")
        print("  Run: earthengine authenticate")
        sys.exit(1)


def fetch_canopy_stats():
    """Fetch canopy height statistics for Al Karama from Meta 1m dataset."""

    aoi = ee.Geometry.Rectangle(BBOX)

    print("\n[1/3] Loading Meta 1m canopy height dataset...")

    # Try the community catalog path first, then alternatives
    dataset_paths = [
        'projects/meta-forest-monitoring-okw37/assets/v1/canopy_height',
        'projects/sat-io/open-datasets/facebook/meta-canopy-height',
    ]

    canopy = None
    for path in dataset_paths:
        try:
            img_col = ee.ImageCollection(path)
            # Get mosaic for our area
            canopy = img_col.filterBounds(aoi).mosaic().clip(aoi)
            # Test if it works by getting a small stat
            test = canopy.reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=aoi,
                scale=10,  # coarse for quick test
                maxPixels=1e6
            ).getInfo()
            print(f"  Found dataset at: {path}")
            print(f"  Test pixel count (at 10m): {test}")
            break
        except Exception as e:
            print(f"  Tried {path}: {e}")
            canopy = None

    if canopy is None:
        print("\n  Could not load Meta canopy height dataset.")
        print("  Trying ETH 10m canopy height as fallback...")
        try:
            eth_canopy = ee.Image('users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1').clip(aoi)
            canopy = eth_canopy
            print("  Loaded ETH 10m canopy height")
        except Exception as e:
            print(f"  ETH also failed: {e}")
            print("\n  Trying WRI/Meta via different asset path...")
            try:
                # Another known path
                canopy = ee.ImageCollection(
                    'projects/meta-forest-monitoring-okw37/assets/v1/canopy_height'
                ).mosaic().clip(aoi)
                print("  Loaded via meta-forest-monitoring path")
            except Exception as e2:
                print(f"  Also failed: {e2}")
                return None

    return canopy, aoi


def compute_statistics(canopy, aoi):
    """Compute canopy height statistics for Al Karama."""

    print("\n[2/3] Computing canopy height statistics...")

    # Basic stats at 5m resolution (balance detail vs compute)
    stats = canopy.reduceRegion(
        reducer=ee.Reducer.mean()
            .combine(ee.Reducer.max(), '', True)
            .combine(ee.Reducer.min(), '', True)
            .combine(ee.Reducer.stdDev(), '', True)
            .combine(ee.Reducer.count(), '', True),
        geometry=aoi,
        scale=5,
        maxPixels=1e8
    ).getInfo()
    print(f"  Raw stats: {json.dumps(stats, indent=2)}")

    # Count pixels with canopy > threshold
    tree_mask = canopy.gt(CANOPY_MIN_HEIGHT)
    tree_stats = tree_mask.reduceRegion(
        reducer=ee.Reducer.sum().combine(ee.Reducer.count(), '', True),
        geometry=aoi,
        scale=5,
        maxPixels=1e8
    ).getInfo()
    print(f"  Tree mask stats (>{CANOPY_MIN_HEIGHT}m): {json.dumps(tree_stats, indent=2)}")

    # Height distribution — count pixels in bins
    print("\n  Height distribution:")
    bins = [(0, 2, 'No canopy / low (<2m)'),
            (2, 5, 'Small trees (2-5m)'),
            (5, 10, 'Medium trees (5-10m)'),
            (10, 15, 'Tall trees (10-15m)'),
            (15, 30, 'Very tall (15-30m)')]

    for lo, hi, label in bins:
        mask = canopy.gte(lo).And(canopy.lt(hi))
        count = mask.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=5,
            maxPixels=1e8
        ).getInfo()
        # Get first band value
        val = list(count.values())[0] if count else 0
        print(f"    {label}: {val} pixels (at 5m)")

    return stats, tree_stats


def export_tree_vectors(canopy, aoi):
    """Convert canopy height raster to tree polygons and export as GeoJSON."""

    print("\n[3/3] Extracting tree canopy polygons...")

    # Create binary mask for tree canopy at 5m scale
    tree_mask = canopy.gt(CANOPY_MIN_HEIGHT).selfMask()
    tree_mask = tree_mask.reproject(crs='EPSG:4326', scale=5)

    # Convert to vectors
    try:
        vectors = tree_mask.reduceToVectors(
            geometry=aoi,
            scale=5,
            geometryType='polygon',
            eightConnected=True,
            maxPixels=1e8,
            bestEffort=True
        )

        # Get the features
        features = vectors.getInfo()
        n_features = len(features.get('features', []))
        print(f"  Extracted {n_features} canopy polygons")

        # Save as GeoJSON
        out_path = os.path.join(OUT_DIR, 'canopy_polygons.geojson')
        with open(out_path, 'w') as f:
            json.dump(features, f)
        print(f"  Saved: {out_path}")

        return features, out_path

    except Exception as e:
        print(f"  Vector extraction failed: {e}")
        print("  This may happen if there are too few/many tree pixels.")
        return None, None


def sample_canopy_heights(canopy, aoi):
    """Sample canopy heights at a grid of points, keeping coordinates."""

    print("\n  Sampling canopy heights on grid...")

    # Sample the image directly — returns points with geometry preserved
    sampled = canopy.sample(
        region=aoi,
        scale=5,
        numPixels=3000,
        seed=42,
        geometries=True
    )

    features = sampled.getInfo()
    all_pts = features.get('features', [])
    n = len(all_pts)
    print(f"  Sampled {n} points")

    # Filter to only points with canopy > 0
    tree_points = [f for f in all_pts
                   if f.get('geometry') and any(v and v > 0 for v in f['properties'].values())]
    print(f"  Points with canopy > 0m: {len(tree_points)}")
    tree_tall = [f for f in all_pts
                 if f.get('geometry') and any(v and v > CANOPY_MIN_HEIGHT for v in f['properties'].values())]
    print(f"  Points with canopy > {CANOPY_MIN_HEIGHT}m: {len(tree_tall)}")

    # Rename property for clarity
    for f in tree_points:
        h = f['properties'].pop('cover_code', 0)
        f['properties']['canopy_height_m'] = h

    # Save sample points
    out_path = os.path.join(OUT_DIR, 'canopy_samples.geojson')
    with open(out_path, 'w') as f:
        json.dump({'type': 'FeatureCollection', 'features': tree_points}, f)
    print(f"  Saved: {out_path}")

    return tree_points


def assign_heights_to_polygons():
    """
    Assign canopy heights to polygons using IDW interpolation from sample points.

    Loads canopy_polygons.geojson (no height) and canopy_samples.geojson (points
    with heights), converts to UTM, finds k=3 nearest sample points for each
    polygon centroid, and assigns IDW-interpolated height.

    Outputs canopy_polygons_with_height.geojson.
    """
    from scipy.spatial import cKDTree

    polygons_path = os.path.join(OUT_DIR, 'canopy_polygons.geojson')
    samples_path = os.path.join(OUT_DIR, 'canopy_samples.geojson')
    out_path = os.path.join(OUT_DIR, 'canopy_polygons_with_height.geojson')

    print("\n[4/4] Assigning heights to canopy polygons via IDW interpolation...")

    if not os.path.exists(polygons_path):
        print(f"  ERROR: {polygons_path} not found")
        return
    if not os.path.exists(samples_path):
        print(f"  ERROR: {samples_path} not found")
        return

    # Load data
    polys = gpd.read_file(polygons_path)
    samples = gpd.read_file(samples_path)
    print(f"  Polygons: {len(polys)}, Sample points: {len(samples)}")

    # Convert to UTM for metric distance calculations
    utm_crs = 'EPSG:32640'  # UTM Zone 40N for Dubai
    polys_utm = polys.to_crs(utm_crs)
    samples_utm = samples.to_crs(utm_crs)

    # Extract sample point coordinates and heights
    sample_coords = np.array([(g.x, g.y) for g in samples_utm.geometry])
    sample_heights = np.array([
        f.get('canopy_height_m', 0) for f in samples['properties']
    ] if 'properties' in samples.columns else [
        samples_utm.iloc[i].get('canopy_height_m', 0) for i in range(len(samples_utm))
    ])
    # Handle case where canopy_height_m is a direct column
    if 'canopy_height_m' in samples_utm.columns:
        sample_heights = samples_utm['canopy_height_m'].values.astype(float)

    median_height = float(np.median(sample_heights[sample_heights > 0]))
    print(f"  Median sample height: {median_height:.1f}m")

    # Build KD-tree from sample points
    tree = cKDTree(sample_coords)

    # For each polygon centroid, find k=3 nearest and IDW-interpolate
    k = 3
    max_dist = 500.0  # meters
    heights = []

    for i, row in polys_utm.iterrows():
        centroid = row.geometry.centroid
        cx, cy = centroid.x, centroid.y
        dists, idxs = tree.query([cx, cy], k=k)

        # cKDTree returns scalar if k=1, ensure array
        if k == 1:
            dists = [dists]
            idxs = [idxs]

        # Filter to within max_dist
        valid = [(d, j) for d, j in zip(dists, idxs) if d <= max_dist and j < len(sample_heights)]

        if not valid:
            heights.append(median_height)
            continue

        # IDW: weight = 1/d^2 (avoid division by zero)
        total_weight = 0.0
        weighted_sum = 0.0
        for d, j in valid:
            if d < 0.1:
                d = 0.1  # avoid inf weight
            w = 1.0 / (d * d)
            weighted_sum += w * sample_heights[j]
            total_weight += w

        h = weighted_sum / total_weight if total_weight > 0 else median_height
        heights.append(round(h, 1))

    polys['canopy_height_m'] = heights

    # Drop the 'count' and 'label' columns from GEE if present
    for col in ['count', 'label']:
        if col in polys.columns:
            polys = polys.drop(columns=[col])

    # Save
    polys.to_file(out_path, driver='GeoJSON')
    h_arr = np.array(heights)
    print(f"  Assigned heights: min={h_arr.min():.1f}m, max={h_arr.max():.1f}m, "
          f"mean={h_arr.mean():.1f}m, median={np.median(h_arr):.1f}m")
    print(f"  Saved: {out_path}")


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  META 1m CANOPY HEIGHT - Al Karama, Dubai")
    print("=" * 60)

    print("\n  Initializing Google Earth Engine...")
    initialize_ee()

    result = fetch_canopy_stats()
    if result is None:
        print("\n  FAILED: Could not load canopy height dataset.")
        return

    canopy, aoi = result

    stats, tree_stats = compute_statistics(canopy, aoi)
    sample_canopy_heights(canopy, aoi)
    export_tree_vectors(canopy, aoi)
    assign_heights_to_polygons()

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  COMPLETE - {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
