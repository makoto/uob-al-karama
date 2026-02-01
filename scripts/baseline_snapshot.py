#!/usr/bin/env python3
"""
Save a baseline snapshot of all Al Karama analysis data.

Captures record counts, completeness, summary statistics for every
numeric field, and the temporal distribution of Mapillary source images.
Run this BEFORE and AFTER fetching new data, then compare the two JSON files.

Usage:
    python scripts/baseline_snapshot.py                # saves with auto timestamp
    python scripts/baseline_snapshot.py --tag before   # saves as baseline_before.json
    python scripts/baseline_snapshot.py --tag after    # saves as baseline_after.json
    python scripts/baseline_snapshot.py --compare before after  # prints diff
"""

import argparse
import csv
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / 'docs' / 'data' / 'al_karama'
SVI_CSV = PROJECT_DIR / 'data' / 'svi_images' / 'pids_urls.csv'
OUT_DIR = PROJECT_DIR / 'baselines'


def percentile(sorted_vals, p):
    """Return the p-th percentile from a sorted list."""
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_vals) else f
    d = k - f
    return sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f])


def field_stats(values):
    """Compute summary stats for a list of numeric values (Nones excluded)."""
    clean = [v for v in values if v is not None]
    if not clean:
        return {'count': 0, 'null_count': len(values)}
    s = sorted(clean)
    return {
        'count': len(clean),
        'null_count': len(values) - len(clean),
        'mean': round(statistics.mean(s), 6),
        'median': round(statistics.median(s), 6),
        'std': round(statistics.stdev(s), 6) if len(s) > 1 else 0,
        'min': round(s[0], 6),
        'max': round(s[-1], 6),
        'p5': round(percentile(s, 5), 6),
        'p25': round(percentile(s, 25), 6),
        'p75': round(percentile(s, 75), 6),
        'p95': round(percentile(s, 95), 6),
    }


def analyse_json(path, numeric_fields):
    """Load a JSON array file and compute stats for the given fields."""
    with open(path) as f:
        data = json.load(f)

    result = {'record_count': len(data), 'fields': {}}
    for fld in numeric_fields:
        values = []
        for d in data:
            v = d.get(fld)
            if v is not None:
                try:
                    values.append(float(v))
                except (ValueError, TypeError):
                    values.append(None)
            else:
                values.append(None)
        result['fields'][fld] = field_stats(values)
    return result


def analyse_geojson(path, numeric_fields):
    """Load a GeoJSON FeatureCollection and compute stats for properties."""
    with open(path) as f:
        data = json.load(f)

    features = data.get('features', [])
    result = {'record_count': len(features), 'fields': {}}
    for fld in numeric_fields:
        values = []
        for feat in features:
            v = feat.get('properties', {}).get(fld)
            if v is not None:
                try:
                    values.append(float(v))
                except (ValueError, TypeError):
                    values.append(None)
            else:
                values.append(None)
        result['fields'][fld] = field_stats(values)
    return result


def analyse_mapillary_dates(csv_path):
    """Analyse temporal distribution of Mapillary images from local CSV."""
    ref_date = datetime.now(timezone.utc)

    dates = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = float(row['captured_at']) / 1000.0
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                dates.append(dt)
            except (ValueError, KeyError):
                continue

    if not dates:
        return {'total_images': 0}

    dates.sort()
    ages_days = [(ref_date - d).days for d in dates]
    ages_sorted = sorted(ages_days)
    total = len(dates)

    # Year counts
    year_counts = {}
    for d in dates:
        y = str(d.year)
        year_counts[y] = year_counts.get(y, 0) + 1

    # Age buckets
    bucket_defs = [
        ('lt_1y', 365),
        ('1_2y', 730),
        ('2_3y', 1095),
        ('3_5y', 1825),
        ('5_7y', 2555),
        ('7_10y', 3650),
        ('gt_10y', 999999),
    ]
    buckets = {}
    prev = 0
    for label, days in bucket_defs:
        count = sum(1 for a in ages_days if prev < a <= days)
        buckets[label] = {'count': count, 'pct': round(count / total * 100, 1)}
        prev = days

    return {
        'total_images': total,
        'date_range': {
            'earliest': dates[0].strftime('%Y-%m-%d'),
            'latest': dates[-1].strftime('%Y-%m-%d'),
        },
        'age_days': {
            'median': round(statistics.median(ages_sorted)),
            'p25': round(percentile(ages_sorted, 25)),
            'p75': round(percentile(ages_sorted, 75)),
            'median_years': round(statistics.median(ages_sorted) / 365, 1),
        },
        'by_year': dict(sorted(year_counts.items())),
        'age_buckets': buckets,
        'reference_date': ref_date.strftime('%Y-%m-%d'),
    }


def build_snapshot():
    """Build the complete baseline snapshot dict."""
    snapshot = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
    }

    # --- Analysis data files ---
    files_config = {
        'priority_points': {
            'file': 'priority_points.json',
            'type': 'json',
            'fields': ['lat', 'lon', 'svf', 'gvi', 'lst', 'ndvi', 'ndbi', 'priority_score'],
        },
        'combined_svi': {
            'file': 'combined_svi.json',
            'type': 'json',
            'fields': ['lat', 'lon', 'svf', 'gvi', 'lst', 'ndvi_satellite'],
        },
        'distance_to_green': {
            'file': 'distance_to_green.json',
            'type': 'json',
            'fields': ['lat', 'lon', 'dist_to_green_m'],
        },
        'clusters': {
            'file': 'clusters.json',
            'type': 'json',
            'fields': ['gvi', 'svf', 'lst', 'ndvi', 'cluster'],
        },
        'segment_comfort': {
            'file': 'segment_comfort.json',
            'type': 'json',
            'fields': ['pci_mean', 'pci_std', 'point_count', 'lst_mean', 'gvi_mean', 'svf_mean', 'shade_mean'],
        },
        'satellite_grid': {
            'file': 'satellite_grid.json',
            'type': 'json',
            'fields': ['lat', 'lon', 'lst', 'ndvi', 'ndbi'],
        },
        'streets': {
            'file': 'streets.geojson',
            'type': 'geojson',
            'fields': ['length', 'betweenness', 'closeness', 'degree', 'centrality', 'pci', 'lst', 'priority'],
        },
    }

    snapshot['analysis_data'] = {}
    for key, cfg in files_config.items():
        path = DATA_DIR / cfg['file']
        if not path.exists():
            print(f'  SKIP {cfg["file"]} (not found)')
            continue
        print(f'  {cfg["file"]}...')
        if cfg['type'] == 'geojson':
            snapshot['analysis_data'][key] = analyse_geojson(path, cfg['fields'])
        else:
            snapshot['analysis_data'][key] = analyse_json(path, cfg['fields'])

    # --- Priority level distribution ---
    pp_path = DATA_DIR / 'priority_points.json'
    if pp_path.exists():
        with open(pp_path) as f:
            pp = json.load(f)
        levels = {}
        for d in pp:
            lv = d.get('priority_level', 'null') or 'null'
            levels[lv] = levels.get(lv, 0) + 1
        snapshot['analysis_data']['priority_points']['priority_level_distribution'] = levels

    # --- Cluster distribution ---
    cl_path = DATA_DIR / 'clusters.json'
    if cl_path.exists():
        with open(cl_path) as f:
            cl = json.load(f)
        cluster_dist = {}
        for d in cl:
            label = d.get('cluster_label', f'Cluster {d.get("cluster", "?")}')
            cluster_dist[label] = cluster_dist.get(label, 0) + 1
        snapshot['analysis_data']['clusters']['cluster_distribution'] = cluster_dist

    # --- Mapillary image dates ---
    if SVI_CSV.exists():
        print(f'  pids_urls.csv (Mapillary dates)...')
        snapshot['mapillary_images'] = analyse_mapillary_dates(SVI_CSV)
    else:
        print(f'  SKIP pids_urls.csv (not found)')
        snapshot['mapillary_images'] = None

    return snapshot


def compare_snapshots(before_path, after_path):
    """Print a comparison table between two baseline snapshots."""
    with open(before_path) as f:
        before = json.load(f)
    with open(after_path) as f:
        after = json.load(f)

    print(f'\n{"="*70}')
    print(f'BASELINE COMPARISON')
    print(f'  Before: {before["generated_at"]}')
    print(f'  After:  {after["generated_at"]}')
    print(f'{"="*70}\n')

    # Record counts
    print(f'{"Dataset":<22} {"Before":>8} {"After":>8} {"Delta":>8}')
    print(f'{"-"*22} {"-"*8} {"-"*8} {"-"*8}')
    for key in before.get('analysis_data', {}):
        b_count = before['analysis_data'][key]['record_count']
        a_count = after.get('analysis_data', {}).get(key, {}).get('record_count', '?')
        delta = a_count - b_count if isinstance(a_count, int) else '?'
        sign = '+' if isinstance(delta, int) and delta > 0 else ''
        print(f'{key:<22} {b_count:>8} {a_count:>8} {sign}{delta:>7}')

    # Mapillary images
    b_img = before.get('mapillary_images', {}).get('total_images', 0)
    a_img = after.get('mapillary_images', {}).get('total_images', 0)
    print(f'{"mapillary_images":<22} {b_img:>8} {a_img:>8} {"+"+str(a_img-b_img) if a_img-b_img>0 else a_img-b_img:>7}')

    # Key metrics comparison
    key_metrics = [
        ('priority_points', 'gvi', 'GVI (mean)'),
        ('priority_points', 'svf', 'SVF (mean)'),
        ('priority_points', 'lst', 'LST (mean)'),
        ('priority_points', 'priority_score', 'Priority Score (mean)'),
        ('segment_comfort', 'pci_mean', 'PCI (mean)'),
        ('segment_comfort', 'shade_mean', 'Shade (mean)'),
        ('distance_to_green', 'dist_to_green_m', 'Dist to Green (mean)'),
    ]

    print(f'\n{"Metric":<25} {"Before":>10} {"After":>10} {"Delta":>10}')
    print(f'{"-"*25} {"-"*10} {"-"*10} {"-"*10}')
    for dataset, field, label in key_metrics:
        b_val = before.get('analysis_data', {}).get(dataset, {}).get('fields', {}).get(field, {}).get('mean', '?')
        a_val = after.get('analysis_data', {}).get(dataset, {}).get('fields', {}).get(field, {}).get('mean', '?')
        if isinstance(b_val, (int, float)) and isinstance(a_val, (int, float)):
            delta = round(a_val - b_val, 6)
            sign = '+' if delta > 0 else ''
            print(f'{label:<25} {b_val:>10.4f} {a_val:>10.4f} {sign}{delta:>9.4f}')
        else:
            print(f'{label:<25} {str(b_val):>10} {str(a_val):>10} {"?":>10}')

    # Completeness comparison (null counts)
    null_fields = [
        ('priority_points', 'gvi', 'GVI nulls'),
        ('priority_points', 'svf', 'SVF nulls'),
        ('priority_points', 'priority_score', 'Priority nulls'),
        ('streets', 'pci', 'Street PCI nulls'),
    ]

    print(f'\n{"Completeness":<25} {"Before":>10} {"After":>10} {"Delta":>10}')
    print(f'{"-"*25} {"-"*10} {"-"*10} {"-"*10}')
    for dataset, field, label in null_fields:
        b_null = before.get('analysis_data', {}).get(dataset, {}).get('fields', {}).get(field, {}).get('null_count', '?')
        a_null = after.get('analysis_data', {}).get(dataset, {}).get('fields', {}).get(field, {}).get('null_count', '?')
        if isinstance(b_null, int) and isinstance(a_null, int):
            delta = a_null - b_null
            sign = '+' if delta > 0 else ''
            print(f'{label:<25} {b_null:>10} {a_null:>10} {sign}{delta:>9}')
        else:
            print(f'{label:<25} {str(b_null):>10} {str(a_null):>10} {"?":>10}')

    # Mapillary age comparison
    print(f'\n{"Image Age":<25} {"Before":>10} {"After":>10}')
    print(f'{"-"*25} {"-"*10} {"-"*10}')
    b_med = before.get('mapillary_images', {}).get('age_days', {}).get('median_years', '?')
    a_med = after.get('mapillary_images', {}).get('age_days', {}).get('median_years', '?')
    print(f'{"Median age (years)":<25} {str(b_med):>10} {str(a_med):>10}')

    for bucket in ['lt_1y', '1_2y', '2_3y', '3_5y', '5_7y', '7_10y', 'gt_10y']:
        b_pct = before.get('mapillary_images', {}).get('age_buckets', {}).get(bucket, {}).get('pct', '?')
        a_pct = after.get('mapillary_images', {}).get('age_buckets', {}).get(bucket, {}).get('pct', '?')
        print(f'{bucket:<25} {str(b_pct)+"%":>10} {str(a_pct)+"%":>10}')

    print()


def main():
    parser = argparse.ArgumentParser(description='Save or compare baseline snapshots')
    parser.add_argument('--tag', default=None, help='Tag for output filename (e.g. "before", "after")')
    parser.add_argument('--compare', nargs=2, metavar=('BEFORE', 'AFTER'),
                        help='Compare two snapshots by tag name')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.compare:
        before_path = OUT_DIR / f'baseline_{args.compare[0]}.json'
        after_path = OUT_DIR / f'baseline_{args.compare[1]}.json'
        if not before_path.exists():
            print(f'Error: {before_path} not found')
            sys.exit(1)
        if not after_path.exists():
            print(f'Error: {after_path} not found')
            sys.exit(1)
        compare_snapshots(before_path, after_path)
        return

    tag = args.tag or datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = OUT_DIR / f'baseline_{tag}.json'

    print(f'Building baseline snapshot...')
    snapshot = build_snapshot()

    with open(out_path, 'w') as f:
        json.dump(snapshot, f, indent=2)

    size_kb = os.path.getsize(out_path) / 1024
    print(f'\nSaved: {out_path} ({size_kb:.0f} KB)')
    print(f'\nTo compare later:')
    print(f'  python scripts/baseline_snapshot.py --tag after')
    print(f'  python scripts/baseline_snapshot.py --compare {tag} after')


if __name__ == '__main__':
    main()
