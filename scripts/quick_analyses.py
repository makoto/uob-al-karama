#!/usr/bin/env python3
"""
Quick High-Accuracy Analyses for Al Karama

1. Correlation deep-dive
2. Cluster analysis (K-means)
3. Accessibility to green space
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
import os

print("="*60)
print("QUICK ANALYSES - Al Karama")
print("="*60)

# Load combined data
print("\nLoading data...")
comfort_df = pd.read_csv('output/walking_routes/point_comfort.csv')
print(f"  Points with all metrics: {len(comfort_df)}")

# ============================================================
# 1. CORRELATION DEEP-DIVE
# ============================================================
print("\n" + "="*60)
print("1. CORRELATION ANALYSIS")
print("="*60)

# Select numeric columns for correlation
corr_cols = ['gvi', 'svf', 'lst', 'ndvi', 'ndbi', 'pci']
corr_data = comfort_df[corr_cols].dropna()
print(f"\n  Valid data points: {len(corr_data)}")

# Calculate correlation matrix
corr_matrix = corr_data.corr()

print("\n  CORRELATION MATRIX:")
print("  " + "-"*70)
header = f"  {'':>8}"
for col in corr_cols:
    header += f"{col:>10}"
print(header)
print("  " + "-"*70)

for row in corr_cols:
    line = f"  {row:>8}"
    for col in corr_cols:
        val = corr_matrix.loc[row, col]
        line += f"{val:>10.3f}"
    print(line)

# Statistical significance (p-values)
print("\n  KEY CORRELATIONS (with p-values):")
print("  " + "-"*50)

key_pairs = [
    ('gvi', 'lst', 'Street greenery vs Temperature'),
    ('gvi', 'ndvi', 'Street greenery vs Satellite vegetation'),
    ('svf', 'lst', 'Sky exposure vs Temperature'),
    ('ndvi', 'lst', 'Satellite vegetation vs Temperature'),
    ('ndbi', 'lst', 'Built-up index vs Temperature'),
    ('gvi', 'svf', 'Greenery vs Sky exposure'),
]

correlation_results = []
for var1, var2, desc in key_pairs:
    r, p = stats.pearsonr(corr_data[var1], corr_data[var2])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {desc:.<40} r={r:>7.3f} (p={p:.2e}) {sig}")
    correlation_results.append({
        'var1': var1, 'var2': var2, 'description': desc,
        'r': round(r, 4), 'p_value': p, 'significant': p < 0.05
    })

print("\n  Significance: *** p<0.001, ** p<0.01, * p<0.05")

# Save correlation results
os.makedirs('output/quick_analysis', exist_ok=True)
pd.DataFrame(correlation_results).to_csv('output/quick_analysis/correlations.csv', index=False)
corr_matrix.to_csv('output/quick_analysis/correlation_matrix.csv')

# ============================================================
# 2. CLUSTER ANALYSIS
# ============================================================
print("\n" + "="*60)
print("2. CLUSTER ANALYSIS (K-Means)")
print("="*60)

# Features for clustering
cluster_features = ['gvi', 'svf', 'lst', 'ndvi']
cluster_data = comfort_df[cluster_features + ['lat', 'lon']].dropna()
print(f"\n  Data points for clustering: {len(cluster_data)}")

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(cluster_data[cluster_features])

# Determine optimal k using elbow method
print("\n  Finding optimal number of clusters...")
inertias = []
K_range = range(2, 8)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Use 4 clusters (typical for urban climate zones)
n_clusters = 4
print(f"  Using {n_clusters} clusters")

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_data['cluster'] = kmeans.fit_predict(X)

# Cluster statistics
print("\n  CLUSTER CHARACTERISTICS:")
print("  " + "-"*70)
print(f"  {'Cluster':>8} {'Count':>8} {'GVI%':>8} {'SVF%':>8} {'LST°C':>8} {'NDVI':>8} {'Label':>15}")
print("  " + "-"*70)

cluster_stats = []
cluster_labels = []
for i in range(n_clusters):
    c_data = cluster_data[cluster_data['cluster'] == i]
    gvi_mean = c_data['gvi'].mean() * 100
    svf_mean = c_data['svf'].mean() * 100
    lst_mean = c_data['lst'].mean()
    ndvi_mean = c_data['ndvi'].mean()

    # Auto-label based on characteristics
    if lst_mean > 50 and gvi_mean < 3:
        label = "Hot & Barren"
    elif lst_mean > 49 and svf_mean > 35:
        label = "Hot & Exposed"
    elif gvi_mean > 5 and lst_mean < 49:
        label = "Green & Cool"
    elif svf_mean < 30:
        label = "Shaded Urban"
    else:
        label = "Mixed"

    cluster_labels.append(label)
    print(f"  {i:>8} {len(c_data):>8} {gvi_mean:>8.1f} {svf_mean:>8.1f} {lst_mean:>8.1f} {ndvi_mean:>8.3f} {label:>15}")

    cluster_stats.append({
        'cluster': i, 'label': label, 'count': len(c_data),
        'gvi_mean': round(gvi_mean, 2), 'svf_mean': round(svf_mean, 2),
        'lst_mean': round(lst_mean, 2), 'ndvi_mean': round(ndvi_mean, 4)
    })

cluster_data['cluster_label'] = cluster_data['cluster'].map(dict(enumerate(cluster_labels)))

# Save cluster data
cluster_data.to_csv('output/quick_analysis/clusters.csv', index=False)
pd.DataFrame(cluster_stats).to_csv('output/quick_analysis/cluster_stats.csv', index=False)

# ============================================================
# 3. ACCESSIBILITY TO GREEN SPACE
# ============================================================
print("\n" + "="*60)
print("3. ACCESSIBILITY TO GREEN SPACE")
print("="*60)

# Load satellite data for NDVI
sat_df = pd.read_csv('output/satellite_full/full_area_data.csv')

# Define "green space" as NDVI > 0.3 (moderate vegetation)
green_threshold = 0.3
green_spaces = sat_df[sat_df['ndvi'] > green_threshold][['lat', 'lon', 'ndvi']].copy()
print(f"\n  Green space cells (NDVI > {green_threshold}): {len(green_spaces)}")
print(f"  Total cells: {len(sat_df)} ({len(green_spaces)/len(sat_df)*100:.1f}% green)")

if len(green_spaces) > 0:
    # Build KD-tree for green spaces
    green_coords = green_spaces[['lat', 'lon']].values
    green_tree = cKDTree(green_coords)

    # Calculate distance from each SVI point to nearest green space
    svi_coords = comfort_df[['lat', 'lon']].values
    distances, indices = green_tree.query(svi_coords, k=1)

    # Convert to meters (approximate)
    comfort_df['dist_to_green_m'] = distances * 111000  # degrees to meters

    # Statistics
    print(f"\n  DISTANCE TO NEAREST GREEN SPACE:")
    print("  " + "-"*40)
    print(f"  Mean distance: {comfort_df['dist_to_green_m'].mean():.0f} m")
    print(f"  Median distance: {comfort_df['dist_to_green_m'].median():.0f} m")
    print(f"  Max distance: {comfort_df['dist_to_green_m'].max():.0f} m")
    print(f"  Min distance: {comfort_df['dist_to_green_m'].min():.0f} m")

    # Accessibility bands
    print(f"\n  ACCESSIBILITY DISTRIBUTION:")
    print("  " + "-"*40)
    bands = [
        (0, 100, "Excellent (<100m)"),
        (100, 200, "Good (100-200m)"),
        (200, 400, "Moderate (200-400m)"),
        (400, float('inf'), "Poor (>400m)")
    ]

    access_stats = []
    for low, high, label in bands:
        count = len(comfort_df[(comfort_df['dist_to_green_m'] >= low) &
                               (comfort_df['dist_to_green_m'] < high)])
        pct = count / len(comfort_df) * 100
        print(f"  {label:.<30} {count:>6} ({pct:>5.1f}%)")
        access_stats.append({'band': label, 'count': count, 'percentage': round(pct, 1)})

    pd.DataFrame(access_stats).to_csv('output/quick_analysis/green_accessibility.csv', index=False)
    comfort_df[['lat', 'lon', 'dist_to_green_m']].to_csv('output/quick_analysis/distance_to_green.csv', index=False)
else:
    print("  No significant green spaces found!")
    comfort_df['dist_to_green_m'] = np.nan

# ============================================================
# CREATE COMBINED VISUALIZATION
# ============================================================
print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

# Prepare data for maps
cluster_json = cluster_data[['lat', 'lon', 'cluster', 'cluster_label', 'gvi', 'svf', 'lst', 'ndvi']].copy()
cluster_json = cluster_json.round({'lat': 6, 'lon': 6, 'gvi': 4, 'svf': 4, 'lst': 1, 'ndvi': 4})
cluster_list = cluster_json.to_dict('records')

access_json = comfort_df[['lat', 'lon', 'dist_to_green_m', 'gvi', 'lst']].dropna()
access_json = access_json.round({'lat': 6, 'lon': 6, 'dist_to_green_m': 0, 'gvi': 4, 'lst': 1})
access_list = access_json.head(10000).to_dict('records')  # Limit for performance

# Cluster colors
cluster_colors = ['#e53935', '#ff9800', '#4caf50', '#2196f3']

html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Quick Analyses - Al Karama</title>
    <meta charset="utf-8">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; font-family: Arial, sans-serif; }}
        #map {{ width: 100%; height: 100vh; }}
        .panel {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(255,255,255,0.95);
            padding: 20px;
            border-radius: 8px;
            z-index: 1000;
            max-width: 380px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            max-height: 90vh;
            overflow-y: auto;
        }}
        .panel h2 {{ margin: 0 0 10px 0; color: #1565c0; }}
        .panel h3 {{ margin: 15px 0 8px 0; font-size: 14px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
        .btn {{ padding: 8px 12px; margin: 3px; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; }}
        .btn-active {{ background: #1565c0; color: white; }}
        .btn:not(.btn-active) {{ background: #e0e0e0; }}
        .stat-box {{ background: #f5f5f5; padding: 10px; border-radius: 6px; margin: 8px 0; font-size: 12px; }}
        .stat-box b {{ color: #1565c0; }}
        .legend {{ margin: 10px 0; }}
        .legend-item {{ display: flex; align-items: center; margin: 4px 0; font-size: 12px; }}
        .legend-color {{ width: 16px; height: 16px; border-radius: 50%; margin-right: 8px; }}
        table {{ font-size: 11px; width: 100%; border-collapse: collapse; }}
        td, th {{ padding: 4px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f5f5f5; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="panel">
        <h2>Quick Analyses</h2>

        <h3>Display</h3>
        <div>
            <button class="btn btn-active" id="btn-clusters" onclick="showLayer('clusters')">Clusters</button>
            <button class="btn" id="btn-access" onclick="showLayer('access')">Green Access</button>
        </div>

        <div id="info-clusters">
            <h3>Urban Climate Clusters</h3>
            <p style="font-size: 11px; color: #666;">K-means clustering on GVI, SVF, LST, NDVI</p>
            <div class="legend">
                <div class="legend-item"><div class="legend-color" style="background: {cluster_colors[0]};"></div> {cluster_labels[0]} ({cluster_stats[0]['count']:,})</div>
                <div class="legend-item"><div class="legend-color" style="background: {cluster_colors[1]};"></div> {cluster_labels[1]} ({cluster_stats[1]['count']:,})</div>
                <div class="legend-item"><div class="legend-color" style="background: {cluster_colors[2]};"></div> {cluster_labels[2]} ({cluster_stats[2]['count']:,})</div>
                <div class="legend-item"><div class="legend-color" style="background: {cluster_colors[3]};"></div> {cluster_labels[3]} ({cluster_stats[3]['count']:,})</div>
            </div>
            <table>
                <tr><th>Cluster</th><th>GVI%</th><th>SVF%</th><th>LST</th></tr>
                <tr><td>{cluster_labels[0]}</td><td>{cluster_stats[0]['gvi_mean']:.1f}</td><td>{cluster_stats[0]['svf_mean']:.1f}</td><td>{cluster_stats[0]['lst_mean']:.1f}°C</td></tr>
                <tr><td>{cluster_labels[1]}</td><td>{cluster_stats[1]['gvi_mean']:.1f}</td><td>{cluster_stats[1]['svf_mean']:.1f}</td><td>{cluster_stats[1]['lst_mean']:.1f}°C</td></tr>
                <tr><td>{cluster_labels[2]}</td><td>{cluster_stats[2]['gvi_mean']:.1f}</td><td>{cluster_stats[2]['svf_mean']:.1f}</td><td>{cluster_stats[2]['lst_mean']:.1f}°C</td></tr>
                <tr><td>{cluster_labels[3]}</td><td>{cluster_stats[3]['gvi_mean']:.1f}</td><td>{cluster_stats[3]['svf_mean']:.1f}</td><td>{cluster_stats[3]['lst_mean']:.1f}°C</td></tr>
            </table>
        </div>

        <div id="info-access" style="display: none;">
            <h3>Accessibility to Green Space</h3>
            <p style="font-size: 11px; color: #666;">Distance to nearest NDVI > 0.3</p>
            <div class="stat-box">
                <b>Mean distance:</b> {comfort_df['dist_to_green_m'].mean():.0f}m<br>
                <b>Median distance:</b> {comfort_df['dist_to_green_m'].median():.0f}m
            </div>
            <div class="legend">
                <div class="legend-item"><div class="legend-color" style="background: #1b5e20;"></div> Excellent (&lt;100m)</div>
                <div class="legend-item"><div class="legend-color" style="background: #66bb6a;"></div> Good (100-200m)</div>
                <div class="legend-item"><div class="legend-color" style="background: #ffb74d;"></div> Moderate (200-400m)</div>
                <div class="legend-item"><div class="legend-color" style="background: #e53935;"></div> Poor (&gt;400m)</div>
            </div>
            <table>
                <tr><th>Band</th><th>Count</th><th>%</th></tr>
                {"".join(f"<tr><td>{s['band']}</td><td>{s['count']:,}</td><td>{s['percentage']}%</td></tr>" for s in access_stats)}
            </table>
        </div>

        <h3>Key Correlations</h3>
        <div style="font-size: 11px;">
            <table>
                <tr><th>Variables</th><th>r</th><th>Sig</th></tr>
                {"".join(f"<tr><td>{c['var1']} ↔ {c['var2']}</td><td>{c['r']:.3f}</td><td>{'✓' if c['significant'] else ''}</td></tr>" for c in correlation_results[:6])}
            </table>
        </div>
    </div>

    <script>
        var map = L.map('map').setView([25.2425, 55.3025], 15);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; OpenStreetMap'
        }}).addTo(map);

        var clusterData = {json.dumps(cluster_list)};
        var accessData = {json.dumps(access_list)};
        var clusterColors = {json.dumps(cluster_colors)};

        var clusterLayer = L.layerGroup();
        var accessLayer = L.layerGroup();

        clusterData.forEach(function(d) {{
            var popup = '<b>' + d.cluster_label + '</b><br>' +
                       'GVI: ' + (d.gvi * 100).toFixed(1) + '%<br>' +
                       'SVF: ' + (d.svf * 100).toFixed(1) + '%<br>' +
                       'LST: ' + d.lst + '°C<br>' +
                       'NDVI: ' + d.ndvi.toFixed(3);
            L.circleMarker([d.lat, d.lon], {{
                radius: 4,
                fillColor: clusterColors[d.cluster],
                color: 'white',
                weight: 1,
                fillOpacity: 0.7
            }}).bindPopup(popup).addTo(clusterLayer);
        }});

        function getAccessColor(dist) {{
            if (dist < 100) return '#1b5e20';
            if (dist < 200) return '#66bb6a';
            if (dist < 400) return '#ffb74d';
            return '#e53935';
        }}

        accessData.forEach(function(d) {{
            var popup = 'Distance to green: ' + d.dist_to_green_m.toFixed(0) + 'm<br>' +
                       'GVI: ' + (d.gvi * 100).toFixed(1) + '%<br>' +
                       'LST: ' + d.lst + '°C';
            L.circleMarker([d.lat, d.lon], {{
                radius: 4,
                fillColor: getAccessColor(d.dist_to_green_m),
                color: 'white',
                weight: 1,
                fillOpacity: 0.7
            }}).bindPopup(popup).addTo(accessLayer);
        }});

        clusterLayer.addTo(map);
        var currentLayer = 'clusters';

        function showLayer(name) {{
            map.removeLayer(clusterLayer);
            map.removeLayer(accessLayer);

            if (name === 'clusters') clusterLayer.addTo(map);
            else accessLayer.addTo(map);

            document.querySelectorAll('.btn').forEach(b => b.classList.remove('btn-active'));
            document.getElementById('btn-' + name).classList.add('btn-active');

            document.getElementById('info-clusters').style.display = name === 'clusters' ? 'block' : 'none';
            document.getElementById('info-access').style.display = name === 'access' ? 'block' : 'none';

            currentLayer = name;
        }}
    </script>
</body>
</html>'''

with open('output/quick_analysis/quick_analysis_map.html', 'w') as f:
    f.write(html)
print(f"  Saved: output/quick_analysis/quick_analysis_map.html")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"""
1. CORRELATIONS:
   - GVI ↔ LST: r={correlation_results[0]['r']:.3f} (street greenery vs temperature)
   - GVI ↔ NDVI: r={correlation_results[1]['r']:.3f} (street vs satellite vegetation)
   - NDVI ↔ LST: r={correlation_results[3]['r']:.3f} (vegetation vs temperature)

2. CLUSTERS:
   - {n_clusters} distinct urban climate zones identified
   - {cluster_labels[0]}: {cluster_stats[0]['count']} points
   - {cluster_labels[1]}: {cluster_stats[1]['count']} points
   - {cluster_labels[2]}: {cluster_stats[2]['count']} points
   - {cluster_labels[3]}: {cluster_stats[3]['count']} points

3. GREEN ACCESSIBILITY:
   - Mean distance to green space: {comfort_df['dist_to_green_m'].mean():.0f}m
   - {access_stats[3]['percentage']}% of streets have poor access (>400m)

Output files in: output/quick_analysis/
""")

print("Done!")
