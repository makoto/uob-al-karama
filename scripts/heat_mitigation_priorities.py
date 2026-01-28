#!/usr/bin/env python3
"""
Heat Mitigation Priority Analysis for Al Karama

Identifies locations where interventions (tree planting, shade structures)
would have the most impact based on:
- High Land Surface Temperature (LST)
- Low Green View Index (GVI)
- Low NDVI (satellite vegetation)
- High Sky View Factor (SVF) = less shade
"""

import pandas as pd
import numpy as np
import json
import os

print("="*60)
print("HEAT MITIGATION PRIORITY ANALYSIS")
print("Al Karama, Dubai")
print("="*60)

# Load SVI data (GVI, SVF)
print("\nLoading street-level data...")
svi_df = pd.read_csv('output/gvi_svf_combined.csv')
print(f"  SVI points: {len(svi_df)}")

# Load satellite data (10m resolution)
print("\nLoading satellite data...")
sat_df = pd.read_csv('output/satellite_full/full_area_data.csv')
print(f"  Satellite cells: {len(sat_df)}")

# Calculate statistics
print("\n" + "-"*40)
print("BASELINE STATISTICS")
print("-"*40)

lst_mean = sat_df['lst'].mean()
lst_std = sat_df['lst'].std()
ndvi_mean = sat_df['ndvi'].mean()
gvi_mean = svi_df['gvi'].mean()
svf_mean = svi_df['svf'].mean()

print(f"LST:  mean={lst_mean:.1f}°C, std={lst_std:.1f}°C")
print(f"NDVI: mean={ndvi_mean:.3f}")
print(f"GVI:  mean={gvi_mean:.3f}")
print(f"SVF:  mean={svf_mean:.3f}")

# For each SVI point, find nearest satellite data
print("\nMatching SVI points to satellite data...")

from scipy.spatial import cKDTree

# Build KD-tree for satellite points
sat_coords = sat_df[['lat', 'lon']].values
sat_tree = cKDTree(sat_coords)

# Find nearest satellite cell for each SVI point
svi_coords = svi_df[['lat', 'lon']].values
distances, indices = sat_tree.query(svi_coords, k=1)

# Add satellite data to SVI points
svi_df['lst'] = sat_df.iloc[indices]['lst'].values
svi_df['ndvi'] = sat_df.iloc[indices]['ndvi'].values
svi_df['ndbi'] = sat_df.iloc[indices]['ndbi'].values

print(f"  Matched {len(svi_df)} points")

# Calculate priority score
# Higher score = higher priority for intervention
print("\nCalculating priority scores...")

# Normalize each factor to 0-1 scale
# For LST: higher temp = higher priority
svi_df['lst_norm'] = (svi_df['lst'] - svi_df['lst'].min()) / (svi_df['lst'].max() - svi_df['lst'].min())

# For GVI: lower vegetation = higher priority (invert)
svi_df['gvi_norm'] = 1 - (svi_df['gvi'] - svi_df['gvi'].min()) / (svi_df['gvi'].max() - svi_df['gvi'].min())

# For NDVI: lower vegetation = higher priority (invert)
svi_df['ndvi_norm'] = 1 - (svi_df['ndvi'] - svi_df['ndvi'].min()) / (svi_df['ndvi'].max() - svi_df['ndvi'].min())

# For SVF: higher sky view = less shade = higher priority
svi_df['svf_norm'] = (svi_df['svf'] - svi_df['svf'].min()) / (svi_df['svf'].max() - svi_df['svf'].min())

# Combined priority score (weighted average)
# Weights: LST (40%), GVI (25%), NDVI (20%), SVF (15%)
svi_df['priority_score'] = (
    0.40 * svi_df['lst_norm'] +
    0.25 * svi_df['gvi_norm'] +
    0.20 * svi_df['ndvi_norm'] +
    0.15 * svi_df['svf_norm']
)

# Classify priority levels
svi_df['priority_level'] = pd.cut(
    svi_df['priority_score'],
    bins=[0, 0.4, 0.6, 0.75, 1.0],
    labels=['Low', 'Medium', 'High', 'Critical']
)

# Statistics
print("\n" + "-"*40)
print("PRIORITY DISTRIBUTION")
print("-"*40)
priority_counts = svi_df['priority_level'].value_counts()
for level in ['Critical', 'High', 'Medium', 'Low']:
    if level in priority_counts.index:
        count = priority_counts[level]
        pct = count / len(svi_df) * 100
        print(f"  {level:10s}: {count:5d} points ({pct:.1f}%)")

# Top priority locations
print("\n" + "-"*40)
print("TOP 20 PRIORITY LOCATIONS")
print("-"*40)

top_priorities = svi_df.nlargest(20, 'priority_score')[
    ['lat', 'lon', 'lst', 'gvi', 'ndvi', 'svf', 'priority_score']
].reset_index(drop=True)

print(f"{'#':>3} {'Lat':>10} {'Lon':>10} {'LST':>6} {'GVI':>6} {'NDVI':>6} {'SVF':>6} {'Score':>6}")
print("-" * 70)
for i, row in top_priorities.iterrows():
    print(f"{i+1:>3} {row['lat']:>10.6f} {row['lon']:>10.6f} {row['lst']:>6.1f} {row['gvi']:>6.3f} {row['ndvi']:>6.3f} {row['svf']:>6.3f} {row['priority_score']:>6.3f}")

# Save priority data
os.makedirs('output/heat_mitigation', exist_ok=True)
svi_df.to_csv('output/heat_mitigation/priority_scores.csv', index=False)
print(f"\nSaved: output/heat_mitigation/priority_scores.csv")

# Create interactive map
print("\nCreating interactive map...")

# Prepare data for map
map_data = svi_df[['lat', 'lon', 'lst', 'gvi', 'ndvi', 'svf', 'priority_score', 'priority_level']].copy()
map_data['priority_level'] = map_data['priority_level'].astype(str)

# Sample for visualization (performance)
if len(map_data) > 5000:
    # Keep all Critical and High priority, sample the rest
    critical_high = map_data[map_data['priority_level'].isin(['Critical', 'High'])]
    other = map_data[~map_data['priority_level'].isin(['Critical', 'High'])].sample(n=min(3000, len(map_data) - len(critical_high)), random_state=42)
    map_data = pd.concat([critical_high, other])
    print(f"  Displaying {len(map_data)} points (all Critical/High + sampled others)")

data_json = map_data.to_dict('records')

# Priority statistics for the map
critical_count = len(svi_df[svi_df['priority_level'] == 'Critical'])
high_count = len(svi_df[svi_df['priority_level'] == 'High'])
critical_pct = critical_count / len(svi_df) * 100
high_pct = high_count / len(svi_df) * 100

# Hotspot clusters (areas with high concentration of priority points)
print("\nIdentifying hotspot clusters...")

# Grid-based clustering
grid_size = 0.001  # ~100m cells
svi_df['grid_lat'] = (svi_df['lat'] / grid_size).round() * grid_size
svi_df['grid_lon'] = (svi_df['lon'] / grid_size).round() * grid_size

hotspots = svi_df[svi_df['priority_level'].isin(['Critical', 'High'])].groupby(
    ['grid_lat', 'grid_lon']
).agg({
    'priority_score': ['mean', 'count'],
    'lst': 'mean',
    'gvi': 'mean'
}).reset_index()

hotspots.columns = ['lat', 'lon', 'avg_score', 'point_count', 'avg_lst', 'avg_gvi']
hotspots = hotspots[hotspots['point_count'] >= 5].nlargest(10, 'avg_score')

print(f"\nTop 10 Hotspot Areas (clusters of {'>'}=5 high-priority points):")
print(f"{'#':>3} {'Lat':>10} {'Lon':>10} {'Points':>7} {'Avg LST':>8} {'Avg GVI':>8} {'Score':>6}")
print("-" * 60)
for i, row in hotspots.iterrows():
    print(f"{hotspots.index.get_loc(i)+1:>3} {row['lat']:>10.5f} {row['lon']:>10.5f} {int(row['point_count']):>7} {row['avg_lst']:>8.1f} {row['avg_gvi']:>8.3f} {row['avg_score']:>6.3f}")

hotspot_data = hotspots.to_dict('records')

html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Heat Mitigation Priorities - Al Karama</title>
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
        }}
        .panel h2 {{ margin: 0 0 5px 0; color: #d32f2f; }}
        .panel h3 {{ margin: 15px 0 10px 0; font-size: 14px; }}
        .stats {{ margin: 10px 0; }}
        .stat-row {{ display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #eee; }}
        .stat-label {{ color: #666; }}
        .stat-value {{ font-weight: bold; }}
        .critical {{ color: #d32f2f; }}
        .high {{ color: #f57c00; }}
        .medium {{ color: #fbc02d; }}
        .low {{ color: #388e3c; }}
        .legend {{ margin-top: 15px; }}
        .legend-item {{ display: flex; align-items: center; margin: 5px 0; font-size: 13px; }}
        .legend-color {{ width: 20px; height: 20px; border-radius: 50%; margin-right: 10px; }}
        .hotspot-list {{ max-height: 150px; overflow-y: auto; font-size: 12px; }}
        .hotspot-item {{ padding: 5px; background: #fff3e0; margin: 3px 0; border-radius: 4px; cursor: pointer; }}
        .hotspot-item:hover {{ background: #ffe0b2; }}
        .btn {{ padding: 8px 12px; margin: 3px; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; }}
        .btn-active {{ background: #1565c0; color: white; }}
        .btn:not(.btn-active) {{ background: #e0e0e0; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="panel">
        <h2>Heat Mitigation Priorities</h2>
        <p style="margin: 5px 0; color: #666; font-size: 13px;">Locations for urban greening interventions</p>

        <div class="stats">
            <div class="stat-row">
                <span class="stat-label">Total points analyzed:</span>
                <span class="stat-value">{len(svi_df):,}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label critical">Critical priority:</span>
                <span class="stat-value critical">{critical_count:,} ({critical_pct:.1f}%)</span>
            </div>
            <div class="stat-row">
                <span class="stat-label high">High priority:</span>
                <span class="stat-value high">{high_count:,} ({high_pct:.1f}%)</span>
            </div>
        </div>

        <h3>Priority Score Components:</h3>
        <div style="font-size: 12px; color: #666;">
            <div>• Land Surface Temp (40%)</div>
            <div>• Low Street Greenery (25%)</div>
            <div>• Low Satellite NDVI (20%)</div>
            <div>• High Sky Exposure (15%)</div>
        </div>

        <h3>Display:</h3>
        <div>
            <button class="btn btn-active" id="btn-all" onclick="showLayer('all')">All Points</button>
            <button class="btn" id="btn-critical" onclick="showLayer('critical')">Critical Only</button>
            <button class="btn" id="btn-hotspots" onclick="showLayer('hotspots')">Hotspot Areas</button>
        </div>

        <div class="legend">
            <div class="legend-item"><div class="legend-color" style="background: #d32f2f;"></div> Critical (>{75}%)</div>
            <div class="legend-item"><div class="legend-color" style="background: #f57c00;"></div> High (60-75%)</div>
            <div class="legend-item"><div class="legend-color" style="background: #fbc02d;"></div> Medium (40-60%)</div>
            <div class="legend-item"><div class="legend-color" style="background: #388e3c;"></div> Low (<40%)</div>
        </div>

        <h3>Top Hotspot Clusters:</h3>
        <div class="hotspot-list" id="hotspots-list"></div>
    </div>

    <script>
        var map = L.map('map').setView([25.2405, 55.3045], 15);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; OpenStreetMap'
        }}).addTo(map);

        var data = {json.dumps(data_json)};
        var hotspots = {json.dumps(hotspot_data)};

        function getPriorityColor(level) {{
            switch(level) {{
                case 'Critical': return '#d32f2f';
                case 'High': return '#f57c00';
                case 'Medium': return '#fbc02d';
                default: return '#388e3c';
            }}
        }}

        var allLayer = L.layerGroup();
        var criticalLayer = L.layerGroup();
        var hotspotLayer = L.layerGroup();

        data.forEach(function(d) {{
            var popup = '<b>Priority: ' + d.priority_level + '</b><br>' +
                       'Score: ' + d.priority_score.toFixed(3) + '<br><hr>' +
                       'LST: ' + d.lst.toFixed(1) + '°C<br>' +
                       'GVI: ' + (d.gvi * 100).toFixed(1) + '%<br>' +
                       'NDVI: ' + d.ndvi.toFixed(3) + '<br>' +
                       'SVF: ' + (d.svf * 100).toFixed(1) + '%';

            var marker = L.circleMarker([d.lat, d.lon], {{
                radius: d.priority_level === 'Critical' ? 6 : (d.priority_level === 'High' ? 5 : 4),
                fillColor: getPriorityColor(d.priority_level),
                color: 'white',
                weight: 1,
                fillOpacity: 0.8
            }}).bindPopup(popup);

            allLayer.addLayer(marker);

            if (d.priority_level === 'Critical') {{
                criticalLayer.addLayer(L.circleMarker([d.lat, d.lon], {{
                    radius: 8,
                    fillColor: '#d32f2f',
                    color: 'white',
                    weight: 2,
                    fillOpacity: 0.9
                }}).bindPopup(popup));
            }}
        }});

        // Hotspot circles
        hotspots.forEach(function(h, i) {{
            var circle = L.circle([h.lat, h.lon], {{
                radius: 80,
                fillColor: '#d32f2f',
                color: '#b71c1c',
                weight: 3,
                fillOpacity: 0.3
            }}).bindPopup('<b>Hotspot #' + (i+1) + '</b><br>' +
                         'High-priority points: ' + h.point_count + '<br>' +
                         'Avg LST: ' + h.avg_lst.toFixed(1) + '°C<br>' +
                         'Avg GVI: ' + (h.avg_gvi * 100).toFixed(1) + '%');
            hotspotLayer.addLayer(circle);
        }});

        // Populate hotspot list
        var listHtml = '';
        hotspots.forEach(function(h, i) {{
            listHtml += '<div class="hotspot-item" onclick="map.setView([' + h.lat + ',' + h.lon + '], 17)">' +
                       '#' + (i+1) + ': ' + h.point_count + ' pts, ' + h.avg_lst.toFixed(1) + '°C, GVI ' + (h.avg_gvi*100).toFixed(0) + '%</div>';
        }});
        document.getElementById('hotspots-list').innerHTML = listHtml;

        allLayer.addTo(map);
        var currentLayer = 'all';

        function showLayer(name) {{
            map.removeLayer(allLayer);
            map.removeLayer(criticalLayer);
            map.removeLayer(hotspotLayer);

            document.querySelectorAll('.btn').forEach(b => b.classList.remove('btn-active'));
            document.getElementById('btn-' + name).classList.add('btn-active');

            if (name === 'all') allLayer.addTo(map);
            else if (name === 'critical') criticalLayer.addTo(map);
            else if (name === 'hotspots') {{ hotspotLayer.addTo(map); allLayer.addTo(map); }}

            currentLayer = name;
        }}
    </script>
</body>
</html>'''

with open('output/heat_mitigation/priority_map.html', 'w') as f:
    f.write(html)
print(f"Saved: output/heat_mitigation/priority_map.html")

# Summary recommendations
print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)

print(f"""
Based on the analysis of {len(svi_df):,} street-level points:

1. CRITICAL AREAS ({critical_count} locations, {critical_pct:.1f}%):
   - These have the highest combination of heat + low vegetation
   - Priority for immediate intervention (tree planting, shade structures)

2. HIGH PRIORITY AREAS ({high_count} locations, {high_pct:.1f}%):
   - Significant heat stress with inadequate greenery
   - Consider for medium-term urban greening projects

3. TOP 10 HOTSPOT CLUSTERS identified:
   - Concentrated areas with multiple high-priority points
   - Best candidates for area-wide interventions
   - Click hotspots in the map to explore

4. INTERVENTION SUGGESTIONS:
   - Tree planting: Focus on areas with GVI < 2% and LST > 50°C
   - Shade structures: Where SVF > 40% (high sky exposure)
   - Green walls/roofs: High NDBI areas with limited ground space

Open the map to explore: output/heat_mitigation/priority_map.html
""")

print("Done!")
