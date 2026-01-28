#!/usr/bin/env python3
"""
Walking Route / Pedestrian Comfort Analysis for Al Karama

Analyzes street-level comfort for pedestrians based on:
- Shade (inverse of SVF)
- Vegetation (GVI)
- Temperature (LST)

Creates comfort-ranked streets and pedestrian comfort map.
"""

import pandas as pd
import numpy as np
import json
import os
from collections import defaultdict

print("="*60)
print("PEDESTRIAN COMFORT / WALKING ROUTE ANALYSIS")
print("Al Karama, Dubai")
print("="*60)

# Load data with satellite info
print("\nLoading data...")
df = pd.read_csv('output/heat_mitigation/priority_scores.csv')
print(f"  Points: {len(df)}")

# Calculate Pedestrian Comfort Index (PCI)
# Higher = more comfortable for walking
print("\nCalculating Pedestrian Comfort Index...")

# Normalize factors (0-1 scale)
# Shade: 1 - SVF (lower SVF = more shade = better)
df['shade'] = 1 - df['svf']

# Temperature comfort: invert and normalize LST
lst_min, lst_max = df['lst'].min(), df['lst'].max()
df['temp_comfort'] = 1 - (df['lst'] - lst_min) / (lst_max - lst_min)

# Vegetation: normalize GVI
gvi_min, gvi_max = df['gvi'].min(), df['gvi'].max()
df['veg_comfort'] = (df['gvi'] - gvi_min) / (gvi_max - gvi_min)

# Combined Pedestrian Comfort Index
# Weights: Temperature (40%), Shade (35%), Vegetation (25%)
df['pci'] = (
    0.40 * df['temp_comfort'] +
    0.35 * df['shade'] +
    0.25 * df['veg_comfort']
)

# Classify comfort levels
df['comfort_level'] = pd.cut(
    df['pci'],
    bins=[0, 0.3, 0.45, 0.6, 1.0],
    labels=['Uncomfortable', 'Moderate', 'Comfortable', 'Very Comfortable']
)

# Statistics
print("\n" + "-"*40)
print("COMFORT DISTRIBUTION")
print("-"*40)
comfort_counts = df['comfort_level'].value_counts()
for level in ['Very Comfortable', 'Comfortable', 'Moderate', 'Uncomfortable']:
    if level in comfort_counts.index:
        count = comfort_counts[level]
        pct = count / len(df) * 100
        print(f"  {level:18s}: {count:5d} points ({pct:.1f}%)")

print(f"\nPCI Statistics:")
print(f"  Mean: {df['pci'].mean():.3f}")
print(f"  Min:  {df['pci'].min():.3f}")
print(f"  Max:  {df['pci'].max():.3f}")

# Group points into street segments (grid-based clustering)
print("\nGrouping into street segments...")
grid_size = 0.0005  # ~50m cells
df['seg_lat'] = (df['lat'] / grid_size).round() * grid_size
df['seg_lon'] = (df['lon'] / grid_size).round() * grid_size
df['segment_id'] = df['seg_lat'].astype(str) + '_' + df['seg_lon'].astype(str)

# Calculate segment statistics
segments = df.groupby('segment_id').agg({
    'lat': 'mean',
    'lon': 'mean',
    'pci': ['mean', 'std', 'count'],
    'lst': 'mean',
    'gvi': 'mean',
    'svf': 'mean',
    'shade': 'mean'
}).reset_index()

segments.columns = ['segment_id', 'lat', 'lon', 'pci_mean', 'pci_std', 'point_count',
                    'lst_mean', 'gvi_mean', 'svf_mean', 'shade_mean']

# Filter segments with enough points
segments = segments[segments['point_count'] >= 3]
print(f"  Street segments: {len(segments)}")

# Rank segments
segments = segments.sort_values('pci_mean', ascending=False).reset_index(drop=True)
segments['rank'] = range(1, len(segments) + 1)

# Top comfortable streets
print("\n" + "-"*40)
print("TOP 15 MOST COMFORTABLE STREET SEGMENTS")
print("-"*40)
print(f"{'Rank':>4} {'Lat':>10} {'Lon':>10} {'PCI':>6} {'LST':>6} {'GVI%':>6} {'Shade%':>7} {'Pts':>4}")
print("-" * 65)
for _, row in segments.head(15).iterrows():
    print(f"{int(row['rank']):>4} {row['lat']:>10.5f} {row['lon']:>10.5f} {row['pci_mean']:>6.3f} "
          f"{row['lst_mean']:>6.1f} {row['gvi_mean']*100:>6.1f} {row['shade_mean']*100:>7.1f} {int(row['point_count']):>4}")

# Least comfortable
print("\n" + "-"*40)
print("TOP 15 LEAST COMFORTABLE (AVOID)")
print("-"*40)
print(f"{'Rank':>4} {'Lat':>10} {'Lon':>10} {'PCI':>6} {'LST':>6} {'GVI%':>6} {'Shade%':>7} {'Pts':>4}")
print("-" * 65)
for _, row in segments.tail(15).iloc[::-1].iterrows():
    print(f"{int(row['rank']):>4} {row['lat']:>10.5f} {row['lon']:>10.5f} {row['pci_mean']:>6.3f} "
          f"{row['lst_mean']:>6.1f} {row['gvi_mean']*100:>6.1f} {row['shade_mean']*100:>7.1f} {int(row['point_count']):>4}")

# Build simple graph for route finding
print("\nBuilding street network graph...")

from scipy.spatial import cKDTree

# Connect nearby segments (within ~100m)
coords = segments[['lat', 'lon']].values
tree = cKDTree(coords)

# Find connections
max_dist = 0.001  # ~100m
edges = []
for i, (lat, lon) in enumerate(coords):
    nearby = tree.query_ball_point([lat, lon], max_dist)
    for j in nearby:
        if i < j:  # Avoid duplicates
            dist = np.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2) * 111000  # to meters
            edges.append((i, j, dist))

print(f"  Network edges: {len(edges)}")

# Simple pathfinding using Dijkstra
import heapq

def build_graph(segments, edges, weight_type='comfort'):
    """Build adjacency list with weights."""
    graph = defaultdict(list)
    for i, j, dist in edges:
        if weight_type == 'distance':
            weight = dist
        elif weight_type == 'comfort':
            # Lower weight = better path
            # Invert PCI so comfortable paths have lower weight
            avg_pci = (segments.iloc[i]['pci_mean'] + segments.iloc[j]['pci_mean']) / 2
            weight = dist * (2 - avg_pci)  # Uncomfortable paths cost more
        graph[i].append((j, weight))
        graph[j].append((i, weight))
    return graph

def dijkstra(graph, start, end):
    """Find shortest path."""
    dist = {start: 0}
    prev = {start: None}
    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if u == end:
            break
        if d > dist.get(u, float('inf')):
            continue
        for v, w in graph[u]:
            new_dist = d + w
            if new_dist < dist.get(v, float('inf')):
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))

    # Reconstruct path
    if end not in prev:
        return None, float('inf')
    path = []
    u = end
    while u is not None:
        path.append(u)
        u = prev[u]
    return path[::-1], dist[end]

# Find example routes
print("\nFinding example routes...")

# Pick start and end points (opposite corners of the area)
lat_sorted = segments.sort_values('lat')
lon_sorted = segments.sort_values('lon')

# Southwest to Northeast corners
sw_idx = segments[(segments['lat'] < segments['lat'].quantile(0.2)) &
                   (segments['lon'] < segments['lon'].quantile(0.2))].index
ne_idx = segments[(segments['lat'] > segments['lat'].quantile(0.8)) &
                   (segments['lon'] > segments['lon'].quantile(0.8))].index

if len(sw_idx) > 0 and len(ne_idx) > 0:
    start_idx = segments.index.get_loc(sw_idx[0])
    end_idx = segments.index.get_loc(ne_idx[0])

    # Build graphs
    dist_graph = build_graph(segments, edges, 'distance')
    comfort_graph = build_graph(segments, edges, 'comfort')

    # Find routes
    short_path, short_cost = dijkstra(dist_graph, start_idx, end_idx)
    comfort_path, comfort_cost = dijkstra(comfort_graph, start_idx, end_idx)

    if short_path and comfort_path:
        # Calculate actual metrics for each route
        def route_metrics(path):
            total_dist = 0
            pcis = []
            lsts = []
            for i in range(len(path) - 1):
                p1, p2 = path[i], path[i+1]
                dist = np.sqrt((coords[p1][0] - coords[p2][0])**2 +
                              (coords[p1][1] - coords[p2][1])**2) * 111000
                total_dist += dist
                pcis.append(segments.iloc[p1]['pci_mean'])
                lsts.append(segments.iloc[p1]['lst_mean'])
            pcis.append(segments.iloc[path[-1]]['pci_mean'])
            lsts.append(segments.iloc[path[-1]]['lst_mean'])
            return total_dist, np.mean(pcis), np.mean(lsts)

        short_dist, short_pci, short_lst = route_metrics(short_path)
        comfort_dist, comfort_pci, comfort_lst = route_metrics(comfort_path)

        print("\n" + "-"*40)
        print("ROUTE COMPARISON EXAMPLE")
        print("-"*40)
        print(f"From: ({segments.iloc[start_idx]['lat']:.5f}, {segments.iloc[start_idx]['lon']:.5f})")
        print(f"To:   ({segments.iloc[end_idx]['lat']:.5f}, {segments.iloc[end_idx]['lon']:.5f})")
        print()
        print(f"{'Route':<20} {'Distance':>10} {'Avg PCI':>10} {'Avg LST':>10}")
        print("-" * 55)
        print(f"{'Shortest':<20} {short_dist:>9.0f}m {short_pci:>10.3f} {short_lst:>9.1f}°C")
        print(f"{'Most Comfortable':<20} {comfort_dist:>9.0f}m {comfort_pci:>10.3f} {comfort_lst:>9.1f}°C")
        print()
        if comfort_dist > short_dist:
            extra = (comfort_dist - short_dist) / short_dist * 100
            temp_saved = short_lst - comfort_lst
            print(f"Comfort route is {extra:.0f}% longer but {temp_saved:.1f}°C cooler on average")

# Save data
os.makedirs('output/walking_routes', exist_ok=True)
df.to_csv('output/walking_routes/point_comfort.csv', index=False)
segments.to_csv('output/walking_routes/segment_comfort.csv', index=False)
print(f"\nSaved: output/walking_routes/point_comfort.csv")
print(f"Saved: output/walking_routes/segment_comfort.csv")

# Create interactive map
print("\nCreating interactive map...")

# Prepare segment data for map
seg_data = segments[['lat', 'lon', 'pci_mean', 'lst_mean', 'gvi_mean', 'shade_mean', 'point_count', 'rank']].copy()
seg_data = seg_data.round({'lat': 6, 'lon': 6, 'pci_mean': 3, 'lst_mean': 1, 'gvi_mean': 4, 'shade_mean': 3})
seg_json = seg_data.to_dict('records')

# Route data if available
route_data = {}
if short_path and comfort_path:
    route_data = {
        'shortest': [[float(segments.iloc[i]['lat']), float(segments.iloc[i]['lon'])] for i in short_path],
        'comfort': [[float(segments.iloc[i]['lat']), float(segments.iloc[i]['lon'])] for i in comfort_path],
        'start': [float(segments.iloc[start_idx]['lat']), float(segments.iloc[start_idx]['lon'])],
        'end': [float(segments.iloc[end_idx]['lat']), float(segments.iloc[end_idx]['lon'])],
        'short_dist': round(short_dist),
        'short_pci': round(short_pci, 3),
        'short_lst': round(short_lst, 1),
        'comfort_dist': round(comfort_dist),
        'comfort_pci': round(comfort_pci, 3),
        'comfort_lst': round(comfort_lst, 1)
    }

# Statistics
very_comfortable = len(df[df['comfort_level'] == 'Very Comfortable'])
comfortable = len(df[df['comfort_level'] == 'Comfortable'])
uncomfortable = len(df[df['comfort_level'] == 'Uncomfortable'])

html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Pedestrian Comfort Map - Al Karama</title>
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
            max-width: 350px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            max-height: 90vh;
            overflow-y: auto;
        }}
        .panel h2 {{ margin: 0 0 5px 0; color: #1565c0; }}
        .panel h3 {{ margin: 15px 0 8px 0; font-size: 14px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
        .stats {{ font-size: 13px; }}
        .stat-row {{ display: flex; justify-content: space-between; padding: 4px 0; }}
        .legend {{ margin: 10px 0; }}
        .legend-item {{ display: flex; align-items: center; margin: 4px 0; font-size: 12px; }}
        .legend-color {{ width: 18px; height: 18px; border-radius: 50%; margin-right: 8px; border: 1px solid #999; }}
        .btn {{ padding: 8px 12px; margin: 3px; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; }}
        .btn-active {{ background: #1565c0; color: white; }}
        .btn:not(.btn-active) {{ background: #e0e0e0; }}
        .route-info {{ background: #e3f2fd; padding: 10px; border-radius: 6px; margin: 10px 0; font-size: 12px; }}
        .route-info b {{ color: #1565c0; }}
        .comfort-high {{ color: #2e7d32; }}
        .comfort-low {{ color: #c62828; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="panel">
        <h2>Pedestrian Comfort Map</h2>
        <p style="margin: 5px 0; color: #666; font-size: 12px;">Walking comfort based on shade, vegetation & temperature</p>

        <h3>Comfort Distribution</h3>
        <div class="stats">
            <div class="stat-row"><span class="comfort-high">Very Comfortable:</span><span>{very_comfortable:,} pts</span></div>
            <div class="stat-row"><span class="comfort-high">Comfortable:</span><span>{comfortable:,} pts</span></div>
            <div class="stat-row"><span class="comfort-low">Uncomfortable:</span><span>{uncomfortable:,} pts</span></div>
        </div>

        <h3>Comfort Index (PCI)</h3>
        <div style="font-size: 11px; color: #666; margin-bottom: 8px;">
            Temperature (40%) + Shade (35%) + Vegetation (25%)
        </div>
        <div class="legend">
            <div class="legend-item"><div class="legend-color" style="background: #1b5e20;"></div> Very Comfortable (>0.6)</div>
            <div class="legend-item"><div class="legend-color" style="background: #66bb6a;"></div> Comfortable (0.45-0.6)</div>
            <div class="legend-item"><div class="legend-color" style="background: #ffb74d;"></div> Moderate (0.3-0.45)</div>
            <div class="legend-item"><div class="legend-color" style="background: #e53935;"></div> Uncomfortable (<0.3)</div>
        </div>

        <h3>Display</h3>
        <div>
            <button class="btn btn-active" id="btn-segments" onclick="showLayer('segments')">Street Segments</button>
            <button class="btn" id="btn-routes" onclick="showLayer('routes')">Route Example</button>
        </div>

        <div id="route-panel" style="display: none;">
            <h3>Route Comparison</h3>
            <div class="route-info">
                <div style="margin-bottom: 8px;"><b>Shortest Route</b> (blue dashed)</div>
                <div>Distance: <span id="short-dist"></span>m</div>
                <div>Avg Comfort: <span id="short-pci"></span></div>
                <div>Avg Temperature: <span id="short-lst"></span>°C</div>
            </div>
            <div class="route-info" style="background: #e8f5e9;">
                <div style="margin-bottom: 8px;"><b style="color: #2e7d32;">Comfort Route</b> (green solid)</div>
                <div>Distance: <span id="comfort-dist"></span>m</div>
                <div>Avg Comfort: <span id="comfort-pci"></span></div>
                <div>Avg Temperature: <span id="comfort-lst"></span>°C</div>
            </div>
            <div id="comparison" style="font-size: 12px; margin-top: 10px; padding: 8px; background: #fff3e0; border-radius: 4px;"></div>
        </div>

        <h3>Top Comfortable Areas</h3>
        <div id="top-areas" style="font-size: 11px; max-height: 120px; overflow-y: auto;"></div>
    </div>

    <script>
        var map = L.map('map').setView([25.2405, 55.3045], 15);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; OpenStreetMap'
        }}).addTo(map);

        var segments = {json.dumps(seg_json)};
        var routeData = {json.dumps(route_data)};

        function getPciColor(pci) {{
            if (pci >= 0.6) return '#1b5e20';
            if (pci >= 0.45) return '#66bb6a';
            if (pci >= 0.3) return '#ffb74d';
            return '#e53935';
        }}

        // Segment layer
        var segmentLayer = L.layerGroup();
        segments.forEach(function(s) {{
            var popup = '<b>Street Segment #' + s.rank + '</b><br>' +
                       'Comfort Index: ' + s.pci_mean.toFixed(3) + '<br>' +
                       'Temperature: ' + s.lst_mean + '°C<br>' +
                       'Vegetation: ' + (s.gvi_mean * 100).toFixed(1) + '%<br>' +
                       'Shade: ' + (s.shade_mean * 100).toFixed(0) + '%<br>' +
                       'Data points: ' + s.point_count;

            L.circleMarker([s.lat, s.lon], {{
                radius: Math.min(12, 5 + s.point_count / 3),
                fillColor: getPciColor(s.pci_mean),
                color: 'white',
                weight: 2,
                fillOpacity: 0.85
            }}).bindPopup(popup).addTo(segmentLayer);
        }});

        // Route layer
        var routeLayer = L.layerGroup();
        if (routeData.shortest) {{
            // Shortest route (blue dashed)
            L.polyline(routeData.shortest, {{
                color: '#1565c0',
                weight: 4,
                dashArray: '10, 10',
                opacity: 0.8
            }}).addTo(routeLayer);

            // Comfort route (green solid)
            L.polyline(routeData.comfort, {{
                color: '#2e7d32',
                weight: 5,
                opacity: 0.9
            }}).addTo(routeLayer);

            // Start/end markers
            L.marker(routeData.start, {{
                icon: L.divIcon({{
                    className: 'start-marker',
                    html: '<div style="background:#1565c0;color:white;padding:5px 10px;border-radius:4px;font-weight:bold;">START</div>',
                    iconSize: [60, 30]
                }})
            }}).addTo(routeLayer);

            L.marker(routeData.end, {{
                icon: L.divIcon({{
                    className: 'end-marker',
                    html: '<div style="background:#c62828;color:white;padding:5px 10px;border-radius:4px;font-weight:bold;">END</div>',
                    iconSize: [50, 30]
                }})
            }}).addTo(routeLayer);

            // Update route panel
            document.getElementById('short-dist').textContent = routeData.short_dist;
            document.getElementById('short-pci').textContent = routeData.short_pci;
            document.getElementById('short-lst').textContent = routeData.short_lst;
            document.getElementById('comfort-dist').textContent = routeData.comfort_dist;
            document.getElementById('comfort-pci').textContent = routeData.comfort_pci;
            document.getElementById('comfort-lst').textContent = routeData.comfort_lst;

            var extraDist = ((routeData.comfort_dist - routeData.short_dist) / routeData.short_dist * 100).toFixed(0);
            var tempSaved = (routeData.short_lst - routeData.comfort_lst).toFixed(1);
            document.getElementById('comparison').innerHTML =
                'Comfort route is <b>' + extraDist + '% longer</b> but <b>' + tempSaved + '°C cooler</b>';
        }}

        // Top areas list
        var topHtml = '';
        segments.slice(0, 10).forEach(function(s, i) {{
            topHtml += '<div style="padding:3px 0;cursor:pointer;" onclick="map.setView([' + s.lat + ',' + s.lon + '],17)">' +
                      (i+1) + '. PCI ' + s.pci_mean.toFixed(2) + ' - ' + s.lst_mean + '°C, ' + (s.shade_mean*100).toFixed(0) + '% shade</div>';
        }});
        document.getElementById('top-areas').innerHTML = topHtml;

        segmentLayer.addTo(map);
        var currentLayer = 'segments';

        function showLayer(name) {{
            map.removeLayer(segmentLayer);
            map.removeLayer(routeLayer);

            document.querySelectorAll('.btn').forEach(b => b.classList.remove('btn-active'));
            document.getElementById('btn-' + name).classList.add('btn-active');
            document.getElementById('route-panel').style.display = name === 'routes' ? 'block' : 'none';

            if (name === 'segments') {{
                segmentLayer.addTo(map);
            }} else if (name === 'routes') {{
                segmentLayer.addTo(map);
                routeLayer.addTo(map);
                if (routeData.start) {{
                    map.fitBounds([routeData.start, routeData.end], {{padding: [50, 50]}});
                }}
            }}
            currentLayer = name;
        }}
    </script>
</body>
</html>'''

with open('output/walking_routes/pedestrian_comfort_map.html', 'w') as f:
    f.write(html)
print(f"Saved: output/walking_routes/pedestrian_comfort_map.html")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"""
Pedestrian Comfort Analysis Complete:

1. COMFORT INDEX (PCI):
   - Combines temperature, shade, and vegetation
   - Scale 0-1 (higher = more comfortable)
   - Mean PCI: {df['pci'].mean():.3f}

2. STREET SEGMENTS:
   - {len(segments)} segments analyzed
   - Best segment PCI: {segments['pci_mean'].max():.3f}
   - Worst segment PCI: {segments['pci_mean'].min():.3f}

3. COMFORT DISTRIBUTION:
   - Very Comfortable: {very_comfortable:,} points ({very_comfortable/len(df)*100:.1f}%)
   - Comfortable: {comfortable:,} points ({comfortable/len(df)*100:.1f}%)
   - Uncomfortable: {uncomfortable:,} points ({uncomfortable/len(df)*100:.1f}%)

4. ROUTE COMPARISON:
   - Shortest vs comfort-optimized routes shown
   - Comfort route trades distance for cooler walking

Open the map: output/walking_routes/pedestrian_comfort_map.html
""")

print("Done!")
