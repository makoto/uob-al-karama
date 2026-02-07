#!/usr/bin/env python3
"""
Origin-Destination Weighted Betweenness for Al Karama
======================================================

Implements OD-weighted betweenness centrality using NetworkX.
Shows which streets people ACTUALLY walk on based on land use.

This is the key insight from Madina's approach, implemented directly
without the buggy third-party library.
"""

import osmnx as ox
import networkx as nx
import pandas as pd
import geopandas as gpd
import numpy as np
import json
import os
from scipy.spatial import cKDTree
from shapely.geometry import Point, box
from collections import defaultdict

print("="*60)
print("OD-WEIGHTED BETWEENNESS ANALYSIS")
print("Al Karama, Dubai")
print("="*60)

# Paths
BASE = os.path.dirname(__file__)
POIS_PATH = os.path.join(BASE, '..', 'docs', 'data', 'al_karama', 'pois.json')
OUT_DIR = os.path.join(BASE, '..', 'output', 'od_betweenness')
os.makedirs(OUT_DIR, exist_ok=True)

# Al Karama boundary
ak_north, ak_south = 25.255, 25.230
ak_east, ak_west = 55.315, 55.290

# Download street network
print("\nDownloading street network from OpenStreetMap...")
G = ox.graph_from_bbox(bbox=(ak_north, ak_south, ak_east, ak_west), network_type='walk')
G_undir = G.to_undirected()
print(f"  Nodes: {G_undir.number_of_nodes()}")
print(f"  Edges: {G_undir.number_of_edges()}")

# Load POIs
print("\nLoading POIs...")
with open(POIS_PATH) as f:
    pois_raw = json.load(f)

# Define origin and destination types
ORIGIN_TYPES = ['hotel', 'bus_station', 'company', 'diplomatic']
DESTINATION_TYPES = ['restaurant', 'cafe', 'fast_food', 'supermarket', 'pharmacy',
                     'bank', 'clothes', 'furniture', 'hairdresser', 'beauty',
                     'laundry', 'copyshop']

# Categorize POIs
origins = []
destinations = []
for p in pois_raw:
    lon, lat = p['position']
    poi_type = p.get('type', '')
    if poi_type in ORIGIN_TYPES:
        origins.append((lat, lon, poi_type))
    elif poi_type in DESTINATION_TYPES:
        destinations.append((lat, lon, poi_type))

print(f"  Origins (hotels, offices, transit): {len(origins)}")
print(f"  Destinations (shops, food, services): {len(destinations)}")

# Find nearest network nodes for each POI
print("\nMapping POIs to network nodes...")
node_coords = np.array([[G_undir.nodes[n]['y'], G_undir.nodes[n]['x']] for n in G_undir.nodes()])
node_ids = list(G_undir.nodes())
tree = cKDTree(node_coords)

def get_nearest_node(lat, lon):
    _, idx = tree.query([lat, lon])
    return node_ids[idx]

origin_nodes = [get_nearest_node(lat, lon) for lat, lon, _ in origins]
destination_nodes = [get_nearest_node(lat, lon) for lat, lon, _ in destinations]

# Remove duplicates
origin_nodes = list(set(origin_nodes))
destination_nodes = list(set(destination_nodes))

print(f"  Unique origin nodes: {len(origin_nodes)}")
print(f"  Unique destination nodes: {len(destination_nodes)}")

# Calculate OD-weighted betweenness
print("\n" + "-"*60)
print("CALCULATING OD-WEIGHTED BETWEENNESS")
print("-"*60)

print("\nThis calculates which edges lie on paths from origins to destinations...")

# Initialize edge betweenness
edge_betweenness = defaultdict(float)

# For each origin, find shortest paths to all destinations
total_paths = 0
for i, origin in enumerate(origin_nodes):
    if (i + 1) % 20 == 0:
        print(f"  Processing origin {i+1}/{len(origin_nodes)}...")

    for dest in destination_nodes:
        if origin == dest:
            continue

        try:
            path = nx.shortest_path(G_undir, origin, dest, weight='length')
            total_paths += 1

            # Count edge traversals (with decay based on path length)
            path_length = sum(G_undir[path[j]][path[j+1]].get('length', 1) for j in range(len(path)-1))

            # Distance decay (exp decay with beta=0.003 means ~50% at 230m)
            decay_factor = np.exp(-0.003 * path_length)

            for j in range(len(path) - 1):
                edge = tuple(sorted([path[j], path[j+1]]))
                edge_betweenness[edge] += decay_factor

        except nx.NetworkXNoPath:
            continue

print(f"\n  Total valid paths: {total_paths}")
print(f"  Edges with traffic: {len(edge_betweenness)}")

# Also calculate standard betweenness for comparison
print("\nCalculating standard betweenness (all-pairs) for comparison...")
standard_bc = nx.edge_betweenness_centrality(G_undir, weight='length', normalized=True)

# Convert to GeoDataFrame
print("\nCreating GeoDataFrame...")
nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_undir)

# Add betweenness values to edges
edges_gdf['od_betweenness'] = 0.0
edges_gdf['std_betweenness'] = 0.0

for idx, row in edges_gdf.iterrows():
    # idx is (u, v, key) tuple for MultiGraph edges
    u, v, k = idx[0], idx[1], idx[2] if len(idx) > 2 else 0
    edge_key = tuple(sorted([u, v]))

    # OD-weighted (uses sorted 2-tuple)
    edges_gdf.at[idx, 'od_betweenness'] = edge_betweenness.get(edge_key, 0)

    # Standard BC returns 3-tuples (u, v, key) for multigraphs
    std_val = standard_bc.get((u, v, k), 0)
    if std_val == 0:
        std_val = standard_bc.get((v, u, k), 0)
    edges_gdf.at[idx, 'std_betweenness'] = std_val

# Debug: check if any standard values were found
std_nonzero = (edges_gdf['std_betweenness'] > 0).sum()
print(f"  Edges with standard betweenness > 0: {std_nonzero}")

# Normalize OD betweenness
od_max = edges_gdf['od_betweenness'].max()
if od_max > 0:
    edges_gdf['od_norm'] = edges_gdf['od_betweenness'] / od_max
else:
    edges_gdf['od_norm'] = 0

# Calculate midpoints
edges_gdf['mid_lon'] = edges_gdf.geometry.centroid.x
edges_gdf['mid_lat'] = edges_gdf.geometry.centroid.y

# Compare top streets
print("\n" + "-"*60)
print("COMPARISON: OD-WEIGHTED vs STANDARD BETWEENNESS")
print("-"*60)

# Top by OD-weighted
print("\nTOP 10 by OD-Weighted (actual pedestrian routes):")
top_od = edges_gdf.nlargest(10, 'od_betweenness')
for i, (idx, row) in enumerate(top_od.iterrows()):
    print(f"{i+1:>3}. OD: {row['od_betweenness']:>8.2f}  Std: {row['std_betweenness']:.6f}  at ({row['mid_lat']:.5f}, {row['mid_lon']:.5f})")

# Top by standard
print("\nTOP 10 by Standard (theoretical centrality):")
top_std = edges_gdf.nlargest(10, 'std_betweenness')
for i, (idx, row) in enumerate(top_std.iterrows()):
    print(f"{i+1:>3}. OD: {row['od_betweenness']:>8.2f}  Std: {row['std_betweenness']:.6f}  at ({row['mid_lat']:.5f}, {row['mid_lon']:.5f})")

# Correlation between the two metrics
correlation = edges_gdf['od_betweenness'].corr(edges_gdf['std_betweenness'])
print(f"\nCorrelation between OD and Standard: {correlation:.3f}")

# Find streets that are high in OD but low in standard (underrated by traditional analysis)
edges_gdf['od_rank'] = edges_gdf['od_betweenness'].rank(ascending=False)
edges_gdf['std_rank'] = edges_gdf['std_betweenness'].rank(ascending=False)
edges_gdf['rank_diff'] = edges_gdf['std_rank'] - edges_gdf['od_rank']  # positive = underrated by standard

print("\nSTREETS UNDERRATED BY STANDARD ANALYSIS (high OD, low standard):")
underrated = edges_gdf[edges_gdf['od_betweenness'] > edges_gdf['od_betweenness'].quantile(0.75)]
underrated = underrated.nlargest(10, 'rank_diff')
for i, (idx, row) in enumerate(underrated.iterrows()):
    print(f"{i+1:>3}. OD rank: {int(row['od_rank']):>4}  Std rank: {int(row['std_rank']):>4}  diff: {int(row['rank_diff']):>4}  at ({row['mid_lat']:.5f}, {row['mid_lon']:.5f})")

# Save results
print("\nSaving results...")
edges_export = edges_gdf[['geometry', 'length', 'od_betweenness', 'std_betweenness',
                          'od_norm', 'mid_lat', 'mid_lon']].copy()
edges_export = edges_export.reset_index(drop=True)
edges_export.to_file(os.path.join(OUT_DIR, 'od_betweenness.geojson'), driver='GeoJSON')
print(f"  Saved: {OUT_DIR}/od_betweenness.geojson")

# Create visualization HTML
print("\nCreating interactive map...")

map_data = []
for idx, row in edges_gdf.reset_index(drop=True).iterrows():
    coords = list(row.geometry.coords)
    map_data.append({
        'coords': [[c[1], c[0]] for c in coords],
        'od': round(float(row['od_betweenness']), 2),
        'std': round(float(row['std_betweenness']), 6),
        'od_norm': round(float(row['od_norm']), 4),
        'length': round(float(row['length']), 1) if pd.notna(row['length']) else 0,
        'lat': round(float(row['mid_lat']), 5)
    })

# Origin/destination markers
origin_markers = [{'lat': lat, 'lon': lon, 'type': t} for lat, lon, t in origins]
dest_markers = [{'lat': lat, 'lon': lon, 'type': t} for lat, lon, t in destinations]

od_max = float(edges_gdf['od_betweenness'].max())
std_max = float(edges_gdf['std_betweenness'].max())

html = f'''<!DOCTYPE html>
<html>
<head>
    <title>OD-Weighted Betweenness - Al Karama</title>
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
        .panel h2 {{ margin: 0 0 5px 0; color: #d32f2f; }}
        .panel h3 {{ margin: 15px 0 8px 0; font-size: 14px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
        .stats {{ font-size: 13px; }}
        .stat-row {{ display: flex; justify-content: space-between; padding: 4px 0; }}
        .legend {{ margin: 10px 0; }}
        .legend-item {{ display: flex; align-items: center; margin: 4px 0; font-size: 12px; }}
        .legend-color {{ width: 30px; height: 4px; margin-right: 8px; }}
        .btn {{ padding: 8px 12px; margin: 3px; border: none; border-radius: 4px; cursor: pointer; font-size: 11px; }}
        .btn-active {{ background: #d32f2f; color: white; }}
        .btn:not(.btn-active) {{ background: #e0e0e0; }}
        .insight {{ background: #fff3e0; padding: 10px; border-radius: 6px; font-size: 12px; margin: 10px 0; line-height: 1.5; }}
        .metric-desc {{ font-size: 11px; color: #666; margin: 5px 0 10px 0; line-height: 1.4; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="panel">
        <h2>OD-Weighted Betweenness</h2>
        <p style="margin: 5px 0; color: #666; font-size: 12px;">Where people actually walk vs theoretical centrality</p>

        <div class="insight">
            <b>Key Insight:</b> OD-weighted analysis shows streets used for actual trips
            (hotel → restaurant, office → shop) rather than theoretical shortest paths.
            <br><br>
            Correlation between methods: <b>{correlation:.2f}</b>
        </div>

        <h3>Display Mode</h3>
        <div>
            <button class="btn btn-active" id="btn-od" onclick="showMetric('od')">OD-Weighted</button>
            <button class="btn" id="btn-std" onclick="showMetric('std')">Standard</button>
            <button class="btn" id="btn-diff" onclick="showMetric('diff')">Difference</button>
        </div>

        <div id="metric-info">
            <h3 id="metric-title">OD-Weighted Betweenness</h3>
            <div class="metric-desc" id="metric-desc">
                Streets weighted by actual trips from hotels/offices to shops/restaurants.
                Shows where pedestrians actually walk.
            </div>
        </div>

        <h3>Legend</h3>
        <div class="legend" id="legend-od">
            <div class="legend-item"><div class="legend-color" style="background: #ffcdd2;"></div> Low pedestrian traffic</div>
            <div class="legend-item"><div class="legend-color" style="background: #e53935;"></div> Medium traffic</div>
            <div class="legend-item"><div class="legend-color" style="background: #b71c1c;"></div> High traffic corridor</div>
        </div>
        <div class="legend" id="legend-std" style="display:none;">
            <div class="legend-item"><div class="legend-color" style="background: #bbdefb;"></div> Low centrality</div>
            <div class="legend-item"><div class="legend-color" style="background: #1976d2;"></div> Medium centrality</div>
            <div class="legend-item"><div class="legend-color" style="background: #0d47a1;"></div> High centrality</div>
        </div>
        <div class="legend" id="legend-diff" style="display:none;">
            <div class="legend-item"><div class="legend-color" style="background: #4caf50;"></div> Underrated by standard</div>
            <div class="legend-item"><div class="legend-color" style="background: #9e9e9e;"></div> Similar ranking</div>
            <div class="legend-item"><div class="legend-color" style="background: #9c27b0;"></div> Overrated by standard</div>
        </div>

        <h3>Show POIs</h3>
        <div>
            <button class="btn" id="btn-origins" onclick="toggleMarkers('origins')">Origins ({len(origins)})</button>
            <button class="btn" id="btn-dests" onclick="toggleMarkers('dests')">Destinations ({len(destinations)})</button>
        </div>

        <h3>Statistics</h3>
        <div class="stats">
            <div class="stat-row"><span>Total paths analyzed:</span><span>{total_paths:,}</span></div>
            <div class="stat-row"><span>Edges with OD traffic:</span><span>{len(edge_betweenness):,}</span></div>
            <div class="stat-row"><span>Max OD betweenness:</span><span>{od_max:.1f}</span></div>
        </div>
    </div>

    <script>
        var map = L.map('map').setView([25.2425, 55.3025], 15);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; OpenStreetMap'
        }}).addTo(map);

        var edges = {json.dumps(map_data)};
        var origins = {json.dumps(origin_markers)};
        var dests = {json.dumps(dest_markers)};
        var odMax = {od_max};
        var stdMax = {std_max};

        var layers = {{}};
        ['od', 'std', 'diff'].forEach(m => layers[m] = L.layerGroup());

        function getOdColor(val) {{
            var ratio = val / odMax;
            if (ratio < 0.1) return '#ffcdd2';
            if (ratio < 0.3) return '#ef5350';
            if (ratio < 0.6) return '#e53935';
            return '#b71c1c';
        }}

        function getStdColor(val) {{
            var ratio = val / stdMax;
            if (ratio < 0.2) return '#bbdefb';
            if (ratio < 0.5) return '#42a5f5';
            if (ratio < 0.75) return '#1976d2';
            return '#0d47a1';
        }}

        function getDiffColor(od, std) {{
            var odRank = od / odMax;
            var stdRank = std / stdMax;
            var diff = odRank - stdRank;
            if (diff > 0.2) return '#4caf50';  // Underrated
            if (diff < -0.2) return '#9c27b0'; // Overrated
            return '#9e9e9e';
        }}

        edges.forEach(function(e) {{
            var popup = '<b>Street Segment</b><br>' +
                       'Length: ' + e.length + 'm<br>' +
                       'OD Betweenness: ' + e.od.toFixed(1) + '<br>' +
                       'Std Betweenness: ' + e.std.toFixed(5);

            var odW = 2 + (e.od / odMax) * 8;
            var stdW = 2 + (e.std / stdMax) * 6;
            var diffW = 3;

            L.polyline(e.coords, {{ color: getOdColor(e.od), weight: odW, opacity: 0.8 }}).bindPopup(popup).addTo(layers.od);
            L.polyline(e.coords, {{ color: getStdColor(e.std), weight: stdW, opacity: 0.8 }}).bindPopup(popup).addTo(layers.std);
            L.polyline(e.coords, {{ color: getDiffColor(e.od, e.std), weight: diffW, opacity: 0.8 }}).bindPopup(popup).addTo(layers.diff);
        }});

        // POI markers
        var originMarkers = L.layerGroup();
        var destMarkers = L.layerGroup();

        origins.forEach(function(o) {{
            L.circleMarker([o.lat, o.lon], {{
                radius: 6, fillColor: '#1565c0', color: 'white', weight: 2, fillOpacity: 0.9
            }}).bindPopup('Origin: ' + o.type).addTo(originMarkers);
        }});

        dests.forEach(function(d) {{
            L.circleMarker([d.lat, d.lon], {{
                radius: 5, fillColor: '#2e7d32', color: 'white', weight: 1, fillOpacity: 0.8
            }}).bindPopup('Destination: ' + d.type).addTo(destMarkers);
        }});

        layers.od.addTo(map);
        var currentMetric = 'od';
        var showOrigins = false;
        var showDests = false;

        var metricInfo = {{
            od: {{ title: 'OD-Weighted Betweenness', desc: 'Streets weighted by actual trips from hotels/offices to shops/restaurants. Shows where pedestrians actually walk.' }},
            std: {{ title: 'Standard Betweenness', desc: 'Traditional graph centrality - how often a street lies on shortest paths between ALL node pairs equally.' }},
            diff: {{ title: 'Difference View', desc: 'Green = underrated by standard analysis (high actual traffic). Purple = overrated (theoretical but not used).' }}
        }};

        function showMetric(name) {{
            map.removeLayer(layers[currentMetric]);
            layers[name].addTo(map);
            document.querySelectorAll('.btn').forEach(b => {{
                if (b.id.startsWith('btn-') && !b.id.includes('origins') && !b.id.includes('dests')) {{
                    b.classList.remove('btn-active');
                }}
            }});
            document.getElementById('btn-' + name).classList.add('btn-active');
            document.querySelectorAll('.legend').forEach(l => l.style.display = 'none');
            document.getElementById('legend-' + name).style.display = 'block';
            document.getElementById('metric-title').textContent = metricInfo[name].title;
            document.getElementById('metric-desc').textContent = metricInfo[name].desc;
            currentMetric = name;
        }}

        function toggleMarkers(type) {{
            var btn = document.getElementById('btn-' + type);
            if (type === 'origins') {{
                if (showOrigins) {{
                    map.removeLayer(originMarkers);
                    btn.classList.remove('btn-active');
                }} else {{
                    originMarkers.addTo(map);
                    btn.classList.add('btn-active');
                }}
                showOrigins = !showOrigins;
            }} else {{
                if (showDests) {{
                    map.removeLayer(destMarkers);
                    btn.classList.remove('btn-active');
                }} else {{
                    destMarkers.addTo(map);
                    btn.classList.add('btn-active');
                }}
                showDests = !showDests;
            }}
        }}
    </script>
</body>
</html>'''

with open(os.path.join(OUT_DIR, 'od_betweenness_map.html'), 'w') as f:
    f.write(html)
print(f"  Saved: {OUT_DIR}/od_betweenness_map.html")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"""
Results saved to: {OUT_DIR}/

KEY FINDINGS:
- OD-weighted analysis shows actual pedestrian traffic based on land use
- Standard betweenness shows theoretical network importance
- Correlation: {correlation:.2f} (1.0 = identical, lower = more different)

Streets with HIGH OD but LOW standard are underrated by traditional
analysis - they're important for actual pedestrians but not obvious
from network topology alone.

Open: {OUT_DIR}/od_betweenness_map.html
""")
