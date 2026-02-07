#!/usr/bin/env python3
"""
Angular Segment Analysis (Space Syntax) for Al Karama
=======================================================

Implements proper angular analysis using turn angles as edge weights.
This is closer to true space syntax than the previous approximation.

Key concept: Use ANGULAR distance (turn angles) not metric distance.
"""

import osmnx as ox
import networkx as nx
import pandas as pd
import geopandas as gpd
import numpy as np
import json
import os
from collections import defaultdict
import math

print("="*60)
print("ANGULAR SEGMENT ANALYSIS (Space Syntax)")
print("Al Karama, Dubai")
print("="*60)

# Paths
BASE = os.path.dirname(__file__)
OUT_DIR = os.path.join(BASE, '..', 'output', 'angular_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

# Al Karama boundary
ak_north, ak_south = 25.255, 25.230
ak_east, ak_west = 55.315, 55.290

# Download street network
print("\n1. DOWNLOADING STREET NETWORK")
print("-"*40)
G = ox.graph_from_bbox(bbox=(ak_north, ak_south, ak_east, ak_west), network_type='walk')
G_undir = G.to_undirected()
print(f"   Nodes: {G_undir.number_of_nodes()}")
print(f"   Edges: {G_undir.number_of_edges()}")


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing between two points in degrees (0-360)."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.atan2(x, y)
    return (math.degrees(bearing) + 360) % 360


def angle_difference(bearing1, bearing2):
    """Calculate the angular difference between two bearings (0-180)."""
    diff = abs(bearing1 - bearing2)
    if diff > 180:
        diff = 360 - diff
    return diff


# Calculate edge bearings
print("\n2. CALCULATING EDGE BEARINGS")
print("-"*40)

edge_bearings = {}
for u, v, k in G_undir.edges(keys=True):
    u_lat, u_lon = G_undir.nodes[u]['y'], G_undir.nodes[u]['x']
    v_lat, v_lon = G_undir.nodes[v]['y'], G_undir.nodes[v]['x']
    bearing = calculate_bearing(u_lat, u_lon, v_lat, v_lon)
    edge_bearings[(u, v, k)] = bearing
    edge_bearings[(v, u, k)] = (bearing + 180) % 360  # Reverse direction

print(f"   Calculated bearings for {len(edge_bearings)//2} edges")

# Calculate turn angles at each node
print("\n3. CALCULATING TURN ANGLES")
print("-"*40)

turn_angles = defaultdict(dict)  # turn_angles[node][(from_edge, to_edge)] = angle

for node in G_undir.nodes():
    edges = list(G_undir.edges(node, keys=True))
    if len(edges) < 2:
        continue

    for i, (u1, v1, k1) in enumerate(edges):
        for u2, v2, k2 in edges[i+1:]:
            # Get bearings of edges leaving this node
            if u1 == node:
                b1 = edge_bearings[(u1, v1, k1)]
            else:
                b1 = edge_bearings[(v1, u1, k1)]

            if u2 == node:
                b2 = edge_bearings[(u2, v2, k2)]
            else:
                b2 = edge_bearings[(v2, u2, k2)]

            # Turn angle is deviation from straight (180 degrees)
            turn = angle_difference(b1, b2)
            deviation = abs(180 - turn)  # 0 = straight, 90 = right angle, 180 = U-turn

            turn_angles[node][((u1, v1, k1), (u2, v2, k2))] = deviation
            turn_angles[node][((u2, v2, k2), (u1, v1, k1))] = deviation

print(f"   Nodes with turn data: {len(turn_angles)}")

# Create angular graph - edge weight is turn angle at destination
print("\n4. BUILDING ANGULAR GRAPH")
print("-"*40)

# For space syntax, we weight paths by the sum of turn angles
# We create a line graph where edges become nodes, connected by turns
# Simpler approach: weight each edge by the average turn angle to reach it

# Calculate "angular cost" for each edge based on turns required to enter it
edge_angular_cost = {}

for u, v, k in G_undir.edges(keys=True):
    # Average turn angle to enter this edge from any neighbor
    entry_turns = []

    for prev_node in G_undir.neighbors(u):
        if prev_node == v:
            continue
        prev_edges = [(prev_node, u, kk) for kk in G_undir[prev_node][u].keys()]
        for prev_edge in prev_edges:
            key = (prev_edge, (u, v, k))
            if key in turn_angles[u]:
                entry_turns.append(turn_angles[u][key])

    for prev_node in G_undir.neighbors(v):
        if prev_node == u:
            continue
        prev_edges = [(prev_node, v, kk) for kk in G_undir[prev_node][v].keys()]
        for prev_edge in prev_edges:
            key = (prev_edge, (v, u, k))
            if key in turn_angles[v]:
                entry_turns.append(turn_angles[v][key])

    if entry_turns:
        avg_turn = np.mean(entry_turns)
    else:
        avg_turn = 0

    # Angular cost = base cost (1) + turn penalty
    # Normalize turn to 0-1 range (0=straight, 1=U-turn)
    edge_angular_cost[(u, v, k)] = 1 + (avg_turn / 180)

# Add angular weight to graph
for u, v, k in G_undir.edges(keys=True):
    G_undir[u][v][k]['angular'] = edge_angular_cost.get((u, v, k), 1)

print(f"   Angular costs calculated for {len(edge_angular_cost)} edges")

# Calculate angular betweenness (Choice in space syntax terms)
print("\n5. CALCULATING ANGULAR BETWEENNESS (CHOICE)")
print("-"*40)

angular_bc = nx.edge_betweenness_centrality(G_undir, weight='angular', normalized=True)
print(f"   Calculated angular betweenness")

# Calculate angular closeness (Integration in space syntax terms)
print("\n6. CALCULATING ANGULAR CLOSENESS (INTEGRATION)")
print("-"*40)

# Global integration (to all nodes)
angular_closeness = nx.closeness_centrality(G_undir, distance='angular')
print(f"   Calculated global angular closeness")

# Local integration (radius-limited) - often more revealing for grid patterns
print("\n6b. CALCULATING LOCAL INTEGRATION (radius=3 steps)")
print("-"*40)

# Use radius=3 (3 topological steps) for local integration
# This shows which streets are well-connected within walking neighborhood
local_integration = {}
for node in G_undir.nodes():
    # Get subgraph within 3 steps
    neighbors_at_radius = nx.single_source_shortest_path_length(G_undir, node, cutoff=3)
    if len(neighbors_at_radius) > 1:
        # Calculate closeness within this local neighborhood
        total_angular = 0
        count = 0
        for target, steps in neighbors_at_radius.items():
            if target != node:
                try:
                    path_length = nx.shortest_path_length(G_undir, node, target, weight='angular')
                    total_angular += path_length
                    count += 1
                except:
                    pass
        if count > 0 and total_angular > 0:
            local_integration[node] = count / total_angular  # inverse of mean angular depth
        else:
            local_integration[node] = 0
    else:
        local_integration[node] = 0

print(f"   Calculated local integration for {len(local_integration)} nodes")

# Also calculate metric betweenness for comparison
print("\n7. CALCULATING METRIC BETWEENNESS (for comparison)")
print("-"*40)

metric_bc = nx.edge_betweenness_centrality(G_undir, weight='length', normalized=True)
print(f"   Calculated metric betweenness")

# Convert to GeoDataFrame
print("\n8. CREATING OUTPUT")
print("-"*40)

nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_undir)
edges_gdf = edges_gdf.reset_index()

# Add metrics
edges_gdf['angular_bc'] = 0.0
edges_gdf['angular_int'] = 0.0
edges_gdf['local_int'] = 0.0
edges_gdf['metric_bc'] = 0.0

for idx, row in edges_gdf.iterrows():
    u, v, k = row['u'], row['v'], row['key']

    # Angular betweenness
    val = angular_bc.get((u, v, k), angular_bc.get((v, u, k), 0))
    edges_gdf.at[idx, 'angular_bc'] = val

    # Global angular integration (average of node closeness)
    int_val = (angular_closeness.get(u, 0) + angular_closeness.get(v, 0)) / 2
    edges_gdf.at[idx, 'angular_int'] = int_val

    # Local integration (radius-3)
    local_val = (local_integration.get(u, 0) + local_integration.get(v, 0)) / 2
    edges_gdf.at[idx, 'local_int'] = local_val

    # Metric betweenness
    m_val = metric_bc.get((u, v, k), metric_bc.get((v, u, k), 0))
    edges_gdf.at[idx, 'metric_bc'] = m_val

# Normalize
for col in ['angular_bc', 'angular_int', 'local_int', 'metric_bc']:
    max_val = edges_gdf[col].max()
    if max_val > 0:
        edges_gdf[f'{col}_norm'] = edges_gdf[col] / max_val
    else:
        edges_gdf[f'{col}_norm'] = 0

# Calculate correlation
corr_ang_metric = edges_gdf['angular_bc_norm'].corr(edges_gdf['metric_bc_norm'])

print(f"\n   Correlation (Angular vs Metric): {corr_ang_metric:.3f}")

# Stats
print("\n" + "-"*40)
print("ANGULAR VS METRIC COMPARISON")
print("-"*40)

print(f"\n   Angular Betweenness (Choice):")
print(f"      Max: {edges_gdf['angular_bc'].max():.6f}")
print(f"      Mean: {edges_gdf['angular_bc'].mean():.6f}")
print(f"      Std: {edges_gdf['angular_bc'].std():.6f}")

print(f"\n   Metric Betweenness:")
print(f"      Max: {edges_gdf['metric_bc'].max():.6f}")
print(f"      Mean: {edges_gdf['metric_bc'].mean():.6f}")
print(f"      Std: {edges_gdf['metric_bc'].std():.6f}")

print(f"\n   Global Angular Integration:")
print(f"      Max: {edges_gdf['angular_int'].max():.6f}")
print(f"      Mean: {edges_gdf['angular_int'].mean():.6f}")
print(f"      Std: {edges_gdf['angular_int'].std():.6f}")
print(f"      CV: {edges_gdf['angular_int'].std()/edges_gdf['angular_int'].mean()*100:.1f}%")

print(f"\n   Local Integration (radius=3):")
print(f"      Max: {edges_gdf['local_int'].max():.6f}")
print(f"      Mean: {edges_gdf['local_int'].mean():.6f}")
print(f"      Std: {edges_gdf['local_int'].std():.6f}")
print(f"      CV: {edges_gdf['local_int'].std()/edges_gdf['local_int'].mean()*100:.1f}%")

# Top streets
print("\n" + "-"*40)
print("TOP 10 BY ANGULAR CHOICE (turns minimize)")
print("-"*40)
edges_gdf['mid_lat'] = edges_gdf.geometry.centroid.y
edges_gdf['mid_lon'] = edges_gdf.geometry.centroid.x

top_angular = edges_gdf.nlargest(10, 'angular_bc')
for i, (_, row) in enumerate(top_angular.iterrows()):
    print(f"{i+1:>3}. Angular: {row['angular_bc']:.6f}  Metric: {row['metric_bc']:.6f}  ({row['mid_lat']:.5f}, {row['mid_lon']:.5f})")

print("\n" + "-"*40)
print("TOP 10 BY METRIC BETWEENNESS")
print("-"*40)
top_metric = edges_gdf.nlargest(10, 'metric_bc')
for i, (_, row) in enumerate(top_metric.iterrows()):
    print(f"{i+1:>3}. Angular: {row['angular_bc']:.6f}  Metric: {row['metric_bc']:.6f}  ({row['mid_lat']:.5f}, {row['mid_lon']:.5f})")

# Save
edges_gdf.to_file(os.path.join(OUT_DIR, 'angular_analysis.geojson'), driver='GeoJSON')
print(f"\nSaved: {OUT_DIR}/angular_analysis.geojson")

# Create map
print("\nCreating interactive map...")

map_data = []
for idx, row in edges_gdf.iterrows():
    coords = list(row.geometry.coords)
    map_data.append({
        'coords': [[c[1], c[0]] for c in coords],
        'ang': round(float(row['angular_bc_norm']), 4),
        'met': round(float(row['metric_bc_norm']), 4),
        'int': round(float(row['angular_int_norm']), 4),
        'loc': round(float(row['local_int_norm']), 4),
    })

# Calculate CV (coefficient of variation) for display
global_cv = edges_gdf['angular_int'].std()/edges_gdf['angular_int'].mean()*100
local_cv = edges_gdf['local_int'].std()/edges_gdf['local_int'].mean()*100

html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Angular Analysis (Space Syntax) - Al Karama</title>
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
            max-width: 400px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            max-height: 90vh;
            overflow-y: auto;
        }}
        .panel h2 {{ margin: 0 0 10px 0; color: #00695c; }}
        .panel h3 {{ margin: 15px 0 8px 0; font-size: 14px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
        .btn {{ padding: 10px 14px; margin: 4px; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; }}
        .btn-ang {{ background: #00897b; color: white; }}
        .btn-met {{ background: #5e35b1; color: white; }}
        .btn-int {{ background: #f57c00; color: white; }}
        .btn-loc {{ background: #e65100; color: white; }}
        .btn:not(.btn-active) {{ opacity: 0.5; }}
        .btn.btn-active {{ opacity: 1; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }}
        .metric-desc {{ font-size: 12px; color: #666; margin: 10px 0; line-height: 1.5; padding: 10px; border-radius: 6px; }}
        .desc-ang {{ background: #e0f2f1; }}
        .desc-met {{ background: #ede7f6; }}
        .desc-int {{ background: #fff3e0; }}
        .desc-loc {{ background: #fff3e0; border: 2px solid #e65100; }}
        .stats {{ background: #f5f5f5; padding: 12px; border-radius: 6px; margin: 10px 0; font-size: 12px; }}
        .btn-basemap {{ background: #424242; color: white; }}
        .highlight {{ background: #ffeb3b; padding: 2px 4px; border-radius: 3px; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="panel">
        <h2>Angular Analysis</h2>
        <p style="color: #666; font-size: 12px;">Space Syntax: Angular vs Metric distance</p>

        <h3>Metric</h3>
        <div>
            <button class="btn btn-ang btn-active" id="btn-ang" onclick="showMetric('ang')">Angular Choice</button>
            <button class="btn btn-met" id="btn-met" onclick="showMetric('met')">Metric Betweenness</button>
        </div>
        <div style="margin-top: 8px;">
            <button class="btn btn-int" id="btn-int" onclick="showMetric('int')">Global Integration</button>
            <button class="btn btn-loc" id="btn-loc" onclick="showMetric('loc')">Local Integration ★</button>
        </div>

        <div id="desc-ang" class="metric-desc desc-ang">
            <b>Angular Choice (NACH)</b><br>
            Betweenness weighted by turn angles. High values = streets on paths that minimize turns.
            Core space syntax metric.
        </div>
        <div id="desc-met" class="metric-desc desc-met" style="display:none;">
            <b>Metric Betweenness</b><br>
            Standard betweenness using distance in meters. For comparison with angular.
        </div>
        <div id="desc-int" class="metric-desc desc-int" style="display:none;">
            <b>Global Angular Integration</b><br>
            How few turns to reach <i>everywhere</i>. Low variance ({global_cv:.0f}% CV) in grid patterns because all streets have similar global access.
        </div>
        <div id="desc-loc" class="metric-desc desc-loc" style="display:none;">
            <b>Local Integration (radius=3)</b> <span class="highlight">★ Recommended</span><br>
            How few turns to reach nearby streets (within 3 steps). Higher variance ({local_cv:.0f}% CV) reveals local accessibility differences that global integration misses.
        </div>

        <div class="stats">
            <b>Correlation:</b> Angular vs Metric = <b>{corr_ang_metric:.2f}</b><br>
            <span style="font-size: 11px; color: #666;">
                Lower = angular reveals different patterns
            </span>
        </div>

        <h3>Legend</h3>
        <div style="display: flex; align-items: center; font-size: 12px;">
            <div style="width: 100px; height: 8px; background: linear-gradient(to right, #eee, currentColor); margin-right: 10px;"></div>
            Low → High
        </div>

        <h3>Background</h3>
        <div>
            <button class="btn btn-basemap btn-active" id="btn-basemap" onclick="toggleBasemap()">Basemap</button>
        </div>

        <h3>Why Integration looks uniform?</h3>
        <div style="font-size: 11px; color: #666; line-height: 1.5; background: #fafafa; padding: 10px; border-radius: 6px;">
            <p>Al Karama has a <b>regular grid pattern</b> where most streets intersect at 90°. This means every street can reach everywhere with similar number of turns.</p>
            <p><b>Try Local Integration</b> - it shows which streets are well-connected within walking distance, revealing differences that global integration misses.</p>
        </div>
    </div>

    <script>
        var map = L.map('map').setView([25.2425, 55.3025], 15);
        var basemapLayer = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; OpenStreetMap'
        }}).addTo(map);
        var basemapOn = true;

        function toggleBasemap() {{
            var btn = document.getElementById('btn-basemap');
            if (basemapOn) {{
                map.removeLayer(basemapLayer);
                btn.classList.remove('btn-active');
                document.getElementById('map').style.background = '#1a1a2e';
            }} else {{
                basemapLayer.addTo(map);
                btn.classList.add('btn-active');
                document.getElementById('map').style.background = 'white';
            }}
            basemapOn = !basemapOn;
        }}

        var edges = {json.dumps(map_data)};
        var layers = {{ ang: L.layerGroup(), met: L.layerGroup(), int: L.layerGroup(), loc: L.layerGroup() }};
        var colors = {{ ang: '#00897b', met: '#5e35b1', int: '#f57c00', loc: '#e65100' }};

        function getColor(val, baseColor) {{
            var r = parseInt(baseColor.slice(1,3), 16);
            var g = parseInt(baseColor.slice(3,5), 16);
            var b = parseInt(baseColor.slice(5,7), 16);
            var factor = 0.2 + val * 0.8;
            return 'rgb(' + Math.round(r*factor) + ',' + Math.round(g*factor) + ',' + Math.round(b*factor) + ')';
        }}

        edges.forEach(function(e) {{
            var popup = 'Angular Choice: ' + e.ang.toFixed(3) + '<br>' +
                       'Metric Betweenness: ' + e.met.toFixed(3) + '<br>' +
                       'Global Integration: ' + e.int.toFixed(3) + '<br>' +
                       'Local Integration: ' + e.loc.toFixed(3);

            ['ang', 'met', 'int', 'loc'].forEach(function(m) {{
                var val = e[m];
                var w = 2 + val * 6;
                L.polyline(e.coords, {{ color: getColor(val, colors[m]), weight: w, opacity: 0.85 }})
                 .bindPopup(popup).addTo(layers[m]);
            }});
        }});

        layers.ang.addTo(map);
        var current = 'ang';

        function showMetric(m) {{
            map.removeLayer(layers[current]);
            layers[m].addTo(map);

            ['ang', 'met', 'int', 'loc'].forEach(function(x) {{
                document.getElementById('btn-' + x).classList.remove('btn-active');
                document.getElementById('desc-' + x).style.display = 'none';
            }});
            document.getElementById('btn-' + m).classList.add('btn-active');
            document.getElementById('desc-' + m).style.display = 'block';

            current = m;
        }}
    </script>
</body>
</html>'''

with open(os.path.join(OUT_DIR, 'angular_map.html'), 'w') as f:
    f.write(html)
print(f"Saved: {OUT_DIR}/angular_map.html")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"""
This analysis uses TURN ANGLES as edge weights instead of distance.

Key insight: Streets with high Angular Choice are important for
navigation because paths through them require fewer turns.

Correlation: {corr_ang_metric:.2f}
- If low: Angular analysis reveals different important streets
- If high: Grid is regular, angular doesn't add much

Open: {OUT_DIR}/angular_map.html
""")
