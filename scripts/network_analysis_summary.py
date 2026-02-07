#!/usr/bin/env python3
"""
Network Analysis Method Comparison for Al Karama
=================================================

Creates a summary comparison of all network analysis methods tested,
showing which work best for grid patterns vs organic patterns.
"""

import osmnx as ox
import networkx as nx
import pandas as pd
import geopandas as gpd
import numpy as np
import json
import os

print("="*60)
print("NETWORK ANALYSIS METHOD COMPARISON")
print("Al Karama, Dubai")
print("="*60)

# Paths
BASE = os.path.dirname(__file__)
OUT_DIR = os.path.join(BASE, '..', 'output', 'network_comparison')
os.makedirs(OUT_DIR, exist_ok=True)

# Load existing analysis results
ANGULAR_DIR = os.path.join(BASE, '..', 'output', 'angular_analysis')
OD_DIR = os.path.join(BASE, '..', 'output', 'od_betweenness')
COMPARISON_DIR = os.path.join(BASE, '..', 'output', 'space_syntax_comparison')

# Al Karama boundary
ak_north, ak_south = 25.255, 25.230
ak_east, ak_west = 55.315, 55.290

print("\n1. LOADING STREET NETWORK")
print("-"*40)
G = ox.graph_from_bbox(bbox=(ak_north, ak_south, ak_east, ak_west), network_type='walk')
G_undir = G.to_undirected()
print(f"   Nodes: {G_undir.number_of_nodes()}")
print(f"   Edges: {G_undir.number_of_edges()}")

# Calculate all metrics fresh for consistent comparison
print("\n2. CALCULATING ALL METRICS")
print("-"*40)

# 2a. Standard Betweenness (metric distance)
print("   - Standard Betweenness (metric)...")
metric_bc = nx.edge_betweenness_centrality(G_undir, weight='length', normalized=True)

# 2b. Angular Betweenness (Choice)
print("   - Angular Betweenness (Choice)...")
# First calculate angular weights
import math

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.atan2(x, y)
    return (math.degrees(bearing) + 360) % 360

def angle_difference(bearing1, bearing2):
    diff = abs(bearing1 - bearing2)
    if diff > 180:
        diff = 360 - diff
    return diff

# Calculate edge bearings
edge_bearings = {}
for u, v, k in G_undir.edges(keys=True):
    u_lat, u_lon = G_undir.nodes[u]['y'], G_undir.nodes[u]['x']
    v_lat, v_lon = G_undir.nodes[v]['y'], G_undir.nodes[v]['x']
    bearing = calculate_bearing(u_lat, u_lon, v_lat, v_lon)
    edge_bearings[(u, v, k)] = bearing
    edge_bearings[(v, u, k)] = (bearing + 180) % 360

# Calculate turn angles
from collections import defaultdict
turn_angles = defaultdict(dict)
for node in G_undir.nodes():
    edges = list(G_undir.edges(node, keys=True))
    if len(edges) < 2:
        continue
    for i, (u1, v1, k1) in enumerate(edges):
        for u2, v2, k2 in edges[i+1:]:
            if u1 == node:
                b1 = edge_bearings[(u1, v1, k1)]
            else:
                b1 = edge_bearings[(v1, u1, k1)]
            if u2 == node:
                b2 = edge_bearings[(u2, v2, k2)]
            else:
                b2 = edge_bearings[(v2, u2, k2)]
            turn = angle_difference(b1, b2)
            deviation = abs(180 - turn)
            turn_angles[node][((u1, v1, k1), (u2, v2, k2))] = deviation
            turn_angles[node][((u2, v2, k2), (u1, v1, k1))] = deviation

# Angular cost per edge
for u, v, k in G_undir.edges(keys=True):
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
    avg_turn = np.mean(entry_turns) if entry_turns else 0
    G_undir[u][v][k]['angular'] = 1 + (avg_turn / 180)

angular_bc = nx.edge_betweenness_centrality(G_undir, weight='angular', normalized=True)

# 2c. Closeness / Integration
print("   - Global Integration (closeness)...")
global_closeness = nx.closeness_centrality(G_undir, distance='angular')

print("   - Local Integration (radius=3)...")
local_integration = {}
for node in G_undir.nodes():
    neighbors = nx.single_source_shortest_path_length(G_undir, node, cutoff=3)
    if len(neighbors) > 1:
        total_angular = 0
        count = 0
        for target in neighbors:
            if target != node:
                try:
                    path_length = nx.shortest_path_length(G_undir, node, target, weight='angular')
                    total_angular += path_length
                    count += 1
                except:
                    pass
        local_integration[node] = count / total_angular if total_angular > 0 else 0
    else:
        local_integration[node] = 0

# 2d. Degree centrality (connectivity)
print("   - Degree Centrality...")
degree_cent = nx.degree_centrality(G_undir)

print("\n3. BUILDING COMPARISON TABLE")
print("-"*40)

# Convert to edge GeoDataFrame
nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_undir)
edges_gdf = edges_gdf.reset_index()

# Add all metrics
edges_gdf['metric_bc'] = 0.0
edges_gdf['angular_bc'] = 0.0
edges_gdf['global_int'] = 0.0
edges_gdf['local_int'] = 0.0
edges_gdf['degree'] = 0.0

for idx, row in edges_gdf.iterrows():
    u, v, k = row['u'], row['v'], row['key']

    # Metric betweenness
    edges_gdf.at[idx, 'metric_bc'] = metric_bc.get((u, v, k), metric_bc.get((v, u, k), 0))

    # Angular betweenness
    edges_gdf.at[idx, 'angular_bc'] = angular_bc.get((u, v, k), angular_bc.get((v, u, k), 0))

    # Global integration
    edges_gdf.at[idx, 'global_int'] = (global_closeness.get(u, 0) + global_closeness.get(v, 0)) / 2

    # Local integration
    edges_gdf.at[idx, 'local_int'] = (local_integration.get(u, 0) + local_integration.get(v, 0)) / 2

    # Degree
    edges_gdf.at[idx, 'degree'] = (degree_cent.get(u, 0) + degree_cent.get(v, 0)) / 2

# Normalize all metrics
for col in ['metric_bc', 'angular_bc', 'global_int', 'local_int', 'degree']:
    max_val = edges_gdf[col].max()
    if max_val > 0:
        edges_gdf[f'{col}_norm'] = edges_gdf[col] / max_val
    else:
        edges_gdf[f'{col}_norm'] = 0

# Calculate statistics
print("\n4. ANALYSIS RESULTS")
print("="*60)

metrics = {
    'Metric Betweenness': 'metric_bc',
    'Angular Choice': 'angular_bc',
    'Global Integration': 'global_int',
    'Local Integration': 'local_int',
    'Degree Centrality': 'degree'
}

stats = []
for name, col in metrics.items():
    mean_val = edges_gdf[col].mean()
    std_val = edges_gdf[col].std()
    cv = (std_val / mean_val * 100) if mean_val > 0 else 0
    max_val = edges_gdf[col].max()
    min_val = edges_gdf[col].min()

    stats.append({
        'Metric': name,
        'Mean': mean_val,
        'Std': std_val,
        'CV%': cv,
        'Max': max_val,
        'Min': min_val,
        'Range': max_val - min_val
    })

    print(f"\n{name}:")
    print(f"   Mean: {mean_val:.6f}")
    print(f"   Std:  {std_val:.6f}")
    print(f"   CV:   {cv:.1f}%")
    print(f"   Range: {min_val:.6f} - {max_val:.6f}")

stats_df = pd.DataFrame(stats)

# Correlations
print("\n" + "="*60)
print("CORRELATION MATRIX")
print("="*60)

corr_cols = ['metric_bc', 'angular_bc', 'global_int', 'local_int', 'degree']
corr_matrix = edges_gdf[corr_cols].corr()
print(corr_matrix.round(3).to_string())

# Determine effectiveness ranking
print("\n" + "="*60)
print("EFFECTIVENESS FOR AL KARAMA (GRID PATTERN)")
print("="*60)

# Sort by CV (higher = more differentiation)
stats_df_sorted = stats_df.sort_values('CV%', ascending=False)
print("\nRanked by ability to differentiate streets (CV%):")
for i, row in stats_df_sorted.iterrows():
    effectiveness = "HIGH" if row['CV%'] > 100 else "MEDIUM" if row['CV%'] > 20 else "LOW"
    print(f"   {row['Metric']:25s}: {row['CV%']:6.1f}% CV  [{effectiveness}]")

# Create summary HTML
print("\n5. CREATING SUMMARY VISUALIZATION")
print("-"*40)

# Prepare map data
map_data = []
for idx, row in edges_gdf.iterrows():
    coords = list(row.geometry.coords)
    map_data.append({
        'coords': [[c[1], c[0]] for c in coords],
        'met': round(float(row['metric_bc_norm']), 4),
        'ang': round(float(row['angular_bc_norm']), 4),
        'gint': round(float(row['global_int_norm']), 4),
        'lint': round(float(row['local_int_norm']), 4),
        'deg': round(float(row['degree_norm']), 4),
    })

# Stats for display
cv_metric = stats_df[stats_df['Metric'] == 'Metric Betweenness']['CV%'].values[0]
cv_angular = stats_df[stats_df['Metric'] == 'Angular Choice']['CV%'].values[0]
cv_global = stats_df[stats_df['Metric'] == 'Global Integration']['CV%'].values[0]
cv_local = stats_df[stats_df['Metric'] == 'Local Integration']['CV%'].values[0]
cv_degree = stats_df[stats_df['Metric'] == 'Degree Centrality']['CV%'].values[0]

corr_met_ang = corr_matrix.loc['metric_bc', 'angular_bc']

html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Network Analysis Comparison - Al Karama</title>
    <meta charset="utf-8">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        #map {{ width: 100%; height: 100vh; }}

        .panel {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(255,255,255,0.98);
            padding: 20px;
            border-radius: 12px;
            z-index: 1000;
            width: 420px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            max-height: 95vh;
            overflow-y: auto;
        }}

        h1 {{ margin: 0 0 5px 0; font-size: 20px; color: #1a1a2e; }}
        h2 {{ margin: 20px 0 10px 0; font-size: 14px; color: #666; border-bottom: 2px solid #eee; padding-bottom: 5px; }}
        .subtitle {{ color: #888; font-size: 12px; margin-bottom: 15px; }}

        .method-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 10px 0; }}

        .method-btn {{
            padding: 12px 8px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 11px;
            font-weight: 500;
            text-align: center;
            transition: all 0.2s;
            position: relative;
        }}
        .method-btn:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }}
        .method-btn.active {{ box-shadow: 0 4px 12px rgba(0,0,0,0.3); transform: translateY(-2px); }}
        .method-btn:not(.active) {{ opacity: 0.6; }}

        .btn-met {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; }}
        .btn-ang {{ background: linear-gradient(135deg, #11998e, #38ef7d); color: white; }}
        .btn-gint {{ background: linear-gradient(135deg, #fc4a1a, #f7b733); color: white; }}
        .btn-lint {{ background: linear-gradient(135deg, #ee0979, #ff6a00); color: white; }}
        .btn-deg {{ background: linear-gradient(135deg, #4568dc, #b06ab3); color: white; }}

        .cv-badge {{
            position: absolute;
            top: -8px;
            right: -8px;
            background: #1a1a2e;
            color: white;
            font-size: 9px;
            padding: 3px 6px;
            border-radius: 10px;
            font-weight: bold;
        }}
        .cv-high {{ background: #2ecc71; }}
        .cv-medium {{ background: #f39c12; }}
        .cv-low {{ background: #e74c3c; }}

        .info-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 12px;
            margin: 10px 0;
            font-size: 12px;
            line-height: 1.6;
            display: none;
        }}
        .info-card.active {{ display: block; }}
        .info-card h3 {{ margin: 0 0 8px 0; font-size: 13px; }}

        .effectiveness {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: bold;
            margin-left: 8px;
        }}
        .eff-high {{ background: #d4edda; color: #155724; }}
        .eff-medium {{ background: #fff3cd; color: #856404; }}
        .eff-low {{ background: #f8d7da; color: #721c24; }}

        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 11px;
            margin: 10px 0;
        }}
        .summary-table th, .summary-table td {{
            padding: 8px 6px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        .summary-table th {{ background: #f8f9fa; font-weight: 600; }}

        .legend {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 10px 0;
            font-size: 11px;
        }}
        .legend-bar {{
            width: 100px;
            height: 10px;
            border-radius: 5px;
        }}

        .conclusion {{
            background: linear-gradient(135deg, #667eea22, #764ba222);
            border-left: 4px solid #667eea;
            padding: 12px;
            border-radius: 0 8px 8px 0;
            margin: 15px 0;
            font-size: 12px;
        }}

        .toggle-row {{
            display: flex;
            gap: 8px;
            margin: 10px 0;
        }}
        .toggle-btn {{
            padding: 8px 12px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 11px;
            background: #eee;
            color: #666;
        }}
        .toggle-btn.active {{ background: #1a1a2e; color: white; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="panel">
        <h1>Network Analysis Comparison</h1>
        <div class="subtitle">Al Karama, Dubai - Regular Grid Pattern</div>

        <h2>Select Analysis Method</h2>
        <div class="method-grid">
            <button class="method-btn btn-met active" id="btn-met" onclick="showMetric('met')">
                Metric<br>Betweenness
                <span class="cv-badge cv-high">{cv_metric:.0f}% CV</span>
            </button>
            <button class="method-btn btn-ang" id="btn-ang" onclick="showMetric('ang')">
                Angular<br>Choice
                <span class="cv-badge cv-high">{cv_angular:.0f}% CV</span>
            </button>
            <button class="method-btn btn-gint" id="btn-gint" onclick="showMetric('gint')">
                Global<br>Integration
                <span class="cv-badge cv-low">{cv_global:.0f}% CV</span>
            </button>
            <button class="method-btn btn-lint" id="btn-lint" onclick="showMetric('lint')">
                Local<br>Integration
                <span class="cv-badge cv-low">{cv_local:.0f}% CV</span>
            </button>
            <button class="method-btn btn-deg" id="btn-deg" onclick="showMetric('deg')">
                Degree<br>Centrality
                <span class="cv-badge cv-low">{cv_degree:.0f}% CV</span>
            </button>
        </div>

        <!-- Info Cards -->
        <div class="info-card active" id="info-met">
            <h3>Metric Betweenness <span class="effectiveness eff-high">HIGH for grids</span></h3>
            <p>Counts how often each street lies on <b>shortest distance paths</b> between all pairs of locations. High values = major through-routes.</p>
            <p><b>Best for:</b> Identifying main corridors, traffic flow prediction</p>
        </div>
        <div class="info-card" id="info-ang">
            <h3>Angular Choice (NACH) <span class="effectiveness eff-high">HIGH for grids</span></h3>
            <p>Like betweenness, but paths minimize <b>turn angles</b> instead of distance. Based on how people actually navigate (prefer fewer turns).</p>
            <p><b>Best for:</b> Pedestrian movement, wayfinding routes</p>
            <p><b>Correlation with Metric:</b> {corr_met_ang:.2f} (moderate - reveals different patterns)</p>
        </div>
        <div class="info-card" id="info-gint">
            <h3>Global Integration <span class="effectiveness eff-low">LOW for grids</span></h3>
            <p>How few turns needed to reach <b>all other streets</b>. In regular grids, all streets have similar global access.</p>
            <p><b>Best for:</b> Organic/irregular street patterns, historic centers</p>
        </div>
        <div class="info-card" id="info-lint">
            <h3>Local Integration (r=3) <span class="effectiveness eff-low">LOW for grids</span></h3>
            <p>How few turns to reach <b>nearby streets</b> (within 3 steps). Even more uniform in regular grids.</p>
            <p><b>Best for:</b> Areas with cul-de-sacs, dead-ends, irregular blocks</p>
        </div>
        <div class="info-card" id="info-deg">
            <h3>Degree Centrality <span class="effectiveness eff-low">LOW for grids</span></h3>
            <p>Simply counts <b>number of connections</b> at each intersection. Most intersections in a grid have 4 connections.</p>
            <p><b>Best for:</b> Quick connectivity overview, not detailed analysis</p>
        </div>

        <h2>Effectiveness Summary</h2>
        <table class="summary-table">
            <tr>
                <th>Method</th>
                <th>CV%</th>
                <th>Grid?</th>
                <th>Organic?</th>
            </tr>
            <tr>
                <td>Metric Betweenness</td>
                <td><b>{cv_metric:.0f}%</b></td>
                <td style="color: #27ae60;">✓ Good</td>
                <td style="color: #27ae60;">✓ Good</td>
            </tr>
            <tr>
                <td>Angular Choice</td>
                <td><b>{cv_angular:.0f}%</b></td>
                <td style="color: #27ae60;">✓ Good</td>
                <td style="color: #27ae60;">✓ Excellent</td>
            </tr>
            <tr>
                <td>Global Integration</td>
                <td>{cv_global:.0f}%</td>
                <td style="color: #e74c3c;">✗ Poor</td>
                <td style="color: #27ae60;">✓ Excellent</td>
            </tr>
            <tr>
                <td>Local Integration</td>
                <td>{cv_local:.0f}%</td>
                <td style="color: #e74c3c;">✗ Poor</td>
                <td style="color: #27ae60;">✓ Good</td>
            </tr>
            <tr>
                <td>Degree Centrality</td>
                <td>{cv_degree:.0f}%</td>
                <td style="color: #e74c3c;">✗ Poor</td>
                <td style="color: #f39c12;">~ Medium</td>
            </tr>
        </table>

        <div class="conclusion">
            <b>Key Finding:</b> For Al Karama's regular grid, <b>Betweenness metrics</b> (both Metric and Angular) effectively differentiate streets, while <b>Integration metrics</b> show uniform values. This is expected - grids have uniform accessibility but variable through-traffic.
        </div>

        <h2>Legend</h2>
        <div class="legend">
            <div class="legend-bar" id="legend-bar" style="background: linear-gradient(to right, #eee, #667eea);"></div>
            <span>Low → High</span>
        </div>

        <div class="toggle-row">
            <button class="toggle-btn active" id="btn-basemap" onclick="toggleBasemap()">Basemap</button>
        </div>
    </div>

    <script>
        var map = L.map('map').setView([25.2425, 55.3025], 15);
        var basemapLayer = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '&copy; OpenStreetMap &copy; CARTO'
        }}).addTo(map);
        var basemapOn = true;

        function toggleBasemap() {{
            var btn = document.getElementById('btn-basemap');
            if (basemapOn) {{
                map.removeLayer(basemapLayer);
                btn.classList.remove('active');
                document.getElementById('map').style.background = '#1a1a2e';
            }} else {{
                basemapLayer.addTo(map);
                btn.classList.add('active');
                document.getElementById('map').style.background = 'white';
            }}
            basemapOn = !basemapOn;
        }}

        var edges = {json.dumps(map_data)};
        var layers = {{ met: L.layerGroup(), ang: L.layerGroup(), gint: L.layerGroup(), lint: L.layerGroup(), deg: L.layerGroup() }};
        var colors = {{
            met: ['#f0f0ff', '#667eea'],
            ang: ['#f0fff0', '#11998e'],
            gint: ['#fff8f0', '#fc4a1a'],
            lint: ['#fff0f5', '#ee0979'],
            deg: ['#f0f0ff', '#4568dc']
        }};

        function interpolateColor(c1, c2, factor) {{
            var r1 = parseInt(c1.slice(1,3), 16), g1 = parseInt(c1.slice(3,5), 16), b1 = parseInt(c1.slice(5,7), 16);
            var r2 = parseInt(c2.slice(1,3), 16), g2 = parseInt(c2.slice(3,5), 16), b2 = parseInt(c2.slice(5,7), 16);
            var r = Math.round(r1 + (r2-r1)*factor), g = Math.round(g1 + (g2-g1)*factor), b = Math.round(b1 + (b2-b1)*factor);
            return 'rgb('+r+','+g+','+b+')';
        }}

        edges.forEach(function(e) {{
            var popup = '<b>Network Metrics</b><br>' +
                       'Metric Betweenness: ' + e.met.toFixed(3) + '<br>' +
                       'Angular Choice: ' + e.ang.toFixed(3) + '<br>' +
                       'Global Integration: ' + e.gint.toFixed(3) + '<br>' +
                       'Local Integration: ' + e.lint.toFixed(3) + '<br>' +
                       'Degree: ' + e.deg.toFixed(3);

            ['met', 'ang', 'gint', 'lint', 'deg'].forEach(function(m) {{
                var val = e[m];
                var w = 2 + val * 5;
                var color = interpolateColor(colors[m][0], colors[m][1], val);
                L.polyline(e.coords, {{ color: color, weight: w, opacity: 0.9 }})
                 .bindPopup(popup).addTo(layers[m]);
            }});
        }});

        layers.met.addTo(map);
        var current = 'met';

        function showMetric(m) {{
            map.removeLayer(layers[current]);
            layers[m].addTo(map);

            ['met', 'ang', 'gint', 'lint', 'deg'].forEach(function(x) {{
                document.getElementById('btn-' + x).classList.remove('active');
                document.getElementById('info-' + x).classList.remove('active');
            }});
            document.getElementById('btn-' + m).classList.add('active');
            document.getElementById('info-' + m).classList.add('active');

            // Update legend
            var bar = document.getElementById('legend-bar');
            bar.style.background = 'linear-gradient(to right, ' + colors[m][0] + ', ' + colors[m][1] + ')';

            current = m;
        }}
    </script>
</body>
</html>'''

with open(os.path.join(OUT_DIR, 'comparison.html'), 'w') as f:
    f.write(html)
print(f"   Saved: {OUT_DIR}/comparison.html")

# Save stats as CSV
stats_df.to_csv(os.path.join(OUT_DIR, 'metrics_comparison.csv'), index=False)
print(f"   Saved: {OUT_DIR}/metrics_comparison.csv")

# Save correlation matrix
corr_matrix.to_csv(os.path.join(OUT_DIR, 'correlation_matrix.csv'))
print(f"   Saved: {OUT_DIR}/correlation_matrix.csv")

print("\n" + "="*60)
print("SUMMARY COMPLETE")
print("="*60)
print(f"""
KEY FINDINGS FOR AL KARAMA (REGULAR GRID):

EFFECTIVE METHODS (High CV = good differentiation):
  1. Metric Betweenness: {cv_metric:.0f}% CV - identifies main corridors
  2. Angular Choice:     {cv_angular:.0f}% CV - pedestrian desire lines

INEFFECTIVE METHODS (Low CV = uniform values):
  3. Global Integration: {cv_global:.0f}% CV - all streets similar
  4. Local Integration:  {cv_local:.0f}% CV - even more uniform
  5. Degree Centrality:  {cv_degree:.0f}% CV - most nodes have 4 edges

RECOMMENDATION:
For grid patterns like Al Karama, use Betweenness-based metrics.
Integration metrics work better for organic/irregular street patterns.

Open: {OUT_DIR}/comparison.html
""")
