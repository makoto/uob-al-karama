#!/usr/bin/env python3
"""
Street Network Analysis for Al Karama - v2

Fixed version with proper centrality calculations.
"""

import osmnx as ox
import networkx as nx
import pandas as pd
import geopandas as gpd
import numpy as np
import json
import os
from scipy.spatial import cKDTree

print("="*60)
print("STREET NETWORK ANALYSIS")
print("Al Karama, Dubai")
print("="*60)

# Al Karama boundary
north, south = 25.255, 25.230
east, west = 55.315, 55.290

# Download street network
print("\nDownloading street network from OpenStreetMap...")
G = ox.graph_from_bbox(bbox=(north, south, east, west), network_type='walk')
print(f"  Nodes: {G.number_of_nodes()}")
print(f"  Edges: {G.number_of_edges()}")

# Convert to undirected for centrality analysis
G_undir = G.to_undirected()

# Basic stats
print("\n" + "-"*40)
print("NETWORK STATISTICS")
print("-"*40)
stats = ox.stats.basic_stats(G)
print(f"  Street segments: {stats['m']}")
print(f"  Intersections: {stats['n']}")
print(f"  Total length: {stats['street_length_total']/1000:.1f} km")

# Calculate NODE centrality (more reliable)
print("\nCalculating centrality metrics (this may take a moment)...")

# Betweenness centrality (node-based, normalized)
print("  Node betweenness centrality...")
node_bc = nx.betweenness_centrality(G_undir, weight='length', normalized=True)

# Closeness centrality (node-based)
print("  Node closeness centrality...")
node_cc = nx.closeness_centrality(G_undir, distance='length')

# Degree centrality
print("  Degree centrality...")
node_dc = nx.degree_centrality(G_undir)

# Add to nodes
nx.set_node_attributes(G_undir, node_bc, 'betweenness')
nx.set_node_attributes(G_undir, node_cc, 'closeness')
nx.set_node_attributes(G_undir, node_dc, 'degree')

# Convert to GeoDataFrames
print("\nConverting to GeoDataFrame...")
nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_undir)

# Transfer node centrality to edges (average of endpoints)
def get_edge_centrality(row, metric_dict):
    u, v = row.name[0], row.name[1]
    return (metric_dict.get(u, 0) + metric_dict.get(v, 0)) / 2

edges_gdf['betweenness'] = edges_gdf.apply(lambda r: get_edge_centrality(r, node_bc), axis=1)
edges_gdf['closeness'] = edges_gdf.apply(lambda r: get_edge_centrality(r, node_cc), axis=1)
edges_gdf['degree'] = edges_gdf.apply(lambda r: get_edge_centrality(r, node_dc), axis=1)

# Calculate edge midpoints
edges_gdf['mid_lon'] = edges_gdf.geometry.centroid.x
edges_gdf['mid_lat'] = edges_gdf.geometry.centroid.y

# Statistics
print("\n" + "-"*40)
print("CENTRALITY STATISTICS")
print("-"*40)
print(f"Betweenness (node-based, transferred to edges):")
print(f"  Mean: {edges_gdf['betweenness'].mean():.6f}")
print(f"  Max:  {edges_gdf['betweenness'].max():.6f}")
print(f"  Top 5%: {edges_gdf['betweenness'].quantile(0.95):.6f}")
print(f"Closeness:")
print(f"  Mean: {edges_gdf['closeness'].mean():.6f}")
print(f"  Max:  {edges_gdf['closeness'].max():.6f}")

# Load pedestrian comfort data
print("\nLoading pedestrian comfort data...")
comfort_df = pd.read_csv('output/walking_routes/segment_comfort.csv')
print(f"  Comfort segments: {len(comfort_df)}")

# Match edges to comfort data
print("\nMatching network edges to comfort data...")
comfort_coords = comfort_df[['lat', 'lon']].values
comfort_tree = cKDTree(comfort_coords)

edge_coords = edges_gdf[['mid_lat', 'mid_lon']].values
distances, indices = comfort_tree.query(edge_coords, k=1)

# Match within 50m
max_dist = 0.0005
edges_gdf['pci'] = np.where(distances < max_dist, comfort_df.iloc[indices]['pci_mean'].values, np.nan)
edges_gdf['lst'] = np.where(distances < max_dist, comfort_df.iloc[indices]['lst_mean'].values, np.nan)
edges_gdf['shade'] = np.where(distances < max_dist, comfort_df.iloc[indices]['shade_mean'].values, np.nan)

matched = edges_gdf['pci'].notna().sum()
print(f"  Matched edges: {matched} / {len(edges_gdf)} ({matched/len(edges_gdf)*100:.1f}%)")

# Normalize centrality metrics
bc_min, bc_max = edges_gdf['betweenness'].min(), edges_gdf['betweenness'].max()
cc_min, cc_max = edges_gdf['closeness'].min(), edges_gdf['closeness'].max()

if bc_max > bc_min:
    edges_gdf['bc_norm'] = (edges_gdf['betweenness'] - bc_min) / (bc_max - bc_min)
else:
    edges_gdf['bc_norm'] = 0.5

if cc_max > cc_min:
    edges_gdf['cc_norm'] = (edges_gdf['closeness'] - cc_min) / (cc_max - cc_min)
else:
    edges_gdf['cc_norm'] = 0.5

# Combined centrality score
edges_gdf['centrality'] = 0.6 * edges_gdf['bc_norm'] + 0.4 * edges_gdf['cc_norm']

# Discomfort (inverted PCI)
edges_gdf['discomfort'] = 1 - edges_gdf['pci'].fillna(0.5)

# Priority = high centrality AND high discomfort
edges_gdf['priority'] = edges_gdf['centrality'] * edges_gdf['discomfort']

# For edges without comfort data, use centrality only (halved)
edges_gdf.loc[edges_gdf['pci'].isna(), 'priority'] = edges_gdf.loc[edges_gdf['pci'].isna(), 'centrality'] * 0.5

# Top priority streets
print("\n" + "-"*40)
print("TOP 15 HIGHEST CENTRALITY STREETS")
print("-"*40)
top_central = edges_gdf.nlargest(15, 'centrality')
print(f"{'#':>3} {'Betweenness':>12} {'Closeness':>10} {'Centrality':>10}")
print("-" * 40)
for i, (idx, row) in enumerate(top_central.iterrows()):
    print(f"{i+1:>3} {row['betweenness']:>12.6f} {row['closeness']:>10.6f} {row['centrality']:>10.4f}")

print("\n" + "-"*40)
print("TOP 15 PRIORITY STREETS (High Traffic + Uncomfortable)")
print("-"*40)
priority_edges = edges_gdf.dropna(subset=['pci']).nlargest(15, 'priority')
print(f"{'#':>3} {'Centrality':>10} {'PCI':>6} {'LST':>6} {'Priority':>8}")
print("-" * 45)
for i, (idx, row) in enumerate(priority_edges.iterrows()):
    print(f"{i+1:>3} {row['centrality']:>10.4f} {row['pci']:>6.3f} {row['lst']:>6.1f} {row['priority']:>8.4f}")

# Save outputs
os.makedirs('output/network_analysis', exist_ok=True)

edges_export = edges_gdf[['geometry', 'length', 'betweenness', 'closeness', 'degree',
                          'centrality', 'pci', 'lst', 'priority', 'mid_lat', 'mid_lon']].copy()
edges_export = edges_export.reset_index(drop=True)
edges_export.to_file('output/network_analysis/street_network.geojson', driver='GeoJSON')
print(f"\nSaved: output/network_analysis/street_network.geojson")

# Create map
print("\nCreating interactive map...")

map_data = []
for idx, row in edges_gdf.reset_index(drop=True).iterrows():
    coords = list(row.geometry.coords)
    map_data.append({
        'coords': [[c[1], c[0]] for c in coords],
        'bc': round(float(row['betweenness']), 6),
        'cc': round(float(row['closeness']), 6),
        'centrality': round(float(row['centrality']), 4),
        'pci': round(float(row['pci']), 3) if pd.notna(row['pci']) else None,
        'lst': round(float(row['lst']), 1) if pd.notna(row['lst']) else None,
        'priority': round(float(row['priority']), 4),
        'length': round(float(row['length']), 1)
    })

bc_max = float(edges_gdf['betweenness'].max())
cc_max = float(edges_gdf['closeness'].max())
total_length = float(edges_gdf['length'].sum() / 1000)
n_segments = len(edges_gdf)

# High centrality count
high_centrality = len(edges_gdf[edges_gdf['centrality'] > 0.7])
high_priority = len(edges_gdf[(edges_gdf['priority'] > 0.4) & (edges_gdf['pci'].notna())])

html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Street Network Analysis - Al Karama</title>
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
            max-width: 360px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            max-height: 90vh;
            overflow-y: auto;
        }}
        .panel h2 {{ margin: 0 0 5px 0; color: #6a1b9a; }}
        .panel h3 {{ margin: 15px 0 8px 0; font-size: 14px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
        .stats {{ font-size: 13px; }}
        .stat-row {{ display: flex; justify-content: space-between; padding: 4px 0; }}
        .legend {{ margin: 10px 0; }}
        .legend-item {{ display: flex; align-items: center; margin: 4px 0; font-size: 12px; }}
        .legend-color {{ width: 30px; height: 4px; margin-right: 8px; }}
        .btn {{ padding: 8px 12px; margin: 3px; border: none; border-radius: 4px; cursor: pointer; font-size: 11px; }}
        .btn-active {{ background: #6a1b9a; color: white; }}
        .btn:not(.btn-active) {{ background: #e0e0e0; }}
        .metric-desc {{ font-size: 11px; color: #666; margin: 5px 0 10px 0; line-height: 1.4; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="panel">
        <h2>Street Network Analysis</h2>
        <p style="margin: 5px 0; color: #666; font-size: 12px;">Centrality metrics from OpenStreetMap</p>

        <h3>Network Stats</h3>
        <div class="stats">
            <div class="stat-row"><span>Street segments:</span><span>{n_segments:,}</span></div>
            <div class="stat-row"><span>Total length:</span><span>{total_length:.1f} km</span></div>
            <div class="stat-row"><span>High centrality streets:</span><span>{high_centrality}</span></div>
            <div class="stat-row"><span>Priority intervention:</span><span>{high_priority}</span></div>
        </div>

        <h3>Display Metric</h3>
        <div>
            <button class="btn btn-active" id="btn-bc" onclick="showMetric('bc')">Betweenness</button>
            <button class="btn" id="btn-cc" onclick="showMetric('cc')">Closeness</button>
            <button class="btn" id="btn-centrality" onclick="showMetric('centrality')">Combined</button>
            <button class="btn" id="btn-priority" onclick="showMetric('priority')">Priority</button>
        </div>

        <div id="metric-info">
            <h3 id="metric-title">Betweenness Centrality</h3>
            <div class="metric-desc" id="metric-desc">
                How often a street lies on shortest paths between locations.
                High values indicate main through-routes that pedestrians must use to traverse the area.
            </div>
            <div class="legend" id="legend-bc">
                <div class="legend-item"><div class="legend-color" style="background: #e1bee7;"></div> Low (local streets)</div>
                <div class="legend-item"><div class="legend-color" style="background: #ab47bc;"></div> Medium</div>
                <div class="legend-item"><div class="legend-color" style="background: #4a148c;"></div> High (main corridors)</div>
            </div>
            <div class="legend" id="legend-cc" style="display:none;">
                <div class="legend-item"><div class="legend-color" style="background: #bbdefb;"></div> Low (peripheral)</div>
                <div class="legend-item"><div class="legend-color" style="background: #42a5f5;"></div> Medium</div>
                <div class="legend-item"><div class="legend-color" style="background: #0d47a1;"></div> High (central/accessible)</div>
            </div>
            <div class="legend" id="legend-centrality" style="display:none;">
                <div class="legend-item"><div class="legend-color" style="background: #f3e5f5;"></div> Low importance</div>
                <div class="legend-item"><div class="legend-color" style="background: #7b1fa2;"></div> Medium importance</div>
                <div class="legend-item"><div class="legend-color" style="background: #311b92;"></div> High importance</div>
            </div>
            <div class="legend" id="legend-priority" style="display:none;">
                <div class="legend-item"><div class="legend-color" style="background: #c8e6c9;"></div> Low priority</div>
                <div class="legend-item"><div class="legend-color" style="background: #ffb74d;"></div> Medium priority</div>
                <div class="legend-item"><div class="legend-color" style="background: #d32f2f;"></div> High priority (intervene!)</div>
                <div class="legend-item"><div class="legend-color" style="background: #9e9e9e;"></div> No comfort data</div>
            </div>
        </div>

        <h3>How to Use</h3>
        <div style="font-size: 11px; color: #666; line-height: 1.5;">
            <p><b>Betweenness:</b> Main pedestrian corridors - streets people use to get around.</p>
            <p><b>Closeness:</b> Most accessible locations - easy to reach from anywhere.</p>
            <p><b>Priority:</b> High traffic + uncomfortable = most people affected. Focus interventions here.</p>
        </div>
    </div>

    <script>
        var map = L.map('map').setView([25.2425, 55.3025], 15);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; OpenStreetMap'
        }}).addTo(map);

        var edges = {json.dumps(map_data)};
        var bcMax = {bc_max};
        var ccMax = {cc_max};

        var layers = {{}};
        ['bc', 'cc', 'centrality', 'priority'].forEach(m => layers[m] = L.layerGroup());

        function getBcColor(val) {{
            var ratio = val / bcMax;
            if (ratio < 0.2) return '#e1bee7';
            if (ratio < 0.5) return '#ab47bc';
            if (ratio < 0.75) return '#7b1fa2';
            return '#4a148c';
        }}

        function getCcColor(val) {{
            var ratio = val / ccMax;
            if (ratio < 0.3) return '#bbdefb';
            if (ratio < 0.6) return '#42a5f5';
            if (ratio < 0.8) return '#1976d2';
            return '#0d47a1';
        }}

        function getCentralityColor(val) {{
            if (val < 0.3) return '#f3e5f5';
            if (val < 0.5) return '#ce93d8';
            if (val < 0.7) return '#7b1fa2';
            return '#311b92';
        }}

        function getPriorityColor(val, hasPci) {{
            if (!hasPci) return '#9e9e9e';
            if (val < 0.2) return '#c8e6c9';
            if (val < 0.35) return '#fff176';
            if (val < 0.5) return '#ffb74d';
            return '#d32f2f';
        }}

        edges.forEach(function(e) {{
            var popup = '<b>Street Segment</b><br>' +
                       'Length: ' + e.length + 'm<br>' +
                       'Betweenness: ' + e.bc.toFixed(5) + '<br>' +
                       'Closeness: ' + e.cc.toFixed(5) + '<br>' +
                       'Combined Centrality: ' + e.centrality.toFixed(3) + '<br>' +
                       (e.pci !== null ? 'Comfort (PCI): ' + e.pci.toFixed(3) + '<br>' : 'Comfort: No data<br>') +
                       (e.lst !== null ? 'Temperature: ' + e.lst + '°C<br>' : '') +
                       'Priority: ' + e.priority.toFixed(3);

            var bcWeight = 2 + (e.bc / bcMax) * 6;
            var ccWeight = 2 + (e.cc / ccMax) * 5;
            var centWeight = 2 + e.centrality * 6;
            var prioWeight = 2 + e.priority * 8;

            L.polyline(e.coords, {{ color: getBcColor(e.bc), weight: bcWeight, opacity: 0.8 }}).bindPopup(popup).addTo(layers.bc);
            L.polyline(e.coords, {{ color: getCcColor(e.cc), weight: ccWeight, opacity: 0.8 }}).bindPopup(popup).addTo(layers.cc);
            L.polyline(e.coords, {{ color: getCentralityColor(e.centrality), weight: centWeight, opacity: 0.8 }}).bindPopup(popup).addTo(layers.centrality);
            L.polyline(e.coords, {{ color: getPriorityColor(e.priority, e.pci !== null), weight: prioWeight, opacity: 0.85 }}).bindPopup(popup).addTo(layers.priority);
        }});

        layers.bc.addTo(map);
        var currentMetric = 'bc';

        var metricInfo = {{
            bc: {{ title: 'Betweenness Centrality', desc: 'How often a street lies on shortest paths between locations. High values indicate main through-routes that pedestrians must use to traverse the area.' }},
            cc: {{ title: 'Closeness Centrality', desc: 'How close (accessible) a street is to all other streets. High values indicate central, well-connected locations easy to reach from anywhere.' }},
            centrality: {{ title: 'Combined Centrality', desc: 'Weighted combination of betweenness (60%) and closeness (40%). Identifies the most important streets in the pedestrian network.' }},
            priority: {{ title: 'Intervention Priority', desc: 'High centrality + Low comfort = Priority. Red streets are heavily used but uncomfortable. These affect the most pedestrians and should be prioritized for tree planting or shade structures.' }}
        }};

        function showMetric(name) {{
            map.removeLayer(layers[currentMetric]);
            layers[name].addTo(map);
            document.querySelectorAll('.btn').forEach(b => b.classList.remove('btn-active'));
            document.getElementById('btn-' + name).classList.add('btn-active');
            document.querySelectorAll('.legend').forEach(l => l.style.display = 'none');
            document.getElementById('legend-' + name).style.display = 'block';
            document.getElementById('metric-title').textContent = metricInfo[name].title;
            document.getElementById('metric-desc').textContent = metricInfo[name].desc;
            currentMetric = name;
        }}
    </script>
</body>
</html>'''

with open('output/network_analysis/network_analysis_map.html', 'w') as f:
    f.write(html)
print(f"Saved: output/network_analysis/network_analysis_map.html")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"""
Key findings:
- {n_segments} street segments in pedestrian network
- {high_centrality} high-centrality streets (main corridors)
- {high_priority} priority streets for intervention

Open: output/network_analysis/network_analysis_map.html
""")
