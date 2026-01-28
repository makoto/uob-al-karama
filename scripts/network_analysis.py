#!/usr/bin/env python3
"""
Street Network Analysis for Al Karama

Uses OSMnx, NetworkX, and momepy to analyze:
- Network centrality (betweenness, closeness)
- Space syntax metrics (integration, choice)
- Combined with pedestrian comfort data

Identifies high-traffic uncomfortable streets for priority intervention.
"""

import osmnx as ox
import networkx as nx
import momepy
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

# Al Karama boundary (approximate)
north, south = 25.255, 25.230
east, west = 55.315, 55.290

# Download street network from OSM
print("\nDownloading street network from OpenStreetMap...")
G = ox.graph_from_bbox(bbox=(north, south, east, west), network_type='walk')
print(f"  Nodes: {G.number_of_nodes()}")
print(f"  Edges: {G.number_of_edges()}")

# Convert to undirected for analysis
G_undir = ox.convert.to_undirected(G)

# Basic network stats
print("\n" + "-"*40)
print("NETWORK STATISTICS")
print("-"*40)
stats = ox.stats.basic_stats(G)
print(f"  Street segments: {stats['m']}")
print(f"  Intersections: {stats['n']}")
print(f"  Total length: {stats['street_length_total']/1000:.1f} km")
print(f"  Avg street length: {stats['street_length_avg']:.1f} m")

# Calculate centrality metrics
print("\nCalculating centrality metrics...")

# Betweenness centrality (edge-based)
print("  Betweenness centrality...")
edge_bc = nx.edge_betweenness_centrality(G_undir, weight='length')

# Closeness centrality (node-based)
print("  Closeness centrality...")
node_cc = nx.closeness_centrality(G_undir, distance='length')

# Add to graph
nx.set_edge_attributes(G_undir, edge_bc, 'betweenness')
nx.set_node_attributes(G_undir, node_cc, 'closeness')

# Convert to GeoDataFrame for spatial analysis
print("\nConverting to GeoDataFrame...")
nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_undir)

# Calculate edge closeness as average of endpoint nodes
edges_gdf['closeness'] = edges_gdf.apply(
    lambda row: (node_cc.get(row.name[0], 0) + node_cc.get(row.name[1], 0)) / 2,
    axis=1
)
edges_gdf['betweenness'] = edges_gdf.index.map(lambda x: edge_bc.get((x[0], x[1]), edge_bc.get((x[1], x[0]), 0)))

# Space Syntax analysis using momepy
print("\nCalculating space syntax metrics...")

# Need to prepare edges for momepy
edges_for_momepy = edges_gdf.reset_index(drop=True).copy()
edges_for_momepy = edges_for_momepy[['geometry', 'length', 'betweenness', 'closeness']].copy()
edges_for_momepy['mm_id'] = range(len(edges_for_momepy))

# Create momepy graph
print("  Building spatial weights...")
try:
    # Create connectivity graph based on shared nodes
    sw = momepy.sw_high(k=3, gdf=edges_for_momepy, ids='mm_id')

    # Calculate reach (local integration approximation)
    print("  Calculating reach (local integration)...")
    edges_for_momepy['reach'] = momepy.Reach(edges_for_momepy, sw, 'mm_id', mode='count').series

    # Straightness (meshedness proxy)
    print("  Calculating straightness...")
    edges_for_momepy['straightness'] = momepy.Straightness(edges_for_momepy).series

    momepy_success = True
except Exception as e:
    print(f"  momepy calculation failed: {e}")
    print("  Continuing with NetworkX metrics only...")
    momepy_success = False

# Calculate edge midpoints for joining with comfort data
print("\nCalculating edge midpoints...")
edges_gdf['mid_lat'] = edges_gdf.geometry.centroid.y
edges_gdf['mid_lon'] = edges_gdf.geometry.centroid.x

if momepy_success:
    edges_gdf['reach'] = edges_for_momepy['reach'].values
    edges_gdf['straightness'] = edges_for_momepy['straightness'].values

# Load pedestrian comfort data
print("\nLoading pedestrian comfort data...")
comfort_df = pd.read_csv('output/walking_routes/segment_comfort.csv')
print(f"  Comfort segments: {len(comfort_df)}")

# Match edges to nearest comfort segment
print("\nMatching network edges to comfort data...")
comfort_coords = comfort_df[['lat', 'lon']].values
comfort_tree = cKDTree(comfort_coords)

edge_coords = edges_gdf[['mid_lat', 'mid_lon']].values
distances, indices = comfort_tree.query(edge_coords, k=1)

# Add comfort data to edges (only if within 50m)
max_dist = 0.0005  # ~50m
edges_gdf['pci'] = np.where(distances < max_dist, comfort_df.iloc[indices]['pci_mean'].values, np.nan)
edges_gdf['lst'] = np.where(distances < max_dist, comfort_df.iloc[indices]['lst_mean'].values, np.nan)
edges_gdf['shade'] = np.where(distances < max_dist, comfort_df.iloc[indices]['shade_mean'].values, np.nan)

matched = edges_gdf['pci'].notna().sum()
print(f"  Matched edges: {matched} / {len(edges_gdf)} ({matched/len(edges_gdf)*100:.1f}%)")

# Calculate priority score for intervention
# High priority = High centrality + Low comfort
print("\nCalculating intervention priority...")

# Normalize metrics
edges_gdf['bc_norm'] = (edges_gdf['betweenness'] - edges_gdf['betweenness'].min()) / \
                       (edges_gdf['betweenness'].max() - edges_gdf['betweenness'].min())
edges_gdf['cc_norm'] = (edges_gdf['closeness'] - edges_gdf['closeness'].min()) / \
                       (edges_gdf['closeness'].max() - edges_gdf['closeness'].min())

# Invert PCI (low comfort = high priority)
edges_gdf['discomfort'] = 1 - edges_gdf['pci'].fillna(0.5)

# Combined priority: centrality * discomfort
edges_gdf['priority'] = (0.5 * edges_gdf['bc_norm'] + 0.5 * edges_gdf['cc_norm']) * edges_gdf['discomfort']

# Statistics
print("\n" + "-"*40)
print("CENTRALITY STATISTICS")
print("-"*40)
print(f"Betweenness centrality:")
print(f"  Mean: {edges_gdf['betweenness'].mean():.6f}")
print(f"  Max:  {edges_gdf['betweenness'].max():.6f}")
print(f"Closeness centrality:")
print(f"  Mean: {edges_gdf['closeness'].mean():.4f}")
print(f"  Max:  {edges_gdf['closeness'].max():.4f}")

if momepy_success:
    print(f"Reach (local integration):")
    print(f"  Mean: {edges_gdf['reach'].mean():.1f}")
    print(f"  Max:  {edges_gdf['reach'].max():.1f}")

# Top priority streets (high traffic + uncomfortable)
print("\n" + "-"*40)
print("TOP 15 PRIORITY STREETS (High Traffic + Uncomfortable)")
print("-"*40)

priority_edges = edges_gdf.dropna(subset=['pci']).nlargest(15, 'priority')
print(f"{'#':>3} {'Betweenness':>12} {'Closeness':>10} {'PCI':>6} {'LST':>6} {'Priority':>8}")
print("-" * 55)
for i, (idx, row) in enumerate(priority_edges.iterrows()):
    print(f"{i+1:>3} {row['betweenness']:>12.6f} {row['closeness']:>10.4f} "
          f"{row['pci']:>6.3f} {row['lst']:>6.1f} {row['priority']:>8.4f}")

# Save data
os.makedirs('output/network_analysis', exist_ok=True)

# Save edges as GeoJSON
edges_export = edges_gdf[['geometry', 'length', 'betweenness', 'closeness',
                          'pci', 'lst', 'shade', 'priority', 'mid_lat', 'mid_lon']].copy()
edges_export = edges_export.reset_index(drop=True)
edges_export.to_file('output/network_analysis/street_network.geojson', driver='GeoJSON')
print(f"\nSaved: output/network_analysis/street_network.geojson")

# Save as CSV (without geometry)
edges_csv = edges_gdf[['length', 'betweenness', 'closeness', 'pci', 'lst',
                       'shade', 'priority', 'mid_lat', 'mid_lon']].copy()
edges_csv = edges_csv.reset_index(drop=True)
edges_csv.to_csv('output/network_analysis/street_metrics.csv', index=False)
print(f"Saved: output/network_analysis/street_metrics.csv")

# Create interactive map
print("\nCreating interactive map...")

# Prepare data for map
edges_for_map = edges_gdf[['geometry', 'betweenness', 'closeness', 'pci', 'lst',
                           'priority', 'length']].copy()
edges_for_map = edges_for_map.reset_index(drop=True)

# Convert to list of line coordinates
map_data = []
for idx, row in edges_for_map.iterrows():
    coords = list(row.geometry.coords)
    map_data.append({
        'coords': [[c[1], c[0]] for c in coords],  # [lat, lon]
        'bc': round(float(row['betweenness']), 6) if pd.notna(row['betweenness']) else 0,
        'cc': round(float(row['closeness']), 4) if pd.notna(row['closeness']) else 0,
        'pci': round(float(row['pci']), 3) if pd.notna(row['pci']) else None,
        'lst': round(float(row['lst']), 1) if pd.notna(row['lst']) else None,
        'priority': round(float(row['priority']), 4) if pd.notna(row['priority']) else 0,
        'length': round(float(row['length']), 1)
    })

# Statistics for display
bc_max = float(edges_gdf['betweenness'].max())
cc_max = float(edges_gdf['closeness'].max())
total_length = float(edges_gdf['length'].sum() / 1000)
n_segments = len(edges_gdf)

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
        .metric-desc {{ font-size: 11px; color: #666; margin: 5px 0 10px 0; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="panel">
        <h2>Street Network Analysis</h2>
        <p style="margin: 5px 0; color: #666; font-size: 12px;">Centrality & Space Syntax metrics from OSM</p>

        <h3>Network Stats</h3>
        <div class="stats">
            <div class="stat-row"><span>Street segments:</span><span>{n_segments:,}</span></div>
            <div class="stat-row"><span>Total length:</span><span>{total_length:.1f} km</span></div>
        </div>

        <h3>Display Metric</h3>
        <div>
            <button class="btn btn-active" id="btn-bc" onclick="showMetric('bc')">Betweenness</button>
            <button class="btn" id="btn-cc" onclick="showMetric('cc')">Closeness</button>
            <button class="btn" id="btn-priority" onclick="showMetric('priority')">Priority</button>
            <button class="btn" id="btn-comfort" onclick="showMetric('comfort')">Comfort</button>
        </div>

        <div id="metric-info">
            <h3 id="metric-title">Betweenness Centrality</h3>
            <div class="metric-desc" id="metric-desc">
                How often a street is on the shortest path between other locations.
                High values = major through-routes.
            </div>
            <div class="legend" id="legend-bc">
                <div class="legend-item"><div class="legend-color" style="background: #f3e5f5;"></div> Low (local streets)</div>
                <div class="legend-item"><div class="legend-color" style="background: #9c27b0;"></div> Medium</div>
                <div class="legend-item"><div class="legend-color" style="background: #4a148c;"></div> High (main routes)</div>
            </div>
            <div class="legend" id="legend-cc" style="display:none;">
                <div class="legend-item"><div class="legend-color" style="background: #e3f2fd;"></div> Low (peripheral)</div>
                <div class="legend-item"><div class="legend-color" style="background: #1976d2;"></div> Medium</div>
                <div class="legend-item"><div class="legend-color" style="background: #0d47a1;"></div> High (central)</div>
            </div>
            <div class="legend" id="legend-priority" style="display:none;">
                <div class="legend-item"><div class="legend-color" style="background: #c8e6c9;"></div> Low priority</div>
                <div class="legend-item"><div class="legend-color" style="background: #ff9800;"></div> Medium priority</div>
                <div class="legend-item"><div class="legend-color" style="background: #d32f2f;"></div> High priority (intervene)</div>
            </div>
            <div class="legend" id="legend-comfort" style="display:none;">
                <div class="legend-item"><div class="legend-color" style="background: #1b5e20;"></div> Comfortable</div>
                <div class="legend-item"><div class="legend-color" style="background: #ffb74d;"></div> Moderate</div>
                <div class="legend-item"><div class="legend-color" style="background: #e53935;"></div> Uncomfortable</div>
            </div>
        </div>

        <h3>Interpretation</h3>
        <div style="font-size: 11px; color: #666;">
            <p><b>Betweenness:</b> Identifies main pedestrian routes - streets people must use to get around.</p>
            <p><b>Closeness:</b> Measures accessibility - how easy to reach everywhere from this street.</p>
            <p><b>Priority:</b> High centrality + Low comfort = streets where interventions help most people.</p>
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

        var layers = {{
            bc: L.layerGroup(),
            cc: L.layerGroup(),
            priority: L.layerGroup(),
            comfort: L.layerGroup()
        }};

        function getBcColor(val) {{
            var ratio = val / bcMax;
            if (ratio < 0.1) return '#f3e5f5';
            if (ratio < 0.3) return '#ce93d8';
            if (ratio < 0.5) return '#9c27b0';
            return '#4a148c';
        }}

        function getCcColor(val) {{
            var ratio = val / ccMax;
            if (ratio < 0.3) return '#e3f2fd';
            if (ratio < 0.6) return '#64b5f6';
            if (ratio < 0.8) return '#1976d2';
            return '#0d47a1';
        }}

        function getPriorityColor(val) {{
            if (val < 0.2) return '#c8e6c9';
            if (val < 0.4) return '#fff176';
            if (val < 0.6) return '#ff9800';
            return '#d32f2f';
        }}

        function getComfortColor(pci) {{
            if (pci === null) return '#9e9e9e';
            if (pci >= 0.5) return '#1b5e20';
            if (pci >= 0.35) return '#ffb74d';
            return '#e53935';
        }}

        edges.forEach(function(e) {{
            var popup = '<b>Street Segment</b><br>' +
                       'Length: ' + e.length + 'm<br>' +
                       'Betweenness: ' + e.bc.toFixed(6) + '<br>' +
                       'Closeness: ' + e.cc.toFixed(4) + '<br>' +
                       (e.pci !== null ? 'Comfort (PCI): ' + e.pci.toFixed(3) + '<br>' : '') +
                       (e.lst !== null ? 'Temperature: ' + e.lst + '°C<br>' : '') +
                       'Priority: ' + e.priority.toFixed(4);

            L.polyline(e.coords, {{
                color: getBcColor(e.bc),
                weight: 3 + (e.bc / bcMax) * 5,
                opacity: 0.8
            }}).bindPopup(popup).addTo(layers.bc);

            L.polyline(e.coords, {{
                color: getCcColor(e.cc),
                weight: 3 + (e.cc / ccMax) * 4,
                opacity: 0.8
            }}).bindPopup(popup).addTo(layers.cc);

            L.polyline(e.coords, {{
                color: getPriorityColor(e.priority),
                weight: 3 + e.priority * 6,
                opacity: 0.85
            }}).bindPopup(popup).addTo(layers.priority);

            L.polyline(e.coords, {{
                color: getComfortColor(e.pci),
                weight: 4,
                opacity: 0.8
            }}).bindPopup(popup).addTo(layers.comfort);
        }});

        layers.bc.addTo(map);
        var currentMetric = 'bc';

        var metricInfo = {{
            bc: {{
                title: 'Betweenness Centrality',
                desc: 'How often a street is on the shortest path between other locations. High values = major through-routes that many pedestrians use.'
            }},
            cc: {{
                title: 'Closeness Centrality',
                desc: 'How accessible a street is to all other locations. High values = central, well-connected streets.'
            }},
            priority: {{
                title: 'Intervention Priority',
                desc: 'Combines high traffic (centrality) with low comfort. Red streets = high traffic but uncomfortable, need intervention.'
            }},
            comfort: {{
                title: 'Pedestrian Comfort',
                desc: 'Walking comfort based on shade, vegetation, and temperature. Gray = no data.'
            }}
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

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"""
Network Analysis Complete:

1. NETWORK METRICS:
   - {n_segments} street segments analyzed
   - {total_length:.1f} km total street length
   - Betweenness & closeness centrality calculated

2. COMBINED WITH COMFORT:
   - {matched} segments matched to comfort data
   - Priority score = Centrality x Discomfort

3. KEY FINDINGS:
   - High betweenness streets: Main pedestrian corridors
   - High priority streets: Busy but uncomfortable (need intervention)

4. OUTPUT FILES:
   - network_analysis_map.html: Interactive visualization
   - street_network.geojson: Full network with metrics (for GIS)
   - street_metrics.csv: Metrics table

Open the map: output/network_analysis/network_analysis_map.html
""")

print("Done!")
