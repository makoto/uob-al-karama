#!/usr/bin/env python3
"""
Isochrone Analysis for Al Karama
=================================

Shows what areas are reachable within 5/10/15 minute walks from:
- Metro stations
- Bus stations

Isochrones reveal transit accessibility and can be combined with
comfort data to find "comfortable walk zones" from transit.
"""

import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np
import json
import os
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from scipy.spatial import ConvexHull, Delaunay
import alphashape

print("="*60)
print("ISOCHRONE ANALYSIS")
print("Walking Distance from Transit Stations")
print("Al Karama, Dubai")
print("="*60)

# Paths
BASE = os.path.dirname(__file__)
POIS_PATH = os.path.join(BASE, '..', 'docs', 'data', 'al_karama', 'pois.json')
OUT_DIR = os.path.join(BASE, '..', 'output', 'isochrone_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

# Al Karama boundary (slightly expanded for isochrones)
ak_north, ak_south = 25.260, 25.225
ak_east, ak_west = 55.320, 55.285

# Transit stations
TRANSIT_STATIONS = [
    # Metro stations
    {'name': 'ADCB Metro Station', 'lat': 25.2445, 'lon': 55.2981, 'type': 'metro'},  # Red Line (formerly Al Karama)
    {'name': 'Burjuman Metro Station', 'lat': 25.2547, 'lon': 55.3043, 'type': 'metro'},  # Red/Green interchange
    {'name': 'Oud Metha Metro Station', 'lat': 25.2436, 'lon': 55.3160, 'type': 'metro'},  # Green Line

    # Bus stations from POIs
    {'name': 'Big Bus Stop 4', 'lat': 25.25228, 'lon': 55.302412, 'type': 'bus'},
    {'name': 'Oud Metha Bus Station', 'lat': 25.244543, 'lon': 55.313219, 'type': 'bus'},
    {'name': 'Al Karama Bus Station', 'lat': 25.237757, 'lon': 55.303022, 'type': 'bus'},
]

# Walking speeds and times
WALK_SPEED_MPS = 1.2  # meters per second (~4.3 km/h, relaxed pace in heat)
WALK_TIMES_MIN = [5, 10, 15]  # isochrone intervals

# Download street network
print("\n1. DOWNLOADING STREET NETWORK")
print("-"*40)
G = ox.graph_from_bbox(bbox=(ak_north, ak_south, ak_east, ak_west), network_type='walk')
print(f"   Nodes: {G.number_of_nodes()}")
print(f"   Edges: {G.number_of_edges()}")

# Find nearest node for each transit station
print("\n2. MAPPING TRANSIT STATIONS TO NETWORK")
print("-"*40)

for station in TRANSIT_STATIONS:
    nearest = ox.nearest_nodes(G, station['lon'], station['lat'])
    station['node'] = nearest
    print(f"   {station['name']} ({station['type']}): node {nearest}")


def get_isochrone_nodes(G, center_node, max_distance_m):
    """Get all nodes reachable within max_distance_m from center_node."""
    # Use Dijkstra to find shortest paths from center
    lengths = nx.single_source_dijkstra_path_length(G, center_node, cutoff=max_distance_m, weight='length')
    return list(lengths.keys())


def nodes_to_polygon(G, nodes, alpha=0.0003):
    """Convert a set of nodes to a polygon using alpha shapes."""
    if len(nodes) < 3:
        return None

    # Get coordinates
    coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in nodes]

    try:
        # Use alphashape for concave hull
        points = [(x, y) for x, y in coords]
        hull = alphashape.alphashape(points, alpha)
        if hull.is_empty:
            # Fallback to convex hull
            from shapely.geometry import MultiPoint
            hull = MultiPoint(points).convex_hull
        return hull
    except Exception as e:
        # Fallback to convex hull
        from shapely.geometry import MultiPoint
        return MultiPoint(coords).convex_hull


# Calculate isochrones for each station
print("\n3. CALCULATING ISOCHRONES")
print("-"*40)

all_isochrones = []

for station in TRANSIT_STATIONS:
    print(f"\n   {station['name']}:")

    for walk_time in WALK_TIMES_MIN:
        max_dist = walk_time * 60 * WALK_SPEED_MPS  # meters

        # Get reachable nodes
        reachable = get_isochrone_nodes(G, station['node'], max_dist)
        print(f"      {walk_time} min ({max_dist:.0f}m): {len(reachable)} nodes")

        # Convert to polygon
        polygon = nodes_to_polygon(G, reachable)

        if polygon:
            all_isochrones.append({
                'station': station['name'],
                'station_type': station['type'],
                'walk_time': walk_time,
                'distance_m': max_dist,
                'node_count': len(reachable),
                'geometry': polygon,
                'lat': station['lat'],
                'lon': station['lon']
            })

# Create GeoDataFrame
isochrones_gdf = gpd.GeoDataFrame(all_isochrones, crs='EPSG:4326')

# Calculate areas (in UTM for accurate measurements)
isochrones_utm = isochrones_gdf.to_crs('EPSG:32640')
isochrones_gdf['area_sqm'] = isochrones_utm.geometry.area
isochrones_gdf['area_ha'] = isochrones_gdf['area_sqm'] / 10000

print("\n4. ISOCHRONE STATISTICS")
print("-"*40)

for station in TRANSIT_STATIONS:
    print(f"\n   {station['name']}:")
    station_iso = isochrones_gdf[isochrones_gdf['station'] == station['name']]
    for _, row in station_iso.iterrows():
        print(f"      {row['walk_time']:>2} min: {row['area_ha']:.1f} ha ({row['node_count']} nodes)")

# Combined coverage (union of all isochrones by time)
print("\n5. COMBINED COVERAGE")
print("-"*40)

combined_coverage = []
for walk_time in WALK_TIMES_MIN:
    time_polys = isochrones_gdf[isochrones_gdf['walk_time'] == walk_time]['geometry'].tolist()
    if time_polys:
        combined = unary_union(time_polys)
        combined_gdf = gpd.GeoDataFrame([{'geometry': combined}], crs='EPSG:4326')
        combined_utm = combined_gdf.to_crs('EPSG:32640')
        area_ha = combined_utm.geometry.area.iloc[0] / 10000
        print(f"   {walk_time} min walk from ANY transit: {area_ha:.1f} ha")
        combined_coverage.append({
            'walk_time': walk_time,
            'geometry': combined,
            'area_ha': area_ha
        })

# Save results
print("\n6. SAVING RESULTS")
print("-"*40)

isochrones_gdf.to_file(os.path.join(OUT_DIR, 'isochrones.geojson'), driver='GeoJSON')
print(f"   Saved: {OUT_DIR}/isochrones.geojson")

# Create interactive map
print("   Creating interactive map...")

# Prepare data for JS
iso_data = []
for _, row in isochrones_gdf.iterrows():
    geom = row['geometry']
    if geom.geom_type == 'Polygon':
        coords = [[[c[1], c[0]] for c in geom.exterior.coords]]
    elif geom.geom_type == 'MultiPolygon':
        coords = [[[c[1], c[0]] for c in poly.exterior.coords] for poly in geom.geoms]
    else:
        continue

    iso_data.append({
        'station': row['station'],
        'type': row['station_type'],
        'time': row['walk_time'],
        'coords': coords,
        'area': round(row['area_ha'], 1)
    })

station_markers = [{'name': s['name'], 'lat': s['lat'], 'lon': s['lon'], 'type': s['type']} for s in TRANSIT_STATIONS]

html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Isochrone Analysis - Al Karama</title>
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
        .panel h2 {{ margin: 0 0 5px 0; color: #6a1b9a; }}
        .panel h3 {{ margin: 15px 0 8px 0; font-size: 14px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
        .btn {{ padding: 8px 12px; margin: 3px; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; }}
        .btn-active {{ box-shadow: 0 2px 8px rgba(0,0,0,0.3); }}
        .btn-5 {{ background: #c8e6c9; }}
        .btn-10 {{ background: #81c784; color: white; }}
        .btn-15 {{ background: #388e3c; color: white; }}
        .btn-all {{ background: #6a1b9a; color: white; }}
        .btn:not(.btn-active) {{ opacity: 0.6; }}
        .legend {{ margin: 10px 0; }}
        .legend-item {{ display: flex; align-items: center; margin: 6px 0; font-size: 12px; }}
        .legend-color {{ width: 20px; height: 20px; margin-right: 10px; border-radius: 3px; border: 1px solid #999; }}
        .station-list {{ font-size: 12px; }}
        .station-item {{ padding: 6px 0; border-bottom: 1px solid #eee; }}
        .station-item:last-child {{ border-bottom: none; }}
        .metro {{ color: #d32f2f; }}
        .bus {{ color: #1976d2; }}
        .insight {{ background: #f3e5f5; padding: 10px; border-radius: 6px; font-size: 12px; margin: 10px 0; line-height: 1.5; }}
        .btn-basemap {{ background: #424242; color: white; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="panel">
        <h2>Isochrone Analysis</h2>
        <p style="color: #666; font-size: 12px;">Walking distance from transit stations</p>

        <div class="insight">
            <b>What are isochrones?</b><br>
            Areas reachable within X minutes of walking. Shows transit accessibility - where people can easily walk to/from public transport.
        </div>

        <h3>Walk Time</h3>
        <div>
            <button class="btn btn-5" id="btn-5" onclick="showTime(5)">5 min</button>
            <button class="btn btn-10" id="btn-10" onclick="showTime(10)">10 min</button>
            <button class="btn btn-15" id="btn-15" onclick="showTime(15)">15 min</button>
            <button class="btn btn-all btn-active" id="btn-all" onclick="showTime('all')">All</button>
        </div>

        <h3>Legend</h3>
        <div class="legend">
            <div class="legend-item"><div class="legend-color" style="background: rgba(200, 230, 201, 0.6);"></div> 5 min walk</div>
            <div class="legend-item"><div class="legend-color" style="background: rgba(129, 199, 132, 0.6);"></div> 10 min walk</div>
            <div class="legend-item"><div class="legend-color" style="background: rgba(56, 142, 60, 0.6);"></div> 15 min walk</div>
            <div class="legend-item"><div class="legend-color" style="background: #d32f2f; border-radius: 50%;"></div> Metro station</div>
            <div class="legend-item"><div class="legend-color" style="background: #1976d2; border-radius: 50%;"></div> Bus station</div>
        </div>

        <h3>Transit Type</h3>
        <div>
            <button class="btn btn-active" id="btn-metro" style="background: #d32f2f; color: white;" onclick="toggleTransit('metro')">Metro</button>
            <button class="btn btn-active" id="btn-bus" style="background: #1976d2; color: white;" onclick="toggleTransit('bus')">Bus</button>
        </div>

        <h3>Transit Stations</h3>
        <div class="station-list">
            {' '.join([f'<div class="station-item"><span class="{s["type"]}">{s["name"]}</span></div>' for s in TRANSIT_STATIONS])}
        </div>

        <h3>Background</h3>
        <div>
            <button class="btn btn-basemap btn-active" id="btn-basemap" onclick="toggleBasemap()">Basemap</button>
        </div>

        <h3>Insights</h3>
        <div style="font-size: 11px; color: #666; line-height: 1.5;">
            <p>• Areas outside all isochrones have poor transit accessibility</p>
            <p>• Overlap zones are well-served by multiple stations</p>
            <p>• Combine with comfort data to find "cool transit corridors"</p>
        </div>
    </div>

    <script>
        var map = L.map('map').setView([25.245, 55.303], 15);
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

        var isochrones = {json.dumps(iso_data)};
        var stations = {json.dumps(station_markers)};

        var timeColors = {{
            5: 'rgba(200, 230, 201, 0.6)',
            10: 'rgba(129, 199, 132, 0.6)',
            15: 'rgba(56, 142, 60, 0.6)'
        }};

        var timeLayers = {{
            metro: {{ 5: L.layerGroup(), 10: L.layerGroup(), 15: L.layerGroup() }},
            bus: {{ 5: L.layerGroup(), 10: L.layerGroup(), 15: L.layerGroup() }}
        }};
        var stationLayers = {{
            metro: L.layerGroup(),
            bus: L.layerGroup()
        }};
        var transitVisible = {{ metro: true, bus: true }};

        // Add isochrones
        isochrones.forEach(function(iso) {{
            var popup = '<b>' + iso.station + '</b><br>' +
                       iso.time + ' min walk<br>' +
                       'Area: ' + iso.area + ' ha';

            iso.coords.forEach(function(ring) {{
                L.polygon(ring, {{
                    color: iso.time === 5 ? '#a5d6a7' : (iso.time === 10 ? '#66bb6a' : '#2e7d32'),
                    weight: 2,
                    fillColor: timeColors[iso.time],
                    fillOpacity: 0.5
                }}).bindPopup(popup).addTo(timeLayers[iso.type][iso.time]);
            }});
        }});

        // Add station markers
        stations.forEach(function(s) {{
            var color = s.type === 'metro' ? '#d32f2f' : '#1976d2';
            L.circleMarker([s.lat, s.lon], {{
                radius: 10,
                fillColor: color,
                color: 'white',
                weight: 3,
                fillOpacity: 1
            }}).bindPopup('<b>' + s.name + '</b><br>' + s.type.toUpperCase()).addTo(stationLayers[s.type]);
        }});

        // Add all layers initially
        ['metro', 'bus'].forEach(function(type) {{
            [5, 10, 15].forEach(function(t) {{ timeLayers[type][t].addTo(map); }});
            stationLayers[type].addTo(map);
        }});

        var currentTime = 'all';

        function showTime(time) {{
            // Remove all time layers
            ['metro', 'bus'].forEach(function(type) {{
                [5, 10, 15].forEach(function(t) {{ map.removeLayer(timeLayers[type][t]); }});
            }});

            // Update buttons
            ['5', '10', '15', 'all'].forEach(function(t) {{
                document.getElementById('btn-' + t).classList.remove('btn-active');
            }});
            document.getElementById('btn-' + time).classList.add('btn-active');

            // Add selected layers for visible transit types
            ['metro', 'bus'].forEach(function(type) {{
                if (transitVisible[type]) {{
                    if (time === 'all') {{
                        [5, 10, 15].forEach(function(t) {{ timeLayers[type][t].addTo(map); }});
                    }} else {{
                        timeLayers[type][time].addTo(map);
                    }}
                }}
            }});

            currentTime = time;
        }}

        function toggleTransit(type) {{
            var btn = document.getElementById('btn-' + type);
            if (transitVisible[type]) {{
                // Hide this transit type
                [5, 10, 15].forEach(function(t) {{ map.removeLayer(timeLayers[type][t]); }});
                map.removeLayer(stationLayers[type]);
                btn.classList.remove('btn-active');
                btn.style.opacity = '0.4';
            }} else {{
                // Show this transit type
                if (currentTime === 'all') {{
                    [5, 10, 15].forEach(function(t) {{ timeLayers[type][t].addTo(map); }});
                }} else {{
                    timeLayers[type][currentTime].addTo(map);
                }}
                stationLayers[type].addTo(map);
                btn.classList.add('btn-active');
                btn.style.opacity = '1';
            }}
            transitVisible[type] = !transitVisible[type];
        }}
    </script>
</body>
</html>'''

with open(os.path.join(OUT_DIR, 'isochrone_map.html'), 'w') as f:
    f.write(html)
print(f"   Saved: {OUT_DIR}/isochrone_map.html")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"""
ISOCHRONE SUMMARY:

Walk speed: {WALK_SPEED_MPS} m/s ({WALK_SPEED_MPS * 3.6:.1f} km/h)
(Slower than normal to account for Dubai heat)

Stations analyzed:
- 1 Metro station (ADCB)
- 3 Bus stations

Key insight: Areas within 10-min walk of transit are prime
candidates for pedestrian comfort improvements - they get
the most foot traffic to/from public transport.

Open: {OUT_DIR}/isochrone_map.html
""")
