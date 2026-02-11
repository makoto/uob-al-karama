#!/usr/bin/env python3
"""
Export Axial Map for DepthMapX
==============================

Creates a simplified axial map from the street network by:
1. Joining collinear street segments into longer axial lines
2. Removing very short segments
3. Exporting to DXF format

Axial maps represent the longest lines of sight/movement through urban space.
"""

import os
import math
import numpy as np
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point
from shapely.ops import linemerge
from collections import defaultdict

print("="*60)
print("AXIAL MAP EXPORT FOR DEPTHMAPX")
print("Al Karama, Dubai")
print("="*60)

# Paths
BASE = os.path.dirname(__file__)
STREETS_PATH = os.path.join(BASE, '..', 'docs', 'data', 'al_karama', 'streets.geojson')
OUT_DIR = os.path.join(BASE, '..', 'output', 'depthmapx_export')
os.makedirs(OUT_DIR, exist_ok=True)

# Load streets
print("\n1. Loading street network...")
streets_gdf = gpd.read_file(STREETS_PATH)
streets_utm = streets_gdf.to_crs('EPSG:32640')
print(f"   Loaded {len(streets_utm)} street segments")

# Parameters
MIN_LENGTH = 10  # Minimum axial line length (meters)
ANGLE_TOLERANCE = 15  # Degrees - segments within this angle are considered collinear

def calculate_bearing(p1, p2):
    """Calculate bearing between two points in degrees (0-360)."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    bearing = math.degrees(math.atan2(dx, dy))
    return (bearing + 360) % 360

def normalize_bearing(bearing):
    """Normalize bearing to 0-180 range (direction-agnostic)."""
    if bearing > 180:
        bearing = bearing - 180
    return bearing

def bearings_similar(b1, b2, tolerance):
    """Check if two bearings are within tolerance (direction-agnostic)."""
    b1 = normalize_bearing(b1)
    b2 = normalize_bearing(b2)
    diff = abs(b1 - b2)
    if diff > 90:
        diff = 180 - diff
    return diff <= tolerance

print("\n2. Building network graph...")
# Create graph from street segments
G = nx.Graph()
node_id = 0
coord_to_node = {}
edge_bearings = {}

for idx, row in streets_utm.iterrows():
    geom = row.geometry
    if geom.geom_type == 'LineString':
        coords = list(geom.coords)
        # Add nodes for endpoints
        for coord in [coords[0], coords[-1]]:
            key = (round(coord[0], 1), round(coord[1], 1))
            if key not in coord_to_node:
                coord_to_node[key] = node_id
                G.add_node(node_id, pos=coord[:2])
                node_id += 1

        # Add edge
        start_key = (round(coords[0][0], 1), round(coords[0][1], 1))
        end_key = (round(coords[-1][0], 1), round(coords[-1][1], 1))
        start_node = coord_to_node[start_key]
        end_node = coord_to_node[end_key]

        if start_node != end_node:
            bearing = calculate_bearing(coords[0][:2], coords[-1][:2])
            G.add_edge(start_node, end_node, geometry=geom, bearing=bearing, idx=idx)
            edge_bearings[(start_node, end_node)] = bearing
            edge_bearings[(end_node, start_node)] = (bearing + 180) % 360

print(f"   Nodes: {G.number_of_nodes()}")
print(f"   Edges: {G.number_of_edges()}")

print("\n3. Creating axial lines by joining collinear segments...")

# Find chains of collinear edges
visited_edges = set()
axial_lines = []

def get_edge_key(u, v):
    return (min(u, v), max(u, v))

def extend_chain(start_node, end_node, bearing, chain_coords):
    """Recursively extend chain in one direction."""
    # Look for collinear continuation from end_node
    for neighbor in G.neighbors(end_node):
        edge_key = get_edge_key(end_node, neighbor)
        if edge_key in visited_edges:
            continue
        if neighbor == start_node:
            continue

        # Check if bearing is similar
        edge_data = G.edges[end_node, neighbor]
        neighbor_bearing = edge_bearings.get((end_node, neighbor), 0)

        if bearings_similar(bearing, neighbor_bearing, ANGLE_TOLERANCE):
            visited_edges.add(edge_key)
            neighbor_pos = G.nodes[neighbor]['pos']
            chain_coords.append(neighbor_pos)
            extend_chain(end_node, neighbor, bearing, chain_coords)
            return

# Process each edge
for u, v, data in G.edges(data=True):
    edge_key = get_edge_key(u, v)
    if edge_key in visited_edges:
        continue

    visited_edges.add(edge_key)
    bearing = data['bearing']

    # Start chain with this edge
    u_pos = G.nodes[u]['pos']
    v_pos = G.nodes[v]['pos']

    # Extend in both directions
    chain_backward = [u_pos]
    chain_forward = [v_pos]

    # Extend backward from u
    for neighbor in G.neighbors(u):
        neighbor_key = get_edge_key(u, neighbor)
        if neighbor_key in visited_edges:
            continue
        if neighbor == v:
            continue
        neighbor_bearing = edge_bearings.get((u, neighbor), 0)
        if bearings_similar(bearing, neighbor_bearing, ANGLE_TOLERANCE):
            visited_edges.add(neighbor_key)
            neighbor_pos = G.nodes[neighbor]['pos']
            chain_backward.insert(0, neighbor_pos)
            # Continue extending
            extend_chain(u, neighbor, bearing, chain_backward)
            chain_backward.reverse()
            break

    # Extend forward from v
    extend_chain(u, v, bearing, chain_forward)

    # Combine into single chain
    full_chain = chain_backward + chain_forward

    if len(full_chain) >= 2:
        line = LineString(full_chain)
        if line.length >= MIN_LENGTH:
            axial_lines.append(line)

print(f"   Created {len(axial_lines)} axial lines")

# Also add remaining unvisited edges as short axial lines
print("\n4. Adding remaining segments...")
for u, v, data in G.edges(data=True):
    edge_key = get_edge_key(u, v)
    # Check both directions since we might have missed some
    geom = data.get('geometry')
    if geom and geom.length >= MIN_LENGTH:
        # Check if this geometry overlaps significantly with existing axial lines
        already_covered = False
        for ax_line in axial_lines:
            if geom.distance(ax_line) < 5:  # Within 5 meters
                already_covered = True
                break
        if not already_covered:
            axial_lines.append(geom)

print(f"   Total axial lines: {len(axial_lines)}")

# Remove duplicates and very similar lines
print("\n5. Removing duplicate/similar lines...")
unique_lines = []
for line in axial_lines:
    is_duplicate = False
    for existing in unique_lines:
        # Check if centroids are close and bearings are similar
        if line.centroid.distance(existing.centroid) < 20:
            c1 = list(line.coords)
            c2 = list(existing.coords)
            b1 = normalize_bearing(calculate_bearing(c1[0], c1[-1]))
            b2 = normalize_bearing(calculate_bearing(c2[0], c2[-1]))
            if abs(b1 - b2) < ANGLE_TOLERANCE or abs(b1 - b2) > (180 - ANGLE_TOLERANCE):
                is_duplicate = True
                break
    if not is_duplicate:
        unique_lines.append(line)

axial_lines = unique_lines
print(f"   Unique axial lines: {len(axial_lines)}")

# Statistics
lengths = [line.length for line in axial_lines]
print(f"\n   Length statistics:")
print(f"      Min: {min(lengths):.1f}m")
print(f"      Max: {max(lengths):.1f}m")
print(f"      Mean: {np.mean(lengths):.1f}m")
print(f"      Median: {np.median(lengths):.1f}m")

# Create GeoDataFrame
axial_gdf = gpd.GeoDataFrame(
    {'id': range(len(axial_lines)), 'length': lengths},
    geometry=axial_lines,
    crs='EPSG:32640'
)

# Export to DXF
print("\n6. Exporting to DXF...")
import ezdxf

doc = ezdxf.new('R2010')
msp = doc.modelspace()

for line in axial_lines:
    coords = list(line.coords)
    if len(coords) == 2:
        msp.add_line(coords[0][:2], coords[1][:2])
    else:
        points = [c[:2] for c in coords]
        msp.add_lwpolyline(points)

dxf_path = os.path.join(OUT_DIR, 'al_karama_axial.dxf')
doc.saveas(dxf_path)
print(f"   ✓ Saved: {dxf_path}")

# Export to GeoJSON for verification
print("\n7. Exporting to GeoJSON...")
geojson_path = os.path.join(OUT_DIR, 'al_karama_axial.geojson')
axial_gdf.to_file(geojson_path, driver='GeoJSON')
print(f"   ✓ Saved: {geojson_path}")

# Export to CSV
print("\n8. Exporting to CSV...")
csv_data = []
for idx, line in enumerate(axial_lines):
    coords = list(line.coords)
    csv_data.append({
        'id': idx,
        'x1': coords[0][0],
        'y1': coords[0][1],
        'x2': coords[-1][0],
        'y2': coords[-1][1],
        'length': line.length
    })

import pandas as pd
csv_df = pd.DataFrame(csv_data)
csv_path = os.path.join(OUT_DIR, 'al_karama_axial.csv')
csv_df.to_csv(csv_path, index=False)
print(f"   ✓ Saved: {csv_path}")

print("\n" + "="*60)
print("AXIAL MAP EXPORT COMPLETE")
print("="*60)
print(f"""
Files saved to: {OUT_DIR}/

Axial Map Files:
1. al_karama_axial.dxf     - RECOMMENDED for DepthMapX axial analysis
2. al_karama_axial.geojson - For verification in GIS
3. al_karama_axial.csv     - Simple coordinate format

Segment Map Files (from previous export):
1. al_karama_streets.dxf   - For DepthMapX segment analysis

Comparison:
- Segment map: {len(streets_utm)} segments (detailed)
- Axial map:   {len(axial_lines)} lines (simplified, longer lines)

DepthMapX Usage:
- Use axial.dxf for AXIAL ANALYSIS (traditional space syntax)
- Use streets.dxf for SEGMENT ANALYSIS (more detailed)
""")
