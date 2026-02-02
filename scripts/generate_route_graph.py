#!/usr/bin/env python3
"""Generate a compact route graph JSON from streets.geojson + street_metrics.json.

Output: docs/data/al_karama/route_graph.json
Format:
  nodes: [[lat, lon], ...]
  edges: [[u, v, len, pci, lst, shade, feature_index], ...]
  edge_keys: ["u","v","len","pci","lst","shade","fi"]
  default_pci: float (mean of non-null PCI values)
  stats: {node_count, edge_count, pci_coverage}
"""

import json
import os

BASE = os.path.join(os.path.dirname(__file__), '..', 'docs', 'data', 'al_karama')

# Load data
with open(os.path.join(BASE, 'streets.geojson')) as f:
    streets = json.load(f)

with open(os.path.join(BASE, 'street_metrics.json')) as f:
    metrics = json.load(f)

# Build shade lookup by feature index
shade_by_idx = {}
for i, m in enumerate(metrics):
    shade_val = m.get('shade', '')
    if shade_val != '' and shade_val is not None:
        shade_by_idx[i] = float(shade_val)

# Assign integer node IDs to unique endpoint coordinate tuples
node_map = {}  # (lon, lat) rounded -> node_id
nodes = []     # [[lat, lon], ...]

def get_node_id(coord):
    """Get or create node ID for a coordinate [lon, lat]."""
    # Round to 7 decimal places to match coordinate precision
    key = (round(coord[0], 7), round(coord[1], 7))
    if key not in node_map:
        node_map[key] = len(nodes)
        nodes.append([round(coord[1], 6), round(coord[0], 6)])  # [lat, lon]
    return node_map[key]

edges = []
pci_values = []
features = streets['features']

for fi, feat in enumerate(features):
    coords = feat['geometry']['coordinates']
    if len(coords) < 2:
        continue

    start = coords[0]
    end = coords[-1]
    u = get_node_id(start)
    v = get_node_id(end)

    # Skip self-loops
    if u == v:
        continue

    props = feat['properties']
    length = round(props.get('length', 0), 2)
    pci = props.get('pci')
    lst = props.get('lst')
    shade = shade_by_idx.get(fi)

    if pci is not None:
        pci = round(pci, 4)
        pci_values.append(pci)
    if lst is not None:
        lst = round(lst, 2)
    if shade is not None:
        shade = round(shade, 4)

    edges.append([u, v, length, pci, lst, shade, fi])

# Compute default PCI
default_pci = round(sum(pci_values) / len(pci_values), 4) if pci_values else 0.43
pci_coverage = round(len(pci_values) / len(edges), 3) if edges else 0

result = {
    'nodes': nodes,
    'edges': edges,
    'edge_keys': ['u', 'v', 'len', 'pci', 'lst', 'shade', 'fi'],
    'default_pci': default_pci,
    'stats': {
        'node_count': len(nodes),
        'edge_count': len(edges),
        'pci_coverage': pci_coverage,
        'total_features': len(features),
        'skipped_self_loops': len(features) - len(edges) - sum(1 for f in features if len(f['geometry']['coordinates']) < 2)
    }
}

out_path = os.path.join(BASE, 'route_graph.json')
with open(out_path, 'w') as f:
    json.dump(result, f, separators=(',', ':'))

print(f"Route graph generated: {out_path}")
print(f"  Nodes: {len(nodes)}")
print(f"  Edges: {len(edges)}")
print(f"  PCI coverage: {pci_coverage:.1%}")
print(f"  Default PCI: {default_pci}")
print(f"  File size: {os.path.getsize(out_path) / 1024:.0f} KB")
