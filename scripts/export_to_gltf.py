#!/usr/bin/env python3
"""Export Al Karama 3D model to glTF/glB format for Autodesk Forma and Rhino 8+.

Reads buildings.geojson and canopy.geojson, extrudes polygons by height,
and exports a combined 3D model.

Output: output/3d_export/al_karama.glb (binary glTF)

Usage:
    python scripts/export_to_gltf.py
    python scripts/export_to_gltf.py --buildings-only
    python scripts/export_to_gltf.py --canopy-only
"""

import argparse
import json
import os
import numpy as np
import trimesh
from shapely.geometry import shape, Polygon
from shapely.ops import triangulate
from scipy.spatial import Delaunay

# Paths
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'docs', 'data', 'al_karama')
OUTPUT_DIR = os.path.join(BASE, 'output', '3d_export')

# Reference point for local coordinates (Al Karama center)
REF_LON, REF_LAT = 55.3025, 25.2485

# Approximate meters per degree at this latitude
M_PER_DEG_LAT = 111320
M_PER_DEG_LON = 111320 * np.cos(np.radians(REF_LAT))


def lonlat_to_meters(lon, lat):
    """Convert lon/lat to local meters from reference point."""
    x = (lon - REF_LON) * M_PER_DEG_LON
    y = (lat - REF_LAT) * M_PER_DEG_LAT
    return x, y


def extrude_polygon(coords, height, base_height=0):
    """Extrude a 2D polygon to 3D mesh.

    Args:
        coords: List of [lon, lat] coordinates (exterior ring)
        height: Extrusion height in meters
        base_height: Base elevation (default 0)

    Returns:
        trimesh.Trimesh object
    """
    # Convert to local meters
    points_2d = np.array([lonlat_to_meters(c[0], c[1]) for c in coords])

    # Remove duplicate last point if present
    if np.allclose(points_2d[0], points_2d[-1]):
        points_2d = points_2d[:-1]

    n = len(points_2d)
    if n < 3:
        return None

    # Create vertices for bottom and top faces
    bottom_verts = np.column_stack([points_2d, np.full(n, base_height)])
    top_verts = np.column_stack([points_2d, np.full(n, base_height + height)])
    vertices = np.vstack([bottom_verts, top_verts])

    faces = []

    # Side faces (quads split into triangles)
    for i in range(n):
        j = (i + 1) % n
        # Bottom-left, bottom-right, top-right, top-left
        bl, br = i, j
        tl, tr = i + n, j + n
        # Two triangles per quad
        faces.append([bl, br, tr])
        faces.append([bl, tr, tl])

    # Top and bottom faces using ear clipping triangulation
    try:
        # Use Shapely for triangulation
        poly = Polygon(points_2d)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < 0.1:
            return None

        # Simple fan triangulation for convex-ish polygons
        # For complex polygons, use constrained Delaunay
        centroid = poly.centroid
        cx, cy = centroid.x, centroid.y

        # Add centroid as extra vertex for fan triangulation
        centroid_bottom = len(vertices)
        centroid_top = centroid_bottom + 1
        vertices = np.vstack([
            vertices,
            [[cx, cy, base_height]],
            [[cx, cy, base_height + height]]
        ])

        # Bottom face (fan from centroid, reversed winding)
        for i in range(n):
            j = (i + 1) % n
            faces.append([centroid_bottom, j, i])

        # Top face (fan from centroid)
        for i in range(n):
            j = (i + 1) % n
            faces.append([centroid_top, i + n, j + n])

    except Exception as e:
        # Fallback: skip top/bottom faces
        pass

    faces = np.array(faces)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Fix normals
    mesh.fix_normals()

    return mesh


def load_and_extrude_geojson(filepath, height_key='height', color=None, name_prefix=''):
    """Load GeoJSON and extrude all polygons.

    Args:
        filepath: Path to GeoJSON file
        height_key: Property key for height value
        color: Optional RGBA color [r, g, b, a] (0-255)
        name_prefix: Prefix for mesh names

    Returns:
        List of (mesh, name) tuples
    """
    with open(filepath) as f:
        data = json.load(f)

    meshes = []
    for i, feat in enumerate(data['features']):
        geom = feat['geometry']
        props = feat['properties']
        height = props.get(height_key, 10)

        if height is None or height <= 0:
            height = 3  # Default height

        # Get polygon coordinates
        if geom['type'] == 'Polygon':
            coords_list = [geom['coordinates'][0]]  # Exterior ring only
        elif geom['type'] == 'MultiPolygon':
            coords_list = [poly[0] for poly in geom['coordinates']]
        else:
            continue

        for coords in coords_list:
            mesh = extrude_polygon(coords, height)
            if mesh is not None:
                # Set color if provided
                if color is not None:
                    mesh.visual.face_colors = color

                # Generate name
                name = props.get('name') or props.get('id') or f'{name_prefix}{i}'
                meshes.append((mesh, str(name)))

    return meshes


def main():
    parser = argparse.ArgumentParser(description='Export Al Karama to glTF/glB')
    parser.add_argument('--buildings-only', action='store_true', help='Export only buildings')
    parser.add_argument('--canopy-only', action='store_true', help='Export only canopy')
    parser.add_argument('--output', '-o', default=None, help='Output file path')
    parser.add_argument('--format', choices=['glb', 'gltf'], default='glb', help='Output format')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_meshes = []

    # Load buildings
    if not args.canopy_only:
        buildings_path = os.path.join(DATA_DIR, 'buildings.geojson')
        if os.path.exists(buildings_path):
            print(f'Loading buildings from {buildings_path}...')
            building_meshes = load_and_extrude_geojson(
                buildings_path,
                height_key='height',
                color=[200, 200, 200, 255],  # Light gray
                name_prefix='building_'
            )
            print(f'  Extruded {len(building_meshes)} buildings')
            all_meshes.extend(building_meshes)
        else:
            print(f'Warning: {buildings_path} not found')

    # Load canopy
    if not args.buildings_only:
        canopy_path = os.path.join(DATA_DIR, 'canopy.geojson')
        if os.path.exists(canopy_path):
            print(f'Loading canopy from {canopy_path}...')
            canopy_meshes = load_and_extrude_geojson(
                canopy_path,
                height_key='canopy_height_m',
                color=[34, 139, 34, 200],  # Forest green, slightly transparent
                name_prefix='tree_'
            )
            print(f'  Extruded {len(canopy_meshes)} tree canopies')
            all_meshes.extend(canopy_meshes)
        else:
            print(f'Warning: {canopy_path} not found')

    if not all_meshes:
        print('Error: No meshes to export')
        return

    # Combine all meshes
    print(f'Combining {len(all_meshes)} meshes...')
    combined = trimesh.util.concatenate([m[0] for m in all_meshes])

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        suffix = ''
        if args.buildings_only:
            suffix = '_buildings'
        elif args.canopy_only:
            suffix = '_canopy'
        output_path = os.path.join(OUTPUT_DIR, f'al_karama{suffix}.{args.format}')

    # Export
    print(f'Exporting to {output_path}...')
    combined.export(output_path, file_type=args.format)

    # Stats
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f'\nExport complete!')
    print(f'  File: {output_path}')
    print(f'  Size: {file_size:.1f} MB')
    print(f'  Vertices: {len(combined.vertices):,}')
    print(f'  Faces: {len(combined.faces):,}')
    print(f'\nCoordinate system: Local meters from ({REF_LAT}°N, {REF_LON}°E)')
    print(f'  X = East (+) / West (-)')
    print(f'  Y = North (+) / South (-)')
    print(f'  Z = Height (meters)')


if __name__ == '__main__':
    main()
