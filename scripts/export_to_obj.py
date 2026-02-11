#!/usr/bin/env python3
"""Export Al Karama 3D model to OBJ format for Rhino and other 3D software.

Reads buildings.geojson and canopy.geojson, extrudes polygons by height,
and exports a combined 3D model with material file.

Output: output/3d_export/al_karama.obj (+ .mtl material file)

Usage:
    python scripts/export_to_obj.py
    python scripts/export_to_obj.py --buildings-only
    python scripts/export_to_obj.py --canopy-only
    python scripts/export_to_obj.py --separate  # Export buildings and canopy as separate files
"""

import argparse
import json
import os
import numpy as np
from shapely.geometry import Polygon

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


class OBJWriter:
    """Simple OBJ file writer with material support."""

    def __init__(self):
        self.vertices = []
        self.faces = []  # List of (face_verts, material_name)
        self.materials = {}  # name -> (r, g, b)
        self.groups = []  # List of (group_name, start_face_idx)

    def add_material(self, name, r, g, b):
        """Add a material (RGB values 0-1)."""
        self.materials[name] = (r, g, b)

    def start_group(self, name):
        """Start a new object group."""
        self.groups.append((name, len(self.faces)))

    def add_mesh(self, vertices, faces, material=None):
        """Add a mesh to the scene.

        Args:
            vertices: Nx3 array of vertex positions
            faces: Mx3 array of face indices (0-based)
            material: Material name to use
        """
        offset = len(self.vertices)
        self.vertices.extend(vertices.tolist())
        for face in faces:
            self.faces.append((
                [f + offset + 1 for f in face],  # OBJ is 1-indexed
                material
            ))

    def write(self, obj_path):
        """Write OBJ and MTL files."""
        mtl_path = obj_path.rsplit('.', 1)[0] + '.mtl'
        mtl_name = os.path.basename(mtl_path)

        # Write MTL file
        with open(mtl_path, 'w') as f:
            f.write('# Al Karama Materials\n')
            for name, (r, g, b) in self.materials.items():
                f.write(f'\nnewmtl {name}\n')
                f.write(f'Kd {r:.3f} {g:.3f} {b:.3f}\n')
                f.write(f'Ka {r*0.2:.3f} {g*0.2:.3f} {b*0.2:.3f}\n')
                f.write('Ks 0.1 0.1 0.1\n')
                f.write('Ns 10\n')
                f.write('d 1.0\n')

        # Write OBJ file
        with open(obj_path, 'w') as f:
            f.write('# Al Karama Urban Digital Twin\n')
            f.write(f'# Exported from GeoJSON data\n')
            f.write(f'# Reference: {REF_LAT}N, {REF_LON}E\n')
            f.write(f'# Units: meters\n\n')
            f.write(f'mtllib {mtl_name}\n\n')

            # Write vertices
            f.write(f'# {len(self.vertices)} vertices\n')
            for v in self.vertices:
                f.write(f'v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n')
            f.write('\n')

            # Write faces with groups and materials
            current_material = None
            group_idx = 0
            next_group_start = self.groups[0][1] if self.groups else len(self.faces) + 1

            f.write(f'# {len(self.faces)} faces\n')
            for i, (face_verts, material) in enumerate(self.faces):
                # Check for new group
                if group_idx < len(self.groups) and i >= next_group_start:
                    group_name, _ = self.groups[group_idx]
                    f.write(f'\ng {group_name}\n')
                    group_idx += 1
                    next_group_start = self.groups[group_idx][1] if group_idx < len(self.groups) else len(self.faces) + 1

                # Check for material change
                if material != current_material:
                    if material:
                        f.write(f'usemtl {material}\n')
                    current_material = material

                # Write face
                f.write(f'f {" ".join(str(v) for v in face_verts)}\n')

        return obj_path, mtl_path


def extrude_polygon_to_obj(coords, height, base_height=0):
    """Extrude a 2D polygon and return vertices and faces.

    Returns:
        (vertices, faces) or (None, None) if invalid
    """
    # Convert to local meters
    points_2d = np.array([lonlat_to_meters(c[0], c[1]) for c in coords])

    # Remove duplicate last point if present
    if np.allclose(points_2d[0], points_2d[-1]):
        points_2d = points_2d[:-1]

    n = len(points_2d)
    if n < 3:
        return None, None

    # Create vertices for bottom and top faces
    bottom_verts = np.column_stack([points_2d, np.full(n, base_height)])
    top_verts = np.column_stack([points_2d, np.full(n, base_height + height)])
    vertices = np.vstack([bottom_verts, top_verts])

    faces = []

    # Side faces (quads split into triangles)
    for i in range(n):
        j = (i + 1) % n
        bl, br = i, j
        tl, tr = i + n, j + n
        faces.append([bl, br, tr])
        faces.append([bl, tr, tl])

    # Top and bottom faces using fan triangulation
    try:
        poly = Polygon(points_2d)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < 0.1:
            return None, None

        centroid = poly.centroid
        cx, cy = centroid.x, centroid.y

        # Add centroid vertices
        centroid_bottom = len(vertices)
        centroid_top = centroid_bottom + 1
        vertices = np.vstack([
            vertices,
            [[cx, cy, base_height]],
            [[cx, cy, base_height + height]]
        ])

        # Bottom face (reversed winding for correct normals)
        for i in range(n):
            j = (i + 1) % n
            faces.append([centroid_bottom, j, i])

        # Top face
        for i in range(n):
            j = (i + 1) % n
            faces.append([centroid_top, i + n, j + n])

    except Exception:
        pass

    return vertices, np.array(faces)


def load_and_extrude_geojson(writer, filepath, height_key='height', material=None, name_prefix=''):
    """Load GeoJSON and add extruded meshes to OBJ writer."""
    with open(filepath) as f:
        data = json.load(f)

    count = 0
    for i, feat in enumerate(data['features']):
        geom = feat['geometry']
        props = feat['properties']
        height = props.get(height_key, 10)

        if height is None or height <= 0:
            height = 3

        # Get polygon coordinates
        if geom['type'] == 'Polygon':
            coords_list = [geom['coordinates'][0]]
        elif geom['type'] == 'MultiPolygon':
            coords_list = [poly[0] for poly in geom['coordinates']]
        else:
            continue

        # Generate group name
        name = props.get('name') or props.get('id') or f'{name_prefix}{i}'
        name = str(name).replace(' ', '_').replace('/', '_')

        for j, coords in enumerate(coords_list):
            vertices, faces = extrude_polygon_to_obj(coords, height)
            if vertices is not None:
                if len(coords_list) > 1:
                    writer.start_group(f'{name}_{j}')
                else:
                    writer.start_group(name)
                writer.add_mesh(vertices, faces, material)
                count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description='Export Al Karama to OBJ format')
    parser.add_argument('--buildings-only', action='store_true', help='Export only buildings')
    parser.add_argument('--canopy-only', action='store_true', help='Export only canopy')
    parser.add_argument('--separate', action='store_true', help='Export buildings and canopy as separate files')
    parser.add_argument('--output', '-o', default=None, help='Output file path')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.separate and not (args.buildings_only or args.canopy_only):
        # Export both as separate files
        for layer in ['buildings', 'canopy']:
            writer = OBJWriter()
            if layer == 'buildings':
                writer.add_material('building', 0.78, 0.78, 0.78)
                path = os.path.join(DATA_DIR, 'buildings.geojson')
                if os.path.exists(path):
                    print(f'Exporting buildings...')
                    count = load_and_extrude_geojson(writer, path, 'height', 'building', 'building_')
                    print(f'  Extruded {count} buildings')
            else:
                writer.add_material('canopy', 0.13, 0.55, 0.13)
                path = os.path.join(DATA_DIR, 'canopy.geojson')
                if os.path.exists(path):
                    print(f'Exporting canopy...')
                    count = load_and_extrude_geojson(writer, path, 'canopy_height_m', 'canopy', 'tree_')
                    print(f'  Extruded {count} tree canopies')

            output_path = os.path.join(OUTPUT_DIR, f'al_karama_{layer}.obj')
            obj_path, mtl_path = writer.write(output_path)
            file_size = os.path.getsize(obj_path) / (1024 * 1024)
            print(f'  Output: {obj_path} ({file_size:.1f} MB)')
        return

    # Combined export
    writer = OBJWriter()
    writer.add_material('building', 0.78, 0.78, 0.78)  # Light gray
    writer.add_material('canopy', 0.13, 0.55, 0.13)    # Forest green

    total_count = 0

    # Load buildings
    if not args.canopy_only:
        buildings_path = os.path.join(DATA_DIR, 'buildings.geojson')
        if os.path.exists(buildings_path):
            print(f'Loading buildings from {buildings_path}...')
            count = load_and_extrude_geojson(writer, buildings_path, 'height', 'building', 'building_')
            print(f'  Extruded {count} buildings')
            total_count += count
        else:
            print(f'Warning: {buildings_path} not found')

    # Load canopy
    if not args.buildings_only:
        canopy_path = os.path.join(DATA_DIR, 'canopy.geojson')
        if os.path.exists(canopy_path):
            print(f'Loading canopy from {canopy_path}...')
            count = load_and_extrude_geojson(writer, canopy_path, 'canopy_height_m', 'canopy', 'tree_')
            print(f'  Extruded {count} tree canopies')
            total_count += count
        else:
            print(f'Warning: {canopy_path} not found')

    if total_count == 0:
        print('Error: No meshes to export')
        return

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        suffix = ''
        if args.buildings_only:
            suffix = '_buildings'
        elif args.canopy_only:
            suffix = '_canopy'
        output_path = os.path.join(OUTPUT_DIR, f'al_karama{suffix}.obj')

    # Export
    print(f'Writing OBJ file...')
    obj_path, mtl_path = writer.write(output_path)

    # Stats
    obj_size = os.path.getsize(obj_path) / (1024 * 1024)
    mtl_size = os.path.getsize(mtl_path) / 1024

    print(f'\nExport complete!')
    print(f'  OBJ: {obj_path} ({obj_size:.1f} MB)')
    print(f'  MTL: {mtl_path} ({mtl_size:.1f} KB)')
    print(f'  Objects: {total_count}')
    print(f'  Vertices: {len(writer.vertices):,}')
    print(f'  Faces: {len(writer.faces):,}')
    print(f'\nCoordinate system: Local meters from ({REF_LAT}°N, {REF_LON}°E)')
    print(f'  X = East (+) / West (-)')
    print(f'  Y = North (+) / South (-)')
    print(f'  Z = Height (meters)')
    print(f'\nTo import in Rhino:')
    print(f'  1. File > Import > select {os.path.basename(obj_path)}')
    print(f'  2. Units: Meters')
    print(f'  3. The model is centered at the Al Karama reference point')


if __name__ == '__main__':
    main()
