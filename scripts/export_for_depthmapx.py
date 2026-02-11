#!/usr/bin/env python3
"""
Export Al Karama Street Network for DepthMapX
==============================================

Exports the street network in formats compatible with DepthMapX:
1. DXF (AutoCAD) - recommended for segment analysis
2. MIF (MapInfo) - alternative format
3. CSV - simple coordinate export

DepthMapX: https://github.com/SpaceGroupUCL/depthmapX
"""

import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

print("="*60)
print("EXPORT FOR DEPTHMAPX")
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
print(f"   Loaded {len(streets_gdf)} street segments")
print(f"   CRS: {streets_gdf.crs}")

# Convert to projected CRS for accurate measurements (UTM Zone 40N for Dubai)
print("\n2. Converting to projected CRS (UTM 40N)...")
streets_utm = streets_gdf.to_crs('EPSG:32640')
print(f"   CRS: {streets_utm.crs}")

# Get bounds for reference
bounds = streets_utm.total_bounds
print(f"   Bounds: {bounds}")
print(f"   Width: {bounds[2]-bounds[0]:.0f}m, Height: {bounds[3]-bounds[1]:.0f}m")

# Export 1: DXF format (recommended for DepthMapX)
print("\n3. Exporting to DXF format...")
try:
    import ezdxf

    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    # Add each street segment as a LINE or POLYLINE
    for idx, row in streets_utm.iterrows():
        geom = row.geometry
        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
            if len(coords) == 2:
                # Simple line
                msp.add_line(coords[0][:2], coords[1][:2])
            else:
                # Polyline for multi-point linestrings
                points = [c[:2] for c in coords]
                msp.add_lwpolyline(points)
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                coords = list(line.coords)
                if len(coords) == 2:
                    msp.add_line(coords[0][:2], coords[1][:2])
                else:
                    points = [c[:2] for c in coords]
                    msp.add_lwpolyline(points)

    dxf_path = os.path.join(OUT_DIR, 'al_karama_streets.dxf')
    doc.saveas(dxf_path)
    print(f"   ✓ Saved: {dxf_path}")

except ImportError:
    print("   ⚠ ezdxf not installed. Install with: pip install ezdxf")
    print("   Skipping DXF export...")

# Export 2: MIF/MID format (MapInfo)
print("\n4. Exporting to MIF format...")
try:
    mif_path = os.path.join(OUT_DIR, 'al_karama_streets.mif')
    streets_utm.to_file(mif_path, driver='MapInfo File')
    print(f"   ✓ Saved: {mif_path}")
except Exception as e:
    print(f"   ⚠ MIF export failed: {e}")

# Export 3: CSV with WKT geometry
print("\n5. Exporting to CSV (with WKT)...")
csv_data = []
for idx, row in streets_utm.iterrows():
    geom = row.geometry
    if geom.geom_type == 'LineString':
        coords = list(geom.coords)
        csv_data.append({
            'id': idx,
            'start_x': coords[0][0],
            'start_y': coords[0][1],
            'end_x': coords[-1][0],
            'end_y': coords[-1][1],
            'length': geom.length,
            'wkt': geom.wkt
        })

csv_df = pd.DataFrame(csv_data)
csv_path = os.path.join(OUT_DIR, 'al_karama_streets.csv')
csv_df.to_csv(csv_path, index=False)
print(f"   ✓ Saved: {csv_path}")

# Export 4: Simple line list (start-end coordinates only)
print("\n6. Exporting simple line list...")
lines_data = []
for idx, row in streets_utm.iterrows():
    geom = row.geometry
    if geom.geom_type == 'LineString':
        coords = list(geom.coords)
        # For each segment in the linestring
        for i in range(len(coords) - 1):
            lines_data.append({
                'x1': coords[i][0],
                'y1': coords[i][1],
                'x2': coords[i+1][0],
                'y2': coords[i+1][1]
            })
    elif geom.geom_type == 'MultiLineString':
        for line in geom.geoms:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                lines_data.append({
                    'x1': coords[i][0],
                    'y1': coords[i][1],
                    'x2': coords[i+1][0],
                    'y2': coords[i+1][1]
                })

lines_df = pd.DataFrame(lines_data)
lines_path = os.path.join(OUT_DIR, 'al_karama_lines.csv')
lines_df.to_csv(lines_path, index=False)
print(f"   ✓ Saved: {lines_path}")
print(f"   Total line segments: {len(lines_df)}")

# Export 5: GeoJSON (for reference/verification)
print("\n7. Exporting GeoJSON (for verification)...")
geojson_path = os.path.join(OUT_DIR, 'al_karama_streets_utm.geojson')
streets_utm.to_file(geojson_path, driver='GeoJSON')
print(f"   ✓ Saved: {geojson_path}")

print("\n" + "="*60)
print("EXPORT COMPLETE")
print("="*60)
print(f"""
Files saved to: {OUT_DIR}/

For DepthMapX:
1. al_karama_streets.dxf  - RECOMMENDED: Import directly into DepthMapX
2. al_karama_streets.mif  - Alternative MapInfo format
3. al_karama_streets.csv  - CSV with WKT geometry
4. al_karama_lines.csv    - Simple x1,y1,x2,y2 format

DepthMapX Import Instructions:
1. Open DepthMapX
2. File > Import > Choose .dxf file
3. Select "Segment Map" or "Axial Map" analysis type
4. Run analysis (Tools > Segment Analysis or Axial Analysis)

Note: Coordinates are in UTM Zone 40N (EPSG:32640) - meters
Bounds: {bounds[0]:.0f}, {bounds[1]:.0f} to {bounds[2]:.0f}, {bounds[3]:.0f}
""")
