"""
PoC: Estimate building heights from depth maps + segmentation.
Uses the 26 test images to demonstrate the approach.
"""

import os
import glob
import numpy as np
from PIL import Image
import pandas as pd
import json
from collections import defaultdict

# Directories
depth_dir = "data/depth_test/depth"
input_dir = "data/depth_test/images"
seg_dir = "data/segmentation/images"
output_dir = "output/height_estimation"
os.makedirs(output_dir, exist_ok=True)

# Load test area images
test_images = pd.read_csv("output/test_area_images.csv")
print(f"Test images: {len(test_images)}")

# Mapillary segmentation color for Building (approximate - may need adjustment)
# From Mapillary Vistas dataset: Building is typically a brownish color
# Let's detect it by analyzing the segmentation images
BUILDING_COLORS = [
    (70, 70, 70),      # Cityscapes building color
    (102, 102, 156),   # Alternative
    (128, 64, 128),    # Another variant
]

def find_building_mask(seg_image_path, tolerance=30):
    """
    Find building pixels in segmentation image.
    Buildings are typically gray/brown colored in Mapillary/Cityscapes.
    """
    img = Image.open(seg_image_path)
    arr = np.array(img)

    # Try to find building-like colors (grays/browns in upper portion of image)
    # Buildings typically appear in upper-middle part of panorama
    h, w = arr.shape[:2]

    # Create mask for potential building pixels
    # Look for grayish colors (R≈G≈B, medium intensity)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

    # Building detection heuristics:
    # 1. Gray-ish colors (low saturation)
    # 2. Medium to dark intensity
    # 3. Not sky (sky is usually bright blue or white)
    # 4. Not road (road is usually darker, at bottom)

    gray_diff = np.abs(r.astype(int) - g.astype(int)) + np.abs(g.astype(int) - b.astype(int))
    is_grayish = gray_diff < 60

    intensity = (r.astype(int) + g.astype(int) + b.astype(int)) / 3
    is_medium = (intensity > 50) & (intensity < 200)

    # Not in bottom 20% (likely road) or top 10% (likely sky in equirectangular)
    row_mask = np.zeros_like(r, dtype=bool)
    row_mask[int(h*0.1):int(h*0.8), :] = True

    # Combine heuristics
    building_mask = is_grayish & is_medium & row_mask

    # Also check for specific building colors
    for color in BUILDING_COLORS:
        color_match = (
            (np.abs(r.astype(int) - color[0]) < tolerance) &
            (np.abs(g.astype(int) - color[1]) < tolerance) &
            (np.abs(b.astype(int) - color[2]) < tolerance)
        )
        building_mask = building_mask | color_match

    return building_mask

def estimate_height_from_depth(depth_path, building_mask, camera_height=2.0):
    """
    Estimate building height from depth map and building mask.

    Approach:
    - Find building pixels at different vertical positions
    - Use depth gradient to estimate height
    - Calibrate assuming camera at ~2m height
    """
    depth_img = Image.open(depth_path)
    depth = np.array(depth_img, dtype=np.float32)

    h, w = depth.shape

    # Normalize depth (model outputs inverse depth - closer = higher value)
    depth_norm = depth.copy()
    depth_norm = (depth_norm - depth_norm.min()) / (depth_norm.max() - depth_norm.min() + 1e-6)

    # For each column, find building pixels and their vertical extent
    heights = []

    for col in range(0, w, 10):  # Sample every 10th column
        col_mask = building_mask[:, col]
        col_depth = depth_norm[:, col]

        building_rows = np.where(col_mask)[0]
        if len(building_rows) < 10:
            continue

        # Find top and bottom of building in this column
        top_row = building_rows.min()
        bottom_row = building_rows.max()

        # Get depth at building (use median for robustness)
        building_depth = np.median(col_depth[col_mask])

        if building_depth < 0.1:  # Too close, skip
            continue

        # Estimate height using vertical angle in equirectangular projection
        # In equirectangular: row maps to elevation angle
        # Row 0 = +90° (zenith), Row h/2 = 0° (horizon), Row h = -90° (nadir)

        # Angle to top of building
        top_angle = (0.5 - top_row / h) * np.pi  # radians from horizon
        bottom_angle = (0.5 - bottom_row / h) * np.pi

        # Relative distance (inverse of normalized depth)
        rel_distance = 1.0 / (building_depth + 0.1)

        # Height estimate using trigonometry
        # Assuming camera at horizon level
        if top_angle > 0:  # Building extends above horizon
            # Height above camera = distance * tan(angle)
            height_above = rel_distance * np.tan(top_angle) * 10  # Scale factor

            # Total building height (assuming we see from ground to top)
            est_height = height_above + camera_height

            if 3 < est_height < 200:  # Reasonable building height
                heights.append(est_height)

    if heights:
        return np.median(heights)
    return None

def match_to_osm_building(lat, lon, osm_buildings, max_distance=30):
    """Find nearest OSM building to this viewpoint."""
    best_match = None
    best_distance = max_distance

    for building in osm_buildings:
        # Get building centroid
        coords = building['coordinates']
        if not coords:
            continue

        b_lon = np.mean([c[0] for c in coords])
        b_lat = np.mean([c[1] for c in coords])

        # Simple distance calculation
        dlat = (lat - b_lat) * 111000  # meters
        dlon = (lon - b_lon) * 111000 * np.cos(np.radians(lat))
        dist = np.sqrt(dlat**2 + dlon**2)

        if dist < best_distance:
            best_distance = dist
            best_match = building

    return best_match, best_distance

# Load OSM buildings
print("\nLoading OSM building data...")
with open("output/osm_3d/al_karama_buildings.geojson") as f:
    osm_data = json.load(f)

osm_buildings = []
for feature in osm_data['features']:
    osm_buildings.append({
        'id': feature['properties']['id'],
        'name': feature['properties']['name'],
        'height': feature['properties']['height'],
        'coordinates': feature['geometry']['coordinates'][0] if feature['geometry']['coordinates'] else []
    })

print(f"Loaded {len(osm_buildings)} OSM buildings")

# Process each test image
print("\nEstimating building heights...")
height_estimates = defaultdict(list)  # building_id -> list of height estimates

for idx, row in test_images.iterrows():
    img_id = str(int(row['id']))

    depth_path = os.path.join(depth_dir, f"{img_id}.tiff")

    # Find segmentation image (check all batch directories)
    seg_path = None
    for batch_dir in glob.glob(os.path.join(seg_dir, "batch_*")):
        potential_path = os.path.join(batch_dir, f"{img_id}_colored_segmented.png")
        if os.path.exists(potential_path):
            seg_path = potential_path
            break

    if not os.path.exists(depth_path):
        print(f"  {img_id}: Missing depth map")
        continue

    if not seg_path:
        print(f"  {img_id}: Missing segmentation")
        continue

    # Find building mask
    building_mask = find_building_mask(seg_path)
    building_ratio = building_mask.sum() / building_mask.size

    if building_ratio < 0.01:
        print(f"  {img_id}: No buildings detected ({building_ratio*100:.1f}%)")
        continue

    # Estimate height
    est_height = estimate_height_from_depth(depth_path, building_mask)

    if est_height is None:
        print(f"  {img_id}: Could not estimate height")
        continue

    # Match to OSM building
    osm_match, dist = match_to_osm_building(row['lat'], row['lon'], osm_buildings)

    if osm_match:
        height_estimates[osm_match['id']].append(est_height)
        print(f"  {img_id}: Est. height={est_height:.1f}m -> Building {osm_match['id']} (dist={dist:.0f}m)")
    else:
        print(f"  {img_id}: Est. height={est_height:.1f}m -> No OSM match")

# Aggregate height estimates per building
print("\n" + "="*50)
print("HEIGHT ESTIMATION RESULTS")
print("="*50)

updated_buildings = 0
for building_id, estimates in height_estimates.items():
    if estimates:
        median_height = np.median(estimates)
        print(f"Building {building_id}: {len(estimates)} observations, median height = {median_height:.1f}m")

        # Update OSM data
        for feature in osm_data['features']:
            if feature['properties']['id'] == building_id:
                old_height = feature['properties']['height']
                feature['properties']['height'] = round(median_height, 1)
                feature['properties']['estimated'] = True
                feature['properties']['observations'] = len(estimates)
                print(f"  Updated: {old_height}m -> {median_height:.1f}m")
                updated_buildings += 1
                break

print(f"\nUpdated {updated_buildings} buildings with estimated heights")

# Save updated GeoJSON
updated_geojson_path = os.path.join(output_dir, "buildings_with_heights.geojson")
with open(updated_geojson_path, 'w') as f:
    json.dump(osm_data, f)
print(f"Saved: {updated_geojson_path}")

# Create updated 3D viewer
html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Al Karama 3D - Height Estimation PoC</title>
    <meta charset="utf-8">
    <script src="https://unpkg.com/deck.gl@8.9.0/dist.min.js"></script>
    <script src="https://unpkg.com/maplibre-gl@2.4.0/dist/maplibre-gl.js"></script>
    <link href="https://unpkg.com/maplibre-gl@2.4.0/dist/maplibre-gl.css" rel="stylesheet">
    <style>
        body { margin: 0; padding: 0; }
        #container { width: 100vw; height: 100vh; }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.85);
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-family: Arial, sans-serif;
            z-index: 1;
            max-width: 320px;
        }
        #info h3 { margin: 0 0 10px 0; color: #4fc3f7; }
        #info p { margin: 5px 0; font-size: 13px; }
        .stat { color: #81c784; font-weight: bold; }
        .legend { margin-top: 15px; border-top: 1px solid #444; padding-top: 10px; }
        .legend-item { display: flex; align-items: center; margin: 5px 0; }
        .legend-color { width: 20px; height: 20px; margin-right: 10px; border-radius: 3px; }
    </style>
</head>
<body>
    <div id="container"></div>
    <div id="info">
        <h3>Building Height Estimation PoC</h3>
        <p>Source: OSM + Depth Maps</p>
        <p>Test Area: 50m radius (26 images)</p>
        <hr style="border-color:#444">
        <p>Total buildings: <span class="stat" id="total">-</span></p>
        <p>Estimated heights: <span class="stat" id="estimated">-</span></p>
        <p>Default heights: <span class="stat" id="default">-</span></p>
        <div class="legend">
            <p><b>Colors:</b></p>
            <div class="legend-item"><div class="legend-color" style="background:#e74c3c"></div> Estimated from depth</div>
            <div class="legend-item"><div class="legend-color" style="background:#3498db"></div> OSM default (9m)</div>
        </div>
    </div>

    <script>
        const BUILDINGS = """ + json.dumps(osm_data) + """;

        const total = BUILDINGS.features.length;
        const estimated = BUILDINGS.features.filter(f => f.properties.estimated).length;
        document.getElementById('total').textContent = total;
        document.getElementById('estimated').textContent = estimated;
        document.getElementById('default').textContent = total - estimated;

        const deckgl = new deck.DeckGL({
            container: 'container',
            mapStyle: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
            initialViewState: {
                longitude: 55.3048,
                latitude: 25.2421,
                zoom: 17,
                pitch: 60,
                bearing: -20
            },
            controller: true,
            layers: [
                new deck.PolygonLayer({
                    id: 'buildings',
                    data: BUILDINGS.features,
                    extruded: true,
                    opacity: 0.9,
                    getPolygon: f => f.geometry.coordinates[0],
                    getElevation: f => f.properties.height,
                    getFillColor: f => f.properties.estimated ? [231, 76, 60] : [52, 152, 219],
                    getLineColor: [255, 255, 255, 100],
                    lineWidthMinPixels: 1,
                    pickable: true
                })
            ],
            getTooltip: ({object}) => object && {
                html: `<div style="padding:8px">
                    <b>${object.properties.name || 'Building ' + object.properties.id}</b><br>
                    Height: ${object.properties.height}m<br>
                    ${object.properties.estimated ?
                        '<span style="color:#e74c3c">✓ Estimated from ' + object.properties.observations + ' images</span>' :
                        '<span style="color:#3498db">Default OSM height</span>'}
                </div>`,
                style: { backgroundColor: '#222', color: '#fff', borderRadius: '4px' }
            }
        });
    </script>
</body>
</html>"""

html_path = os.path.join(output_dir, "height_estimation_poc.html")
with open(html_path, 'w') as f:
    f.write(html_content)
print(f"Saved 3D viewer: {html_path}")

print("\nDone! Open the HTML to see estimated building heights.")
