"""
PoC v2: Improved building height estimation.
- Better OSM matching (search for buildings visible from viewpoint, not just nearest)
- Visual output showing estimates
"""

import os
import glob
import numpy as np
from PIL import Image
import pandas as pd
import json

# Directories
depth_dir = "data/depth_test/depth"
input_dir = "data/depth_test/images"
seg_dir = "data/segmentation/images"
output_dir = "output/height_estimation"
os.makedirs(output_dir, exist_ok=True)

# Load test area images
test_images = pd.read_csv("output/test_area_images.csv")
print(f"Test images: {len(test_images)}")

def analyze_depth_for_height(depth_path, seg_path):
    """
    Analyze depth map to estimate building heights visible in image.
    Returns height estimates for different directions.
    """
    depth_img = Image.open(depth_path)
    depth = np.array(depth_img, dtype=np.float32)

    seg_img = Image.open(seg_path)
    seg = np.array(seg_img)

    h, w = depth.shape

    # Normalize depth
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

    # Simple building detection: look for consistent vertical structures
    # in the middle portion of the image (not sky at top, not ground at bottom)

    results = []

    # Analyze 8 directions (every 45 degrees in panorama)
    for direction in range(8):
        col_start = int(direction * w / 8)
        col_end = int((direction + 1) * w / 8)

        # Get depth slice for this direction
        depth_slice = depth_norm[:, col_start:col_end]

        # Find where depth is relatively consistent (building facade)
        # Look at rows from horizon (h/2) upward
        horizon = h // 2

        # Calculate depth variance in vertical strips
        for row in range(int(h * 0.15), horizon):
            strip_depth = depth_slice[row, :].mean()

            # If we find consistent depth above horizon, it's likely a building
            if strip_depth > 0.1:  # Not sky (sky is very far = low depth value)
                # Calculate angle above horizon
                angle_above_horizon = ((horizon - row) / h) * 180  # degrees

                # Estimate distance (inverse relationship with depth value)
                rel_distance = 50 / (strip_depth + 0.1)  # rough scale

                # Height = distance * tan(angle)
                height_estimate = rel_distance * np.tan(np.radians(angle_above_horizon))

                if 3 < height_estimate < 150:
                    results.append({
                        'direction': direction * 45,  # degrees from front
                        'height': height_estimate,
                        'distance': rel_distance,
                        'confidence': strip_depth
                    })
                break  # Found building top for this direction

    return results

# Process images and collect estimates
all_estimates = []

print("\nAnalyzing depth maps...")
for idx, row in test_images.iterrows():
    img_id = str(int(row['id']))

    depth_path = os.path.join(depth_dir, f"{img_id}.tiff")

    # Find segmentation
    seg_path = None
    for batch_dir in glob.glob(os.path.join(seg_dir, "batch_*")):
        potential_path = os.path.join(batch_dir, f"{img_id}_colored_segmented.png")
        if os.path.exists(potential_path):
            seg_path = potential_path
            break

    if not os.path.exists(depth_path) or not seg_path:
        continue

    estimates = analyze_depth_for_height(depth_path, seg_path)

    for est in estimates:
        all_estimates.append({
            'image_id': img_id,
            'lat': float(row['lat']),
            'lon': float(row['lon']),
            'direction': float(est['direction']),
            'height': float(est['height']),
            'distance': float(est['distance']),
            'confidence': float(est['confidence'])
        })

    if estimates:
        heights = [e['height'] for e in estimates]
        print(f"  {img_id}: {len(estimates)} directions, heights: {min(heights):.0f}-{max(heights):.0f}m")

print(f"\nTotal estimates: {len(all_estimates)}")

# Create summary
df = pd.DataFrame(all_estimates)
if not df.empty:
    print(f"\nHeight statistics:")
    print(f"  Mean: {df['height'].mean():.1f}m")
    print(f"  Median: {df['height'].median():.1f}m")
    print(f"  Range: {df['height'].min():.1f}m - {df['height'].max():.1f}m")

    # Save estimates
    df.to_csv(os.path.join(output_dir, "height_estimates.csv"), index=False)

# Create visualization showing estimates on map
html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Building Height Estimation PoC - Test Area</title>
    <meta charset="utf-8">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body { margin: 0; font-family: Arial, sans-serif; }
        #map { width: 100%; height: 100vh; }
        .info-panel {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.85);
            color: white;
            padding: 15px;
            border-radius: 8px;
            z-index: 1000;
            max-width: 300px;
        }
        .info-panel h3 { margin: 0 0 10px 0; color: #4fc3f7; }
        .info-panel p { margin: 5px 0; font-size: 13px; }
        .stat { color: #81c784; }
        .legend { margin-top: 10px; }
        .legend-item { display: flex; align-items: center; margin: 3px 0; font-size: 12px; }
        .legend-circle { width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="info-panel">
        <h3>Height Estimation PoC</h3>
        <p>Test Area: 50m radius</p>
        <p>Images analyzed: <span class="stat">""" + str(len(test_images)) + """</span></p>
        <p>Height estimates: <span class="stat">""" + str(len(all_estimates)) + """</span></p>
        """ + (f"""<p>Mean height: <span class="stat">{df['height'].mean():.1f}m</span></p>
        <p>Median height: <span class="stat">{df['height'].median():.1f}m</span></p>""" if not df.empty else "") + """
        <div class="legend">
            <p><b>Estimated Height:</b></p>
            <div class="legend-item"><div class="legend-circle" style="background:#2ecc71"></div> &lt;10m (low-rise)</div>
            <div class="legend-item"><div class="legend-circle" style="background:#f39c12"></div> 10-30m (mid-rise)</div>
            <div class="legend-item"><div class="legend-circle" style="background:#e74c3c"></div> &gt;30m (high-rise)</div>
        </div>
        <p style="margin-top:15px;font-size:11px;color:#888">
            Note: Heights are relative estimates from depth maps.
            Accuracy depends on calibration with known references.
        </p>
    </div>

    <script>
        const map = L.map('map').setView([25.2421, 55.3048], 18);

        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            maxZoom: 20
        }).addTo(map);

        // Test area circle
        L.circle([25.242071, 55.304779], {
            radius: 50,
            color: '#4fc3f7',
            fillOpacity: 0.1,
            weight: 2
        }).addTo(map);

        // Height estimates
        const estimates = """ + json.dumps(all_estimates) + """;

        estimates.forEach(est => {
            const color = est.height < 10 ? '#2ecc71' :
                         est.height < 30 ? '#f39c12' : '#e74c3c';

            // Draw line showing direction and estimated building
            const dirRad = est.direction * Math.PI / 180;
            const dist = Math.min(est.distance, 30) / 111000; // Convert to degrees

            const endLat = est.lat + dist * Math.cos(dirRad);
            const endLon = est.lon + dist * Math.sin(dirRad) / Math.cos(est.lat * Math.PI / 180);

            L.polyline([[est.lat, est.lon], [endLat, endLon]], {
                color: color,
                weight: 3,
                opacity: 0.7
            }).bindPopup(`
                <b>Height estimate: ${est.height.toFixed(1)}m</b><br>
                Direction: ${est.direction}°<br>
                Distance: ~${est.distance.toFixed(0)}m<br>
                Image: ${est.image_id}
            `).addTo(map);
        });

        // Camera positions
        const images = """ + test_images.to_json(orient='records') + """;
        images.forEach(img => {
            L.circleMarker([img.lat, img.lon], {
                radius: 4,
                color: '#fff',
                fillColor: '#4fc3f7',
                fillOpacity: 1,
                weight: 1
            }).addTo(map);
        });
    </script>
</body>
</html>"""

html_path = os.path.join(output_dir, "height_estimates_map.html")
with open(html_path, 'w') as f:
    f.write(html_content)

print(f"\nSaved: {html_path}")
print("\nDone!")
