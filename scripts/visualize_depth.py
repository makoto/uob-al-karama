"""
Visualize depth maps alongside original images.
Creates an HTML gallery comparing original images with depth maps.
"""

import os
import glob
import base64
from PIL import Image
import numpy as np
import io
import pandas as pd
import folium

# Directories
input_dir = "data/depth_test/images"
depth_dir = "data/depth_test/depth"
output_dir = "output"

os.makedirs(output_dir, exist_ok=True)

# Get image pairs
image_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
print(f"Found {len(image_files)} images")

# Function to convert TIFF to displayable PNG base64
def tiff_to_base64_png(tiff_path, colormap=True):
    """Convert TIFF depth map to base64 PNG for HTML display."""
    img = Image.open(tiff_path)
    arr = np.array(img)

    # Normalize to 0-255
    arr_norm = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)

    if colormap:
        # Apply viridis-like colormap (inverted so closer = warmer colors)
        arr_norm = 255 - arr_norm  # Invert: higher values = closer = brighter

        # Create RGB image with colormap
        r = np.clip(arr_norm * 2, 0, 255).astype(np.uint8)
        g = np.clip((arr_norm - 85) * 3, 0, 255).astype(np.uint8)
        b = np.clip(255 - arr_norm * 2, 0, 255).astype(np.uint8)

        rgb = np.stack([r, g, b], axis=-1)
        img_colored = Image.fromarray(rgb)
    else:
        img_colored = Image.fromarray(arr_norm)

    # Save to bytes
    buffer = io.BytesIO()
    img_colored.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

# Function to encode original image
def image_to_base64(img_path):
    with open(img_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

# Build HTML gallery
html_parts = ["""
<!DOCTYPE html>
<html>
<head>
    <title>Depth Map Gallery - Test Area</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: white; }
        h1 { text-align: center; }
        .gallery { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; }
        .pair {
            display: flex;
            flex-direction: column;
            align-items: center;
            background: #2a2a2a;
            padding: 15px;
            border-radius: 10px;
            max-width: 800px;
        }
        .images { display: flex; gap: 10px; }
        .images img { max-width: 380px; border-radius: 5px; }
        .label { font-size: 12px; color: #aaa; margin-top: 5px; }
        .image-id { font-size: 14px; color: #4fc3f7; margin-bottom: 10px; }
        .legend {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
            margin: 20px 0;
            padding: 15px;
            background: #2a2a2a;
            border-radius: 10px;
        }
        .legend-bar {
            width: 200px;
            height: 20px;
            background: linear-gradient(to right, #0000ff, #00ffff, #00ff00, #ffff00, #ff0000);
            border-radius: 3px;
        }
        .legend-labels { display: flex; justify-content: space-between; width: 200px; font-size: 12px; }
    </style>
</head>
<body>
    <h1>Depth Estimation Results - Test Area (50m radius)</h1>
    <p style="text-align: center;">26 images centered around image 827604215891753</p>

    <div class="legend">
        <span>Depth Legend:</span>
        <div>
            <div class="legend-bar"></div>
            <div class="legend-labels">
                <span>Far</span>
                <span>Near</span>
            </div>
        </div>
    </div>

    <div class="gallery">
"""]

for img_path in image_files:
    img_id = os.path.basename(img_path).replace('.png', '')
    depth_path = os.path.join(depth_dir, f"{img_id}.tiff")

    if os.path.exists(depth_path):
        orig_b64 = image_to_base64(img_path)
        depth_b64 = tiff_to_base64_png(depth_path, colormap=True)

        html_parts.append(f"""
        <div class="pair">
            <div class="image-id">ID: {img_id}</div>
            <div class="images">
                <div>
                    <img src="data:image/png;base64,{orig_b64}" alt="Original">
                    <div class="label">Original Image</div>
                </div>
                <div>
                    <img src="data:image/png;base64,{depth_b64}" alt="Depth">
                    <div class="label">Depth Map</div>
                </div>
            </div>
        </div>
        """)
        print(f"Processed {img_id}")

html_parts.append("""
    </div>
</body>
</html>
""")

# Write gallery HTML
gallery_path = os.path.join(output_dir, "depth_gallery.html")
with open(gallery_path, 'w') as f:
    f.write(''.join(html_parts))
print(f"\nGallery saved to: {gallery_path}")

# Create a map showing depth map locations with image file references
print("\nCreating map with depth locations...")

# Save depth maps as PNG files for web display
depth_png_dir = os.path.join(output_dir, "depth_png")
os.makedirs(depth_png_dir, exist_ok=True)

print("Converting depth TIFFs to PNGs...")
for img_path in image_files:
    img_id = os.path.basename(img_path).replace('.png', '')
    depth_path = os.path.join(depth_dir, f"{img_id}.tiff")
    depth_png_path = os.path.join(depth_png_dir, f"{img_id}.png")

    if os.path.exists(depth_path) and not os.path.exists(depth_png_path):
        # Convert TIFF to colored PNG
        img = Image.open(depth_path)
        arr = np.array(img)
        arr_norm = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
        arr_norm = 255 - arr_norm  # Invert: higher = closer

        r = np.clip(arr_norm * 2, 0, 255).astype(np.uint8)
        g = np.clip((arr_norm - 85) * 3, 0, 255).astype(np.uint8)
        b = np.clip(255 - arr_norm * 2, 0, 255).astype(np.uint8)

        rgb = np.stack([r, g, b], axis=-1)
        Image.fromarray(rgb).save(depth_png_path)
        print(f"  Saved {img_id}.png")

# Load test area images data
test_images = pd.read_csv("output/test_area_images.csv")
print(f"Loaded {len(test_images)} test image records")

# Create map centered on test area
center_lat = test_images['lat'].mean()
center_lon = test_images['lon'].mean()
print(f"Map center: {center_lat}, {center_lon}")

m = folium.Map(location=[center_lat, center_lon], zoom_start=19)

# Get absolute paths for file:// URLs
abs_input_dir = os.path.abspath(input_dir)
abs_depth_png_dir = os.path.abspath(depth_png_dir)

# Add markers for each image with local file references
for idx, row in test_images.iterrows():
    img_id = str(int(row['id']))
    lat = row['lat']
    lon = row['lon']

    orig_file = f"file://{abs_input_dir}/{img_id}.png"
    depth_file = f"file://{abs_depth_png_dir}/{img_id}.png"

    is_center = img_id == "827604215891753"

    popup_html = f"""
    <div style="min-width:420px">
        <b>Image ID:</b> {img_id}<br>
        <b>Distance:</b> {row['distance_m']:.1f}m from center
        {'<br><b style="color:red">CENTER IMAGE</b>' if is_center else ''}<br>
        <div style="display:flex;gap:5px;margin-top:8px;">
            <div>
                <img src="{orig_file}" width="200"><br>
                <small>Original</small>
            </div>
            <div>
                <img src="{depth_file}" width="200"><br>
                <small>Depth (red=near, blue=far)</small>
            </div>
        </div>
    </div>
    """

    # Use different colors
    if is_center:
        icon = folium.Icon(color='red', icon='star')
    else:
        icon = folium.Icon(color='blue', icon='camera')

    marker = folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_html, max_width=450),
        icon=icon,
        tooltip=f"Image {img_id}"
    )
    marker.add_to(m)
    print(f"Added marker for {img_id} at ({lat:.6f}, {lon:.6f})")

# Add 50m circle around center
folium.Circle(
    location=[25.242071, 55.304779],
    radius=50,
    color='red',
    fill=True,
    fill_opacity=0.1,
    weight=2,
    tooltip="50m test area"
).add_to(m)

# Save map
map_path = os.path.join(output_dir, "depth_map.html")
m.save(map_path)
print(f"\nMap saved to: {map_path}")

print("\nDone!")
