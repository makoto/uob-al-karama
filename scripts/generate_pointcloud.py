"""
Generate 3D point cloud from depth maps and street view images.
Uses depth estimation + camera position to create 3D reconstruction.
"""

import os
import glob
import numpy as np
from PIL import Image
import pandas as pd
import requests
from dotenv import load_dotenv
import json

load_dotenv()

# Directories
input_dir = "data/depth_test/images"
depth_dir = "data/depth_test/depth"
output_dir = "output/pointcloud"
os.makedirs(output_dir, exist_ok=True)

# Load test area images
test_images = pd.read_csv("output/test_area_images.csv")
print(f"Loaded {len(test_images)} images")

# Get Mapillary API key for fetching camera metadata
mly_api_key = os.getenv("MLY_API_KEY")

def get_camera_metadata(image_id):
    """Fetch camera metadata from Mapillary API."""
    url = f"https://graph.mapillary.com/{image_id}"
    params = {
        "access_token": mly_api_key,
        "fields": "compass_angle,computed_compass_angle,camera_type,geometry,captured_at"
    }
    try:
        resp = requests.get(url, params=params)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"Error fetching metadata for {image_id}: {e}")
    return None

def depth_to_pointcloud(depth_path, image_path, camera_lat, camera_lon, compass_angle,
                         sample_rate=10, max_depth_percentile=95):
    """
    Convert depth map to 3D point cloud.

    Args:
        depth_path: Path to depth TIFF
        image_path: Path to original image (for colors)
        camera_lat, camera_lon: Camera position
        compass_angle: Camera heading in degrees (0=North, 90=East)
        sample_rate: Sample every Nth pixel (reduces point count)
        max_depth_percentile: Clip depth values above this percentile

    Returns:
        points: Nx3 array of XYZ coordinates
        colors: Nx3 array of RGB colors
    """
    # Load depth map
    depth_img = Image.open(depth_path)
    depth = np.array(depth_img, dtype=np.float32)

    # Load color image
    color_img = Image.open(image_path)
    colors = np.array(color_img)

    h, w = depth.shape

    # Normalize depth (relative depth from model)
    # Invert so larger = farther (model outputs inverse depth)
    depth_norm = depth.copy()

    # Clip outliers
    max_depth = np.percentile(depth_norm, max_depth_percentile)
    depth_norm = np.clip(depth_norm, 0, max_depth)

    # Scale to reasonable range (0-50 meters)
    depth_norm = (depth_norm / depth_norm.max()) * 50

    # Camera parameters (assuming equirectangular panorama)
    # Horizontal FOV = 360 degrees, Vertical FOV ~180 degrees
    fov_h = 2 * np.pi  # 360 degrees
    fov_v = np.pi      # 180 degrees

    # Create pixel coordinate grids (sample every Nth pixel)
    rows = np.arange(0, h, sample_rate)
    cols = np.arange(0, w, sample_rate)
    col_grid, row_grid = np.meshgrid(cols, rows)

    # Convert pixel to spherical coordinates
    # Horizontal angle (azimuth): 0 to 2*pi
    theta = (col_grid / w) * fov_h
    # Vertical angle (elevation): pi/2 to -pi/2
    phi = (0.5 - row_grid / h) * fov_v

    # Get depth and color values at sampled points
    d = depth_norm[row_grid, col_grid]
    c = colors[row_grid, col_grid]

    # Convert spherical to Cartesian (camera-relative)
    # X = right, Y = forward, Z = up
    x = d * np.cos(phi) * np.sin(theta)
    y = d * np.cos(phi) * np.cos(theta)
    z = d * np.sin(phi)

    # Rotate by compass angle (convert to world coordinates)
    angle_rad = np.radians(compass_angle)
    x_world = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    y_world = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    z_world = z

    # Offset by camera position (convert lat/lon to local meters)
    # Using simple approximation: 1 degree lat ≈ 111km, 1 degree lon ≈ 111km * cos(lat)
    lat_to_m = 111000
    lon_to_m = 111000 * np.cos(np.radians(camera_lat))

    # Reference point (first camera position)
    ref_lat = test_images['lat'].iloc[0]
    ref_lon = test_images['lon'].iloc[0]

    cam_x = (camera_lon - ref_lon) * lon_to_m
    cam_y = (camera_lat - ref_lat) * lat_to_m
    cam_z = 2.0  # Assume camera height of 2 meters

    x_world += cam_x
    y_world += cam_y
    z_world += cam_z

    # Flatten arrays
    points = np.stack([x_world.flatten(), y_world.flatten(), z_world.flatten()], axis=1)
    colors_flat = c.reshape(-1, 3)

    # Filter out sky points (very high Z or very far depth)
    mask = (z_world.flatten() < 30) & (d.flatten() > 1)
    points = points[mask]
    colors_flat = colors_flat[mask]

    return points, colors_flat

def save_ply(filename, points, colors):
    """Save point cloud to PLY format."""
    n_points = len(points)

    header = f"""ply
format ascii 1.0
element vertex {n_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

    with open(filename, 'w') as f:
        f.write(header)
        for i in range(n_points):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x:.3f} {y:.3f} {z:.3f} {int(r)} {int(g)} {int(b)}\n")

# Process each image
all_points = []
all_colors = []

print("\nFetching camera metadata and generating point clouds...")

for idx, row in test_images.iterrows():
    img_id = str(int(row['id']))

    depth_path = os.path.join(depth_dir, f"{img_id}.tiff")
    image_path = os.path.join(input_dir, f"{img_id}.png")

    if not os.path.exists(depth_path) or not os.path.exists(image_path):
        continue

    # Get camera heading from Mapillary
    metadata = get_camera_metadata(img_id)

    if metadata:
        compass_angle = metadata.get('computed_compass_angle') or metadata.get('compass_angle', 0)
        print(f"Image {img_id}: compass={compass_angle:.1f}°")
    else:
        compass_angle = 0
        print(f"Image {img_id}: using default compass=0°")

    # Generate point cloud for this image
    points, colors = depth_to_pointcloud(
        depth_path, image_path,
        row['lat'], row['lon'], compass_angle,
        sample_rate=8  # Sample every 8th pixel
    )

    all_points.append(points)
    all_colors.append(colors)
    print(f"  Generated {len(points)} points")

# Combine all point clouds
all_points = np.vstack(all_points)
all_colors = np.vstack(all_colors)

print(f"\nTotal points: {len(all_points)}")

# Save combined point cloud
ply_path = os.path.join(output_dir, "test_area_pointcloud.ply")
save_ply(ply_path, all_points, all_colors)
print(f"Saved point cloud to: {ply_path}")

# Also save individual clouds for debugging
print("\nSaving individual point clouds...")
for idx, row in test_images.iterrows():
    img_id = str(int(row['id']))
    depth_path = os.path.join(depth_dir, f"{img_id}.tiff")
    image_path = os.path.join(input_dir, f"{img_id}.png")

    if not os.path.exists(depth_path):
        continue

    metadata = get_camera_metadata(img_id)
    compass_angle = metadata.get('computed_compass_angle', 0) if metadata else 0

    points, colors = depth_to_pointcloud(
        depth_path, image_path,
        row['lat'], row['lon'], compass_angle,
        sample_rate=4
    )

    individual_ply = os.path.join(output_dir, f"{img_id}.ply")
    save_ply(individual_ply, points, colors)

print(f"\nDone! Point clouds saved to {output_dir}/")
print("\nTo view the point cloud, you can use:")
print("  - MeshLab (free): https://www.meshlab.net/")
print("  - CloudCompare (free): https://www.cloudcompare.org/")
print("  - Blender (free): https://www.blender.org/")
print("  - Online: https://3dviewer.net/")
