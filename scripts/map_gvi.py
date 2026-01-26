"""
Script to map Green View Index (GVI) results for Al Karama with image popups.
Two layers: Clustered (for performance) and Non-clustered (to see colors).
"""

import pandas as pd
import geopandas as gpd
import glob
import folium
from folium.plugins import HeatMap, MarkerCluster
import os

# Load image metadata with coordinates
metadata = pd.read_csv("data/svi_images/mly_pids.csv")
print(f"Loaded metadata for {len(metadata)} images")

# Load GVI results from all batches
gvi_files = glob.glob("data/segmentation/summary/*/pixel_ratios.csv")
print(f"Found {len(gvi_files)} batch result files")

if len(gvi_files) == 0:
    print("No GVI results found yet. Wait for segmentation to complete.")
    exit()

# Combine all batch results
dfs = []
for f in gvi_files:
    df = pd.read_csv(f)
    dfs.append(df)
gvi_data = pd.concat(dfs, ignore_index=True)

# Filter for vegetation only
vegetation = gvi_data[gvi_data['label_name'] == 'Vegetation'][['filename_key', 'pixel_ratios']].copy()
vegetation.columns = ['id', 'gvi']
vegetation['id'] = vegetation['id'].astype(str)
metadata['id'] = metadata['id'].astype(str)

print(f"Found GVI values for {len(vegetation)} images")

# Merge with coordinates
merged = metadata.merge(vegetation, on='id', how='inner')
print(f"Merged {len(merged)} images with coordinates")

# Convert GVI to percentage
merged['gvi_percent'] = merged['gvi'] * 100

# Find blend images for each point
def find_blend_image(image_id):
    """Find the blend image path for a given image ID."""
    pattern = f"data/segmentation/images/batch_*/{image_id}_blend.png"
    matches = glob.glob(pattern)
    if matches:
        return os.path.abspath(matches[0])
    return None

print("Finding blend images...")
merged['blend_image_path'] = merged['id'].apply(find_blend_image)
images_found = merged['blend_image_path'].notna().sum()
print(f"Found blend images for {images_found} points")

# Create output directory
os.makedirs("output", exist_ok=True)

# Save merged data
output_csv = "output/gvi_with_coordinates.csv"
merged.to_csv(output_csv, index=False)
print(f"Saved merged data to: {output_csv}")

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(
    merged,
    geometry=gpd.points_from_xy(merged['lon'], merged['lat']),
    crs="EPSG:4326"
)

# Save as GeoJSON
output_geojson = "output/gvi_points.geojson"
gdf.to_file(output_geojson, driver="GeoJSON")
print(f"Saved GeoJSON to: {output_geojson}")

# Create interactive map
center_lat = merged['lat'].mean()
center_lon = merged['lon'].mean()

print("Creating interactive map with toggleable layers...")
m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

# Layer 1: Clustered (for performance when zoomed in)
marker_cluster = MarkerCluster(name="Clustered (with images)", show=False).add_to(m)

# Layer 2: Non-clustered colored points (to see GVI distribution)
color_layer = folium.FeatureGroup(name="GVI Colors (no clustering)", show=True).add_to(m)

# Add points to both layers
for idx, row in merged.iterrows():
    gvi_val = row['gvi_percent']

    # Color from red (low GVI) to green (high GVI)
    if gvi_val < 5:
        color = 'red'
    elif gvi_val < 10:
        color = 'orange'
    elif gvi_val < 20:
        color = 'yellow'
    else:
        color = 'green'

    # Popup with image (for clustered layer)
    if pd.notna(row['blend_image_path']):
        img_path = row['blend_image_path']
        popup_html = f'''
        <div style="width:320px;">
            <b>GVI: {gvi_val:.1f}%</b><br>
            <b>ID:</b> {row['id']}<br>
            <img src="file://{img_path}" width="300" loading="lazy">
        </div>
        '''
    else:
        popup_html = f'''
        <div>
            <b>GVI: {gvi_val:.1f}%</b><br>
            <b>ID:</b> {row['id']}
        </div>
        '''

    # Simple popup for non-clustered layer (lightweight)
    simple_popup = f"GVI: {gvi_val:.1f}%"

    # Add to clustered layer (with image popup)
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=5,
        color=color,
        fill=True,
        fillColor=color,
        fillOpacity=0.7,
        popup=folium.Popup(popup_html, max_width=340)
    ).add_to(marker_cluster)

    # Add to non-clustered color layer (simple popup)
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=4,
        color=color,
        fill=True,
        fillColor=color,
        fillOpacity=0.7,
        popup=simple_popup
    ).add_to(color_layer)

# Add layer control to toggle between views
folium.LayerControl(collapsed=False).add_to(m)

# Add legend
legend_html = '''
<div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px;">
<b>Green View Index</b><br>
<i style="background: green; width: 12px; height: 12px; display: inline-block;"></i> &gt;20%<br>
<i style="background: yellow; width: 12px; height: 12px; display: inline-block;"></i> 10-20%<br>
<i style="background: orange; width: 12px; height: 12px; display: inline-block;"></i> 5-10%<br>
<i style="background: red; width: 12px; height: 12px; display: inline-block;"></i> &lt;5%<br>
<small>Toggle layers top-right</small>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Save point map
point_map_path = "output/gvi_point_map.html"
m.save(point_map_path)
print(f"Saved point map to: {point_map_path}")

# Heatmap
m2 = folium.Map(location=[center_lat, center_lon], zoom_start=15)
heat_data = [[row['lat'], row['lon'], row['gvi']] for _, row in merged.iterrows()]
HeatMap(heat_data, radius=15, blur=10, max_zoom=17).add_to(m2)

heatmap_path = "output/gvi_heatmap.html"
m2.save(heatmap_path)
print(f"Saved heatmap to: {heatmap_path}")

# Print summary statistics
print(f"\n=== GVI Summary Statistics ===")
print(f"Mean GVI: {merged['gvi_percent'].mean():.2f}%")
print(f"Median GVI: {merged['gvi_percent'].median():.2f}%")
print(f"Min GVI: {merged['gvi_percent'].min():.2f}%")
print(f"Max GVI: {merged['gvi_percent'].max():.2f}%")
print(f"Std Dev: {merged['gvi_percent'].std():.2f}%")

print(f"\nDone! Open the maps in your browser:")
print(f"  open {point_map_path}")
print(f"  open {heatmap_path}")
print(f"\nUse the layer control (top-right) to toggle between:")
print(f"  - 'GVI Colors' = See all colored points (default)")
print(f"  - 'Clustered' = Grouped markers with image popups")
