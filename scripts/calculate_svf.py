"""
Script to calculate Sky View Factor (SVF) and create combined GVI/SVF analysis.
Uses existing segmentation data - no additional processing needed.
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

# Load segmentation results from all batches
seg_files = glob.glob("data/segmentation/summary/*/pixel_ratios.csv")
print(f"Found {len(seg_files)} batch result files")

if len(seg_files) == 0:
    print("No segmentation results found. Run calculate_gvi.py first.")
    exit()

# Combine all batch results
dfs = []
for f in seg_files:
    df = pd.read_csv(f)
    dfs.append(df)
seg_data = pd.concat(dfs, ignore_index=True)

# Extract Sky View Factor
sky = seg_data[seg_data['label_name'] == 'Sky'][['filename_key', 'pixel_ratios']].copy()
sky.columns = ['id', 'svf']

# Extract Green View Index
vegetation = seg_data[seg_data['label_name'] == 'Vegetation'][['filename_key', 'pixel_ratios']].copy()
vegetation.columns = ['id', 'gvi']

# Convert IDs to string
sky['id'] = sky['id'].astype(str)
vegetation['id'] = vegetation['id'].astype(str)
metadata['id'] = metadata['id'].astype(str)

print(f"Found SVF values for {len(sky)} images")
print(f"Found GVI values for {len(vegetation)} images")

# Merge SVF and GVI
combined = sky.merge(vegetation, on='id', how='outer')
combined = metadata.merge(combined, on='id', how='inner')
print(f"Combined {len(combined)} images with coordinates")

# Convert to percentages
combined['svf_percent'] = combined['svf'] * 100
combined['gvi_percent'] = combined['gvi'] * 100

# Fill NaN with 0
combined['svf_percent'] = combined['svf_percent'].fillna(0)
combined['gvi_percent'] = combined['gvi_percent'].fillna(0)

# Create output directory
os.makedirs("output", exist_ok=True)

# Save combined data
output_csv = "output/gvi_svf_combined.csv"
combined.to_csv(output_csv, index=False)
print(f"Saved combined data to: {output_csv}")

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(
    combined,
    geometry=gpd.points_from_xy(combined['lon'], combined['lat']),
    crs="EPSG:4326"
)

# Save as GeoJSON
output_geojson = "output/gvi_svf_points.geojson"
gdf.to_file(output_geojson, driver="GeoJSON")
print(f"Saved GeoJSON to: {output_geojson}")

# ============================================
# Create Combined Comparison Map
# ============================================
center_lat = combined['lat'].mean()
center_lon = combined['lon'].mean()

print("Creating combined comparison map...")
m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

# Layer 1: GVI Colors
gvi_layer = folium.FeatureGroup(name="Green View Index (GVI)", show=True).add_to(m)

# Layer 2: SVF Colors
svf_layer = folium.FeatureGroup(name="Sky View Factor (SVF)", show=False).add_to(m)

# Add points to both layers
for idx, row in combined.iterrows():
    gvi_val = row['gvi_percent']
    svf_val = row['svf_percent']

    # GVI color scale (red to green)
    if gvi_val < 5:
        gvi_color = 'red'
    elif gvi_val < 10:
        gvi_color = 'orange'
    elif gvi_val < 20:
        gvi_color = 'yellow'
    else:
        gvi_color = 'green'

    # SVF color scale (red to blue)
    if svf_val < 15:
        svf_color = 'red'
    elif svf_val < 30:
        svf_color = 'orange'
    elif svf_val < 50:
        svf_color = 'lightblue'
    else:
        svf_color = 'blue'

    # Popup with both values
    popup_html = f"GVI: {gvi_val:.1f}% | SVF: {svf_val:.1f}%"

    # Add to GVI layer
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=4,
        color=gvi_color,
        fill=True,
        fillColor=gvi_color,
        fillOpacity=0.7,
        popup=popup_html
    ).add_to(gvi_layer)

    # Add to SVF layer
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=4,
        color=svf_color,
        fill=True,
        fillColor=svf_color,
        fillOpacity=0.7,
        popup=popup_html
    ).add_to(svf_layer)

# Add layer control
folium.LayerControl(collapsed=False).add_to(m)

# Add legend for both
legend_html = '''
<div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px; font-size: 12px;">
<b>Green View Index (GVI)</b><br>
<i style="background: green; width: 12px; height: 12px; display: inline-block;"></i> &gt;20%<br>
<i style="background: yellow; width: 12px; height: 12px; display: inline-block;"></i> 10-20%<br>
<i style="background: orange; width: 12px; height: 12px; display: inline-block;"></i> 5-10%<br>
<i style="background: red; width: 12px; height: 12px; display: inline-block;"></i> &lt;5%<br>
<hr style="margin: 5px 0;">
<b>Sky View Factor (SVF)</b><br>
<i style="background: blue; width: 12px; height: 12px; display: inline-block;"></i> &gt;50% (open)<br>
<i style="background: lightblue; width: 12px; height: 12px; display: inline-block;"></i> 30-50%<br>
<i style="background: orange; width: 12px; height: 12px; display: inline-block;"></i> 15-30%<br>
<i style="background: red; width: 12px; height: 12px; display: inline-block;"></i> &lt;15% (canyon)<br>
<small>Toggle layers top-right</small>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Save comparison map
comparison_map_path = "output/gvi_svf_comparison_map.html"
m.save(comparison_map_path)
print(f"Saved comparison map to: {comparison_map_path}")

# ============================================
# Create SVF Heatmap
# ============================================
m2 = folium.Map(location=[center_lat, center_lon], zoom_start=15)
# Filter out NaN values for heatmap
heat_data = [[row['lat'], row['lon'], row['svf']] for _, row in combined.iterrows() if pd.notna(row['svf'])]
HeatMap(heat_data, radius=15, blur=10, max_zoom=17).add_to(m2)

svf_heatmap_path = "output/svf_heatmap.html"
m2.save(svf_heatmap_path)
print(f"Saved SVF heatmap to: {svf_heatmap_path}")

# ============================================
# Create Scatter Plot (GVI vs SVF)
# ============================================
import json

# Prepare data for scatter plot
scatter_data = combined[['gvi_percent', 'svf_percent', 'id', 'lat', 'lon']].to_dict('records')

scatter_html = f'''
<!DOCTYPE html>
<html>
<head>
    <title>GVI vs SVF Scatter Plot - Al Karama</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        #scatter {{ width: 100%; height: 600px; }}
        .stats {{ background: #f5f5f5; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Green View Index vs Sky View Factor - Al Karama, Dubai</h1>

    <div class="stats">
        <h3>Summary Statistics</h3>
        <table>
            <tr><th></th><th>GVI (%)</th><th>SVF (%)</th></tr>
            <tr><td>Mean</td><td>{combined['gvi_percent'].mean():.2f}</td><td>{combined['svf_percent'].mean():.2f}</td></tr>
            <tr><td>Median</td><td>{combined['gvi_percent'].median():.2f}</td><td>{combined['svf_percent'].median():.2f}</td></tr>
            <tr><td>Min</td><td>{combined['gvi_percent'].min():.2f}</td><td>{combined['svf_percent'].min():.2f}</td></tr>
            <tr><td>Max</td><td>{combined['gvi_percent'].max():.2f}</td><td>{combined['svf_percent'].max():.2f}</td></tr>
            <tr><td>Std Dev</td><td>{combined['gvi_percent'].std():.2f}</td><td>{combined['svf_percent'].std():.2f}</td></tr>
        </table>
        <p><b>Correlation:</b> {combined['gvi_percent'].corr(combined['svf_percent']):.3f}</p>
    </div>

    <div id="scatter"></div>

    <script>
        var data = {json.dumps(scatter_data)};

        var trace = {{
            x: data.map(d => d.gvi_percent),
            y: data.map(d => d.svf_percent),
            mode: 'markers',
            type: 'scatter',
            marker: {{
                size: 5,
                color: data.map(d => d.gvi_percent),
                colorscale: 'RdYlGn',
                opacity: 0.6
            }},
            text: data.map(d => `ID: ${{d.id}}<br>GVI: ${{d.gvi_percent.toFixed(1)}}%<br>SVF: ${{d.svf_percent.toFixed(1)}}%`),
            hoverinfo: 'text'
        }};

        var layout = {{
            title: 'GVI vs SVF Relationship',
            xaxis: {{ title: 'Green View Index (%)', range: [0, 70] }},
            yaxis: {{ title: 'Sky View Factor (%)', range: [0, 100] }},
            hovermode: 'closest'
        }};

        Plotly.newPlot('scatter', [trace], layout);
    </script>
</body>
</html>
'''

scatter_path = "output/gvi_svf_scatter.html"
with open(scatter_path, 'w') as f:
    f.write(scatter_html)
print(f"Saved scatter plot to: {scatter_path}")

# ============================================
# Print Summary Statistics
# ============================================
print(f"\n{'='*50}")
print("SUMMARY STATISTICS")
print('='*50)

print(f"\n--- Green View Index (GVI) ---")
print(f"Mean:   {combined['gvi_percent'].mean():.2f}%")
print(f"Median: {combined['gvi_percent'].median():.2f}%")
print(f"Min:    {combined['gvi_percent'].min():.2f}%")
print(f"Max:    {combined['gvi_percent'].max():.2f}%")
print(f"Std:    {combined['gvi_percent'].std():.2f}%")

print(f"\n--- Sky View Factor (SVF) ---")
print(f"Mean:   {combined['svf_percent'].mean():.2f}%")
print(f"Median: {combined['svf_percent'].median():.2f}%")
print(f"Min:    {combined['svf_percent'].min():.2f}%")
print(f"Max:    {combined['svf_percent'].max():.2f}%")
print(f"Std:    {combined['svf_percent'].std():.2f}%")

print(f"\n--- Correlation ---")
corr = combined['gvi_percent'].corr(combined['svf_percent'])
print(f"GVI-SVF Correlation: {corr:.3f}")
if corr > 0.3:
    print("→ Positive correlation: greener areas tend to have more open sky")
elif corr < -0.3:
    print("→ Negative correlation: greener areas tend to have less open sky")
else:
    print("→ Weak correlation: GVI and SVF are relatively independent")

print(f"\n{'='*50}")
print("OUTPUT FILES")
print('='*50)
print(f"  Combined data:     {output_csv}")
print(f"  GeoJSON:           {output_geojson}")
print(f"  Comparison map:    {comparison_map_path}")
print(f"  SVF heatmap:       {svf_heatmap_path}")
print(f"  Scatter plot:      {scatter_path}")
print(f"\nOpen in browser:")
print(f"  open {comparison_map_path}")
print(f"  open {scatter_path}")
