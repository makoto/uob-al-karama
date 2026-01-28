"""
Satellite-based Urban Climate Analysis for Al Karama
Uses Google Earth Engine to extract:
- Land Surface Temperature (Landsat)
- Vegetation Index (NDVI)
- Built-up Index
- Surface reflectance
"""

import ee
import os
import json
import pandas as pd
import numpy as np

# Initialize Earth Engine
ee.Initialize(project='uobdubai')
print("✅ Connected to Google Earth Engine")

output_dir = "output/satellite_analysis"
os.makedirs(output_dir, exist_ok=True)

# Al Karama bounding box
AL_KARAMA = ee.Geometry.Rectangle([55.290, 25.230, 55.320, 25.260])
center = [25.245, 55.305]

print("\n" + "="*60)
print("SATELLITE ANALYSIS FOR AL KARAMA")
print("="*60)

# ============================================================
# 1. LAND SURFACE TEMPERATURE (Landsat 8/9)
# ============================================================
print("\n1. LAND SURFACE TEMPERATURE")
print("-"*40)

def calculate_lst(image):
    """Calculate Land Surface Temperature from Landsat thermal band."""
    # Scale factors for Collection 2
    thermal = image.select('ST_B10').multiply(0.00341802).add(149.0)
    # Convert Kelvin to Celsius
    lst_celsius = thermal.subtract(273.15)
    return image.addBands(lst_celsius.rename('LST'))

# Get Landsat 8/9 imagery (summer months for heat analysis)
landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
    .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')) \
    .filterBounds(AL_KARAMA) \
    .filterDate('2024-06-01', '2024-09-30') \
    .filter(ee.Filter.lt('CLOUD_COVER', 20)) \
    .map(calculate_lst)

landsat_count = landsat.size().getInfo()
print(f"  Landsat images (Summer 2024, <20% cloud): {landsat_count}")

if landsat_count > 0:
    # Get median LST
    lst_median = landsat.select('LST').median()

    # Sample LST values across the area
    lst_stats = lst_median.reduceRegion(
        reducer=ee.Reducer.mean().combine(
            ee.Reducer.minMax(), sharedInputs=True
        ).combine(ee.Reducer.stdDev(), sharedInputs=True),
        geometry=AL_KARAMA,
        scale=30,
        maxPixels=1e9
    ).getInfo()

    print(f"  Mean LST: {lst_stats.get('LST_mean', 'N/A'):.1f}°C")
    print(f"  Min LST: {lst_stats.get('LST_min', 'N/A'):.1f}°C")
    print(f"  Max LST: {lst_stats.get('LST_max', 'N/A'):.1f}°C")
    print(f"  Std Dev: {lst_stats.get('LST_stdDev', 'N/A'):.1f}°C")

# ============================================================
# 2. VEGETATION INDEX (NDVI) from Sentinel-2
# ============================================================
print("\n2. VEGETATION INDEX (NDVI)")
print("-"*40)

def calculate_ndvi(image):
    """Calculate NDVI from Sentinel-2."""
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

def mask_clouds_s2(image):
    """Mask clouds in Sentinel-2 imagery."""
    qa = image.select('QA60')
    cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(cloud_mask)

# Get Sentinel-2 imagery
sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(AL_KARAMA) \
    .filterDate('2024-01-01', '2024-12-31') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
    .map(mask_clouds_s2) \
    .map(calculate_ndvi)

s2_count = sentinel2.size().getInfo()
print(f"  Sentinel-2 images (2024, <20% cloud): {s2_count}")

if s2_count > 0:
    # Get median NDVI
    ndvi_median = sentinel2.select('NDVI').median()

    ndvi_stats = ndvi_median.reduceRegion(
        reducer=ee.Reducer.mean().combine(
            ee.Reducer.minMax(), sharedInputs=True
        ).combine(ee.Reducer.stdDev(), sharedInputs=True),
        geometry=AL_KARAMA,
        scale=10,
        maxPixels=1e9
    ).getInfo()

    print(f"  Mean NDVI: {ndvi_stats.get('NDVI_mean', 0):.3f}")
    print(f"  Min NDVI: {ndvi_stats.get('NDVI_min', 0):.3f}")
    print(f"  Max NDVI: {ndvi_stats.get('NDVI_max', 0):.3f}")

    # Classify vegetation
    ndvi_mean = ndvi_stats.get('NDVI_mean', 0)
    if ndvi_mean < 0.1:
        veg_class = "Very Low (desert/urban)"
    elif ndvi_mean < 0.2:
        veg_class = "Low (sparse vegetation)"
    elif ndvi_mean < 0.4:
        veg_class = "Moderate (grassland/shrubs)"
    else:
        veg_class = "High (dense vegetation)"
    print(f"  Vegetation class: {veg_class}")

# ============================================================
# 3. BUILT-UP INDEX (NDBI)
# ============================================================
print("\n3. BUILT-UP INDEX (NDBI)")
print("-"*40)

def calculate_ndbi(image):
    """Calculate Normalized Difference Built-up Index."""
    # NDBI = (SWIR - NIR) / (SWIR + NIR)
    ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')
    return image.addBands(ndbi)

sentinel2_ndbi = sentinel2.map(calculate_ndbi)
ndbi_median = sentinel2_ndbi.select('NDBI').median()

ndbi_stats = ndbi_median.reduceRegion(
    reducer=ee.Reducer.mean().combine(
        ee.Reducer.minMax(), sharedInputs=True
    ),
    geometry=AL_KARAMA,
    scale=20,
    maxPixels=1e9
).getInfo()

print(f"  Mean NDBI: {ndbi_stats.get('NDBI_mean', 0):.3f}")
print(f"  Min NDBI: {ndbi_stats.get('NDBI_min', 0):.3f}")
print(f"  Max NDBI: {ndbi_stats.get('NDBI_max', 0):.3f}")

ndbi_mean = ndbi_stats.get('NDBI_mean', 0)
if ndbi_mean > 0.1:
    print(f"  ⚠️ High built-up density (NDBI > 0.1)")

# ============================================================
# 4. URBAN HEAT ISLAND ANALYSIS
# ============================================================
print("\n4. URBAN HEAT ISLAND POTENTIAL")
print("-"*40)

# UHI indicators
print("  Indicators:")
print(f"    - High impervious surface (NDBI): {'Yes' if ndbi_mean > 0 else 'No'}")
print(f"    - Low vegetation (NDVI < 0.2): {'Yes' if ndvi_stats.get('NDVI_mean', 0) < 0.2 else 'No'}")
if landsat_count > 0:
    lst_range = lst_stats.get('LST_max', 0) - lst_stats.get('LST_min', 0)
    print(f"    - LST variation: {lst_range:.1f}°C")
    print(f"    - Potential hotspots: Areas with LST > {lst_stats.get('LST_mean', 0) + lst_stats.get('LST_stdDev', 0):.1f}°C")

# ============================================================
# 5. EXPORT VISUALIZATION MAP
# ============================================================
print("\n5. CREATING VISUALIZATION...")
print("-"*40)

# Create HTML map with results
html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Satellite Analysis - Al Karama</title>
    <meta charset="utf-8">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; font-family: Arial, sans-serif; }}
        #map {{ width: 100%; height: 100vh; }}
        .info-panel {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(255,255,255,0.95);
            padding: 20px;
            border-radius: 8px;
            z-index: 1000;
            max-width: 350px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }}
        .info-panel h2 {{ margin: 0 0 15px 0; color: #d32f2f; }}
        .metric {{
            margin: 15px 0;
            padding: 15px;
            background: #f5f5f5;
            border-radius: 8px;
            border-left: 4px solid #1976d2;
        }}
        .metric h4 {{ margin: 0 0 8px 0; color: #333; }}
        .metric .value {{ font-size: 28px; font-weight: bold; }}
        .metric .unit {{ font-size: 14px; color: #666; }}
        .hot {{ border-left-color: #d32f2f; }}
        .hot .value {{ color: #d32f2f; }}
        .green {{ border-left-color: #388e3c; }}
        .green .value {{ color: #388e3c; }}
        .built {{ border-left-color: #7b1fa2; }}
        .built .value {{ color: #7b1fa2; }}
        .source {{ font-size: 11px; color: #888; margin-top: 15px; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="info-panel">
        <h2>🛰️ Satellite Analysis</h2>
        <p>Al Karama, Dubai | 2024 Data</p>

        <div class="metric hot">
            <h4>🌡️ Land Surface Temperature</h4>
            <div class="value">{lst_stats.get('LST_mean', 0):.1f}<span class="unit">°C</span></div>
            <div>Range: {lst_stats.get('LST_min', 0):.1f}° - {lst_stats.get('LST_max', 0):.1f}°C</div>
            <div style="font-size:12px;color:#666">Summer 2024 (Jun-Sep)</div>
        </div>

        <div class="metric green">
            <h4>🌿 Vegetation Index (NDVI)</h4>
            <div class="value">{ndvi_stats.get('NDVI_mean', 0):.2f}</div>
            <div>{veg_class}</div>
        </div>

        <div class="metric built">
            <h4>🏢 Built-up Index (NDBI)</h4>
            <div class="value">{ndbi_stats.get('NDBI_mean', 0):.2f}</div>
            <div>{'High density urban' if ndbi_mean > 0.1 else 'Moderate density'}</div>
        </div>

        <div class="metric">
            <h4>⚠️ Urban Heat Island Risk</h4>
            <div class="value">{'High' if ndbi_mean > 0 and ndvi_stats.get('NDVI_mean', 0) < 0.2 else 'Moderate'}</div>
            <div>Low vegetation + high impervious surface</div>
        </div>

        <div class="source">
            Data: Landsat 8/9 (thermal), Sentinel-2 (vegetation)<br>
            Source: USGS, ESA Copernicus via Google Earth Engine
        </div>
    </div>

    <script>
        const map = L.map('map').setView([25.245, 55.305], 15);

        // Satellite basemap
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            maxZoom: 19
        }}).addTo(map);

        // Al Karama boundary
        L.rectangle([[25.230, 55.290], [25.260, 55.320]], {{
            color: '#d32f2f',
            weight: 3,
            fill: false
        }}).addTo(map);

        // Add labels
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_only_labels/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            maxZoom: 19
        }}).addTo(map);
    </script>
</body>
</html>"""

html_path = os.path.join(output_dir, "satellite_analysis.html")
with open(html_path, 'w') as f:
    f.write(html_content)

print(f"  Saved: {html_path}")

# Save summary data
summary = {
    'area': 'Al Karama, Dubai',
    'analysis_date': '2024',
    'landsat_images': landsat_count,
    'sentinel2_images': s2_count,
    'lst_mean_celsius': lst_stats.get('LST_mean') if landsat_count > 0 else None,
    'lst_min_celsius': lst_stats.get('LST_min') if landsat_count > 0 else None,
    'lst_max_celsius': lst_stats.get('LST_max') if landsat_count > 0 else None,
    'ndvi_mean': ndvi_stats.get('NDVI_mean'),
    'ndvi_min': ndvi_stats.get('NDVI_min'),
    'ndvi_max': ndvi_stats.get('NDVI_max'),
    'ndbi_mean': ndbi_stats.get('NDBI_mean'),
    'vegetation_class': veg_class,
    'uhi_risk': 'High' if ndbi_mean > 0 and ndvi_stats.get('NDVI_mean', 0) < 0.2 else 'Moderate'
}

json_path = os.path.join(output_dir, "satellite_summary.json")
with open(json_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"  Saved: {json_path}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"\nKey Findings for Al Karama:")
print(f"  🌡️ Summer surface temperature: {lst_stats.get('LST_mean', 0):.1f}°C average")
print(f"  🌿 Vegetation coverage: {veg_class}")
print(f"  🏢 Urban density: {'High' if ndbi_mean > 0.1 else 'Moderate'}")
print(f"  ⚠️ Heat island risk: {'High' if ndbi_mean > 0 and ndvi_stats.get('NDVI_mean', 0) < 0.2 else 'Moderate'}")

print(f"\nOpen visualization: {html_path}")
