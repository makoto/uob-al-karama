"""
Thermal Sharpening: Downscale Landsat LST (30m) to 10m using Sentinel-2 NDVI.
Uses the LST-NDVI relationship to disaggregate thermal data.
"""

import ee
import os
import json
import numpy as np
from scipy import stats

ee.Initialize(project='uobdubai')
print("✅ Connected to Google Earth Engine")

output_dir = "output/thermal_sharpening"
os.makedirs(output_dir, exist_ok=True)

AL_KARAMA = ee.Geometry.Rectangle([55.290, 25.230, 55.320, 25.260])

print("\n" + "="*60)
print("THERMAL SHARPENING: 30m → 10m")
print("="*60)

# ============================================================
# 1. PREPARE DATA
# ============================================================
print("\n1. Preparing satellite data...")

# Sentinel-2 NDVI at 10m
def mask_clouds_s2(image):
    qa = image.select('QA60')
    cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(cloud_mask)

def add_ndvi(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(AL_KARAMA) \
    .filterDate('2024-06-01', '2024-09-30') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
    .map(mask_clouds_s2) \
    .map(add_ndvi)

ndvi_10m = sentinel2.select('NDVI').median().clip(AL_KARAMA)
print(f"  Sentinel-2 NDVI (10m): {sentinel2.size().getInfo()} images")

# Landsat LST at 30m (native 100m)
def calculate_lst(image):
    thermal = image.select('ST_B10').multiply(0.00341802).add(149.0)
    lst_celsius = thermal.subtract(273.15)
    return image.addBands(lst_celsius.rename('LST'))

landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
    .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')) \
    .filterBounds(AL_KARAMA) \
    .filterDate('2024-06-01', '2024-09-30') \
    .filter(ee.Filter.lt('CLOUD_COVER', 20)) \
    .map(calculate_lst)

lst_30m = landsat.select('LST').median().clip(AL_KARAMA)
print(f"  Landsat LST (30m): {landsat.size().getInfo()} images")

# ============================================================
# 2. BUILD REGRESSION MODEL AT 30m
# ============================================================
print("\n2. Building LST-NDVI regression model...")

# Resample NDVI to 30m for regression
# First set default projection, then reduce resolution
ndvi_30m = ndvi_10m.setDefaultProjection(crs='EPSG:4326', scale=10) \
    .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=65536) \
    .reproject(crs='EPSG:4326', scale=30)

# Sample points for regression
sample_points = lst_30m.addBands(ndvi_30m).sample(
    region=AL_KARAMA,
    scale=30,
    numPixels=2000,
    seed=42
)

samples = sample_points.getInfo()
print(f"  Sampled {len(samples['features'])} points for regression")

# Extract values
lst_values = []
ndvi_values = []
for feat in samples['features']:
    props = feat['properties']
    if props.get('LST') is not None and props.get('NDVI') is not None:
        lst_values.append(props['LST'])
        ndvi_values.append(props['NDVI'])

lst_arr = np.array(lst_values)
ndvi_arr = np.array(ndvi_values)

# Linear regression: LST = a + b * NDVI
slope, intercept, r_value, p_value, std_err = stats.linregress(ndvi_arr, lst_arr)

print(f"\n  Regression Model: LST = {intercept:.2f} + ({slope:.2f} × NDVI)")
print(f"  R² = {r_value**2:.3f}")
print(f"  p-value = {p_value:.2e}")
print(f"  Standard Error = {std_err:.2f}°C")

if r_value**2 < 0.3:
    print(f"  ⚠️ Weak correlation - sharpening may be less reliable")
elif r_value**2 < 0.5:
    print(f"  ℹ️ Moderate correlation - reasonable for urban areas")
else:
    print(f"  ✅ Good correlation - sharpening should work well")

# ============================================================
# 3. APPLY SHARPENING
# ============================================================
print("\n3. Applying thermal sharpening...")

# Method: DisTrad-like approach
# 1. Calculate residuals at 30m: residual = LST_observed - LST_predicted
# 2. Add residuals to 10m prediction

# Predicted LST at 30m from regression
lst_predicted_30m = ndvi_30m.multiply(slope).add(intercept)

# Residuals at 30m
residuals_30m = lst_30m.subtract(lst_predicted_30m)

# Predicted LST at 10m from regression
lst_predicted_10m = ndvi_10m.multiply(slope).add(intercept)

# Resample residuals to 10m (bilinear)
residuals_10m = residuals_30m.resample('bilinear').reproject(crs='EPSG:4326', scale=10)

# Final sharpened LST at 10m
lst_sharpened_10m = lst_predicted_10m.add(residuals_10m)

print("  ✅ Sharpening complete")

# ============================================================
# 4. SAMPLE BOTH RESOLUTIONS FOR COMPARISON
# ============================================================
print("\n4. Sampling data for visualization...")

# Create 10m grid
lat_min, lat_max = 25.230, 25.260
lon_min, lon_max = 55.290, 55.320
grid_spacing = 0.0001  # ~11m

lats = np.arange(lat_min, lat_max, grid_spacing)
lons = np.arange(lon_min, lon_max, grid_spacing)

print(f"  Grid: {len(lons)} x {len(lats)} = {len(lons) * len(lats)} cells")

# Sample combined data
combined = lst_30m.rename('lst_30m') \
    .addBands(lst_sharpened_10m.rename('lst_10m')) \
    .addBands(ndvi_10m.rename('ndvi'))

points = []
for lat in lats:
    for lon in lons:
        points.append({'lat': lat, 'lon': lon})

all_data = []
batch_size = 500

print(f"  Sampling {len(points)} points...")

for i in range(0, len(points), batch_size):
    batch = points[i:i+batch_size]
    features = [ee.Feature(ee.Geometry.Point([p['lon'], p['lat']])) for p in batch]
    fc = ee.FeatureCollection(features)

    sampled = combined.sampleRegions(collection=fc, scale=10, geometries=True)

    try:
        results = sampled.getInfo()
        for feat in results['features']:
            props = feat['properties']
            coords = feat['geometry']['coordinates']
            all_data.append({
                'lon': coords[0],
                'lat': coords[1],
                'lst_30m': props.get('lst_30m'),
                'lst_10m': props.get('lst_10m'),
                'ndvi': props.get('ndvi')
            })
    except Exception as e:
        pass

    if (i // batch_size + 1) % 20 == 0:
        print(f"    Processed {i + len(batch)}/{len(points)}")

valid_data = [d for d in all_data if d['lst_10m'] is not None and d['lst_30m'] is not None]
print(f"  Valid cells: {len(valid_data)}")

# ============================================================
# 5. ACCURACY ASSESSMENT
# ============================================================
print("\n" + "="*60)
print("5. ACCURACY ASSESSMENT")
print("="*60)

lst_30m_vals = np.array([d['lst_30m'] for d in valid_data])
lst_10m_vals = np.array([d['lst_10m'] for d in valid_data])

# Statistics
print(f"\n  Original 30m LST:")
print(f"    Mean: {np.mean(lst_30m_vals):.2f}°C")
print(f"    Std Dev: {np.std(lst_30m_vals):.2f}°C")
print(f"    Range: {np.min(lst_30m_vals):.1f} - {np.max(lst_30m_vals):.1f}°C")

print(f"\n  Sharpened 10m LST:")
print(f"    Mean: {np.mean(lst_10m_vals):.2f}°C")
print(f"    Std Dev: {np.std(lst_10m_vals):.2f}°C")
print(f"    Range: {np.min(lst_10m_vals):.1f} - {np.max(lst_10m_vals):.1f}°C")

# Difference
diff = lst_10m_vals - lst_30m_vals
print(f"\n  Difference (10m - 30m):")
print(f"    Mean: {np.mean(diff):.2f}°C")
print(f"    RMSE: {np.sqrt(np.mean(diff**2)):.2f}°C")
print(f"    Range: {np.min(diff):.1f} to {np.max(diff):.1f}°C")

# ============================================================
# 6. CREATE VISUALIZATION
# ============================================================
print("\n6. Creating comparison visualization...")

lst_min = min(np.min(lst_30m_vals), np.min(lst_10m_vals))
lst_max = max(np.max(lst_30m_vals), np.max(lst_10m_vals))

html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Thermal Sharpening Comparison - Al Karama</title>
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
            background: rgba(255,255,255,0.95);
            padding: 20px;
            border-radius: 8px;
            z-index: 1000;
            max-width: 380px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
        .info-panel h2 { margin: 0 0 10px 0; color: #d32f2f; }
        .layer-btn {
            padding: 10px 15px;
            margin: 5px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
        }
        .layer-btn.active { background: #d32f2f; color: white; }
        .layer-btn:not(.active) { background: #e0e0e0; }
        .stats { margin: 15px 0; font-size: 13px; }
        .stats td { padding: 4px 10px; }
        .model { background: #f5f5f5; padding: 10px; border-radius: 5px; margin: 10px 0; font-size: 12px; }
        .legend { margin-top: 15px; }
        .legend-gradient { height: 15px; border-radius: 3px; margin: 5px 0;
            background: linear-gradient(to right, #313695, #4575b4, #abd9e9, #fee090, #f46d43, #a50026); }
        .legend-labels { display: flex; justify-content: space-between; font-size: 11px; }
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="info-panel">
        <h2>Thermal Sharpening</h2>
        <p>30m → 10m Downscaling</p>

        <div style="margin: 15px 0;">
            <button class="layer-btn active" id="btn-10m" onclick="showLayer('10m')">10m Sharpened</button>
            <button class="layer-btn" id="btn-30m" onclick="showLayer('30m')">30m Original</button>
        </div>

        <div class="model">
            <b>Regression Model:</b><br>
            LST = """ + f"{intercept:.1f} + ({slope:.1f} × NDVI)" + """<br>
            R² = """ + f"{r_value**2:.3f}" + """
        </div>

        <table class="stats">
            <tr><th></th><th>30m</th><th>10m</th></tr>
            <tr><td>Mean</td><td>""" + f"{np.mean(lst_30m_vals):.1f}°C" + """</td><td>""" + f"{np.mean(lst_10m_vals):.1f}°C" + """</td></tr>
            <tr><td>Std Dev</td><td>""" + f"{np.std(lst_30m_vals):.1f}°C" + """</td><td>""" + f"{np.std(lst_10m_vals):.1f}°C" + """</td></tr>
            <tr><td>Min</td><td>""" + f"{np.min(lst_30m_vals):.1f}°C" + """</td><td>""" + f"{np.min(lst_10m_vals):.1f}°C" + """</td></tr>
            <tr><td>Max</td><td>""" + f"{np.max(lst_30m_vals):.1f}°C" + """</td><td>""" + f"{np.max(lst_10m_vals):.1f}°C" + """</td></tr>
        </table>

        <p style="font-size:12px;color:#666">
            The 10m sharpened version shows more spatial detail,
            especially around vegetation and buildings.
        </p>

        <div class="legend">
            <div><b>Temperature (°C)</b></div>
            <div class="legend-gradient"></div>
            <div class="legend-labels"><span>""" + f"{lst_min:.0f}" + """</span><span>""" + f"{lst_max:.0f}" + """</span></div>
        </div>
    </div>

    <script>
        var map = L.map('map').setView([25.245, 55.305], 15);

        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            maxZoom: 19
        }).addTo(map);

        var data = """ + json.dumps(valid_data) + """;

        var lstMin = """ + str(lst_min) + """;
        var lstMax = """ + str(lst_max) + """;

        function getColor(v) {
            var ratio = (v - lstMin) / (lstMax - lstMin);
            var colors = [[49,54,149],[69,117,180],[171,217,233],[254,224,144],[244,109,67],[165,0,38]];
            var idx = Math.min(Math.floor(ratio * 5), 4);
            var t = (ratio * 5) - idx;
            var c1 = colors[idx], c2 = colors[idx+1];
            return 'rgb(' + Math.round(c1[0]+(c2[0]-c1[0])*t) + ',' +
                          Math.round(c1[1]+(c2[1]-c1[1])*t) + ',' +
                          Math.round(c1[2]+(c2[2]-c1[2])*t) + ')';
        }

        var layers = { '10m': L.layerGroup(), '30m': L.layerGroup() };

        data.forEach(function(d) {
            L.rectangle([[d.lat-0.00005, d.lon-0.00005], [d.lat+0.00005, d.lon+0.00005]], {
                color: getColor(d.lst_10m), fillColor: getColor(d.lst_10m), fillOpacity: 0.85, weight: 0
            }).bindPopup('10m LST: ' + d.lst_10m.toFixed(1) + '°C<br>30m LST: ' + d.lst_30m.toFixed(1) + '°C<br>NDVI: ' + d.ndvi.toFixed(3)).addTo(layers['10m']);

            L.rectangle([[d.lat-0.00005, d.lon-0.00005], [d.lat+0.00005, d.lon+0.00005]], {
                color: getColor(d.lst_30m), fillColor: getColor(d.lst_30m), fillOpacity: 0.85, weight: 0
            }).bindPopup('30m LST: ' + d.lst_30m.toFixed(1) + '°C<br>10m LST: ' + d.lst_10m.toFixed(1) + '°C').addTo(layers['30m']);
        });

        layers['10m'].addTo(map);
        var currentLayer = '10m';

        function showLayer(name) {
            map.removeLayer(layers[currentLayer]);
            layers[name].addTo(map);
            currentLayer = name;
            document.getElementById('btn-10m').className = 'layer-btn' + (name === '10m' ? ' active' : '');
            document.getElementById('btn-30m').className = 'layer-btn' + (name === '30m' ? ' active' : '');
        }
    </script>
</body>
</html>"""

html_path = os.path.join(output_dir, "thermal_sharpening.html")
with open(html_path, 'w') as f:
    f.write(html_content)

# Save data
import csv
csv_path = os.path.join(output_dir, "sharpened_data.csv")
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['lat', 'lon', 'lst_30m', 'lst_10m', 'ndvi'])
    writer.writeheader()
    writer.writerows(valid_data)

print(f"\n  Saved: {html_path}")
print(f"  Saved: {csv_path}")

print("\n" + "="*60)
print("COMPLETE")
print("="*60)
print(f"\nOpen the map to compare 30m vs 10m: {html_path}")
