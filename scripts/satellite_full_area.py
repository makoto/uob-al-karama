"""
Satellite analysis for the FULL Al Karama boundary.
Creates a grid-based analysis covering the entire area, not just SVI points.
"""

import ee
import os
import json
import numpy as np

ee.Initialize(project='uobdubai')
print("✅ Connected to Google Earth Engine", flush=True)

output_dir = "output/satellite_full"
os.makedirs(output_dir, exist_ok=True)

# Al Karama boundary (full area)
AL_KARAMA = ee.Geometry.Rectangle([55.290, 25.230, 55.320, 25.260])

print("\n" + "="*60, flush=True)
print("FULL AREA SATELLITE ANALYSIS - AL KARAMA", flush=True)
print("="*60, flush=True)

# ============================================================
# PREPARE SATELLITE COMPOSITES
# ============================================================
print("\nPreparing satellite data...", flush=True)

# Sentinel-2 NDVI
def mask_clouds_s2(image):
    qa = image.select('QA60')
    cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(cloud_mask)

def add_indices(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    return image.addBands(ndvi).addBands(ndbi).addBands(ndwi)

sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(AL_KARAMA) \
    .filterDate('2024-01-01', '2024-12-31') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
    .map(mask_clouds_s2) \
    .map(add_indices)

s2_composite = sentinel2.median().clip(AL_KARAMA)
print(f"  Sentinel-2 images: {sentinel2.size().getInfo()}")

# Landsat LST
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

lst_composite = landsat.select('LST').median().clip(AL_KARAMA)
print(f"  Landsat images: {landsat.size().getInfo()}")

# Combine all layers
combined = s2_composite.select(['NDVI', 'NDBI', 'NDWI']).addBands(lst_composite)

# ============================================================
# CREATE GRID AND SAMPLE
# ============================================================
print("\nCreating analysis grid (10m resolution)...")

# Create a grid of points covering Al Karama
# Approximately 10m spacing
lat_min, lat_max = 25.230, 25.260
lon_min, lon_max = 55.290, 55.320

# 10m ≈ 0.00009 degrees
grid_spacing = 0.0001  # ~11m

lats = np.arange(lat_min, lat_max, grid_spacing)
lons = np.arange(lon_min, lon_max, grid_spacing)

print(f"  Grid: {len(lons)} x {len(lats)} = {len(lons) * len(lats)} cells")

# Sample in batches
all_data = []
batch_size = 1000  # Larger batches for efficiency

points = []
for lat in lats:
    for lon in lons:
        points.append({'lat': lat, 'lon': lon})

import sys
print(f"  Sampling {len(points)} grid points...", flush=True)

total_batches = (len(points) - 1) // batch_size + 1
for i in range(0, len(points), batch_size):
    batch = points[i:i+batch_size]
    batch_num = i // batch_size + 1

    print(f"    Batch {batch_num}/{total_batches}...", end=" ", flush=True)

    features = [ee.Feature(ee.Geometry.Point([p['lon'], p['lat']])) for p in batch]
    fc = ee.FeatureCollection(features)

    sampled = combined.sampleRegions(
        collection=fc,
        scale=10,
        geometries=True
    )

    try:
        results = sampled.getInfo()
        for j, feat in enumerate(results['features']):
            props = feat['properties']
            coords = feat['geometry']['coordinates']
            all_data.append({
                'lon': coords[0],
                'lat': coords[1],
                'ndvi': props.get('NDVI'),
                'ndbi': props.get('NDBI'),
                'ndwi': props.get('NDWI'),
                'lst': props.get('LST')
            })
        print(f"got {len(results['features'])} points", flush=True)
    except Exception as e:
        print(f"error: {e}", flush=True)

print(f"  Sampled {len(all_data)} grid cells", flush=True)

# Filter valid data
valid_data = [d for d in all_data if d['lst'] is not None and d['ndvi'] is not None]
print(f"  Valid cells: {len(valid_data)}", flush=True)

# Save CSV immediately (before statistics/viz which might hang)
print("  Saving CSV...", flush=True)
import csv
csv_path = os.path.join(output_dir, "full_area_data.csv")
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['lat', 'lon', 'lst', 'ndvi', 'ndbi', 'ndwi'])
    writer.writeheader()
    writer.writerows(valid_data)
print(f"  Saved: {csv_path}", flush=True)

# ============================================================
# CALCULATE STATISTICS
# ============================================================
print("\n" + "="*60)
print("FULL AREA STATISTICS")
print("="*60)

import statistics

lsts = [d['lst'] for d in valid_data]
ndvis = [d['ndvi'] for d in valid_data]
ndbis = [d['ndbi'] for d in valid_data]

print(f"\nLand Surface Temperature (Summer 2024):")
print(f"  Mean: {statistics.mean(lsts):.1f}°C")
print(f"  Median: {statistics.median(lsts):.1f}°C")
print(f"  Min: {min(lsts):.1f}°C")
print(f"  Max: {max(lsts):.1f}°C")
print(f"  Std Dev: {statistics.stdev(lsts):.1f}°C")

print(f"\nVegetation Index (NDVI):")
print(f"  Mean: {statistics.mean(ndvis):.3f}")
print(f"  Median: {statistics.median(ndvis):.3f}")
print(f"  Min: {min(ndvis):.3f}")
print(f"  Max: {max(ndvis):.3f}")

print(f"\nBuilt-up Index (NDBI):")
print(f"  Mean: {statistics.mean(ndbis):.3f}")
print(f"  Median: {statistics.median(ndbis):.3f}")

# Classify areas
high_veg = len([d for d in valid_data if d['ndvi'] > 0.2])
low_veg = len([d for d in valid_data if d['ndvi'] < 0.1])
hot_spots = len([d for d in valid_data if d['lst'] > statistics.mean(lsts) + statistics.stdev(lsts)])
cool_spots = len([d for d in valid_data if d['lst'] < statistics.mean(lsts) - statistics.stdev(lsts)])

print(f"\nArea Classification:")
print(f"  High vegetation (NDVI > 0.2): {high_veg} cells ({high_veg/len(valid_data)*100:.1f}%)")
print(f"  Low vegetation (NDVI < 0.1): {low_veg} cells ({low_veg/len(valid_data)*100:.1f}%)")
print(f"  Hotspots (>1 std above mean): {hot_spots} cells ({hot_spots/len(valid_data)*100:.1f}%)")
print(f"  Cool spots (<1 std below mean): {cool_spots} cells ({cool_spots/len(valid_data)*100:.1f}%)")

# ============================================================
# CREATE VISUALIZATION
# ============================================================
print("\n" + "="*60, flush=True)
print("Creating visualization...", flush=True)

# Sample data for visualization (limit to 20k for browser performance)
import random
max_viz_points = 20000
if len(valid_data) > max_viz_points:
    viz_data = random.sample(valid_data, max_viz_points)
    print(f"  Sampled {max_viz_points} points for visualization", flush=True)
else:
    viz_data = valid_data

lst_min = min(lsts)
lst_max = max(lsts)
lst_mean = statistics.mean(lsts)

html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Full Area Satellite Analysis - Al Karama</title>
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
            max-width: 350px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
        .info-panel h2 { margin: 0 0 10px 0; color: #d32f2f; }
        .stats { font-size: 13px; margin: 10px 0; }
        .stats td { padding: 3px 8px; }
        .layer-btn {
            padding: 8px 12px;
            margin: 3px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .layer-btn.active { background: #1565c0; color: white; }
        .layer-btn:not(.active) { background: #e0e0e0; }
        .legend { margin-top: 15px; }
        .legend-gradient { height: 15px; border-radius: 3px; margin: 5px 0; }
        .legend-labels { display: flex; justify-content: space-between; font-size: 11px; }
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="info-panel">
        <h2>Full Area Analysis</h2>
        <p>Al Karama (entire boundary)</p>
        <p>Grid cells: """ + str(len(valid_data)) + """ (~30m resolution)</p>

        <div style="margin: 15px 0;">
            <button class="layer-btn active" id="btn-lst" onclick="showLayer('lst')">Temperature</button>
            <button class="layer-btn" id="btn-ndvi" onclick="showLayer('ndvi')">Vegetation</button>
            <button class="layer-btn" id="btn-ndbi" onclick="showLayer('ndbi')">Built-up</button>
        </div>

        <table class="stats">
            <tr><td><b>LST Mean:</b></td><td>""" + f"{lst_mean:.1f}" + """°C</td></tr>
            <tr><td><b>LST Range:</b></td><td>""" + f"{lst_min:.1f} - {lst_max:.1f}" + """°C</td></tr>
            <tr><td><b>NDVI Mean:</b></td><td>""" + f"{statistics.mean(ndvis):.3f}" + """</td></tr>
            <tr><td><b>NDBI Mean:</b></td><td>""" + f"{statistics.mean(ndbis):.3f}" + """</td></tr>
        </table>

        <table class="stats">
            <tr><td>High vegetation:</td><td>""" + f"{high_veg/len(valid_data)*100:.1f}%" + """</td></tr>
            <tr><td>Low vegetation:</td><td>""" + f"{low_veg/len(valid_data)*100:.1f}%" + """</td></tr>
            <tr><td>Hotspots:</td><td>""" + f"{hot_spots/len(valid_data)*100:.1f}%" + """</td></tr>
        </table>

        <div class="legend" id="legend-lst">
            <div><b>Temperature (°C)</b></div>
            <div class="legend-gradient" style="background: linear-gradient(to right, #313695, #4575b4, #abd9e9, #fee090, #f46d43, #a50026);"></div>
            <div class="legend-labels"><span>""" + f"{lst_min:.0f}" + """</span><span>""" + f"{lst_max:.0f}" + """</span></div>
        </div>
        <div class="legend" id="legend-ndvi" style="display:none">
            <div><b>NDVI (Vegetation)</b></div>
            <div class="legend-gradient" style="background: linear-gradient(to right, #8B4513, #D2691E, #F5DEB3, #90EE90, #228B22, #006400);"></div>
            <div class="legend-labels"><span>-0.2</span><span>0.6+</span></div>
        </div>
        <div class="legend" id="legend-ndbi" style="display:none">
            <div><b>NDBI (Built-up)</b></div>
            <div class="legend-gradient" style="background: linear-gradient(to right, #1a9850, #d9ef8b, #fee08b, #d73027);"></div>
            <div class="legend-labels"><span>-0.3</span><span>0.3+</span></div>
        </div>
    </div>

    <script>
        var map = L.map('map').setView([25.245, 55.305], 15);

        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            maxZoom: 19
        }).addTo(map);

        var data = """ + json.dumps(viz_data) + """;

        var lstMin = """ + str(lst_min) + """;
        var lstMax = """ + str(lst_max) + """;

        function getLSTColor(v) {
            var ratio = (v - lstMin) / (lstMax - lstMin);
            var colors = [[49,54,149],[69,117,180],[171,217,233],[254,224,144],[244,109,67],[165,0,38]];
            var idx = Math.min(Math.floor(ratio * 5), 4);
            var t = (ratio * 5) - idx;
            var c1 = colors[idx], c2 = colors[idx+1];
            return 'rgb(' + Math.round(c1[0]+(c2[0]-c1[0])*t) + ',' +
                          Math.round(c1[1]+(c2[1]-c1[1])*t) + ',' +
                          Math.round(c1[2]+(c2[2]-c1[2])*t) + ')';
        }

        function getNDVIColor(v) {
            if (v < 0) return '#8B4513';
            if (v < 0.1) return '#D2691E';
            if (v < 0.2) return '#F5DEB3';
            if (v < 0.3) return '#90EE90';
            if (v < 0.4) return '#228B22';
            return '#006400';
        }

        function getNDBIColor(v) {
            if (v < -0.1) return '#1a9850';
            if (v < 0) return '#91cf60';
            if (v < 0.1) return '#fee08b';
            if (v < 0.2) return '#fc8d59';
            return '#d73027';
        }

        var layers = { lst: L.layerGroup(), ndvi: L.layerGroup(), ndbi: L.layerGroup() };

        data.forEach(function(d) {
            var size = 8;

            L.rectangle([[d.lat-0.00005, d.lon-0.00005], [d.lat+0.00005, d.lon+0.00005]], {
                color: getLSTColor(d.lst), fillColor: getLSTColor(d.lst), fillOpacity: 0.8, weight: 0
            }).bindPopup('LST: ' + d.lst.toFixed(1) + '°C<br>NDVI: ' + d.ndvi.toFixed(3)).addTo(layers.lst);

            L.rectangle([[d.lat-0.00005, d.lon-0.00005], [d.lat+0.00005, d.lon+0.00005]], {
                color: getNDVIColor(d.ndvi), fillColor: getNDVIColor(d.ndvi), fillOpacity: 0.8, weight: 0
            }).bindPopup('NDVI: ' + d.ndvi.toFixed(3) + '<br>LST: ' + d.lst.toFixed(1) + '°C').addTo(layers.ndvi);

            L.rectangle([[d.lat-0.00005, d.lon-0.00005], [d.lat+0.00005, d.lon+0.00005]], {
                color: getNDBIColor(d.ndbi), fillColor: getNDBIColor(d.ndbi), fillOpacity: 0.8, weight: 0
            }).bindPopup('NDBI: ' + d.ndbi.toFixed(3) + '<br>LST: ' + d.lst.toFixed(1) + '°C').addTo(layers.ndbi);
        });

        layers.lst.addTo(map);
        var currentLayer = 'lst';

        function showLayer(name) {
            map.removeLayer(layers[currentLayer]);
            layers[name].addTo(map);
            currentLayer = name;

            ['lst','ndvi','ndbi'].forEach(function(n) {
                document.getElementById('btn-' + n).className = 'layer-btn' + (n === name ? ' active' : '');
                document.getElementById('legend-' + n).style.display = n === name ? 'block' : 'none';
            });
        }
    </script>
</body>
</html>"""

html_path = os.path.join(output_dir, "full_area_analysis.html")
with open(html_path, 'w') as f:
    f.write(html_content)

# Save data
import csv
csv_path = os.path.join(output_dir, "full_area_data.csv")
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['lat', 'lon', 'lst', 'ndvi', 'ndbi', 'ndwi'])
    writer.writeheader()
    writer.writerows(valid_data)

print(f"  Saved: {html_path}")
print(f"  Saved: {csv_path}")

print("\n" + "="*60)
print("COMPLETE")
print("="*60)
