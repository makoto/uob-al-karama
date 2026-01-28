"""
Combined Urban Climate Analysis
Overlays satellite thermal data with street-level GVI/SVF data.
"""

import ee
import os
import json
import pandas as pd
import numpy as np

# Initialize Earth Engine
ee.Initialize(project='uobdubai')
print("✅ Connected to Google Earth Engine")

output_dir = "output/combined_analysis"
os.makedirs(output_dir, exist_ok=True)

# Load street-level data
print("\nLoading street-level data...")
gvi_svf = pd.read_csv("output/gvi_svf_combined.csv")
print(f"  Loaded {len(gvi_svf)} street view locations")

# Filter to valid coordinates
gvi_svf = gvi_svf.dropna(subset=['lat', 'lon', 'gvi', 'svf'])
print(f"  Valid records: {len(gvi_svf)}")

# ============================================================
# SAMPLE SATELLITE DATA AT STREET VIEW LOCATIONS
# ============================================================
print("\n" + "="*60)
print("SAMPLING SATELLITE DATA AT STREET VIEW LOCATIONS")
print("="*60)

# Create feature collection from street view points
# Process in batches to avoid memory issues
batch_size = 500
all_results = []

# Prepare satellite imagery
print("\nPreparing satellite composites...")

# Landsat LST composite (summer 2024)
def calculate_lst(image):
    thermal = image.select('ST_B10').multiply(0.00341802).add(149.0)
    lst_celsius = thermal.subtract(273.15)
    return image.addBands(lst_celsius.rename('LST'))

landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
    .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')) \
    .filterBounds(ee.Geometry.Rectangle([55.29, 25.23, 55.32, 25.26])) \
    .filterDate('2024-06-01', '2024-09-30') \
    .filter(ee.Filter.lt('CLOUD_COVER', 20)) \
    .map(calculate_lst)

lst_composite = landsat.select('LST').median()

# Sentinel-2 NDVI composite
def calculate_ndvi(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

def mask_clouds_s2(image):
    qa = image.select('QA60')
    cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(cloud_mask)

sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(ee.Geometry.Rectangle([55.29, 25.23, 55.32, 25.26])) \
    .filterDate('2024-01-01', '2024-12-31') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
    .map(mask_clouds_s2) \
    .map(calculate_ndvi)

ndvi_composite = sentinel2.select('NDVI').median()

# Combined image
combined = lst_composite.addBands(ndvi_composite)

print("  LST composite: Landsat 8/9 (Summer 2024)")
print("  NDVI composite: Sentinel-2 (2024)")

# Sample in batches
print(f"\nSampling at {len(gvi_svf)} locations (in batches of {batch_size})...")

for i in range(0, len(gvi_svf), batch_size):
    batch = gvi_svf.iloc[i:i+batch_size]

    # Create points
    features = []
    for idx, row in batch.iterrows():
        point = ee.Geometry.Point([row['lon'], row['lat']])
        feature = ee.Feature(point, {'id': str(row['id'])})
        features.append(feature)

    fc = ee.FeatureCollection(features)

    # Sample the combined image
    sampled = combined.sampleRegions(
        collection=fc,
        scale=30,  # Landsat resolution
        geometries=True
    )

    # Get results
    try:
        results = sampled.getInfo()
        for feat in results['features']:
            props = feat['properties']
            all_results.append({
                'id': props.get('id'),
                'lst': props.get('LST'),
                'ndvi_satellite': props.get('NDVI')
            })
    except Exception as e:
        print(f"  Batch {i//batch_size + 1} error: {e}")

    print(f"  Processed batch {i//batch_size + 1}/{(len(gvi_svf)-1)//batch_size + 1}")

print(f"\nSampled {len(all_results)} locations")

# Merge satellite data with street-level data
sat_df = pd.DataFrame(all_results)
sat_df['id'] = sat_df['id'].astype(str)
gvi_svf['id'] = gvi_svf['id'].astype(str)

combined_df = gvi_svf.merge(sat_df, on='id', how='left')
combined_df = combined_df.dropna(subset=['lst'])
print(f"Combined dataset: {len(combined_df)} records with all data")

# ============================================================
# CORRELATION ANALYSIS
# ============================================================
print("\n" + "="*60)
print("CORRELATION ANALYSIS")
print("="*60)

# Calculate correlations
correlations = {
    'GVI vs LST': combined_df['gvi'].corr(combined_df['lst']),
    'SVF vs LST': combined_df['svf'].corr(combined_df['lst']),
    'GVI vs NDVI (satellite)': combined_df['gvi'].corr(combined_df['ndvi_satellite']),
    'SVF vs NDVI (satellite)': combined_df['svf'].corr(combined_df['ndvi_satellite']),
}

print("\nCorrelation coefficients:")
for name, corr in correlations.items():
    interpretation = "strong" if abs(corr) > 0.5 else "moderate" if abs(corr) > 0.3 else "weak"
    direction = "positive" if corr > 0 else "negative"
    print(f"  {name}: {corr:.3f} ({interpretation} {direction})")

# Key finding
gvi_lst_corr = correlations['GVI vs LST']
if gvi_lst_corr < -0.1:
    print(f"\n  ✅ Higher vegetation (GVI) associated with LOWER surface temperature")
elif gvi_lst_corr > 0.1:
    print(f"\n  ⚠️ Unexpected: Higher GVI associated with higher temperature")
else:
    print(f"\n  ℹ️ Weak relationship between street-level vegetation and surface temperature")

# ============================================================
# STATISTICS
# ============================================================
print("\n" + "="*60)
print("COMBINED STATISTICS")
print("="*60)

stats = {
    'count': len(combined_df),
    'gvi_mean': combined_df['gvi'].mean(),
    'svf_mean': combined_df['svf'].mean(),
    'lst_mean': combined_df['lst'].mean(),
    'lst_std': combined_df['lst'].std(),
    'ndvi_satellite_mean': combined_df['ndvi_satellite'].mean(),
}

print(f"\nStreet-level metrics (from SVI):")
print(f"  GVI (vegetation visible): {stats['gvi_mean']*100:.2f}%")
print(f"  SVF (sky visible): {stats['svf_mean']*100:.2f}%")

print(f"\nSatellite metrics (from Landsat/Sentinel):")
print(f"  Land Surface Temperature: {stats['lst_mean']:.1f}°C (±{stats['lst_std']:.1f}°C)")
print(f"  NDVI (vegetation index): {stats['ndvi_satellite_mean']:.3f}")

# Identify hot spots and cool spots
hot_threshold = combined_df['lst'].quantile(0.9)
cool_threshold = combined_df['lst'].quantile(0.1)

hotspots = combined_df[combined_df['lst'] >= hot_threshold]
coolspots = combined_df[combined_df['lst'] <= cool_threshold]

print(f"\nHotspots (top 10% by temperature, >{hot_threshold:.1f}°C):")
print(f"  Count: {len(hotspots)}")
print(f"  Average GVI: {hotspots['gvi'].mean()*100:.2f}%")
print(f"  Average SVF: {hotspots['svf'].mean()*100:.2f}%")

print(f"\nCoolspots (bottom 10% by temperature, <{cool_threshold:.1f}°C):")
print(f"  Count: {len(coolspots)}")
print(f"  Average GVI: {coolspots['gvi'].mean()*100:.2f}%")
print(f"  Average SVF: {coolspots['svf'].mean()*100:.2f}%")

# Save combined data
combined_df.to_csv(os.path.join(output_dir, "combined_data.csv"), index=False)

# ============================================================
# CREATE VISUALIZATION
# ============================================================
print("\n" + "="*60)
print("CREATING VISUALIZATION")
print("="*60)

# Sample for visualization (limit points for performance)
viz_df = combined_df.sample(min(3000, len(combined_df)), random_state=42)

html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Combined Urban Climate Analysis - Al Karama</title>
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
            max-width: 380px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            max-height: 90vh;
            overflow-y: auto;
        }}
        .info-panel h2 {{ margin: 0 0 15px 0; color: #1565c0; }}
        .section {{ margin: 15px 0; padding: 12px; background: #f5f5f5; border-radius: 8px; }}
        .section h4 {{ margin: 0 0 10px 0; color: #333; }}
        .correlation {{ display: flex; justify-content: space-between; margin: 5px 0; }}
        .corr-value {{ font-weight: bold; }}
        .negative {{ color: #2e7d32; }}
        .positive {{ color: #c62828; }}
        .weak {{ color: #757575; }}
        .finding {{ padding: 10px; background: #e3f2fd; border-radius: 5px; margin: 10px 0; }}
        .legend {{ margin-top: 15px; }}
        .legend-title {{ font-weight: bold; margin-bottom: 8px; }}
        .legend-gradient {{
            height: 15px;
            border-radius: 3px;
            margin: 5px 0;
        }}
        .legend-labels {{ display: flex; justify-content: space-between; font-size: 11px; }}
        .layer-controls {{ margin: 15px 0; }}
        .layer-btn {{
            padding: 8px 12px;
            margin: 3px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }}
        .layer-btn.active {{ background: #1565c0; color: white; }}
        .layer-btn:not(.active) {{ background: #e0e0e0; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="info-panel">
        <h2>🌡️🌿 Combined Analysis</h2>
        <p>Street-level + Satellite Data</p>

        <div class="layer-controls">
            <button class="layer-btn active" onclick="showLayer('lst')">🌡️ Temperature</button>
            <button class="layer-btn" onclick="showLayer('gvi')">🌿 GVI</button>
            <button class="layer-btn" onclick="showLayer('svf')">☀️ SVF</button>
        </div>

        <div class="section">
            <h4>📊 Correlations</h4>
            <div class="correlation">
                <span>GVI vs Temperature:</span>
                <span class="corr-value {'negative' if gvi_lst_corr < -0.1 else 'positive' if gvi_lst_corr > 0.1 else 'weak'}">{correlations['GVI vs LST']:.3f}</span>
            </div>
            <div class="correlation">
                <span>SVF vs Temperature:</span>
                <span class="corr-value {'negative' if correlations['SVF vs LST'] < -0.1 else 'positive' if correlations['SVF vs LST'] > 0.1 else 'weak'}">{correlations['SVF vs LST']:.3f}</span>
            </div>
            <div class="correlation">
                <span>GVI vs NDVI (satellite):</span>
                <span class="corr-value">{correlations['GVI vs NDVI (satellite)']:.3f}</span>
            </div>
        </div>

        <div class="finding">
            <b>Key Finding:</b><br>
            {'✅ Areas with more street-level vegetation (GVI) tend to have lower surface temperatures' if gvi_lst_corr < -0.1 else '⚠️ Weak relationship between street vegetation and surface temperature - may need larger sample or finer resolution'}
        </div>

        <div class="section">
            <h4>🔥 Hotspots vs Coolspots</h4>
            <table style="width:100%;font-size:12px">
                <tr><th></th><th>Hotspots</th><th>Coolspots</th></tr>
                <tr><td>Temperature</td><td>>{hot_threshold:.1f}°C</td><td>&lt;{cool_threshold:.1f}°C</td></tr>
                <tr><td>Avg GVI</td><td>{hotspots['gvi'].mean()*100:.1f}%</td><td>{coolspots['gvi'].mean()*100:.1f}%</td></tr>
                <tr><td>Avg SVF</td><td>{hotspots['svf'].mean()*100:.1f}%</td><td>{coolspots['svf'].mean()*100:.1f}%</td></tr>
            </table>
        </div>

        <div class="legend" id="legend-lst">
            <div class="legend-title">Land Surface Temperature</div>
            <div class="legend-gradient" style="background: linear-gradient(to right, #313695, #4575b4, #74add1, #abd9e9, #fee090, #fdae61, #f46d43, #d73027, #a50026);"></div>
            <div class="legend-labels"><span>{combined_df['lst'].min():.0f}°C</span><span>{combined_df['lst'].max():.0f}°C</span></div>
        </div>

        <div class="legend" id="legend-gvi" style="display:none">
            <div class="legend-title">Green View Index</div>
            <div class="legend-gradient" style="background: linear-gradient(to right, #fff5f0, #fee0d2, #fcbba1, #fc9272, #fb6a4a, #ef3b2c, #cb181d, #99000d);"></div>
            <div class="legend-labels"><span>0%</span><span>30%+</span></div>
        </div>

        <p style="font-size:11px;color:#666;margin-top:15px">
            Data: {len(combined_df)} locations | Landsat 8/9 thermal + Sentinel-2 NDVI + Street View imagery
        </p>
    </div>

    <script>
        const map = L.map('map').setView([25.245, 55.305], 15);

        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            maxZoom: 19
        }}).addTo(map);

        const data = {viz_df[['lat', 'lon', 'gvi', 'svf', 'lst']].to_json(orient='records')};

        // Color functions
        function getLSTColor(lst) {{
            const min = {combined_df['lst'].min()};
            const max = {combined_df['lst'].max()};
            const ratio = (lst - min) / (max - min);

            // RdYlBu reversed (blue=cool, red=hot)
            const colors = [
                [49, 54, 149], [69, 117, 180], [116, 173, 209], [171, 217, 233],
                [254, 224, 144], [253, 174, 97], [244, 109, 67], [215, 48, 39], [165, 0, 38]
            ];
            const idx = Math.min(Math.floor(ratio * (colors.length - 1)), colors.length - 2);
            const t = ratio * (colors.length - 1) - idx;
            const c1 = colors[idx], c2 = colors[idx + 1];
            const r = Math.round(c1[0] + t * (c2[0] - c1[0]));
            const g = Math.round(c1[1] + t * (c2[1] - c1[1]));
            const b = Math.round(c1[2] + t * (c2[2] - c1[2]));
            return `rgb(${{r}},${{g}},${{b}})`;
        }}

        function getGVIColor(gvi) {{
            const ratio = Math.min(gvi / 0.3, 1);
            const r = Math.round(255 - ratio * 100);
            const g = Math.round(255 - ratio * 200);
            const b = Math.round(255 - ratio * 240);
            return `rgb(${{r}},${{g}},${{b}})`;
        }}

        function getSVFColor(svf) {{
            const ratio = svf;
            return `rgb(${{Math.round(255 * ratio)}}, ${{Math.round(200 * ratio)}}, ${{Math.round(50 + 150 * ratio)}})`;
        }}

        // Layer groups
        const layers = {{
            lst: L.layerGroup(),
            gvi: L.layerGroup(),
            svf: L.layerGroup()
        }};

        // Create markers for each layer
        data.forEach(d => {{
            // LST layer
            L.circleMarker([d.lat, d.lon], {{
                radius: 4,
                color: getLSTColor(d.lst),
                fillColor: getLSTColor(d.lst),
                fillOpacity: 0.8,
                weight: 0
            }}).bindPopup(`<b>Temperature:</b> ${{d.lst.toFixed(1)}}°C<br><b>GVI:</b> ${{(d.gvi*100).toFixed(1)}}%<br><b>SVF:</b> ${{(d.svf*100).toFixed(1)}}%`).addTo(layers.lst);

            // GVI layer
            L.circleMarker([d.lat, d.lon], {{
                radius: 4,
                color: getGVIColor(d.gvi),
                fillColor: getGVIColor(d.gvi),
                fillOpacity: 0.8,
                weight: 0
            }}).bindPopup(`<b>GVI:</b> ${{(d.gvi*100).toFixed(1)}}%<br><b>Temperature:</b> ${{d.lst.toFixed(1)}}°C`).addTo(layers.gvi);

            // SVF layer
            L.circleMarker([d.lat, d.lon], {{
                radius: 4,
                color: getSVFColor(d.svf),
                fillColor: getSVFColor(d.svf),
                fillOpacity: 0.8,
                weight: 0
            }}).bindPopup(`<b>SVF:</b> ${{(d.svf*100).toFixed(1)}}%<br><b>Temperature:</b> ${{d.lst.toFixed(1)}}°C`).addTo(layers.svf);
        }});

        // Add default layer
        layers.lst.addTo(map);

        // Layer switching
        let currentLayer = 'lst';
        function showLayer(layerName) {{
            map.removeLayer(layers[currentLayer]);
            layers[layerName].addTo(map);
            currentLayer = layerName;

            // Update buttons
            document.querySelectorAll('.layer-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');

            // Update legend
            document.querySelectorAll('.legend').forEach(l => l.style.display = 'none');
            document.getElementById('legend-' + layerName).style.display = 'block';
        }}
    </script>
</body>
</html>"""

html_path = os.path.join(output_dir, "combined_analysis.html")
with open(html_path, 'w') as f:
    f.write(html_content)

print(f"  Saved: {html_path}")
print(f"  Saved: {os.path.join(output_dir, 'combined_data.csv')}")

# Save summary
summary = {
    'total_locations': len(combined_df),
    'correlations': correlations,
    'statistics': stats,
    'hotspots': {
        'threshold': hot_threshold,
        'count': len(hotspots),
        'avg_gvi': hotspots['gvi'].mean(),
        'avg_svf': hotspots['svf'].mean()
    },
    'coolspots': {
        'threshold': cool_threshold,
        'count': len(coolspots),
        'avg_gvi': coolspots['gvi'].mean(),
        'avg_svf': coolspots['svf'].mean()
    }
}

with open(os.path.join(output_dir, "analysis_summary.json"), 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"\nOpen: {html_path}")
