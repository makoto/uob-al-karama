"""
Urban Climate & Environmental Analysis
Extract climate-relevant metrics from existing SVI + segmentation data.
"""

import os
import glob
import pandas as pd
import numpy as np
import json
from math import atan2, degrees

output_dir = "output/climate_analysis"
os.makedirs(output_dir, exist_ok=True)

print("="*60)
print("URBAN CLIMATE & ENVIRONMENTAL ANALYSIS")
print("="*60)

# Load existing data
print("\nLoading data...")
gvi_svf = pd.read_csv("output/gvi_svf_combined.csv")
print(f"Loaded {len(gvi_svf)} image records with GVI/SVF")

# Load all segmentation data for detailed analysis
seg_data = []
for csv_path in glob.glob("data/segmentation/summary/batch_*/pixel_ratios.csv"):
    df = pd.read_csv(csv_path)
    seg_data.append(df)

if seg_data:
    all_seg = pd.concat(seg_data, ignore_index=True)
    print(f"Loaded {len(all_seg)} segmentation records")
else:
    all_seg = pd.DataFrame()
    print("No segmentation data found")

# Pivot segmentation data to get one row per image with all class ratios
if not all_seg.empty:
    seg_pivot = all_seg.pivot_table(
        index='filename_key',
        columns='label_name',
        values='pixel_ratios',
        aggfunc='first'
    ).reset_index()
    seg_pivot.columns.name = None
    print(f"Pivoted to {len(seg_pivot)} images with {len(seg_pivot.columns)-1} classes")

# ============================================================
# 1. GREEN INFRASTRUCTURE ANALYSIS
# ============================================================
print("\n" + "="*60)
print("1. GREEN INFRASTRUCTURE")
print("="*60)

# Already have GVI, but let's break it down further
green_classes = ['Vegetation', 'Terrain']  # From Mapillary/Cityscapes

if not all_seg.empty:
    vegetation = all_seg[all_seg['label_name'] == 'Vegetation'].groupby('filename_key')['pixel_ratios'].sum()
    terrain = all_seg[all_seg['label_name'] == 'Terrain'].groupby('filename_key')['pixel_ratios'].sum()

    print(f"\nVegetation visibility:")
    print(f"  Mean: {vegetation.mean()*100:.2f}%")
    print(f"  Median: {vegetation.median()*100:.2f}%")
    print(f"  Max: {vegetation.max()*100:.2f}%")

    # Classify green infrastructure levels
    green_levels = pd.cut(vegetation, bins=[0, 0.01, 0.05, 0.15, 1.0],
                          labels=['Very Low (<1%)', 'Low (1-5%)', 'Medium (5-15%)', 'High (>15%)'])
    print(f"\nGreen infrastructure distribution:")
    print(green_levels.value_counts())

# ============================================================
# 2. SURFACE MATERIALS & THERMAL PERFORMANCE
# ============================================================
print("\n" + "="*60)
print("2. SURFACE MATERIALS & THERMAL PERFORMANCE")
print("="*60)

# Material classes and their thermal properties (approximate)
material_thermal = {
    'Road': {'albedo': 0.1, 'thermal_mass': 'high', 'heat_risk': 'high'},
    'Sidewalk': {'albedo': 0.2, 'thermal_mass': 'high', 'heat_risk': 'medium'},
    'Building': {'albedo': 0.3, 'thermal_mass': 'high', 'heat_risk': 'medium'},
    'Vegetation': {'albedo': 0.2, 'thermal_mass': 'low', 'heat_risk': 'low'},
    'Terrain': {'albedo': 0.25, 'thermal_mass': 'medium', 'heat_risk': 'medium'},
    'Sky': {'albedo': None, 'thermal_mass': None, 'heat_risk': None},
}

if not all_seg.empty:
    # Calculate surface material breakdown
    surface_classes = ['Road', 'Sidewalk', 'Building', 'Vegetation', 'Terrain']
    surface_data = all_seg[all_seg['label_name'].isin(surface_classes)]
    surface_summary = surface_data.groupby('label_name')['pixel_ratios'].mean() * 100

    print("\nAverage surface material visibility:")
    for material, pct in surface_summary.sort_values(ascending=False).items():
        thermal = material_thermal.get(material, {})
        heat_risk = thermal.get('heat_risk', 'unknown')
        print(f"  {material}: {pct:.1f}% (heat risk: {heat_risk})")

    # Calculate heat stress index based on surface materials
    # Higher impervious surface = higher heat stress potential
    impervious = all_seg[all_seg['label_name'].isin(['Road', 'Sidewalk', 'Building'])]
    impervious_ratio = impervious.groupby('filename_key')['pixel_ratios'].sum()

    print(f"\nImpervious surface ratio:")
    print(f"  Mean: {impervious_ratio.mean()*100:.1f}%")
    print(f"  This indicates potential for urban heat island effect")

# ============================================================
# 3. SHADING & SOLAR EXPOSURE POTENTIAL
# ============================================================
print("\n" + "="*60)
print("3. SHADING & SOLAR EXPOSURE")
print("="*60)

print("\nUsing Sky View Factor (SVF) as proxy for solar exposure:")
print(f"  Mean SVF: {gvi_svf['svf'].mean()*100:.1f}%")
print(f"  Low SVF (<20%): More shading from buildings")
print(f"  High SVF (>40%): More direct sun exposure")

# Classify shading conditions
svf_values = gvi_svf['svf'].dropna()
shading_levels = pd.cut(svf_values, bins=[0, 0.2, 0.3, 0.4, 1.0],
                        labels=['Heavy shade (<20%)', 'Moderate shade (20-30%)',
                                'Light shade (30-40%)', 'Exposed (>40%)'])
print(f"\nShading distribution:")
print(shading_levels.value_counts())

# ============================================================
# 4. STREET ORIENTATION & VENTILATION
# ============================================================
print("\n" + "="*60)
print("4. STREET ORIENTATION & VENTILATION POTENTIAL")
print("="*60)

# Calculate street orientations from consecutive image positions
coords = gvi_svf[['id', 'lat', 'lon']].dropna()
coords = coords.sort_values('id')

# Calculate bearings between consecutive points
orientations = []
for i in range(len(coords)-1):
    lat1, lon1 = coords.iloc[i]['lat'], coords.iloc[i]['lon']
    lat2, lon2 = coords.iloc[i+1]['lat'], coords.iloc[i+1]['lon']

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    if abs(dlat) > 0.00001 or abs(dlon) > 0.00001:  # Minimum movement
        bearing = degrees(atan2(dlon, dlat)) % 360
        orientations.append(bearing)

if orientations:
    orientations = np.array(orientations)

    # Classify into cardinal directions
    def classify_orientation(bearing):
        if bearing < 22.5 or bearing >= 337.5:
            return 'N-S'
        elif 22.5 <= bearing < 67.5:
            return 'NE-SW'
        elif 67.5 <= bearing < 112.5:
            return 'E-W'
        elif 112.5 <= bearing < 157.5:
            return 'SE-NW'
        elif 157.5 <= bearing < 202.5:
            return 'N-S'
        elif 202.5 <= bearing < 247.5:
            return 'NE-SW'
        elif 247.5 <= bearing < 292.5:
            return 'E-W'
        else:
            return 'SE-NW'

    orientation_classes = [classify_orientation(b) for b in orientations]
    orientation_counts = pd.Series(orientation_classes).value_counts()

    print("\nStreet orientation distribution:")
    for orient, count in orientation_counts.items():
        pct = count / len(orientation_classes) * 100
        # Dubai prevailing wind is NW (Shamal)
        if orient == 'SE-NW':
            note = "✓ Good for Shamal wind ventilation"
        elif orient == 'E-W':
            note = "⚠ Morning/evening sun exposure"
        else:
            note = ""
        print(f"  {orient}: {pct:.1f}% {note}")

# ============================================================
# 5. ENVIRONMENTAL COMFORT INDEX
# ============================================================
print("\n" + "="*60)
print("5. ENVIRONMENTAL COMFORT INDEX")
print("="*60)

# Create composite comfort score
# Higher = better comfort (more shade, more green, less impervious)

comfort_data = gvi_svf[['id', 'lat', 'lon', 'gvi', 'svf']].copy()

# Shade score: Lower SVF = more shade = better in hot climate (inverse)
comfort_data['shade_score'] = 1 - comfort_data['svf'].fillna(0.5)

# Green score: Higher GVI = more vegetation = better
comfort_data['green_score'] = comfort_data['gvi'].fillna(0)

# Combine into comfort index (0-100)
comfort_data['comfort_index'] = (
    comfort_data['shade_score'] * 50 +  # 50% weight for shade
    comfort_data['green_score'] * 50    # 50% weight for greenery
) * 100

print(f"\nEnvironmental Comfort Index (0-100):")
print(f"  Mean: {comfort_data['comfort_index'].mean():.1f}")
print(f"  Median: {comfort_data['comfort_index'].median():.1f}")
print(f"  Range: {comfort_data['comfort_index'].min():.1f} - {comfort_data['comfort_index'].max():.1f}")

# Classify comfort levels
comfort_levels = pd.cut(comfort_data['comfort_index'], bins=[0, 25, 50, 75, 100],
                        labels=['Poor', 'Fair', 'Good', 'Excellent'])
print(f"\nComfort distribution:")
print(comfort_levels.value_counts())

# Save comfort data
comfort_data.to_csv(os.path.join(output_dir, "environmental_comfort.csv"), index=False)

# ============================================================
# 6. SUMMARY & RECOMMENDATIONS
# ============================================================
print("\n" + "="*60)
print("6. SUMMARY & DATA GAPS")
print("="*60)

print("\n✅ AVAILABLE FROM CURRENT DATA:")
print("  - Green View Index (vegetation visibility)")
print("  - Sky View Factor (shading potential)")
print("  - Surface material ratios")
print("  - Street orientation")
print("  - Environmental comfort index (composite)")

print("\n⚠️ NEED ADDITIONAL DATA FOR:")
analysis_gaps = {
    "Heat stress mapping": "Landsat/Sentinel thermal imagery (free from USGS/ESA)",
    "Solar radiation": "Solar path calculation + 3D building geometry",
    "Wind/ventilation CFD": "Detailed 3D model + wind data",
    "Blue infrastructure": "OSM water features + drainage maps",
    "Flood risk": "Dubai Municipality flood maps",
    "Air quality": "UAE air quality monitoring stations",
    "Noise pollution": "Traffic data + acoustic modeling",
}

for analysis, source in analysis_gaps.items():
    print(f"  - {analysis}: {source}")

# ============================================================
# CREATE VISUALIZATION
# ============================================================
print("\n" + "="*60)
print("Creating visualization...")

# Create HTML map with comfort index
html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Urban Climate Analysis - Al Karama</title>
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
            padding: 15px;
            border-radius: 8px;
            z-index: 1000;
            max-width: 320px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
        .info-panel h3 { margin: 0 0 10px 0; color: #1565c0; }
        .metric { margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 5px; }
        .metric h4 { margin: 0 0 5px 0; font-size: 13px; color: #333; }
        .metric .value { font-size: 24px; font-weight: bold; }
        .good { color: #2e7d32; }
        .fair { color: #f57c00; }
        .poor { color: #c62828; }
        .legend { margin-top: 15px; }
        .legend-title { font-weight: bold; margin-bottom: 5px; }
        .legend-item { display: flex; align-items: center; margin: 3px 0; font-size: 12px; }
        .legend-color { width: 20px; height: 20px; margin-right: 8px; border-radius: 3px; }
        .tabs { display: flex; margin-bottom: 10px; }
        .tab { padding: 8px 12px; cursor: pointer; background: #e0e0e0; border-radius: 4px 4px 0 0; margin-right: 2px; }
        .tab.active { background: #1565c0; color: white; }
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="info-panel">
        <h3>🌡️ Urban Climate Analysis</h3>
        <p>Al Karama, Dubai</p>

        <div class="metric">
            <h4>Environmental Comfort Index</h4>
            <div class="value """ + ('good' if comfort_data['comfort_index'].mean() > 50 else 'fair') + """">
                """ + f"{comfort_data['comfort_index'].mean():.0f}" + """/100
            </div>
        </div>

        <div class="metric">
            <h4>Green Infrastructure (GVI)</h4>
            <div class="value """ + ('poor' if gvi_svf['gvi'].mean() < 0.05 else 'fair') + """">
                """ + f"{gvi_svf['gvi'].mean()*100:.1f}%" + """
            </div>
        </div>

        <div class="metric">
            <h4>Shading (100 - SVF)</h4>
            <div class="value """ + ('good' if gvi_svf['svf'].mean() < 0.35 else 'fair') + """">
                """ + f"{(1-gvi_svf['svf'].mean())*100:.0f}%" + """
            </div>
        </div>

        <div class="legend">
            <div class="legend-title">Comfort Index</div>
            <div class="legend-item"><div class="legend-color" style="background:#c62828"></div> Poor (0-25)</div>
            <div class="legend-item"><div class="legend-color" style="background:#f57c00"></div> Fair (25-50)</div>
            <div class="legend-item"><div class="legend-color" style="background:#7cb342"></div> Good (50-75)</div>
            <div class="legend-item"><div class="legend-color" style="background:#2e7d32"></div> Excellent (75-100)</div>
        </div>
    </div>

    <script>
        const map = L.map('map').setView([25.242, 55.305], 15);

        L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
            maxZoom: 19
        }).addTo(map);

        const comfortData = """ + comfort_data.dropna().head(2000).to_json(orient='records') + """;

        function getColor(value) {
            if (value < 25) return '#c62828';
            if (value < 50) return '#f57c00';
            if (value < 75) return '#7cb342';
            return '#2e7d32';
        }

        comfortData.forEach(d => {
            L.circleMarker([d.lat, d.lon], {
                radius: 4,
                color: getColor(d.comfort_index),
                fillColor: getColor(d.comfort_index),
                fillOpacity: 0.7,
                weight: 1
            }).bindPopup(`
                <b>Comfort Index: ${d.comfort_index.toFixed(0)}/100</b><br>
                Green (GVI): ${(d.gvi*100).toFixed(1)}%<br>
                Shade: ${((1-d.svf)*100).toFixed(0)}%
            `).addTo(map);
        });
    </script>
</body>
</html>"""

html_path = os.path.join(output_dir, "urban_climate_map.html")
with open(html_path, 'w') as f:
    f.write(html_content)

print(f"\nSaved: {html_path}")
print(f"Saved: {os.path.join(output_dir, 'environmental_comfort.csv')}")
print("\nDone!")
