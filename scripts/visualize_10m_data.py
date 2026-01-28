#!/usr/bin/env python3
"""Generate visualization from 10m satellite data CSV."""

import pandas as pd
import numpy as np
import random
import json

# Load the CSV
print("Loading CSV data...")
df = pd.read_csv('output/satellite_full/full_area_data.csv')
print(f"  Total rows: {len(df)}")

# Filter valid data
df = df.dropna()
print(f"  Valid rows: {len(df)}")

# Calculate statistics
print("\n" + "="*60)
print("STATISTICS (10m resolution)")
print("="*60)
print(f"\nGrid cells: {len(df)}")
print(f"\nLand Surface Temperature (LST):")
print(f"  Mean: {df['lst'].mean():.1f}°C")
print(f"  Min: {df['lst'].min():.1f}°C")
print(f"  Max: {df['lst'].max():.1f}°C")
print(f"  Std: {df['lst'].std():.1f}°C")

print(f"\nNDVI (Vegetation Index):")
print(f"  Mean: {df['ndvi'].mean():.3f}")
print(f"  Min: {df['ndvi'].min():.3f}")
print(f"  Max: {df['ndvi'].max():.3f}")

print(f"\nNDBI (Built-up Index):")
print(f"  Mean: {df['ndbi'].mean():.3f}")
print(f"  Min: {df['ndbi'].min():.3f}")
print(f"  Max: {df['ndbi'].max():.3f}")

# Sample for visualization
max_viz_points = 20000
if len(df) > max_viz_points:
    print(f"\nSampling {max_viz_points} points for visualization...")
    df_viz = df.sample(n=max_viz_points, random_state=42)
else:
    df_viz = df

# Convert to list of dicts for JSON
data_for_json = df_viz.to_dict('records')

# Create HTML
print("\nCreating HTML visualization...")

html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Full Area Satellite Analysis - Al Karama (10m)</title>
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
        .info-panel h2 {{ margin: 0 0 10px 0; color: #d32f2f; }}
        .stats {{ font-size: 13px; margin: 10px 0; }}
        .stats td {{ padding: 3px 8px; }}
        .layer-btn {{
            padding: 8px 12px;
            margin: 3px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }}
        .layer-btn.active {{ background: #1565c0; color: white; }}
        .layer-btn:not(.active) {{ background: #e0e0e0; }}
        .legend {{ margin-top: 15px; }}
        .legend-gradient {{ height: 15px; border-radius: 3px; margin: 5px 0; }}
        .legend-labels {{ display: flex; justify-content: space-between; font-size: 11px; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="info-panel">
        <h2>Full Area Analysis (10m)</h2>
        <p>Al Karama district boundary</p>
        <p>Grid cells: {len(df):,} (10m resolution)</p>
        <p>Displayed: {len(df_viz):,} points</p>

        <table class="stats">
            <tr><td>Mean LST:</td><td><b>{df['lst'].mean():.1f}°C</b></td></tr>
            <tr><td>Mean NDVI:</td><td><b>{df['ndvi'].mean():.3f}</b></td></tr>
            <tr><td>Mean NDBI:</td><td><b>{df['ndbi'].mean():.3f}</b></td></tr>
        </table>

        <div style="margin: 15px 0;">
            <button class="layer-btn active" id="btn-lst" onclick="showLayer('lst')">Temperature</button>
            <button class="layer-btn" id="btn-ndvi" onclick="showLayer('ndvi')">Vegetation</button>
            <button class="layer-btn" id="btn-ndbi" onclick="showLayer('ndbi')">Built-up</button>
        </div>

        <div class="legend" id="legend-lst">
            <b>Land Surface Temperature</b>
            <div class="legend-gradient" style="background: linear-gradient(to right, #313695, #4575b4, #74add1, #fee090, #f46d43, #d73027, #a50026);"></div>
            <div class="legend-labels"><span>{df['lst'].min():.0f}°C</span><span>{df['lst'].max():.0f}°C</span></div>
        </div>
        <div class="legend" id="legend-ndvi" style="display:none;">
            <b>NDVI (Vegetation)</b>
            <div class="legend-gradient" style="background: linear-gradient(to right, #8B4513, #D2691E, #F4A460, #FFFFE0, #90EE90, #228B22, #006400);"></div>
            <div class="legend-labels"><span>-0.2</span><span>0.6</span></div>
        </div>
        <div class="legend" id="legend-ndbi" style="display:none;">
            <b>NDBI (Built-up)</b>
            <div class="legend-gradient" style="background: linear-gradient(to right, #1a9850, #91cf60, #d9ef8b, #fee08b, #fc8d59, #d73027);"></div>
            <div class="legend-labels"><span>-0.3</span><span>0.3</span></div>
        </div>
    </div>

    <script>
        var map = L.map('map').setView([25.2405, 55.3045], 15);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; OpenStreetMap'
        }}).addTo(map);

        var data = {json.dumps(data_for_json)};

        var layers = {{
            lst: L.layerGroup(),
            ndvi: L.layerGroup(),
            ndbi: L.layerGroup()
        }};

        function getTempColor(t) {{
            var min = {df['lst'].min()}, max = {df['lst'].max()};
            var ratio = (t - min) / (max - min);
            if (ratio < 0.17) return '#313695';
            if (ratio < 0.33) return '#4575b4';
            if (ratio < 0.5) return '#74add1';
            if (ratio < 0.67) return '#fee090';
            if (ratio < 0.83) return '#f46d43';
            return '#d73027';
        }}

        function getNdviColor(v) {{
            if (v < 0) return '#8B4513';
            if (v < 0.1) return '#D2691E';
            if (v < 0.2) return '#F4A460';
            if (v < 0.3) return '#FFFFE0';
            if (v < 0.4) return '#90EE90';
            return '#228B22';
        }}

        function getNdbiColor(v) {{
            if (v < -0.1) return '#1a9850';
            if (v < 0) return '#91cf60';
            if (v < 0.1) return '#fee08b';
            if (v < 0.2) return '#fc8d59';
            return '#d73027';
        }}

        data.forEach(function(d) {{
            var popup = 'LST: ' + d.lst.toFixed(1) + '°C<br>' +
                       'NDVI: ' + d.ndvi.toFixed(3) + '<br>' +
                       'NDBI: ' + d.ndbi.toFixed(3);

            L.circleMarker([d.lat, d.lon], {{
                radius: 3,
                fillColor: getTempColor(d.lst),
                color: 'none',
                fillOpacity: 0.7
            }}).bindPopup(popup).addTo(layers.lst);

            L.circleMarker([d.lat, d.lon], {{
                radius: 3,
                fillColor: getNdviColor(d.ndvi),
                color: 'none',
                fillOpacity: 0.7
            }}).bindPopup(popup).addTo(layers.ndvi);

            L.circleMarker([d.lat, d.lon], {{
                radius: 3,
                fillColor: getNdbiColor(d.ndbi),
                color: 'none',
                fillOpacity: 0.7
            }}).bindPopup(popup).addTo(layers.ndbi);
        }});

        layers.lst.addTo(map);
        var currentLayer = 'lst';

        function showLayer(name) {{
            map.removeLayer(layers[currentLayer]);
            layers[name].addTo(map);
            currentLayer = name;

            document.querySelectorAll('.layer-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('btn-' + name).classList.add('active');

            document.querySelectorAll('.legend').forEach(l => l.style.display = 'none');
            document.getElementById('legend-' + name).style.display = 'block';
        }}
    </script>
</body>
</html>'''

with open('output/satellite_full/full_area_analysis_10m.html', 'w') as f:
    f.write(html)

print(f"Saved: output/satellite_full/full_area_analysis_10m.html")
print("\nDone!")
