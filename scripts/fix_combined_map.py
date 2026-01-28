"""
Fix the combined analysis map - create a working version.
"""

import os
import json
import pandas as pd

output_dir = "output/combined_analysis"

# Load the combined data
df = pd.read_csv(os.path.join(output_dir, "combined_data.csv"))
print(f"Loaded {len(df)} records")

# Sample for performance
viz_df = df.sample(min(3000, len(df)), random_state=42)
viz_df = viz_df.dropna(subset=['lat', 'lon', 'gvi', 'svf', 'lst'])
print(f"Visualization sample: {len(viz_df)} records")

# Get statistics
lst_min = df['lst'].min()
lst_max = df['lst'].max()
gvi_mean = df['gvi'].mean()
svf_mean = df['svf'].mean()
lst_mean = df['lst'].mean()

# Convert to JSON
data_json = viz_df[['lat', 'lon', 'gvi', 'svf', 'lst']].to_dict(orient='records')

html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Combined Urban Climate Analysis - Al Karama</title>
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
        .info-panel h2 { margin: 0 0 15px 0; color: #1565c0; }
        .section { margin: 15px 0; padding: 12px; background: #f5f5f5; border-radius: 8px; }
        .section h4 { margin: 0 0 10px 0; }
        .layer-controls { margin: 15px 0; }
        .layer-btn {
            padding: 8px 12px;
            margin: 3px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
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
        <h2>Combined Analysis</h2>
        <p>Street-level + Satellite Data</p>
        <p>Points: """ + str(len(viz_df)) + """</p>

        <div class="layer-controls">
            <button class="layer-btn active" id="btn-lst" onclick="showLayer('lst')">Temperature</button>
            <button class="layer-btn" id="btn-gvi" onclick="showLayer('gvi')">GVI</button>
            <button class="layer-btn" id="btn-svf" onclick="showLayer('svf')">SVF</button>
        </div>

        <div class="section">
            <h4>Statistics</h4>
            <p>Mean LST: """ + f"{lst_mean:.1f}" + """°C</p>
            <p>Mean GVI: """ + f"{gvi_mean*100:.2f}" + """%</p>
            <p>Mean SVF: """ + f"{svf_mean*100:.1f}" + """%</p>
        </div>

        <div class="legend" id="legend-lst">
            <div><b>Temperature (°C)</b></div>
            <div class="legend-gradient" style="background: linear-gradient(to right, blue, cyan, yellow, orange, red);"></div>
            <div class="legend-labels"><span>""" + f"{lst_min:.0f}" + """</span><span>""" + f"{lst_max:.0f}" + """</span></div>
        </div>

        <div class="legend" id="legend-gvi" style="display:none">
            <div><b>Green View Index (%)</b></div>
            <div class="legend-gradient" style="background: linear-gradient(to right, #f7f7f7, #90EE90, #228B22);"></div>
            <div class="legend-labels"><span>0%</span><span>30%+</span></div>
        </div>

        <div class="legend" id="legend-svf" style="display:none">
            <div><b>Sky View Factor (%)</b></div>
            <div class="legend-gradient" style="background: linear-gradient(to right, #333, #87CEEB);"></div>
            <div class="legend-labels"><span>0%</span><span>100%</span></div>
        </div>
    </div>

    <script>
        var map = L.map('map').setView([25.245, 55.305], 15);

        L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
            maxZoom: 19,
            attribution: 'CartoDB'
        }).addTo(map);

        var data = """ + json.dumps(data_json) + """;

        console.log("Data points:", data.length);

        var lstMin = """ + str(lst_min) + """;
        var lstMax = """ + str(lst_max) + """;

        function getLSTColor(lst) {
            var ratio = (lst - lstMin) / (lstMax - lstMin);
            if (ratio < 0.25) return 'blue';
            if (ratio < 0.5) return 'cyan';
            if (ratio < 0.75) return 'orange';
            return 'red';
        }

        function getGVIColor(gvi) {
            if (gvi < 0.02) return '#f7f7f7';
            if (gvi < 0.05) return '#c7e9c0';
            if (gvi < 0.1) return '#74c476';
            if (gvi < 0.2) return '#31a354';
            return '#006d2c';
        }

        function getSVFColor(svf) {
            var r = Math.round(50 + 180 * svf);
            var g = Math.round(50 + 180 * svf);
            var b = Math.round(100 + 155 * svf);
            return 'rgb(' + r + ',' + g + ',' + b + ')';
        }

        var layers = {
            lst: L.layerGroup(),
            gvi: L.layerGroup(),
            svf: L.layerGroup()
        };

        for (var i = 0; i < data.length; i++) {
            var d = data[i];

            L.circleMarker([d.lat, d.lon], {
                radius: 4,
                color: getLSTColor(d.lst),
                fillColor: getLSTColor(d.lst),
                fillOpacity: 0.8,
                weight: 0
            }).bindPopup('Temp: ' + d.lst.toFixed(1) + '°C<br>GVI: ' + (d.gvi*100).toFixed(1) + '%').addTo(layers.lst);

            L.circleMarker([d.lat, d.lon], {
                radius: 4,
                color: getGVIColor(d.gvi),
                fillColor: getGVIColor(d.gvi),
                fillOpacity: 0.8,
                weight: 0
            }).bindPopup('GVI: ' + (d.gvi*100).toFixed(1) + '%<br>Temp: ' + d.lst.toFixed(1) + '°C').addTo(layers.gvi);

            L.circleMarker([d.lat, d.lon], {
                radius: 4,
                color: getSVFColor(d.svf),
                fillColor: getSVFColor(d.svf),
                fillOpacity: 0.8,
                weight: 0
            }).bindPopup('SVF: ' + (d.svf*100).toFixed(1) + '%<br>Temp: ' + d.lst.toFixed(1) + '°C').addTo(layers.svf);
        }

        layers.lst.addTo(map);
        var currentLayer = 'lst';

        function showLayer(name) {
            map.removeLayer(layers[currentLayer]);
            layers[name].addTo(map);
            currentLayer = name;

            document.getElementById('btn-lst').className = 'layer-btn' + (name === 'lst' ? ' active' : '');
            document.getElementById('btn-gvi').className = 'layer-btn' + (name === 'gvi' ? ' active' : '');
            document.getElementById('btn-svf').className = 'layer-btn' + (name === 'svf' ? ' active' : '');

            document.getElementById('legend-lst').style.display = name === 'lst' ? 'block' : 'none';
            document.getElementById('legend-gvi').style.display = name === 'gvi' ? 'block' : 'none';
            document.getElementById('legend-svf').style.display = name === 'svf' ? 'block' : 'none';
        }

        console.log("Map initialized with", data.length, "points");
    </script>
</body>
</html>"""

html_path = os.path.join(output_dir, "combined_analysis.html")
with open(html_path, 'w') as f:
    f.write(html_content)

print(f"Saved: {html_path}")
