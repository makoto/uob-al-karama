"""
Fetch OSM building data for Al Karama and create 3D visualization.
"""

import requests
import json
import os

output_dir = "output/osm_3d"
os.makedirs(output_dir, exist_ok=True)

# Al Karama bounding box (slightly expanded)
bbox = (25.230, 55.290, 25.255, 55.320)

# Overpass query for buildings with all available data
query = f"""
[out:json][timeout:60];
(
  way["building"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
  relation["building"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
);
out body;
>;
out skel qt;
"""

print("Fetching building data from OpenStreetMap...")
response = requests.post(
    "https://overpass-api.de/api/interpreter",
    data={"data": query}
)

data = response.json()
elements = data.get('elements', [])

# Separate nodes and ways
nodes = {e['id']: e for e in elements if e['type'] == 'node'}
ways = [e for e in elements if e['type'] == 'way' and 'tags' in e]

print(f"Found {len(ways)} buildings, {len(nodes)} nodes")

# Process buildings
buildings = []
for way in ways:
    tags = way.get('tags', {})

    # Get height info
    height = None
    if 'height' in tags:
        try:
            height = float(tags['height'].replace('m', '').strip())
        except:
            pass
    elif 'building:levels' in tags:
        try:
            levels = int(tags['building:levels'])
            height = levels * 3.0  # Assume 3m per floor
        except:
            pass

    if height is None:
        height = 9.0  # Default 3 floors

    # Get building footprint coordinates
    coords = []
    for node_id in way.get('nodes', []):
        if node_id in nodes:
            n = nodes[node_id]
            coords.append([n['lon'], n['lat']])

    if len(coords) >= 3:
        buildings.append({
            'id': way['id'],
            'name': tags.get('name', ''),
            'type': tags.get('building', 'yes'),
            'height': height,
            'levels': tags.get('building:levels', ''),
            'coordinates': coords
        })

print(f"Processed {len(buildings)} buildings with footprints")

# Count buildings with height data
with_height = sum(1 for b in buildings if b['height'] != 9.0)
print(f"Buildings with height data: {with_height}")
print(f"Buildings with default height (9m): {len(buildings) - with_height}")

# Save as GeoJSON
geojson = {
    "type": "FeatureCollection",
    "features": []
}

for b in buildings:
    feature = {
        "type": "Feature",
        "properties": {
            "id": b['id'],
            "name": b['name'],
            "building_type": b['type'],
            "height": b['height'],
            "levels": b['levels']
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [b['coordinates']]
        }
    }
    geojson['features'].append(feature)

geojson_path = os.path.join(output_dir, "al_karama_buildings.geojson")
with open(geojson_path, 'w') as f:
    json.dump(geojson, f)
print(f"\nSaved GeoJSON: {geojson_path}")

# Create 3D HTML viewer using deck.gl
html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Al Karama 3D Buildings (OpenStreetMap)</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://unpkg.com/deck.gl@8.9.0/dist.min.js"></script>
    <script src="https://unpkg.com/maplibre-gl@2.4.0/dist/maplibre-gl.js"></script>
    <link href="https://unpkg.com/maplibre-gl@2.4.0/dist/maplibre-gl.css" rel="stylesheet">
    <style>
        body { margin: 0; padding: 0; }
        #container { width: 100vw; height: 100vh; position: relative; }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-family: Arial, sans-serif;
            z-index: 1;
            max-width: 300px;
        }
        #info h3 { margin: 0 0 10px 0; }
        #info p { margin: 5px 0; font-size: 13px; }
        .legend { margin-top: 10px; }
        .legend-item { display: flex; align-items: center; margin: 3px 0; }
        .legend-color { width: 20px; height: 20px; margin-right: 8px; border-radius: 3px; }
    </style>
</head>
<body>
    <div id="container"></div>
    <div id="info">
        <h3>Al Karama 3D Buildings</h3>
        <p>Source: OpenStreetMap</p>
        <p>Buildings: <span id="count">Loading...</span></p>
        <p>With height data: <span id="heightCount">Loading...</span></p>
        <div class="legend">
            <p><b>Height (color)</b></p>
            <div class="legend-item"><div class="legend-color" style="background:#2ecc71"></div> Low (&lt;10m)</div>
            <div class="legend-item"><div class="legend-color" style="background:#3498db"></div> Medium (10-30m)</div>
            <div class="legend-item"><div class="legend-color" style="background:#9b59b6"></div> High (30-60m)</div>
            <div class="legend-item"><div class="legend-color" style="background:#e74c3c"></div> Very High (&gt;60m)</div>
        </div>
        <p style="margin-top:15px"><b>Controls:</b></p>
        <p>Drag: Rotate/Pan</p>
        <p>Scroll: Zoom</p>
        <p>Ctrl+Drag: Tilt</p>
    </div>

    <script>
        const BUILDINGS_DATA = """ + json.dumps(geojson) + """;

        document.getElementById('count').textContent = BUILDINGS_DATA.features.length;
        const withHeight = BUILDINGS_DATA.features.filter(f => f.properties.height !== 9.0).length;
        document.getElementById('heightCount').textContent = withHeight;

        function getColor(height) {
            if (height < 10) return [46, 204, 113];      // Green
            if (height < 30) return [52, 152, 219];      // Blue
            if (height < 60) return [155, 89, 182];      // Purple
            return [231, 76, 60];                         // Red
        }

        const deckgl = new deck.DeckGL({
            container: 'container',
            mapStyle: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
            initialViewState: {
                longitude: 55.305,
                latitude: 25.242,
                zoom: 15,
                pitch: 60,
                bearing: -20
            },
            controller: true,
            layers: [
                new deck.PolygonLayer({
                    id: 'buildings',
                    data: BUILDINGS_DATA.features,
                    extruded: true,
                    wireframe: false,
                    opacity: 0.8,
                    getPolygon: f => f.geometry.coordinates[0],
                    getElevation: f => f.properties.height,
                    getFillColor: f => getColor(f.properties.height),
                    getLineColor: [255, 255, 255],
                    lineWidthMinPixels: 1,
                    pickable: true
                })
            ],
            getTooltip: ({object}) => object && {
                html: `<b>${object.properties.name || 'Building'}</b><br>
                       Type: ${object.properties.building_type}<br>
                       Height: ${object.properties.height}m<br>
                       ${object.properties.levels ? 'Levels: ' + object.properties.levels : ''}`,
                style: {
                    backgroundColor: '#333',
                    color: '#fff',
                    padding: '8px',
                    borderRadius: '4px'
                }
            }
        });
    </script>
</body>
</html>"""

html_path = os.path.join(output_dir, "al_karama_3d.html")
with open(html_path, 'w') as f:
    f.write(html_content)
print(f"Saved 3D viewer: {html_path}")

print("\nDone! Open the HTML file to view 3D buildings.")
