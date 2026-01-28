"""
Improve OSM building heights using Google Earth Engine datasets.

Extracts height data from:
1. GHSL Built-C (10m) - height classes
2. GHSL Built-H (100m) - continuous heights
3. nDSM (ALOS DSM - SRTM, 30m) - surface model difference

Merges with OSM data using priority system:
  OSM real (!=9m) > GHSL Built-C > GHSL Built-H > nDSM > default 9m
"""

import ee
import json
import os
import sys
import numpy as np

# ---- Configuration ----
GEOJSON_INPUT = "docs/osm_3d/al_karama_buildings.geojson"
GEOJSON_OUTPUT = "docs/osm_3d/al_karama_buildings.geojson"  # overwrite
HTML_OUTPUT = "docs/osm_3d/al_karama_3d.html"

# Al Karama bounding box
BBOX = [55.290, 25.230, 55.320, 25.255]

# GHSL Built-C height class midpoints (from the Built-C coding scheme)
# Class 1: <=3m, Class 2: 3-6m, Class 3: 6-15m, Class 4: 15-30m, Class 5: >30m
BUILTC_CLASS_TO_HEIGHT = {
    0: None,   # no building
    1: 3.0,    # <=3m
    2: 4.5,    # 3-6m
    3: 10.5,   # 6-15m
    4: 22.5,   # 15-30m
    5: 45.0,   # >30m
    # The actual classes encode building type + height together.
    # Classes 1-5 are residential, 11-15 non-residential, 21-25 mixed.
    # Height band is encoded as class % 10.
    11: 3.0, 12: 4.5, 13: 10.5, 14: 22.5, 15: 45.0,
    21: 3.0, 22: 4.5, 23: 10.5, 24: 22.5, 25: 45.0,
}

# Batch size for reduceRegions (GEE has limits on feature collection size)
BATCH_SIZE = 500


def initialize_ee():
    """Initialize Earth Engine."""
    try:
        ee.Initialize(project='uobdubai')
        print("Initialized GEE with project 'uobdubai'")
        return
    except Exception:
        pass
    try:
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
        print("Initialized GEE with high-volume endpoint")
        return
    except Exception:
        pass
    try:
        ee.Initialize()
        print("Initialized GEE with defaults")
        return
    except Exception as e:
        print(f"Failed to initialize GEE: {e}")
        print("Run: earthengine authenticate")
        sys.exit(1)


def load_buildings(path):
    """Load GeoJSON and return features list."""
    with open(path) as f:
        data = json.load(f)
    print(f"Loaded {len(data['features'])} buildings from {path}")
    return data


def buildings_to_ee_features(features):
    """Convert GeoJSON features to a list of ee.Feature objects."""
    ee_features = []
    for feat in features:
        coords = feat['geometry']['coordinates']
        props = feat['properties']
        try:
            geom = ee.Geometry.Polygon(coords)
            ee_feat = ee.Feature(geom, {'id': props['id']})
            ee_features.append(ee_feat)
        except Exception:
            continue
    return ee_features


def extract_ghsl_builtc(ee_features):
    """
    Extract GHSL Built-C (10m) height classes per building.
    Uses mode reducer to get the dominant height class.
    """
    print("\n--- GHSL Built-C (10m resolution) ---")
    image = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_C/2018")
    band = image.select('built_characteristics')

    results = {}
    n_batches = (len(ee_features) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(ee_features), BATCH_SIZE):
        batch = ee_features[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  Batch {batch_num}/{n_batches} ({len(batch)} buildings)...")

        fc = ee.FeatureCollection(batch)
        reduced = band.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.mode(),
            scale=10,
        )

        try:
            result_list = reduced.getInfo()['features']
            for r in result_list:
                bid = r['properties']['id']
                mode_val = r['properties'].get('mode')
                if mode_val is not None:
                    # Extract height band: class % 10 gives height category
                    height_band = int(mode_val) % 10
                    height = BUILTC_CLASS_TO_HEIGHT.get(height_band)
                    if height is not None:
                        results[bid] = height
        except Exception as e:
            print(f"    Error in batch {batch_num}: {e}")

    print(f"  Got Built-C heights for {len(results)} buildings")
    return results


def extract_ghsl_builth(ee_features):
    """
    Extract GHSL Built-H (100m) continuous heights per building.
    Uses mean reducer.
    """
    print("\n--- GHSL Built-H (100m resolution) ---")
    image = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_H/2018")
    band = image.select('built_height')

    results = {}
    n_batches = (len(ee_features) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(ee_features), BATCH_SIZE):
        batch = ee_features[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  Batch {batch_num}/{n_batches} ({len(batch)} buildings)...")

        fc = ee.FeatureCollection(batch)
        reduced = band.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.mean(),
            scale=100,
        )

        try:
            result_list = reduced.getInfo()['features']
            for r in result_list:
                bid = r['properties']['id']
                mean_val = r['properties'].get('mean')
                if mean_val is not None and mean_val > 0:
                    results[bid] = round(float(mean_val), 1)
        except Exception as e:
            print(f"    Error in batch {batch_num}: {e}")

    print(f"  Got Built-H heights for {len(results)} buildings")
    return results


def extract_ndsm(ee_features):
    """
    Compute nDSM (ALOS DSM - SRTM DEM) per building.
    30m resolution, noisy but covers more area.
    """
    print("\n--- nDSM (ALOS - SRTM, 30m resolution) ---")
    alos = ee.ImageCollection("JAXA/ALOS/AW3D30/V4_1").mosaic().select('DSM')
    srtm = ee.Image("USGS/SRTMGL1_003").select('elevation')
    ndsm = alos.subtract(srtm).rename('ndsm')

    results = {}
    n_batches = (len(ee_features) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(ee_features), BATCH_SIZE):
        batch = ee_features[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  Batch {batch_num}/{n_batches} ({len(batch)} buildings)...")

        fc = ee.FeatureCollection(batch)
        reduced = ndsm.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.mean(),
            scale=30,
        )

        try:
            result_list = reduced.getInfo()['features']
            for r in result_list:
                bid = r['properties']['id']
                mean_val = r['properties'].get('mean')
                if mean_val is not None and mean_val > 2.0:
                    # Only use positive values above noise threshold
                    results[bid] = round(float(mean_val), 1)
        except Exception as e:
            print(f"    Error in batch {batch_num}: {e}")

    print(f"  Got nDSM heights for {len(results)} buildings")
    return results


def merge_heights(geojson, builtc, builth, ndsm_data):
    """
    Merge height data with priority:
      1. OSM real height (!=9m) - keep as-is
      2. GHSL Built-C (10m)
      3. GHSL Built-H (100m)
      4. nDSM (30m)
      5. Default 9m
    """
    print("\n--- Merging heights ---")
    stats = {
        'osm_real': 0,
        'ghsl_builtc': 0,
        'ghsl_builth': 0,
        'ndsm': 0,
        'default': 0,
        'total': len(geojson['features']),
    }

    for feat in geojson['features']:
        props = feat['properties']
        bid = props['id']
        current_height = props['height']

        # Priority 1: Real OSM height
        if current_height != 9.0:
            props['height_source'] = 'osm'
            stats['osm_real'] += 1
            continue

        # Priority 2: GHSL Built-C
        if bid in builtc:
            props['height'] = builtc[bid]
            props['height_source'] = 'ghsl_builtc'
            stats['ghsl_builtc'] += 1
            continue

        # Priority 3: GHSL Built-H
        if bid in builth:
            props['height'] = builth[bid]
            props['height_source'] = 'ghsl_builth'
            stats['ghsl_builth'] += 1
            continue

        # Priority 4: nDSM
        if bid in ndsm_data:
            props['height'] = ndsm_data[bid]
            props['height_source'] = 'ndsm'
            stats['ndsm'] += 1
            continue

        # Priority 5: Default
        props['height_source'] = 'default'
        stats['default'] += 1

    # Print summary
    print(f"\n{'='*50}")
    print("MERGE STATISTICS")
    print(f"{'='*50}")
    print(f"  Total buildings:     {stats['total']}")
    print(f"  OSM real height:     {stats['osm_real']} ({100*stats['osm_real']/stats['total']:.1f}%)")
    print(f"  GHSL Built-C (10m):  {stats['ghsl_builtc']} ({100*stats['ghsl_builtc']/stats['total']:.1f}%)")
    print(f"  GHSL Built-H (100m): {stats['ghsl_builth']} ({100*stats['ghsl_builth']/stats['total']:.1f}%)")
    print(f"  nDSM (30m):          {stats['ndsm']} ({100*stats['ndsm']/stats['total']:.1f}%)")
    print(f"  Default (9m):        {stats['default']} ({100*stats['default']/stats['total']:.1f}%)")

    improved = stats['total'] - stats['default']
    print(f"\n  Buildings with data:  {improved} ({100*improved/stats['total']:.1f}%)")

    return geojson, stats


def generate_html(geojson, stats, output_path):
    """Generate 3D visualization HTML with source coloring toggle."""
    geojson_str = json.dumps(geojson)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Al Karama 3D Buildings - GEE Enhanced Heights</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://unpkg.com/deck.gl@8.9.0/dist.min.js"></script>
    <script src="https://unpkg.com/maplibre-gl@2.4.0/dist/maplibre-gl.js"></script>
    <link href="https://unpkg.com/maplibre-gl@2.4.0/dist/maplibre-gl.css" rel="stylesheet">
    <style>
        body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; }}
        #container {{ width: 100vw; height: 100vh; position: relative; }}
        #panel {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.88);
            color: #eee;
            padding: 16px;
            border-radius: 10px;
            z-index: 1;
            max-width: 320px;
            font-size: 13px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        }}
        #panel h3 {{ margin: 0 0 8px 0; color: #4fc3f7; font-size: 16px; }}
        #panel h4 {{ margin: 12px 0 6px 0; color: #aaa; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }}
        .stat-row {{ display: flex; justify-content: space-between; margin: 3px 0; }}
        .stat-val {{ color: #81c784; font-weight: bold; }}
        .stat-pct {{ color: #888; font-size: 11px; margin-left: 4px; }}
        hr {{ border: none; border-top: 1px solid #333; margin: 10px 0; }}
        .legend-item {{ display: flex; align-items: center; margin: 4px 0; }}
        .legend-color {{ width: 16px; height: 16px; margin-right: 8px; border-radius: 3px; flex-shrink: 0; }}
        .toggle-group {{ margin: 10px 0; }}
        .toggle-btn {{
            padding: 6px 12px;
            border: 1px solid #555;
            background: #222;
            color: #ccc;
            cursor: pointer;
            font-size: 12px;
            border-radius: 4px;
            margin-right: 4px;
        }}
        .toggle-btn.active {{
            background: #1976d2;
            color: #fff;
            border-color: #1976d2;
        }}
        .controls {{ margin-top: 12px; color: #777; font-size: 11px; }}
    </style>
</head>
<body>
    <div id="container"></div>
    <div id="panel">
        <h3>Al Karama 3D Buildings</h3>
        <p style="color:#aaa; margin:0 0 8px 0;">Heights enhanced with GEE data</p>

        <h4>Statistics</h4>
        <div class="stat-row"><span>Total buildings</span><span class="stat-val">{stats['total']}</span></div>
        <div class="stat-row"><span>OSM real height</span><span><span class="stat-val">{stats['osm_real']}</span><span class="stat-pct">({100*stats['osm_real']/stats['total']:.1f}%)</span></span></div>
        <div class="stat-row"><span>GHSL Built-C (10m)</span><span><span class="stat-val">{stats['ghsl_builtc']}</span><span class="stat-pct">({100*stats['ghsl_builtc']/stats['total']:.1f}%)</span></span></div>
        <div class="stat-row"><span>GHSL Built-H (100m)</span><span><span class="stat-val">{stats['ghsl_builth']}</span><span class="stat-pct">({100*stats['ghsl_builth']/stats['total']:.1f}%)</span></span></div>
        <div class="stat-row"><span>nDSM (30m)</span><span><span class="stat-val">{stats['ndsm']}</span><span class="stat-pct">({100*stats['ndsm']/stats['total']:.1f}%)</span></span></div>
        <div class="stat-row"><span>Default (9m)</span><span><span class="stat-val">{stats['default']}</span><span class="stat-pct">({100*stats['default']/stats['total']:.1f}%)</span></span></div>

        <hr>

        <h4>Color Mode</h4>
        <div class="toggle-group">
            <button class="toggle-btn active" id="btn-height" onclick="setColorMode('height')">By Height</button>
            <button class="toggle-btn" id="btn-source" onclick="setColorMode('source')">By Source</button>
        </div>

        <div id="legend-height">
            <div class="legend-item"><div class="legend-color" style="background:#2ecc71"></div> Low (&lt;10m)</div>
            <div class="legend-item"><div class="legend-color" style="background:#3498db"></div> Medium (10-30m)</div>
            <div class="legend-item"><div class="legend-color" style="background:#9b59b6"></div> High (30-60m)</div>
            <div class="legend-item"><div class="legend-color" style="background:#e74c3c"></div> Very High (&gt;60m)</div>
        </div>

        <div id="legend-source" style="display:none">
            <div class="legend-item"><div class="legend-color" style="background:#e74c3c"></div> OSM real height</div>
            <div class="legend-item"><div class="legend-color" style="background:#f39c12"></div> GHSL Built-C (10m)</div>
            <div class="legend-item"><div class="legend-color" style="background:#2ecc71"></div> GHSL Built-H (100m)</div>
            <div class="legend-item"><div class="legend-color" style="background:#3498db"></div> nDSM (30m)</div>
            <div class="legend-item"><div class="legend-color" style="background:#666"></div> Default (9m)</div>
        </div>

        <div class="controls">
            <b>Controls:</b> Drag: Rotate/Pan | Scroll: Zoom | Ctrl+Drag: Tilt
        </div>
    </div>

    <script>
        const BUILDINGS_DATA = {geojson_str};
        let colorMode = 'height';

        function getHeightColor(height) {{
            if (height < 10) return [46, 204, 113];
            if (height < 30) return [52, 152, 219];
            if (height < 60) return [155, 89, 182];
            return [231, 76, 60];
        }}

        const SOURCE_COLORS = {{
            'osm':         [231, 76, 60],
            'ghsl_builtc': [243, 156, 18],
            'ghsl_builth': [46, 204, 113],
            'ndsm':        [52, 152, 219],
            'default':     [102, 102, 102],
        }};

        function getSourceColor(source) {{
            return SOURCE_COLORS[source] || [102, 102, 102];
        }}

        function getColor(feature) {{
            if (colorMode === 'height') {{
                return getHeightColor(feature.properties.height);
            }} else {{
                return getSourceColor(feature.properties.height_source);
            }}
        }}

        const SOURCE_LABELS = {{
            'osm': 'OSM real height',
            'ghsl_builtc': 'GHSL Built-C (10m)',
            'ghsl_builth': 'GHSL Built-H (100m)',
            'ndsm': 'nDSM (ALOS-SRTM, 30m)',
            'default': 'Default (9m)',
        }};

        function buildLayer() {{
            return new deck.PolygonLayer({{
                id: 'buildings',
                data: BUILDINGS_DATA.features,
                extruded: true,
                wireframe: false,
                opacity: 0.85,
                getPolygon: f => f.geometry.coordinates[0],
                getElevation: f => f.properties.height,
                getFillColor: f => getColor(f),
                getLineColor: [255, 255, 255, 60],
                lineWidthMinPixels: 1,
                pickable: true,
                updateTriggers: {{
                    getFillColor: colorMode
                }}
            }});
        }}

        const deckgl = new deck.DeckGL({{
            container: 'container',
            mapStyle: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
            initialViewState: {{
                longitude: 55.305,
                latitude: 25.242,
                zoom: 15,
                pitch: 60,
                bearing: -20
            }},
            controller: true,
            layers: [buildLayer()],
            getTooltip: ({{object}}) => object && {{
                html: '<div style="padding:8px; max-width:250px;">' +
                    '<b>' + (object.properties.name || 'Building') + '</b><br>' +
                    'Type: ' + (object.properties.building_type || 'unknown') + '<br>' +
                    'Height: ' + object.properties.height + 'm<br>' +
                    (object.properties.levels ? 'Levels: ' + object.properties.levels + '<br>' : '') +
                    'Source: <b>' + (SOURCE_LABELS[object.properties.height_source] || 'unknown') + '</b>' +
                    '</div>',
                style: {{
                    backgroundColor: '#222',
                    color: '#fff',
                    borderRadius: '6px',
                    fontSize: '12px'
                }}
            }}
        }});

        function setColorMode(mode) {{
            colorMode = mode;
            document.getElementById('btn-height').className = mode === 'height' ? 'toggle-btn active' : 'toggle-btn';
            document.getElementById('btn-source').className = mode === 'source' ? 'toggle-btn active' : 'toggle-btn';
            document.getElementById('legend-height').style.display = mode === 'height' ? 'block' : 'none';
            document.getElementById('legend-source').style.display = mode === 'source' ? 'block' : 'none';
            deckgl.setProps({{ layers: [buildLayer()] }});
        }}
    </script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)
    print(f"\nSaved 3D viewer: {output_path}")


def main():
    # Initialize GEE
    initialize_ee()

    # Load buildings
    geojson = load_buildings(GEOJSON_INPUT)
    features = geojson['features']

    # Convert to EE features
    print("\nConverting buildings to EE features...")
    ee_features = buildings_to_ee_features(features)
    print(f"Converted {len(ee_features)} buildings")

    # Extract heights from each source
    builtc_heights = extract_ghsl_builtc(ee_features)
    builth_heights = extract_ghsl_builth(ee_features)
    ndsm_heights = extract_ndsm(ee_features)

    # Merge with priority system
    geojson, stats = merge_heights(geojson, builtc_heights, builth_heights, ndsm_heights)

    # Save updated GeoJSON
    print(f"\nSaving updated GeoJSON to {GEOJSON_OUTPUT}...")
    with open(GEOJSON_OUTPUT, 'w') as f:
        json.dump(geojson, f)
    print("Saved.")

    # Generate HTML visualization
    generate_html(geojson, stats, HTML_OUTPUT)

    print("\nDone! Open the HTML file to view the enhanced 3D buildings.")


if __name__ == '__main__':
    main()
