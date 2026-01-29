#!/usr/bin/env python3
"""
Building Shade Analysis for Al Karama, Dubai
=============================================
Computes shadow projections from 3,243 buildings at 5 times of day
(July 15, peak summer) and calculates shade coverage for street segments.

Uses NOAA solar position algorithm (no external solar library needed).
"""

import json
import math
import os
import time
from datetime import datetime, timezone, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, LineString, mapping
from shapely.ops import unary_union
from shapely import strtree

# ── Configuration ──────────────────────────────────────────────────────
LATITUDE = 25.2485   # Al Karama center
LONGITUDE = 55.3025
TIMEZONE_OFFSET = 4   # UAE = UTC+4
DATE = (2024, 7, 15)  # July 15, peak summer

TIMES_LOCAL = [8, 10, 12, 14, 16]  # Local hours to analyze

BUILDINGS_PATH = os.path.join(os.path.dirname(__file__),
    '..', 'docs', 'osm_3d', 'al_karama_buildings.geojson')
STREETS_PATH = os.path.join(os.path.dirname(__file__),
    '..', 'docs', 'network_analysis', 'street_network.geojson')
CANOPY_PATH = os.path.join(os.path.dirname(__file__),
    '..', 'docs', 'shade_analysis', 'canopy_polygons_with_height.geojson')
OUT_DIR = os.path.join(os.path.dirname(__file__),
    '..', 'docs', 'shade_analysis')

UTM_CRS = 'EPSG:32640'  # UTM Zone 40N for Dubai


# ── 1. NOAA Solar Position Algorithm ──────────────────────────────────

def solar_position(year, month, day, hour_utc, lat, lon):
    """
    Calculate solar altitude and azimuth using NOAA spreadsheet algorithm.

    Parameters
    ----------
    year, month, day : int
        Date components.
    hour_utc : float
        Hour of day in UTC (fractional).
    lat, lon : float
        Observer latitude and longitude in degrees.

    Returns
    -------
    altitude : float
        Solar altitude angle in degrees (0=horizon, 90=zenith).
    azimuth : float
        Solar azimuth in degrees clockwise from north (0=N, 90=E, 180=S, 270=W).
    """
    # Julian Day Number
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    jdn = day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    jd = jdn + (hour_utc - 12.0) / 24.0

    # Julian Century
    jc = (jd - 2451545.0) / 36525.0

    # Geometric mean longitude of sun (degrees)
    L0 = (280.46646 + jc * (36000.76983 + 0.0003032 * jc)) % 360

    # Mean anomaly of sun (degrees)
    M = (357.52911 + jc * (35999.05029 - 0.0001537 * jc)) % 360
    M_rad = math.radians(M)

    # Equation of center
    C = (math.sin(M_rad) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
         + math.sin(2 * M_rad) * (0.019993 - 0.000101 * jc)
         + math.sin(3 * M_rad) * 0.000289)

    # Sun true longitude and apparent longitude
    sun_lon = L0 + C
    omega = 125.04 - 1934.136 * jc
    sun_app_lon = sun_lon - 0.00569 - 0.00478 * math.sin(math.radians(omega))

    # Mean obliquity of ecliptic
    obliq_mean = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    obliq_corr = obliq_mean + 0.00256 * math.cos(math.radians(omega))
    obliq_rad = math.radians(obliq_corr)

    # Solar declination
    sin_dec = math.sin(obliq_rad) * math.sin(math.radians(sun_app_lon))
    dec = math.asin(sin_dec)  # radians

    # Equation of time (minutes)
    tan_half_obliq = math.tan(obliq_rad / 2)
    y_eq = tan_half_obliq ** 2
    L0_rad = math.radians(L0)
    eqt = 4 * math.degrees(
        y_eq * math.sin(2 * L0_rad)
        - 2 * 0.016708634 * math.sin(M_rad)  # eccentricity ≈ 0.016708634
        + 4 * 0.016708634 * y_eq * math.sin(M_rad) * math.cos(2 * L0_rad)
        - 0.5 * y_eq * y_eq * math.sin(4 * L0_rad)
        - 1.25 * 0.016708634 ** 2 * math.sin(2 * M_rad)
    )

    # True solar time
    time_offset = eqt + 4 * lon  # minutes  (lon in degrees)
    tst = hour_utc * 60 + time_offset  # minutes from midnight UTC
    # Hour angle
    ha = (tst / 4) - 180  # degrees
    ha_rad = math.radians(ha)

    lat_rad = math.radians(lat)

    # Solar altitude
    sin_alt = (math.sin(lat_rad) * math.sin(dec)
               + math.cos(lat_rad) * math.cos(dec) * math.cos(ha_rad))
    altitude = math.degrees(math.asin(max(-1, min(1, sin_alt))))

    # Solar azimuth
    cos_az = (math.sin(dec) - math.sin(lat_rad) * sin_alt) / (math.cos(lat_rad) * math.cos(math.radians(altitude)))
    cos_az = max(-1, min(1, cos_az))
    azimuth = math.degrees(math.acos(cos_az))
    if ha > 0:
        azimuth = 360 - azimuth

    return altitude, azimuth


def get_sun_positions():
    """Calculate sun positions for all analysis times."""
    year, month, day = DATE
    positions = {}
    print("\n  Sun Positions for July 15, Dubai:")
    print(f"  {'Local Time':>12} {'Altitude':>10} {'Azimuth':>10} {'Shadow Factor':>14}")
    print("  " + "-" * 50)
    for hour_local in TIMES_LOCAL:
        hour_utc = hour_local - TIMEZONE_OFFSET
        alt, az = solar_position(year, month, day, hour_utc, LATITUDE, LONGITUDE)
        shadow_factor = 1.0 / math.tan(math.radians(alt)) if alt > 0 else float('inf')
        positions[hour_local] = {'altitude': alt, 'azimuth': az, 'shadow_factor': shadow_factor}
        print(f"  {hour_local:>8}:00   {alt:>8.1f}°  {az:>8.1f}°  {shadow_factor:>12.2f}x")
    return positions


# ── 2. Shadow Projection ─────────────────────────────────────────────

def project_shadow(building_geom, height, altitude, azimuth):
    """
    Create a shadow polygon for a building at given sun position.

    The shadow is cast opposite the sun azimuth, with length = height / tan(altitude).
    The shadow polygon is the union of the building footprint and its
    translated copy (the shadow tip), connected by wall trapezoids.
    """
    if altitude <= 0 or height <= 0:
        return building_geom  # No shadow or no height

    shadow_len = height / math.tan(math.radians(altitude))
    if shadow_len < 0.5:
        return building_geom  # Negligible shadow

    # Shadow direction is opposite to sun azimuth
    # Azimuth: 0=N, 90=E, so shadow goes opposite
    shadow_az = math.radians(azimuth + 180)
    dx = shadow_len * math.sin(shadow_az)
    dy = shadow_len * math.cos(shadow_az)

    from shapely.affinity import translate

    # Handle both Polygon and MultiPolygon
    if building_geom.geom_type == 'MultiPolygon':
        shadows = []
        for poly in building_geom.geoms:
            s = _project_single_polygon(poly, dx, dy)
            if s is not None and s.is_valid and not s.is_empty:
                shadows.append(s)
        if not shadows:
            return building_geom
        result = unary_union(shadows + [building_geom])
    else:
        shadow_tip = translate(building_geom, xoff=dx, yoff=dy)
        result = unary_union([building_geom, shadow_tip])
        # Add connecting walls (convex hull is a simple approximation)
        result = building_geom.union(shadow_tip).convex_hull

    # Clean up
    result = result.buffer(0)
    if result.is_valid and not result.is_empty:
        return result.simplify(0.5)
    return building_geom


def _project_single_polygon(poly, dx, dy):
    """Project shadow for a single polygon using convex hull of footprint + tip."""
    from shapely.affinity import translate
    shadow_tip = translate(poly, xoff=dx, yoff=dy)
    return poly.union(shadow_tip).convex_hull


# ── 3. Chunked Shadow Union ──────────────────────────────────────────

def chunked_shadow_union(shadow_gdf, grid_size=200):
    """
    Union shadow polygons using spatial chunks for performance.
    Divides area into grid_size x grid_size meter tiles, unions within each,
    then unions the results.
    """
    bounds = shadow_gdf.total_bounds  # minx, miny, maxx, maxy

    xs = np.arange(bounds[0], bounds[2] + grid_size, grid_size)
    ys = np.arange(bounds[1], bounds[3] + grid_size, grid_size)

    sindex = shadow_gdf.sindex
    chunk_unions = []

    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            box = Polygon([
                (xs[i], ys[j]), (xs[i+1], ys[j]),
                (xs[i+1], ys[j+1]), (xs[i], ys[j+1])
            ])
            candidates = list(sindex.query(box))
            if not candidates:
                continue
            chunk = shadow_gdf.iloc[candidates]
            chunk_geoms = chunk.geometry[chunk.geometry.is_valid & ~chunk.geometry.is_empty]
            if len(chunk_geoms) > 0:
                try:
                    u = unary_union(chunk_geoms.values)
                    if u.is_valid and not u.is_empty:
                        chunk_unions.append(u)
                except Exception:
                    # If union fails, add individual geometries
                    chunk_unions.extend(chunk_geoms.values)

    if not chunk_unions:
        return MultiPolygon()

    # Final union of chunks
    try:
        result = unary_union(chunk_unions)
    except Exception:
        result = MultiPolygon(chunk_unions)

    return result.buffer(0)


# ── 4. Street Shade Calculation ──────────────────────────────────────

def calculate_street_shade(streets_utm, shadow_union):
    """
    Calculate shade fraction for each street segment.
    shade_fraction = length of segment within shadow / total segment length
    """
    shade_fractions = []

    if shadow_union.is_empty:
        return [0.0] * len(streets_utm)

    # Use STRtree for fast spatial queries
    prep_shadow = shadow_union

    for _, street in streets_utm.iterrows():
        geom = street.geometry
        if geom is None or geom.is_empty:
            shade_fractions.append(0.0)
            continue

        total_len = geom.length
        if total_len == 0:
            shade_fractions.append(0.0)
            continue

        try:
            intersection = geom.intersection(prep_shadow)
            shaded_len = intersection.length if not intersection.is_empty else 0.0
            frac = min(shaded_len / total_len, 1.0)
            shade_fractions.append(round(frac, 4))
        except Exception:
            shade_fractions.append(0.0)

    return shade_fractions


# ── 5. HTML Map Generation ───────────────────────────────────────────

def generate_html_map(streets_wgs, shade_data, shadow_geojsons, sun_positions, out_path,
                      canopy_shadow_geojsons=None):
    """Generate interactive Leaflet HTML map with time slider."""

    # Prepare canopy shadow GeoJSON strings
    canopy_shadow_json_strs = {}
    if canopy_shadow_geojsons:
        for h, gjson in canopy_shadow_geojsons.items():
            canopy_shadow_json_strs[h] = gjson

    # Prepare street features with shade data
    street_features = []
    for idx, row in streets_wgs.iterrows():
        props = {
            'length': round(row.get('length', 0), 1),
            'shade_avg': round(shade_data.loc[idx, 'shade_avg'] * 100, 1) if idx in shade_data.index else 0,
        }
        for h in TIMES_LOCAL:
            col = f'shade_{h:02d}'
            props[col] = round(shade_data.loc[idx, col] * 100, 1) if idx in shade_data.index else 0

        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        coords = list(geom.coords)
        feature = {
            'type': 'Feature',
            'properties': props,
            'geometry': {
                'type': 'LineString',
                'coordinates': [[c[0], c[1]] for c in coords]
            }
        }
        street_features.append(feature)

    streets_geojson = json.dumps({'type': 'FeatureCollection', 'features': street_features})

    # Prepare shadow GeoJSONs (simplified for embedding)
    shadow_json_strs = {}
    for h, gjson in shadow_geojsons.items():
        shadow_json_strs[h] = gjson

    # Compute stats
    avg_shade = shade_data['shade_avg'].mean() * 100
    well_shaded = (shade_data['shade_avg'] >= 0.5).sum()
    exposed = (shade_data['shade_avg'] < 0.2).sum()
    total = len(shade_data)

    # Sun position info for display
    sun_info = {}
    for h, sp in sun_positions.items():
        sun_info[h] = {
            'altitude': round(sp['altitude'], 1),
            'azimuth': round(sp['azimuth'], 1),
            'shadow_factor': round(sp['shadow_factor'], 2)
        }

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Building Shade Analysis - Al Karama, Dubai</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        #map {{ position: absolute; top: 0; left: 0; right: 0; bottom: 0; }}

        .control-panel {{
            position: absolute; top: 10px; right: 10px; z-index: 1000;
            background: white; padding: 15px; border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3); max-width: 320px;
            font-size: 13px;
        }}
        .control-panel h3 {{ margin-bottom: 10px; color: #1565c0; font-size: 15px; }}

        .time-slider {{
            position: absolute; bottom: 30px; left: 50%; transform: translateX(-50%);
            z-index: 1000; background: white; padding: 12px 24px;
            border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            text-align: center; min-width: 400px;
        }}
        .time-slider input[type=range] {{ width: 100%; margin: 8px 0; }}
        .time-label {{ font-weight: bold; font-size: 16px; color: #1565c0; }}
        .time-marks {{ display: flex; justify-content: space-between; font-size: 11px; color: #666; }}

        .stats-box {{ background: #f5f5f5; padding: 10px; border-radius: 6px; margin: 8px 0; }}
        .stats-box .stat-row {{ display: flex; justify-content: space-between; margin: 3px 0; }}
        .stats-box .stat-label {{ color: #666; }}
        .stats-box .stat-value {{ font-weight: bold; }}

        .sun-info {{ background: #fff8e1; padding: 8px 10px; border-radius: 6px; margin: 8px 0; border-left: 3px solid #ffa000; }}
        .sun-info .sun-title {{ font-weight: bold; color: #f57f17; margin-bottom: 4px; }}

        .legend {{ margin-top: 10px; }}
        .legend-item {{ display: flex; align-items: center; margin: 3px 0; }}
        .legend-color {{ width: 30px; height: 4px; margin-right: 8px; border-radius: 2px; }}

        .toggle-row {{ margin: 5px 0; }}
        .toggle-row label {{ cursor: pointer; }}

        .back-link {{
            position: absolute; top: 10px; left: 10px; z-index: 1000;
            background: white; padding: 8px 14px; border-radius: 6px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2); text-decoration: none;
            color: #1565c0; font-size: 13px; font-weight: 500;
        }}
        .back-link:hover {{ background: #e3f2fd; }}
    </style>
</head>
<body>
<div id="map"></div>
<a href="../index.html" class="back-link">&#8592; Dashboard</a>
<a href="shade_map_3d.html" class="back-link" style="left:120px;">View in 3D &#8594;</a>

<div class="control-panel">
    <h3>Building Shade Analysis</h3>
    <p style="color:#666; font-size:12px; margin-bottom:8px;">July 15 (peak summer) - Al Karama, Dubai</p>

    <div class="sun-info">
        <div class="sun-title">Sun Position</div>
        <div id="sunAlt">Altitude: --</div>
        <div id="sunAz">Azimuth: --</div>
        <div id="sunShadow">Shadow factor: --</div>
        <div style="margin-top:6px; font-size:11px; color:#795548;">Sunrise: ~5:38 AM &nbsp;|&nbsp; Sunset: ~7:11 PM<br>Day length: ~13.6 hours</div>
    </div>

    <div class="stats-box">
        <div class="stat-row"><span class="stat-label">Daily avg shade:</span><span class="stat-value" id="statAvg">{avg_shade:.1f}%</span></div>
        <div class="stat-row"><span class="stat-label">Well-shaded (&ge;50%):</span><span class="stat-value" id="statShaded">{well_shaded}</span></div>
        <div class="stat-row"><span class="stat-label">Exposed (&lt;20%):</span><span class="stat-value" id="statExposed">{exposed}</span></div>
        <div class="stat-row"><span class="stat-label">Total streets:</span><span class="stat-value">{total}</span></div>
    </div>

    <div class="toggle-row">
        <label><input type="checkbox" id="toggleShadows" checked> Show building shadows</label>
    </div>
    <div class="toggle-row">
        <label><input type="checkbox" id="toggleTreeShadows" checked> Show tree shadows</label>
    </div>
    <div class="toggle-row">
        <label><input type="checkbox" id="toggleStreets" checked> Show street shade %</label>
    </div>

    <div class="legend">
        <strong>Street Shade Coverage</strong>
        <div class="legend-item"><div class="legend-color" style="background:#d32f2f"></div> 0% (fully exposed)</div>
        <div class="legend-item"><div class="legend-color" style="background:#ff9800"></div> 25%</div>
        <div class="legend-item"><div class="legend-color" style="background:#fdd835"></div> 50%</div>
        <div class="legend-item"><div class="legend-color" style="background:#8bc34a"></div> 75%</div>
        <div class="legend-item"><div class="legend-color" style="background:#2e7d32"></div> 100% (fully shaded)</div>
    </div>
</div>

<div class="time-slider">
    <div class="time-label" id="timeLabel">12:00 PM</div>
    <input type="range" id="timeSlider" min="0" max="4" value="2" step="1">
    <div class="time-marks">
        <span>8 AM</span><span>10 AM</span><span>12 PM</span><span>2 PM</span><span>4 PM</span>
    </div>
</div>

<script>
var times = [8, 10, 12, 14, 16];
var timeLabels = ['8:00 AM', '10:00 AM', '12:00 PM', '2:00 PM', '4:00 PM'];
var sunInfo = {json.dumps(sun_info)};

var map = L.map('map').setView([{LATITUDE}, {LONGITUDE}], 15);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
    attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
    maxZoom: 19
}}).addTo(map);

// Street data
var streetsData = {streets_geojson};

// Shadow data per time
var shadowData = {{
"""

    for h in TIMES_LOCAL:
        html += f"    {h}: {shadow_json_strs.get(h, '{}')},\n"

    html += "};\n\n// Tree shadow data per time\nvar treeShadowData = {\n"

    for h in TIMES_LOCAL:
        html += f"    {h}: {canopy_shadow_json_strs.get(h, '{}')},\n"

    html += """};

// Layers
var streetLayer = null;
var shadowLayer = null;
var treeShadowLayer = null;

function shadeColor(pct) {
    if (pct >= 75) return '#2e7d32';
    if (pct >= 50) return '#8bc34a';
    if (pct >= 25) return '#fdd835';
    if (pct >= 10) return '#ff9800';
    return '#d32f2f';
}

function updateMap(timeIdx) {
    var h = times[timeIdx];
    var hStr = h < 10 ? '0' + h : '' + h;
    var shadeKey = 'shade_' + hStr;

    // Update time label
    document.getElementById('timeLabel').textContent = timeLabels[timeIdx];

    // Update sun info
    var si = sunInfo[h];
    document.getElementById('sunAlt').textContent = 'Altitude: ' + si.altitude + '\\u00b0';
    document.getElementById('sunAz').textContent = 'Azimuth: ' + si.azimuth + '\\u00b0';
    document.getElementById('sunShadow').textContent = 'Shadow factor: ' + si.shadow_factor + 'x height';

    // Update streets
    if (streetLayer) map.removeLayer(streetLayer);
    if (document.getElementById('toggleStreets').checked) {
        streetLayer = L.geoJSON(streetsData, {
            style: function(feature) {
                var shade = feature.properties[shadeKey] || 0;
                return {
                    color: shadeColor(shade),
                    weight: 3,
                    opacity: 0.85
                };
            },
            onEachFeature: function(feature, layer) {
                var p = feature.properties;
                var shade = p[shadeKey] || 0;
                layer.bindPopup(
                    '<strong>Street Segment</strong><br>' +
                    'Length: ' + p.length + 'm<br>' +
                    'Shade at ' + timeLabels[timeIdx] + ': <strong>' + shade.toFixed(1) + '%</strong><br>' +
                    'Daily avg shade: ' + p.shade_avg.toFixed(1) + '%'
                );
            }
        }).addTo(map);
    }

    // Update building shadows
    if (shadowLayer) map.removeLayer(shadowLayer);
    if (document.getElementById('toggleShadows').checked && shadowData[h] && shadowData[h].features) {
        shadowLayer = L.geoJSON(shadowData[h], {
            style: {
                fillColor: '#263238',
                fillOpacity: 0.25,
                color: '#455a64',
                weight: 0.5,
                opacity: 0.4
            }
        }).addTo(map);
    }

    // Update tree shadows
    if (treeShadowLayer) map.removeLayer(treeShadowLayer);
    if (document.getElementById('toggleTreeShadows').checked && treeShadowData[h] && treeShadowData[h].features) {
        treeShadowLayer = L.geoJSON(treeShadowData[h], {
            style: {
                fillColor: '#1b5e20',
                fillOpacity: 0.3,
                color: '#2e7d32',
                weight: 0.5,
                opacity: 0.5
            }
        }).addTo(map);
    }

    // Compute time-specific stats
    var shaded = 0, exposed = 0;
    streetsData.features.forEach(function(f) {
        var v = f.properties[shadeKey] || 0;
        if (v >= 50) shaded++;
        else if (v < 20) exposed++;
    });
    document.getElementById('statShaded').textContent = shaded;
    document.getElementById('statExposed').textContent = exposed;
}

// Event listeners
document.getElementById('timeSlider').addEventListener('input', function() {
    updateMap(parseInt(this.value));
});
document.getElementById('toggleShadows').addEventListener('change', function() {
    updateMap(parseInt(document.getElementById('timeSlider').value));
});
document.getElementById('toggleTreeShadows').addEventListener('change', function() {
    updateMap(parseInt(document.getElementById('timeSlider').value));
});
document.getElementById('toggleStreets').addEventListener('change', function() {
    updateMap(parseInt(document.getElementById('timeSlider').value));
});

// Initial render
updateMap(2);
</script>
</body>
</html>"""

    with open(out_path, 'w') as f:
        f.write(html)
    print(f"\n  HTML map saved: {out_path}")


# ── 5b. 3D HTML Map Generation (deck.gl + SunLight) ──────────────────

def generate_3d_html_map(buildings_wgs, streets_wgs, shade_data, sun_positions, out_path, canopy_wgs=None):
    """Generate interactive 3D HTML map with deck.gl _SunLight shadow rendering."""

    # Prepare canopy GeoJSON
    canopy_geojson = '{"type":"FeatureCollection","features":[]}'
    if canopy_wgs is not None and len(canopy_wgs) > 0:
        canopy_features = []
        for _, row in canopy_wgs.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            height = row.get('canopy_height_m', 0)
            if height is None or height <= 0:
                continue
            coords = list(geom.exterior.coords) if geom.geom_type == 'Polygon' else []
            if not coords:
                continue
            feature = {
                'type': 'Feature',
                'properties': {
                    'height': round(float(height), 1),
                    'type': 'tree',
                },
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[[c[0], c[1]] for c in coords]]
                }
            }
            canopy_features.append(feature)
        canopy_geojson = json.dumps({
            'type': 'FeatureCollection',
            'features': canopy_features
        })
        print(f"  Canopy features for 3D map: {len(canopy_features)}")

    # Prepare buildings GeoJSON
    building_features = []
    for _, row in buildings_wgs.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        height = row.get('height', 0)
        if height is None or height <= 0:
            continue
        coords = list(geom.exterior.coords)
        feature = {
            'type': 'Feature',
            'properties': {
                'name': row.get('name', '') or '',
                'height': round(float(height), 1),
                'levels': row.get('levels', '') or '',
                'building_type': row.get('building_type', '') or '',
            },
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[c[0], c[1]] for c in coords]]
            }
        }
        building_features.append(feature)

    buildings_geojson = json.dumps({
        'type': 'FeatureCollection',
        'features': building_features
    })

    # Prepare street features with shade data
    street_features = []
    for idx, row in streets_wgs.iterrows():
        props = {
            'length': round(row.get('length', 0), 1),
            'shade_avg': round(shade_data.loc[idx, 'shade_avg'] * 100, 1) if idx in shade_data.index else 0,
        }
        for h in TIMES_LOCAL:
            col = f'shade_{h:02d}'
            props[col] = round(shade_data.loc[idx, col] * 100, 1) if idx in shade_data.index else 0

        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        coords = list(geom.coords)
        feature = {
            'type': 'Feature',
            'properties': props,
            'geometry': {
                'type': 'LineString',
                'coordinates': [[c[0], c[1]] for c in coords]
            }
        }
        street_features.append(feature)

    streets_geojson = json.dumps({
        'type': 'FeatureCollection',
        'features': street_features
    })

    # Compute stats
    avg_shade = shade_data['shade_avg'].mean() * 100
    well_shaded = int((shade_data['shade_avg'] >= 0.5).sum())
    exposed = int((shade_data['shade_avg'] < 0.2).sum())
    total = len(shade_data)

    # Sun position info
    sun_info = {}
    for h, sp in sun_positions.items():
        sun_info[h] = {
            'altitude': round(sp['altitude'], 1),
            'azimuth': round(sp['azimuth'], 1),
            'shadow_factor': round(sp['shadow_factor'], 2)
        }

    # Build UTC timestamps for _SunLight (July 15, 2024)
    utc_timestamps = {}
    for h in TIMES_LOCAL:
        hour_utc = h - TIMEZONE_OFFSET
        # JavaScript Date.UTC(year, monthIndex, day, hour)
        utc_timestamps[h] = f"Date.UTC(2024, 6, 15, {hour_utc}, 0, 0)"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>3D Shade Analysis - Al Karama, Dubai</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://unpkg.com/deck.gl@8.9.0/dist.min.js"></script>
    <script src="https://unpkg.com/maplibre-gl@2.4.0/dist/maplibre-gl.js"></script>
    <link href="https://unpkg.com/maplibre-gl@2.4.0/dist/maplibre-gl.css" rel="stylesheet">
    <style>
        body {{ margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; overflow: hidden; }}
        #container {{ width: 100vw; height: 100vh; position: relative; }}

        .panel {{
            position: absolute; top: 10px; right: 10px; z-index: 1;
            background: rgba(0,0,0,0.88); color: #eee; padding: 16px;
            border-radius: 10px; max-width: 320px; font-size: 13px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        }}
        .panel h3 {{ margin: 0 0 8px 0; color: #4fc3f7; font-size: 16px; }}
        .panel h4 {{ margin: 12px 0 6px 0; color: #aaa; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }}

        .sun-info {{
            background: rgba(255,160,0,0.12); padding: 8px 10px; border-radius: 6px;
            margin: 8px 0; border-left: 3px solid #ffa000;
        }}
        .sun-info .sun-title {{ font-weight: bold; color: #ffb300; margin-bottom: 4px; }}

        .stats-box {{ background: rgba(255,255,255,0.06); padding: 10px; border-radius: 6px; margin: 8px 0; }}
        .stat-row {{ display: flex; justify-content: space-between; margin: 3px 0; }}
        .stat-label {{ color: #999; }}
        .stat-value {{ font-weight: bold; color: #81c784; }}

        .toggle-row {{ margin: 5px 0; }}
        .toggle-row label {{ cursor: pointer; color: #ccc; }}

        .legend {{ margin-top: 10px; }}
        .legend-title {{ color: #aaa; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }}
        .legend-item {{ display: flex; align-items: center; margin: 3px 0; font-size: 12px; color: #bbb; }}
        .legend-color {{ width: 24px; height: 4px; margin-right: 8px; border-radius: 2px; flex-shrink: 0; }}

        .time-slider {{
            position: absolute; bottom: 30px; left: 50%; transform: translateX(-50%);
            z-index: 1; background: rgba(0,0,0,0.88); padding: 12px 24px;
            border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.4);
            text-align: center; min-width: 420px; color: #eee;
        }}
        .time-slider input[type=range] {{ width: 100%; margin: 8px 0; accent-color: #4fc3f7; }}
        .time-label {{ font-weight: bold; font-size: 16px; color: #4fc3f7; }}
        .time-marks {{ display: flex; justify-content: space-between; font-size: 11px; color: #888; }}

        .nav-links {{
            position: absolute; top: 10px; left: 10px; z-index: 1;
            display: flex; gap: 8px;
        }}
        .nav-link {{
            background: rgba(0,0,0,0.88); padding: 8px 14px; border-radius: 6px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3); text-decoration: none;
            color: #4fc3f7; font-size: 13px; font-weight: 500;
        }}
        .nav-link:hover {{ background: rgba(79,195,247,0.15); }}

        .mode-btn {{
            padding: 5px 10px; border: 1px solid #555; background: #222;
            color: #ccc; cursor: pointer; font-size: 11px; border-radius: 4px;
        }}
        .mode-btn:hover {{ background: #333; }}
        .mode-btn.active {{ background: #1976d2; color: #fff; border-color: #1976d2; }}

        .controls-hint {{
            position: absolute; bottom: 120px; left: 50%; transform: translateX(-50%);
            z-index: 1; color: #555; font-size: 11px; text-align: center;
        }}
    </style>
</head>
<body>
<div id="container"></div>

<div class="nav-links">
    <a href="../index.html" class="nav-link">&#8592; Dashboard</a>
    <a href="shade_map.html" class="nav-link">2D View</a>
</div>

<div class="panel">
    <h3>3D Shade Analysis</h3>
    <p style="color:#888; font-size:12px; margin:0 0 8px;">July 15 (peak summer) &mdash; Al Karama, Dubai</p>
    <p style="color:#666; font-size:11px; margin:0 0 8px;">GPU-rendered sun shadows from building geometry</p>

    <div class="sun-info">
        <div class="sun-title">Sun Position</div>
        <div style="display:flex; align-items:center; gap:10px;">
            <svg id="sunCompass" width="80" height="80" viewBox="0 0 80 80" style="flex-shrink:0;">
                <!-- compass ring -->
                <circle cx="40" cy="40" r="36" fill="none" stroke="#555" stroke-width="1"/>
                <circle cx="40" cy="40" r="2" fill="#888"/>
                <!-- cardinal labels -->
                <text x="40" y="9" text-anchor="middle" font-size="8" fill="#888">N</text>
                <text x="73" y="43" text-anchor="middle" font-size="8" fill="#888">E</text>
                <text x="40" y="78" text-anchor="middle" font-size="8" fill="#888">S</text>
                <text x="7" y="43" text-anchor="middle" font-size="8" fill="#888">W</text>
                <!-- tick marks -->
                <line x1="40" y1="5" x2="40" y2="10" stroke="#666" stroke-width="1"/>
                <line x1="75" y1="40" x2="70" y2="40" stroke="#666" stroke-width="1"/>
                <line x1="40" y1="75" x2="40" y2="70" stroke="#666" stroke-width="1"/>
                <line x1="5" y1="40" x2="10" y2="40" stroke="#666" stroke-width="1"/>
                <!-- sun dot + line (rotated by JS) -->
                <g id="compassNeedle">
                    <line x1="40" y1="40" x2="40" y2="12" stroke="#ffb300" stroke-width="2" stroke-opacity="0.6"/>
                    <circle cx="40" cy="10" r="6" fill="#ffb300" opacity="0.9"/>
                    <text x="40" y="13" text-anchor="middle" font-size="7" font-weight="bold" fill="#333">&#9788;</text>
                </g>
            </svg>
            <div style="flex:1;">
                <div id="sunAlt">Altitude: --</div>
                <div id="sunAz">Azimuth: --</div>
                <div id="sunShadow">Shadow factor: --</div>
            </div>
        </div>
        <div style="margin-top:6px; font-size:11px; color:#a1887f;">
            Sunrise: ~5:38 AM &nbsp;|&nbsp; Sunset: ~7:11 PM<br>Day length: ~13.6 hours
        </div>
    </div>

    <div class="stats-box">
        <h4 style="margin-top:0">Daily Summary</h4>
        <div class="stat-row"><span class="stat-label">Avg shade coverage:</span><span class="stat-value">{avg_shade:.1f}%</span></div>
        <div class="stat-row"><span class="stat-label">Well-shaded (&ge;50%):</span><span class="stat-value" id="statShaded">{well_shaded}</span></div>
        <div class="stat-row"><span class="stat-label">Exposed (&lt;20%):</span><span class="stat-value" id="statExposed">{exposed}</span></div>
        <div class="stat-row"><span class="stat-label">Total streets:</span><span class="stat-value">{total}</span></div>
    </div>

    <div class="toggle-row">
        <label><input type="checkbox" id="toggleBuildings" checked> Buildings (extruded)</label>
    </div>
    <div class="toggle-row">
        <label><input type="checkbox" id="toggleCanopy" checked> Tree canopy (extruded)</label>
    </div>
    <div class="toggle-row">
        <label><input type="checkbox" id="toggleShadows" checked> Sun shadows (GPU)</label>
    </div>

    <h4 style="margin:12px 0 6px 0; color:#aaa; font-size:12px; text-transform:uppercase; letter-spacing:1px;">Street Display</h4>
    <div style="display:flex; gap:4px; flex-wrap:wrap;">
        <button class="mode-btn active" id="modeNeutral" onclick="setStreetMode('neutral')">Shadow view</button>
        <button class="mode-btn" id="modeShade" onclick="setStreetMode('shade')">Shade %</button>
        <button class="mode-btn" id="modeOff" onclick="setStreetMode('off')">Off</button>
    </div>
    <div style="color:#777; font-size:11px; margin-top:4px;" id="modeHint">Semi-transparent streets &mdash; GPU shadows visible on roads</div>

    <div class="legend" id="legendShade" style="display:none;">
        <div class="legend-title">Street Shade Coverage</div>
        <div class="legend-item"><div class="legend-color" style="background:#d32f2f"></div> 0% (fully exposed)</div>
        <div class="legend-item"><div class="legend-color" style="background:#ff9800"></div> 25%</div>
        <div class="legend-item"><div class="legend-color" style="background:#fdd835"></div> 50%</div>
        <div class="legend-item"><div class="legend-color" style="background:#8bc34a"></div> 75%</div>
        <div class="legend-item"><div class="legend-color" style="background:#2e7d32"></div> 100% (fully shaded)</div>
    </div>
    <div class="legend" id="legendNeutral">
        <div class="legend-title">Shadow View</div>
        <div class="legend-item"><div class="legend-color" style="background:rgba(255,255,255,0.5)"></div> Street (light = sun-exposed)</div>
        <div class="legend-item"><div class="legend-color" style="background:rgba(60,60,60,0.7)"></div> Street (dark = in shadow)</div>
    </div>
</div>

<div class="time-slider">
    <div class="time-label" id="timeLabel">12:00 PM</div>
    <input type="range" id="timeSlider" min="0" max="4" value="2" step="1">
    <div class="time-marks">
        <span>8 AM</span><span>10 AM</span><span>12 PM</span><span>2 PM</span><span>4 PM</span>
    </div>
</div>

<div class="controls-hint">Ctrl+drag to tilt &bull; Scroll to zoom &bull; Right-drag to rotate</div>

<script>
// ── Data ──
var BUILDINGS = {buildings_geojson};
var STREETS = {streets_geojson};
var CANOPY = {canopy_geojson};

var TIMES = [8, 10, 12, 14, 16];
var TIME_LABELS = ['8:00 AM', '10:00 AM', '12:00 PM', '2:00 PM', '4:00 PM'];
var SUN_INFO = {json.dumps(sun_info)};

// UTC timestamps for _SunLight (July 15, 2024 — month index 6 = July)
var UTC_TIMESTAMPS = {{
"""

    for h in TIMES_LOCAL:
        hour_utc = h - TIMEZONE_OFFSET
        html += f"    {h}: new Date(Date.UTC(2024, 6, 15, {hour_utc}, 0, 0)),\n"

    html += f"""}};

// ── State ──
var currentTimeIdx = 2; // start at noon
var showBuildings = true;
var showCanopy = true;
var streetMode = 'neutral'; // 'neutral' | 'shade' | 'off'
var showShadows = true;

// ── Color helpers ──
function shadeColor(pct) {{
    if (pct >= 75) return [46, 125, 50];
    if (pct >= 50) return [139, 195, 74];
    if (pct >= 25) return [253, 216, 53];
    if (pct >= 10) return [255, 152, 0];
    return [211, 47, 47];
}}

// ── Street mode switcher ──
function setStreetMode(mode) {{
    streetMode = mode;
    // Update button states
    document.getElementById('modeNeutral').className = mode === 'neutral' ? 'mode-btn active' : 'mode-btn';
    document.getElementById('modeShade').className = mode === 'shade' ? 'mode-btn active' : 'mode-btn';
    document.getElementById('modeOff').className = mode === 'off' ? 'mode-btn active' : 'mode-btn';
    // Update hint text
    var hints = {{
        neutral: 'Semi-transparent streets \\u2014 GPU shadows visible on roads',
        shade: 'Streets colored by pre-computed shade percentage',
        off: 'Streets hidden'
    }};
    document.getElementById('modeHint').textContent = hints[mode];
    // Toggle legends
    document.getElementById('legendNeutral').style.display = mode === 'neutral' ? '' : 'none';
    document.getElementById('legendShade').style.display = mode === 'shade' ? '' : 'none';
    updateMap(currentTimeIdx);
}}

// ── Build layers ──
function buildLayers() {{
    var h = TIMES[currentTimeIdx];
    var shadeKey = 'shade_' + (h < 10 ? '0' + h : '' + h);
    var layers = [];

    // Buildings layer
    if (showBuildings) {{
        layers.push(new deck.PolygonLayer({{
            id: 'buildings',
            data: BUILDINGS.features,
            extruded: true,
            wireframe: false,
            opacity: 0.85,
            getPolygon: function(d) {{ return d.geometry.coordinates; }},
            getElevation: function(d) {{ return d.properties.height || 9; }},
            getFillColor: [160, 170, 180],
            material: {{
                ambient: 0.6,
                diffuse: 0.7,
                shininess: 32,
                specularColor: [120, 125, 130]
            }},
            pickable: true
        }}));
    }}

    // Canopy layer — green extruded polygons
    if (showCanopy && CANOPY.features && CANOPY.features.length > 0) {{
        layers.push(new deck.PolygonLayer({{
            id: 'canopy',
            data: CANOPY.features,
            extruded: true,
            wireframe: false,
            opacity: 0.7,
            getPolygon: function(d) {{ return d.geometry.coordinates; }},
            getElevation: function(d) {{ return d.properties.height || 3; }},
            getFillColor: [70, 180, 90],
            material: {{
                ambient: 0.6,
                diffuse: 0.7,
                shininess: 20,
                specularColor: [60, 140, 70]
            }},
            pickable: true
        }}));
    }}

    // Streets layer — neutral mode: semi-transparent white so GPU shadows show through
    if (streetMode === 'neutral') {{
        layers.push(new deck.PathLayer({{
            id: 'streets-neutral',
            data: STREETS.features,
            getPath: function(d) {{ return d.geometry.coordinates; }},
            getColor: [220, 220, 220, 90],
            getWidth: 4,
            widthUnits: 'pixels',
            pickable: true,
            opacity: 1.0
        }}));
    }}

    // Streets layer — shade % mode: colored by pre-computed shade fraction
    if (streetMode === 'shade') {{
        layers.push(new deck.PathLayer({{
            id: 'streets-shade',
            data: STREETS.features,
            getPath: function(d) {{ return d.geometry.coordinates; }},
            getColor: function(d) {{
                var shade = d.properties[shadeKey] || 0;
                return shadeColor(shade);
            }},
            getWidth: 3,
            widthUnits: 'pixels',
            pickable: true,
            opacity: 0.9,
            updateTriggers: {{
                getColor: [shadeKey]
            }}
        }}));
    }}

    return layers;
}}

// ── Lighting with _SunLight ──
function buildLightingEffect(timeIdx) {{
    var h = TIMES[timeIdx];
    var timestamp = UTC_TIMESTAMPS[h];

    var sunLight = new deck._SunLight({{
        timestamp: timestamp.getTime(),
        color: [255, 255, 255],
        intensity: 1.0
    }});

    var ambientLight = new deck.AmbientLight({{
        color: [255, 255, 255],
        intensity: 0.8
    }});

    var lightingEffect = new deck.LightingEffect({{
        ambientLight: ambientLight,
        sunLight: sunLight
    }});
    lightingEffect.shadowColor = [0, 0, 0, showShadows ? 0.35 : 0.0];

    return lightingEffect;
}}

// ── Initialize deck.gl ──
var deckgl = new deck.DeckGL({{
    container: 'container',
    mapStyle: 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
    initialViewState: {{
        longitude: {LONGITUDE},
        latitude: {LATITUDE},
        zoom: 15,
        pitch: 55,
        bearing: -20,
        maxPitch: 85
    }},
    controller: true,
    layers: buildLayers(),
    effects: [buildLightingEffect(2)],
    getTooltip: function(info) {{
        if (!info.object) return null;
        var d = info.object;
        var p = d.properties;
        if (info.layer && info.layer.id === 'buildings') {{
            return {{
                html: '<div style="padding:6px;max-width:250px;">' +
                    '<b>' + (p.name || 'Building') + '</b><br>' +
                    'Height: ' + p.height + 'm' +
                    (p.levels ? '<br>Levels: ' + p.levels : '') +
                    (p.building_type && p.building_type !== 'yes' ? '<br>Type: ' + p.building_type : '') +
                    '</div>',
                style: {{ background: 'rgba(0,0,0,0.85)', color: '#eee', fontSize: '12px', borderRadius: '6px' }}
            }};
        }}
        if (info.layer && info.layer.id === 'canopy') {{
            return {{
                html: '<div style="padding:6px;max-width:250px;">' +
                    '<b>Tree Canopy</b><br>' +
                    'Height: ' + p.height + 'm' +
                    '</div>',
                style: {{ background: 'rgba(0,0,0,0.85)', color: '#eee', fontSize: '12px', borderRadius: '6px' }}
            }};
        }}
        if (info.layer && (info.layer.id === 'streets-neutral' || info.layer.id === 'streets-shade')) {{
            var h = TIMES[currentTimeIdx];
            var shadeKey = 'shade_' + (h < 10 ? '0' + h : '' + h);
            var shade = p[shadeKey] || 0;
            return {{
                html: '<div style="padding:6px;max-width:250px;">' +
                    '<b>Street Segment</b><br>' +
                    'Length: ' + (p.length || 0) + 'm<br>' +
                    'Shade at ' + TIME_LABELS[currentTimeIdx] + ': <b>' + shade.toFixed(1) + '%</b><br>' +
                    'Daily avg: ' + (p.shade_avg || 0).toFixed(1) + '%' +
                    '</div>',
                style: {{ background: 'rgba(0,0,0,0.85)', color: '#eee', fontSize: '12px', borderRadius: '6px' }}
            }};
        }}
        return null;
    }}
}});

// ── Update function ──
function updateMap(timeIdx) {{
    currentTimeIdx = timeIdx;
    var h = TIMES[timeIdx];

    // Update time label
    document.getElementById('timeLabel').textContent = TIME_LABELS[timeIdx];

    // Update sun info
    var si = SUN_INFO[h];
    document.getElementById('sunAlt').textContent = 'Altitude: ' + si.altitude + '\\u00b0';
    document.getElementById('sunAz').textContent = 'Azimuth: ' + si.azimuth + '\\u00b0';
    document.getElementById('sunShadow').textContent = 'Shadow factor: ' + si.shadow_factor + 'x height';

    // Update compass needle — azimuth is clockwise from north, same as SVG rotate
    document.getElementById('compassNeedle').setAttribute('transform', 'rotate(' + si.azimuth + ', 40, 40)');

    // Update time-specific stats
    var shadeKey = 'shade_' + (h < 10 ? '0' + h : '' + h);
    var shaded = 0, exp = 0;
    STREETS.features.forEach(function(f) {{
        var v = f.properties[shadeKey] || 0;
        if (v >= 50) shaded++;
        else if (v < 20) exp++;
    }});
    document.getElementById('statShaded').textContent = shaded;
    document.getElementById('statExposed').textContent = exp;

    // Update deck.gl layers and lighting
    deckgl.setProps({{
        layers: buildLayers(),
        effects: [buildLightingEffect(timeIdx)]
    }});

}}

// ── Event listeners ──
document.getElementById('timeSlider').addEventListener('input', function() {{
    updateMap(parseInt(this.value));
}});
document.getElementById('toggleBuildings').addEventListener('change', function() {{
    showBuildings = this.checked;
    updateMap(currentTimeIdx);
}});
document.getElementById('toggleCanopy').addEventListener('change', function() {{
    showCanopy = this.checked;
    updateMap(currentTimeIdx);
}});
document.getElementById('toggleShadows').addEventListener('change', function() {{
    showShadows = this.checked;
    updateMap(currentTimeIdx);
}});

// Initial render
updateMap(2);
</script>
</body>
</html>"""

    with open(out_path, 'w') as f:
        f.write(html)
    print(f"\n  3D HTML map saved: {out_path}")


# ── 6. Main Pipeline ─────────────────────────────────────────────────

def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  BUILDING SHADE ANALYSIS - Al Karama, Dubai")
    print("  Date: July 15 (peak summer)")
    print("=" * 60)

    # ── Step 1: Sun positions ──
    print("\n[1/6] Calculating sun positions...")
    sun_positions = get_sun_positions()

    # ── Step 2: Load data ──
    print("\n[2/6] Loading building and street data...")
    buildings = gpd.read_file(BUILDINGS_PATH)
    streets = gpd.read_file(STREETS_PATH)
    print(f"  Buildings: {len(buildings)}")
    print(f"  Streets:   {len(streets)}")

    # Load canopy polygons with heights (optional — graceful degradation)
    canopy = None
    canopy_utm = None
    if os.path.exists(CANOPY_PATH):
        try:
            canopy = gpd.read_file(CANOPY_PATH)
            canopy = canopy[canopy['canopy_height_m'].notna() & (canopy['canopy_height_m'] > 0)].copy()
            print(f"  Canopy polygons: {len(canopy)} (with height > 0)")
        except Exception as e:
            print(f"  WARNING: Could not load canopy data: {e}")
            canopy = None
    else:
        print(f"  NOTE: No canopy data found at {CANOPY_PATH} — running buildings-only")

    # Convert to UTM for metric calculations
    buildings_utm = buildings.to_crs(UTM_CRS)
    streets_utm = streets.to_crs(UTM_CRS)
    if canopy is not None and len(canopy) > 0:
        canopy_utm = canopy.to_crs(UTM_CRS)

    # Filter buildings with valid height
    buildings_utm = buildings_utm[buildings_utm['height'].notna() & (buildings_utm['height'] > 0)].copy()
    print(f"  Buildings with height: {len(buildings_utm)}")

    # ── Step 3: Compute shadows for each time ──
    print("\n[3/6] Computing shadow projections...")
    shadow_unions = {}
    shadow_geojsons_wgs = {}  # For HTML embedding (WGS84)
    canopy_shadow_geojsons_wgs = {}  # Canopy-only shadows for 2D map toggle
    shade_results = {}

    for hour in TIMES_LOCAL:
        sp = sun_positions[hour]
        alt, az = sp['altitude'], sp['azimuth']
        print(f"\n  {hour:02d}:00 - alt={alt:.1f}, az={az:.1f}")

        # Project shadows for all buildings
        shadow_geoms = []
        t_start = time.time()
        for idx, row in buildings_utm.iterrows():
            geom = row.geometry
            height = row['height']
            if geom is None or geom.is_empty or height <= 0:
                continue
            shadow = project_shadow(geom, height, alt, az)
            if shadow is not None and not shadow.is_empty:
                shadow_geoms.append(shadow)

        n_building_shadows = len(shadow_geoms)
        print(f"    Projected {n_building_shadows} building shadows in {time.time()-t_start:.1f}s")

        # Add canopy shadows
        canopy_shadow_geoms = []
        if canopy_utm is not None and len(canopy_utm) > 0:
            t_canopy = time.time()
            for _, crow in canopy_utm.iterrows():
                cgeom = crow.geometry
                cheight = crow['canopy_height_m']
                if cgeom is None or cgeom.is_empty or cheight <= 0:
                    continue
                cshadow = project_shadow(cgeom, cheight, alt, az)
                if cshadow is not None and not cshadow.is_empty:
                    canopy_shadow_geoms.append(cshadow)
            shadow_geoms.extend(canopy_shadow_geoms)
            print(f"    Projected {len(canopy_shadow_geoms)} canopy shadows in {time.time()-t_canopy:.1f}s")

        # Create GeoDataFrame of individual shadows
        shadow_gdf = gpd.GeoDataFrame(geometry=shadow_geoms, crs=UTM_CRS)

        # Chunked union
        t_start = time.time()
        shadow_union = chunked_shadow_union(shadow_gdf, grid_size=200)
        shadow_unions[hour] = shadow_union
        print(f"    Union completed in {time.time()-t_start:.1f}s")

        # Calculate street shade
        t_start = time.time()
        shade_fracs = calculate_street_shade(streets_utm, shadow_union)
        shade_results[hour] = shade_fracs

        avg_shade = np.mean(shade_fracs) * 100
        well_shaded = sum(1 for f in shade_fracs if f >= 0.5)
        print(f"    Street shade: avg={avg_shade:.1f}%, well-shaded={well_shaded}/{len(shade_fracs)}")
        print(f"    Shade calc completed in {time.time()-t_start:.1f}s")

        # Save shadow GeoJSON (simplified for file size)
        shadow_wgs = gpd.GeoDataFrame(
            geometry=[shadow_union.simplify(2)], crs=UTM_CRS
        ).to_crs('EPSG:4326')

        shadow_geojson_path = os.path.join(OUT_DIR, f'shadows_{hour:02d}.geojson')
        shadow_wgs.to_file(shadow_geojson_path, driver='GeoJSON')
        print(f"    Saved: {shadow_geojson_path}")

        # Prepare simplified GeoJSON for HTML embedding
        simplified = shadow_union.simplify(5)  # More aggressive simplification for embedding
        shadow_wgs_embed = gpd.GeoDataFrame(
            geometry=[simplified], crs=UTM_CRS
        ).to_crs('EPSG:4326')
        shadow_geojsons_wgs[hour] = shadow_wgs_embed.to_json()

        # Prepare canopy-only shadow GeoJSON for 2D map toggle
        if canopy_shadow_geoms:
            canopy_union = unary_union(canopy_shadow_geoms).buffer(0)
            canopy_simplified = canopy_union.simplify(5)
            canopy_wgs_embed = gpd.GeoDataFrame(
                geometry=[canopy_simplified], crs=UTM_CRS
            ).to_crs('EPSG:4326')
            canopy_shadow_geojsons_wgs[hour] = canopy_wgs_embed.to_json()

    # ── Step 4: Compile shade CSV ──
    print("\n[4/6] Compiling shade results...")
    shade_df = pd.DataFrame(index=streets.index)
    for hour in TIMES_LOCAL:
        shade_df[f'shade_{hour:02d}'] = shade_results[hour]
    shade_df['shade_avg'] = shade_df[[f'shade_{h:02d}' for h in TIMES_LOCAL]].mean(axis=1)

    # Add street properties
    shade_df['length'] = streets['length'].values
    if 'mid_lat' in streets.columns:
        shade_df['mid_lat'] = streets['mid_lat'].values
        shade_df['mid_lon'] = streets['mid_lon'].values

    csv_path = os.path.join(OUT_DIR, 'street_shade.csv')
    shade_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Summary stats
    print(f"\n  === Daily Summary ===")
    print(f"  Average shade coverage: {shade_df['shade_avg'].mean()*100:.1f}%")
    print(f"  Well-shaded streets (>=50%): {(shade_df['shade_avg'] >= 0.5).sum()} / {len(shade_df)}")
    print(f"  Exposed streets (<20%): {(shade_df['shade_avg'] < 0.2).sum()} / {len(shade_df)}")

    for hour in TIMES_LOCAL:
        col = f'shade_{hour:02d}'
        print(f"  {hour:02d}:00 - avg={shade_df[col].mean()*100:.1f}%, "
              f"max={shade_df[col].max()*100:.1f}%, "
              f"min={shade_df[col].min()*100:.1f}%")

    # ── Step 5: Generate HTML map ──
    print("\n[5/6] Generating interactive 2D map...")
    html_path = os.path.join(OUT_DIR, 'shade_map.html')
    generate_html_map(streets, shade_df, shadow_geojsons_wgs, sun_positions, html_path,
                      canopy_shadow_geojsons=canopy_shadow_geojsons_wgs)

    # ── Step 6: Generate 3D HTML map ──
    print("\n[6/6] Generating 3D interactive map (deck.gl + SunLight)...")
    html_3d_path = os.path.join(OUT_DIR, 'shade_map_3d.html')
    buildings_with_height = buildings[buildings['height'].notna() & (buildings['height'] > 0)].copy()
    generate_3d_html_map(buildings_with_height, streets, shade_df, sun_positions, html_3d_path,
                         canopy_wgs=canopy)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  COMPLETE - Total time: {elapsed:.1f}s")
    print(f"  Outputs in: {OUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
