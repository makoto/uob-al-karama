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

def generate_html_map(streets_wgs, shade_data, shadow_geojsons, sun_positions, out_path):
    """Generate interactive Leaflet HTML map with time slider."""

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

    html += """};

// Layers
var streetLayer = null;
var shadowLayer = null;

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

    // Update shadows
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


# ── 6. Main Pipeline ─────────────────────────────────────────────────

def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  BUILDING SHADE ANALYSIS - Al Karama, Dubai")
    print("  Date: July 15 (peak summer)")
    print("=" * 60)

    # ── Step 1: Sun positions ──
    print("\n[1/5] Calculating sun positions...")
    sun_positions = get_sun_positions()

    # ── Step 2: Load data ──
    print("\n[2/5] Loading building and street data...")
    buildings = gpd.read_file(BUILDINGS_PATH)
    streets = gpd.read_file(STREETS_PATH)
    print(f"  Buildings: {len(buildings)}")
    print(f"  Streets:   {len(streets)}")

    # Convert to UTM for metric calculations
    buildings_utm = buildings.to_crs(UTM_CRS)
    streets_utm = streets.to_crs(UTM_CRS)

    # Filter buildings with valid height
    buildings_utm = buildings_utm[buildings_utm['height'].notna() & (buildings_utm['height'] > 0)].copy()
    print(f"  Buildings with height: {len(buildings_utm)}")

    # ── Step 3: Compute shadows for each time ──
    print("\n[3/5] Computing shadow projections...")
    shadow_unions = {}
    shadow_geojsons_wgs = {}  # For HTML embedding (WGS84)
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

        print(f"    Projected {len(shadow_geoms)} shadows in {time.time()-t_start:.1f}s")

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

    # ── Step 4: Compile shade CSV ──
    print("\n[4/5] Compiling shade results...")
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
    print("\n[5/5] Generating interactive map...")
    html_path = os.path.join(OUT_DIR, 'shade_map.html')
    generate_html_map(streets, shade_df, shadow_geojsons_wgs, sun_positions, html_path)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  COMPLETE - Total time: {elapsed:.1f}s")
    print(f"  Outputs in: {OUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
