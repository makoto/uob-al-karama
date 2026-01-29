#!/usr/bin/env python3
"""
Export Al Karama 2D analysis layers as vector SVG and PDF files.

Reads JSON/GeoJSON data from docs/data/al_karama/ and generates
publication-quality vector maps matching the 2D viewer color schemes.

Usage:
    cd docs
    python scripts/export_vector_maps.py

Output:
    data/al_karama/exports/*.svg
    data/al_karama/exports/*.pdf
"""

import json
import os
import sys
import math
import io
import base64
import urllib.request
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap, BoundaryNorm, ListedColormap
from matplotlib.patches import Circle
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'al_karama')
OUT_DIR = os.path.join(DATA_DIR, 'exports')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(name):
    with open(os.path.join(DATA_DIR, name)) as f:
        return json.load(f)


def save(fig, name):
    svg_path = os.path.join(OUT_DIR, name + '.svg')
    pdf_path = os.path.join(OUT_DIR, name + '.pdf')
    fig.savefig(svg_path, format='svg', bbox_inches='tight', pad_inches=0.1, transparent=True)
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    svg_kb = os.path.getsize(svg_path) / 1024
    pdf_kb = os.path.getsize(pdf_path) / 1024
    print(f'  {name}.svg ({svg_kb:.0f} KB)  {name}.pdf ({pdf_kb:.0f} KB)')


def make_fig(title, aspect='equal'):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect(aspect)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.tick_params(labelsize=8)
    return fig, ax

# ---------------------------------------------------------------------------
# 1. Street Centrality (betweenness + closeness)
# ---------------------------------------------------------------------------

def export_centrality():
    print('Street Centrality...')
    streets = load_json('streets.geojson')

    for metric in ['betweenness', 'closeness']:
        fig, ax = make_fig(f'Street Centrality ({metric.title()})')

        segments = []
        values = []
        for feat in streets['features']:
            coords = feat['geometry']['coordinates']
            if feat['geometry']['type'] == 'LineString':
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                segments.append(list(zip(lons, lats)))
                values.append(feat['properties'].get(metric, 0) or 0)
            elif feat['geometry']['type'] == 'MultiLineString':
                for line in coords:
                    lons = [c[0] for c in line]
                    lats = [c[1] for c in line]
                    segments.append(list(zip(lons, lats)))
                    values.append(feat['properties'].get(metric, 0) or 0)

        values = np.array(values)

        if metric == 'betweenness':
            colors = ['#e1bee7', '#ce93d8', '#ab47bc', '#8e24aa', '#4a148c']
            vmax = 0.196
            label = 'Betweenness Centrality'
        else:
            colors = ['#bbdefb', '#90caf9', '#42a5f5', '#1e88e5', '#0d47a1']
            vmax = 0.00058
            label = 'Closeness Centrality'

        cmap = LinearSegmentedColormap.from_list('cent', colors, N=256)
        norm = Normalize(vmin=0, vmax=vmax)

        # Line widths proportional to value
        widths = 0.5 + (np.clip(values / vmax, 0, 1)) * 3

        lc = LineCollection(segments, linewidths=widths, cmap=cmap, norm=norm)
        lc.set_array(values)
        ax.add_collection(lc)
        ax.autoscale()

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label(label, fontsize=10)

        save(fig, f'centrality_{metric}')


# ---------------------------------------------------------------------------
# 2. Pedestrian Comfort
# ---------------------------------------------------------------------------

def export_comfort():
    print('Pedestrian Comfort...')
    data = load_json('segment_comfort.json')
    fig, ax = make_fig('Pedestrian Comfort Index (PCI)')

    lons = [d['lon'] for d in data]
    lats = [d['lat'] for d in data]
    vals = [d['pci_mean'] for d in data]

    colors = ['#e53935', '#ff9800', '#ffeb3b', '#4caf50', '#1b5e20']
    bounds = [0, 0.3, 0.4, 0.5, 0.6, 1.0]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)

    sc = ax.scatter(lons, lats, c=vals, cmap=cmap, norm=norm,
                    s=30, edgecolors='#333', linewidths=0.3, zorder=2)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02, ticks=[0.15, 0.35, 0.45, 0.55, 0.8])
    cbar.ax.set_yticklabels(['Bad (<0.3)', 'Poor', 'Fair', 'Good', 'High (>0.6)'], fontsize=8)
    cbar.set_label('Pedestrian Comfort Index', fontsize=10)

    save(fig, 'comfort')


# ---------------------------------------------------------------------------
# 3. Heat Mitigation Priority
# ---------------------------------------------------------------------------

def export_priority():
    print('Heat Mitigation Priority...')
    data = load_json('priority_points.json')
    fig, ax = make_fig('Heat Mitigation Priority')

    level_colors = {'Critical': '#d32f2f', 'High': '#f57c00', 'Medium': '#ffeb3b', 'Low': '#388e3c', '': '#999999'}
    level_order = ['Low', 'Medium', 'High', 'Critical']

    for level in level_order:
        pts = [d for d in data if d.get('priority_level') == level]
        if not pts:
            continue
        lons = [d['lon'] for d in pts]
        lats = [d['lat'] for d in pts]
        ax.scatter(lons, lats, c=level_colors[level], s=3, alpha=0.7,
                   label=level, edgecolors='none', zorder=2)

    ax.legend(loc='upper left', fontsize=9, title='Priority Level', framealpha=0.8)
    save(fig, 'priority')


# ---------------------------------------------------------------------------
# 4. Combined SVI (LST / GVI / SVF)
# ---------------------------------------------------------------------------

def export_combined():
    print('Combined SVI + Satellite...')
    data = load_json('combined_svi.json')
    lons = [d['lon'] for d in data]
    lats = [d['lat'] for d in data]

    metrics = {
        'lst': {
            'values': [d.get('lst', 0) for d in data],
            'title': 'Combined SVI - Land Surface Temperature',
            'cmap': LinearSegmentedColormap.from_list('lst',
                ['#3c50c8', '#3cc8c8', '#f0b428', '#dc3220'], N=256),
            'vmin': 46, 'vmax': 53,
            'label': 'Temperature (\u00b0C)'
        },
        'gvi': {
            'values': [d.get('gvi', 0) for d in data],
            'title': 'Combined SVI - Green View Index',
            'cmap': LinearSegmentedColormap.from_list('gvi',
                ['#f7f7f7', '#c7e9c0', '#74c476', '#238b45', '#006d2c'], N=256),
            'vmin': 0, 'vmax': 0.2,
            'label': 'GVI'
        },
        'svf': {
            'values': [d.get('svf', 0) for d in data],
            'title': 'Combined SVI - Sky View Factor',
            'cmap': LinearSegmentedColormap.from_list('svf',
                ['rgb(50,50,100)', 'rgb(140,140,178)', 'rgb(230,230,255)'], N=256) if False else
                LinearSegmentedColormap.from_list('svf',
                    [(50/255,50/255,100/255), (140/255,140/255,178/255), (230/255,230/255,1.0)], N=256),
            'vmin': 0, 'vmax': 1.0,
            'label': 'SVF'
        }
    }

    for key, m in metrics.items():
        fig, ax = make_fig(m['title'])
        norm = Normalize(vmin=m['vmin'], vmax=m['vmax'])
        sc = ax.scatter(lons, lats, c=m['values'], cmap=m['cmap'], norm=norm,
                        s=3, edgecolors='none', alpha=0.75, zorder=2)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label(m['label'], fontsize=10)
        save(fig, f'combined_{key}')


# ---------------------------------------------------------------------------
# 5. Satellite Grid (LST / NDVI / NDBI)
# ---------------------------------------------------------------------------

def export_satellite():
    print('Satellite Grid (30m)...')
    data = load_json('satellite_grid.json')
    lons = [d['lon'] for d in data]
    lats = [d['lat'] for d in data]

    # LST color ramp: RdYlBu reversed
    lst_colors = [
        (49/255,54/255,149/255),
        (69/255,117/255,180/255),
        (171/255,217/255,233/255),
        (254/255,224/255,144/255),
        (244/255,109/255,67/255),
        (165/255,0,38/255)
    ]

    metrics = {
        'lst': {
            'values': [d.get('lst', 0) for d in data],
            'title': 'Satellite Grid - Land Surface Temperature (30m)',
            'cmap': LinearSegmentedColormap.from_list('sat_lst', lst_colors, N=256),
            'vmin': 36, 'vmax': 54,
            'label': 'LST (\u00b0C)'
        },
        'ndvi': {
            'values': [d.get('ndvi', 0) for d in data],
            'title': 'Satellite Grid - Vegetation Index (NDVI)',
            'colors': ['#8B4513', '#D2691E', '#F5DEB3', '#90EE90', '#228B22', '#006400'],
            'bounds': [-0.2, 0, 0.1, 0.2, 0.3, 0.4, 0.6],
            'label': 'NDVI'
        },
        'ndbi': {
            'values': [d.get('ndbi', 0) for d in data],
            'title': 'Satellite Grid - Built-up Index (NDBI)',
            'colors': ['#1a9850', '#91cf60', '#fee08b', '#fc8d59', '#d73027'],
            'bounds': [-0.3, -0.1, 0, 0.1, 0.2, 0.4],
            'label': 'NDBI'
        }
    }

    for key, m in metrics.items():
        fig, ax = make_fig(m['title'])

        if 'cmap' in m:
            norm = Normalize(vmin=m['vmin'], vmax=m['vmax'])
            sc = ax.scatter(lons, lats, c=m['values'], cmap=m['cmap'], norm=norm,
                            s=2, edgecolors='none', alpha=0.8, zorder=2)
            cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
        else:
            cmap = ListedColormap(m['colors'])
            norm = BoundaryNorm(m['bounds'], cmap.N)
            sc = ax.scatter(lons, lats, c=m['values'], cmap=cmap, norm=norm,
                            s=2, edgecolors='none', alpha=0.8, zorder=2)
            cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02,
                                ticks=[(m['bounds'][i]+m['bounds'][i+1])/2 for i in range(len(m['bounds'])-1)])

        cbar.set_label(m['label'], fontsize=10)
        save(fig, f'satellite_{key}')


# ---------------------------------------------------------------------------
# 6. Climate Clusters
# ---------------------------------------------------------------------------

def export_clusters():
    print('Climate Clusters...')
    data = load_json('clusters.json')
    fig, ax = make_fig('Climate Clusters')

    cluster_colors = {0: '#e53935', 1: '#ff9800', 2: '#4caf50', 3: '#2196f3'}
    cluster_names = {0: 'Hot & Barren', 1: 'Warm Urban', 2: 'Shaded Urban', 3: 'Cool & Green'}

    for cid in sorted(cluster_colors.keys()):
        pts = [d for d in data if d.get('cluster') == cid]
        if not pts:
            continue
        lons = [d['lon'] for d in pts]
        lats = [d['lat'] for d in pts]
        label = cluster_names.get(cid, f'Cluster {cid}')
        ax.scatter(lons, lats, c=cluster_colors[cid], s=3, alpha=0.75,
                   label=label, edgecolors='none', zorder=2)

    ax.legend(loc='upper left', fontsize=9, title='Cluster', framealpha=0.8)
    save(fig, 'clusters')


# ---------------------------------------------------------------------------
# 7. Green Space Access
# ---------------------------------------------------------------------------

def export_green_access():
    print('Green Space Access...')
    data = load_json('distance_to_green.json')
    fig, ax = make_fig('Distance to Green Space')

    lons = [d['lon'] for d in data]
    lats = [d['lat'] for d in data]
    vals = [d['dist_to_green_m'] for d in data]

    colors = ['#1b5e20', '#66bb6a', '#ffb74d', '#d32f2f']
    bounds = [0, 100, 200, 400, max(max(vals) + 1, 401)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)

    sc = ax.scatter(lons, lats, c=vals, cmap=cmap, norm=norm,
                    s=3, edgecolors='none', alpha=0.75, zorder=2)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02, ticks=[50, 150, 300, 450])
    cbar.ax.set_yticklabels(['<100m', '100-200m', '200-400m', '>400m'], fontsize=8)
    cbar.set_label('Distance to Green Space (m)', fontsize=10)

    save(fig, 'green_access')


# ---------------------------------------------------------------------------
# 8. Bundled All-Layers SVG with raster basemap
# ---------------------------------------------------------------------------

ZOOM = 16
TILE_SIZE = 256
TILE_URL = 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png'
TILE_SUBDOMAINS = 'abcd'
PAD = 0.002  # degrees padding around data extent


def _deg2tile(lat, lon, z):
    """Convert lat/lon to tile x,y at zoom z."""
    lat_rad = math.radians(lat)
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n)
    return x, y


def _tile2deg(x, y, z):
    """Top-left corner of tile in lat/lon."""
    n = 2 ** z
    lon = x / n * 360 - 180
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def _lat_to_merc(lat):
    """Convert latitude to Web Mercator Y value."""
    lat_rad = math.radians(lat)
    return math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad))


def _download_tiles(tx_min, tx_max, ty_min, ty_max, z):
    """Download and stitch basemap tiles into a single PIL Image."""
    nx = tx_max - tx_min + 1
    ny = ty_max - ty_min + 1
    stitched = Image.new('RGB', (nx * TILE_SIZE, ny * TILE_SIZE))

    count = 0
    total = nx * ny
    for ty in range(ty_min, ty_max + 1):
        for tx in range(tx_min, tx_max + 1):
            s = TILE_SUBDOMAINS[count % len(TILE_SUBDOMAINS)]
            url = TILE_URL.replace('{s}', s).replace('{z}', str(z)).replace('{x}', str(tx)).replace('{y}', str(ty))
            req = urllib.request.Request(url, headers={'User-Agent': 'AlKaramaExport/1.0'})
            resp = urllib.request.urlopen(req)
            tile_img = Image.open(io.BytesIO(resp.read())).convert('RGB')
            px = (tx - tx_min) * TILE_SIZE
            py = (ty - ty_min) * TILE_SIZE
            stitched.paste(tile_img, (px, py))
            count += 1
            print(f'    Tile {count}/{total}', end='\r')

    print(f'    Downloaded {total} tiles ({stitched.size[0]}x{stitched.size[1]} px)')
    return stitched


class SvgBuilder:
    """Build a layered SVG with a raster basemap and vector analysis layers."""

    def __init__(self, img_w, img_h, tl_lat, tl_lon, br_lat, br_lon):
        self.img_w = img_w
        self.img_h = img_h
        self.tl_lat = tl_lat
        self.tl_lon = tl_lon
        self.br_lat = br_lat
        self.br_lon = br_lon
        self.merc_tl = _lat_to_merc(tl_lat)
        self.merc_br = _lat_to_merc(br_lat)
        self.layers = []  # list of (id, label, svg_content)

    def lonlat_to_px(self, lat, lon):
        """Convert lat/lon to pixel coordinates in the SVG."""
        px_x = (lon - self.tl_lon) / (self.br_lon - self.tl_lon) * self.img_w
        merc_pt = _lat_to_merc(lat)
        px_y = (self.merc_tl - merc_pt) / (self.merc_tl - self.merc_br) * self.img_h
        return px_x, px_y

    def add_basemap(self, img):
        """Add raster basemap as base64-encoded image layer."""
        buf = io.BytesIO()
        img.save(buf, format='PNG', optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        content = (f'<image x="0" y="0" width="{self.img_w}" height="{self.img_h}" '
                   f'href="data:image/png;base64,{b64}" />')
        self.layers.append(('basemap', 'Basemap (CARTO Positron)', content))

    def add_layer(self, layer_id, label, content):
        """Add a named vector layer."""
        self.layers.append((layer_id, label, content))

    def build(self):
        """Generate the complete SVG string."""
        parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg"',
            f'     xmlns:xlink="http://www.w3.org/1999/xlink"',
            f'     xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"',
            f'     width="{self.img_w}" height="{self.img_h}"',
            f'     viewBox="0 0 {self.img_w} {self.img_h}">',
            f'<!-- Al Karama Urban Analysis - All Layers -->',
            f'<!-- Basemap: CARTO Positron, Zoom {ZOOM} -->',
            f'<!-- Coordinate space: Web Mercator pixels -->',
            f'<!-- Lat: {self.br_lat:.6f} to {self.tl_lat:.6f} -->',
            f'<!-- Lon: {self.tl_lon:.6f} to {self.br_lon:.6f} -->',
        ]

        for layer_id, label, content in self.layers:
            parts.append(
                f'<g id="{layer_id}" '
                f'inkscape:groupmode="layer" '
                f'inkscape:label="{label}">'
            )
            parts.append(content)
            parts.append('</g>')

        parts.append('</svg>')
        return '\n'.join(parts)


def _color_hex(r, g, b):
    """Convert 0-255 RGB to hex."""
    return f'#{int(r):02x}{int(g):02x}{int(b):02x}'


def _build_centrality_svg(builder, streets, metric):
    """Build SVG polyline elements for street centrality."""
    if metric == 'betweenness':
        colors = [(225, 190, 231), (206, 147, 216), (171, 71, 188), (142, 36, 170), (74, 20, 140)]
        vmax = 0.196
    else:
        colors = [(187, 222, 251), (144, 202, 249), (66, 165, 245), (30, 136, 229), (13, 71, 161)]
        vmax = 0.00058

    parts = []
    for feat in streets['features']:
        val = feat['properties'].get(metric, 0) or 0
        ratio = min(1.0, val / vmax)
        idx = min(int(ratio * 4), 3)
        t = (ratio * 4) - idx
        c1, c2 = colors[idx], colors[min(idx + 1, 4)]
        r = c1[0] + (c2[0] - c1[0]) * t
        g = c1[1] + (c2[1] - c1[1]) * t
        b = c1[2] + (c2[2] - c1[2]) * t
        color = _color_hex(r, g, b)
        width = 0.5 + ratio * 3

        coords_list = []
        geom = feat['geometry']
        if geom['type'] == 'LineString':
            coords_list.append(geom['coordinates'])
        elif geom['type'] == 'MultiLineString':
            coords_list.extend(geom['coordinates'])

        for coords in coords_list:
            points = []
            for c in coords:
                px, py = builder.lonlat_to_px(c[1], c[0])
                points.append(f'{px:.1f},{py:.1f}')
            parts.append(f'<polyline points="{" ".join(points)}" '
                         f'fill="none" stroke="{color}" stroke-width="{width:.1f}" '
                         f'stroke-opacity="0.85" stroke-linecap="round"/>')

    return '\n'.join(parts)


def _scatter_svg(builder, lons, lats, colors_list, radius=3, opacity=0.75):
    """Build SVG circle elements for scatter data."""
    parts = []
    for i in range(len(lons)):
        px, py = builder.lonlat_to_px(lats[i], lons[i])
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius}" '
                     f'fill="{colors_list[i]}" fill-opacity="{opacity}"/>')
    return '\n'.join(parts)


def _comfort_color(pci):
    if pci >= 0.6: return '#1b5e20'
    if pci >= 0.5: return '#4caf50'
    if pci >= 0.4: return '#ffeb3b'
    if pci >= 0.3: return '#ff9800'
    return '#e53935'


def _priority_color(level):
    m = {'Critical': '#d32f2f', 'High': '#f57c00', 'Medium': '#ffeb3b', 'Low': '#388e3c'}
    return m.get(level, '#999999')


def _lst_color(lst):
    ratio = max(0, min(1, (lst - 46) / 7))
    colors = [(60, 80, 200), (60, 200, 200), (240, 180, 40), (220, 50, 32)]
    idx = min(int(ratio * 3), 2)
    t = (ratio * 3) - idx
    c1, c2 = colors[idx], colors[idx + 1]
    return _color_hex(c1[0]+(c2[0]-c1[0])*t, c1[1]+(c2[1]-c1[1])*t, c1[2]+(c2[2]-c1[2])*t)


def _gvi_color(gvi):
    if gvi < 0.02: return '#f7f7f7'
    if gvi < 0.05: return '#c7e9c0'
    if gvi < 0.1: return '#74c476'
    if gvi < 0.2: return '#238b45'
    return '#006d2c'


def _svf_color(svf):
    r = int(50 + 180 * svf)
    g = int(50 + 180 * svf)
    b = int(100 + 155 * svf)
    return _color_hex(min(r, 255), min(g, 255), min(b, 255))


def _cluster_color(c):
    return {0: '#e53935', 1: '#ff9800', 2: '#4caf50', 3: '#2196f3'}.get(c, '#999')


def _green_access_color(dist):
    if dist < 100: return '#1b5e20'
    if dist < 200: return '#66bb6a'
    if dist < 400: return '#ffb74d'
    return '#d32f2f'


def _sat_lst_color(v):
    ratio = max(0, min(1, (v - 36) / 18))
    colors = [(49,54,149),(69,117,180),(171,217,233),(254,224,144),(244,109,67),(165,0,38)]
    idx = min(int(ratio * 5), 4)
    t = (ratio * 5) - idx
    c1, c2 = colors[idx], colors[idx + 1]
    return _color_hex(c1[0]+(c2[0]-c1[0])*t, c1[1]+(c2[1]-c1[1])*t, c1[2]+(c2[2]-c1[2])*t)


def _sat_ndvi_color(v):
    if v < 0: return '#8B4513'
    if v < 0.1: return '#D2691E'
    if v < 0.2: return '#F5DEB3'
    if v < 0.3: return '#90EE90'
    if v < 0.4: return '#228B22'
    return '#006400'


def _sat_ndbi_color(v):
    if v < -0.1: return '#1a9850'
    if v < 0: return '#91cf60'
    if v < 0.1: return '#fee08b'
    if v < 0.2: return '#fc8d59'
    return '#d73027'


def export_all_layers_svg():
    """Generate a single layered SVG with raster basemap + all vector layers."""
    print('=== Bundled All-Layers SVG ===')

    # ------------------------------------------------------------------
    # 1. Compute bounding box from all data
    # ------------------------------------------------------------------
    all_lats = []
    all_lons = []
    for fname in ['priority_points.json', 'combined_svi.json', 'distance_to_green.json',
                   'clusters.json', 'segment_comfort.json', 'satellite_grid.json']:
        d = load_json(fname)
        all_lats.extend(p['lat'] for p in d)
        all_lons.extend(p['lon'] for p in d)

    lat_min, lat_max = min(all_lats) - PAD, max(all_lats) + PAD
    lon_min, lon_max = min(all_lons) - PAD, max(all_lons) + PAD
    print(f'  Bbox: lat [{lat_min:.4f}, {lat_max:.4f}], lon [{lon_min:.4f}, {lon_max:.4f}]')

    # ------------------------------------------------------------------
    # 2. Download basemap tiles
    # ------------------------------------------------------------------
    tx_min, ty_min = _deg2tile(lat_max, lon_min, ZOOM)
    tx_max, ty_max = _deg2tile(lat_min, lon_max, ZOOM)
    print(f'  Downloading tiles (zoom {ZOOM})...')
    basemap_img = _download_tiles(tx_min, tx_max, ty_min, ty_max, ZOOM)

    img_w, img_h = basemap_img.size
    tl_lat, tl_lon = _tile2deg(tx_min, ty_min, ZOOM)
    br_lat, br_lon = _tile2deg(tx_max + 1, ty_max + 1, ZOOM)

    # ------------------------------------------------------------------
    # 3. Build SVG
    # ------------------------------------------------------------------
    builder = SvgBuilder(img_w, img_h, tl_lat, tl_lon, br_lat, br_lon)
    builder.add_basemap(basemap_img)

    # --- Street Centrality (Betweenness) ---
    print('  Layer: Centrality (Betweenness)...')
    streets = load_json('streets.geojson')
    content = _build_centrality_svg(builder, streets, 'betweenness')
    builder.add_layer('centrality_betweenness', 'Street Centrality (Betweenness)', content)

    # --- Street Centrality (Closeness) ---
    print('  Layer: Centrality (Closeness)...')
    content = _build_centrality_svg(builder, streets, 'closeness')
    builder.add_layer('centrality_closeness', 'Street Centrality (Closeness)', content)

    # --- Pedestrian Comfort ---
    print('  Layer: Pedestrian Comfort...')
    data = load_json('segment_comfort.json')
    lons = [d['lon'] for d in data]
    lats = [d['lat'] for d in data]
    colors_list = [_comfort_color(d['pci_mean']) for d in data]
    content = _scatter_svg(builder, lons, lats, colors_list, radius=6, opacity=0.8)
    builder.add_layer('comfort', 'Pedestrian Comfort (PCI)', content)

    # --- Heat Mitigation Priority ---
    print('  Layer: Heat Mitigation Priority...')
    data = load_json('priority_points.json')
    lons = [d['lon'] for d in data]
    lats = [d['lat'] for d in data]
    colors_list = [_priority_color(d.get('priority_level', '')) for d in data]
    content = _scatter_svg(builder, lons, lats, colors_list, radius=2.5, opacity=0.7)
    builder.add_layer('priority', 'Heat Mitigation Priority', content)

    # --- Combined SVI: LST ---
    print('  Layer: Combined LST...')
    data = load_json('combined_svi.json')
    lons = [d['lon'] for d in data]
    lats = [d['lat'] for d in data]
    colors_list = [_lst_color(d.get('lst', 0)) for d in data]
    content = _scatter_svg(builder, lons, lats, colors_list, radius=2.5)
    builder.add_layer('combined_lst', 'Combined SVI - LST', content)

    # --- Combined SVI: GVI ---
    print('  Layer: Combined GVI...')
    colors_list = [_gvi_color(d.get('gvi', 0)) for d in data]
    content = _scatter_svg(builder, lons, lats, colors_list, radius=2.5)
    builder.add_layer('combined_gvi', 'Combined SVI - GVI', content)

    # --- Combined SVI: SVF ---
    print('  Layer: Combined SVF...')
    colors_list = [_svf_color(d.get('svf', 0)) for d in data]
    content = _scatter_svg(builder, lons, lats, colors_list, radius=2.5)
    builder.add_layer('combined_svf', 'Combined SVI - SVF', content)

    # --- Satellite Grid: LST ---
    print('  Layer: Satellite LST...')
    data = load_json('satellite_grid.json')
    lons = [d['lon'] for d in data]
    lats = [d['lat'] for d in data]
    colors_list = [_sat_lst_color(d.get('lst', 0)) for d in data]
    content = _scatter_svg(builder, lons, lats, colors_list, radius=2)
    builder.add_layer('satellite_lst', 'Satellite Grid - LST (30m)', content)

    # --- Satellite Grid: NDVI ---
    print('  Layer: Satellite NDVI...')
    colors_list = [_sat_ndvi_color(d.get('ndvi', 0)) for d in data]
    content = _scatter_svg(builder, lons, lats, colors_list, radius=2)
    builder.add_layer('satellite_ndvi', 'Satellite Grid - NDVI (30m)', content)

    # --- Satellite Grid: NDBI ---
    print('  Layer: Satellite NDBI...')
    colors_list = [_sat_ndbi_color(d.get('ndbi', 0)) for d in data]
    content = _scatter_svg(builder, lons, lats, colors_list, radius=2)
    builder.add_layer('satellite_ndbi', 'Satellite Grid - NDBI (30m)', content)

    # --- Climate Clusters ---
    print('  Layer: Climate Clusters...')
    data = load_json('clusters.json')
    lons = [d['lon'] for d in data]
    lats = [d['lat'] for d in data]
    colors_list = [_cluster_color(d.get('cluster', 0)) for d in data]
    content = _scatter_svg(builder, lons, lats, colors_list, radius=2.5)
    builder.add_layer('clusters', 'Climate Clusters', content)

    # --- Green Space Access ---
    print('  Layer: Green Space Access...')
    data = load_json('distance_to_green.json')
    lons = [d['lon'] for d in data]
    lats = [d['lat'] for d in data]
    colors_list = [_green_access_color(d.get('dist_to_green_m', 0)) for d in data]
    content = _scatter_svg(builder, lons, lats, colors_list, radius=2.5)
    builder.add_layer('green_access', 'Green Space Access', content)

    # ------------------------------------------------------------------
    # 4. Write SVG
    # ------------------------------------------------------------------
    svg_content = builder.build()
    out_path = os.path.join(OUT_DIR, 'all_layers.svg')
    with open(out_path, 'w') as f:
        f.write(svg_content)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f'  => all_layers.svg ({size_mb:.1f} MB)')
    print(f'     {len(builder.layers)} layers (1 raster basemap + {len(builder.layers) - 1} vector)')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f'Exporting vector maps to {OUT_DIR}/')
    print()

    export_centrality()
    export_comfort()
    export_priority()
    export_combined()
    export_satellite()
    export_clusters()
    export_green_access()

    print()
    export_all_layers_svg()

    # List all generated files
    files = sorted(os.listdir(OUT_DIR))
    print()
    print(f'Done. {len(files)} files in {OUT_DIR}/')
    for f in files:
        size = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f'  {f} ({size/1024:.0f} KB)')


if __name__ == '__main__':
    main()
