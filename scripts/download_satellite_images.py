"""
Download actual satellite imagery for Al Karama from Google Earth Engine.
Exports: RGB, NDVI, and Land Surface Temperature as GeoTIFF files.
"""

import ee
import os
import requests
import time

ee.Initialize(project='uobdubai')
print("✅ Connected to Google Earth Engine")

output_dir = "data/satellite"
os.makedirs(output_dir, exist_ok=True)

# Al Karama bounding box
AL_KARAMA = ee.Geometry.Rectangle([55.290, 25.230, 55.320, 25.260])

print("\n" + "="*60)
print("DOWNLOADING SATELLITE IMAGERY FOR AL KARAMA")
print("="*60)

# ============================================================
# 1. Sentinel-2 RGB + NDVI (10m resolution)
# ============================================================
print("\n1. Preparing Sentinel-2 imagery...")

def mask_clouds_s2(image):
    qa = image.select('QA60')
    cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(cloud_mask)

def add_ndvi(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(AL_KARAMA) \
    .filterDate('2024-01-01', '2024-12-31') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
    .map(mask_clouds_s2) \
    .map(add_ndvi)

# Get median composite
s2_composite = sentinel2.median().clip(AL_KARAMA)

# RGB visualization
rgb = s2_composite.select(['B4', 'B3', 'B2'])
ndvi = s2_composite.select('NDVI')

print(f"  Sentinel-2 images used: {sentinel2.size().getInfo()}")

# ============================================================
# 2. Landsat LST (30m resolution)
# ============================================================
print("\n2. Preparing Landsat thermal imagery...")

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

print(f"  Landsat images used: {landsat.size().getInfo()}")

# ============================================================
# 3. Download images
# ============================================================
print("\n3. Generating download URLs...")

def get_download_url(image, name, scale, bands=None):
    """Get download URL for an image."""
    if bands:
        image = image.select(bands)

    url = image.getDownloadURL({
        'name': name,
        'scale': scale,
        'region': AL_KARAMA,
        'format': 'GEO_TIFF',
        'crs': 'EPSG:4326'
    })
    return url

def download_file(url, filepath):
    """Download file from URL."""
    print(f"  Downloading {os.path.basename(filepath)}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"    ✅ Saved: {filepath}")
        return True
    else:
        print(f"    ❌ Error: {response.status_code}")
        return False

# Download RGB (as 8-bit for visualization)
print("\n  RGB (Sentinel-2, 10m):")
rgb_vis = rgb.visualize(min=0, max=3000)
rgb_url = get_download_url(rgb_vis, 'al_karama_rgb', 10)
download_file(rgb_url, os.path.join(output_dir, 'al_karama_rgb.tif'))

# Download NDVI
print("\n  NDVI (Sentinel-2, 10m):")
ndvi_url = get_download_url(ndvi, 'al_karama_ndvi', 10)
download_file(ndvi_url, os.path.join(output_dir, 'al_karama_ndvi.tif'))

# Download LST
print("\n  Land Surface Temperature (Landsat, 30m):")
lst_url = get_download_url(lst_composite, 'al_karama_lst', 30)
download_file(lst_url, os.path.join(output_dir, 'al_karama_lst.tif'))

# ============================================================
# 4. Summary
# ============================================================
print("\n" + "="*60)
print("DOWNLOAD COMPLETE")
print("="*60)

files = os.listdir(output_dir)
for f in files:
    size = os.path.getsize(os.path.join(output_dir, f)) / 1024
    print(f"  {f}: {size:.1f} KB")

print(f"\nFiles saved to: {output_dir}/")
print("\nTo view in QGIS:")
print("  1. Open QGIS")
print("  2. Layer > Add Layer > Add Raster Layer")
print("  3. Select the .tif files")
