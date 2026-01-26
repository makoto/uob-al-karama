"""
Script to download Street View Imagery (SVI) for Al Karama, Dubai using ZenSVI.
"""

import os
from dotenv import load_dotenv
import geopandas as gpd
from zensvi.download import MLYDownloader

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
mly_api_key = os.getenv("YOUR_OWN_MLY_API_KEY")

if not mly_api_key:
    raise ValueError("Mapillary API key not found in .env file")

# Read the GeoJSON file
geojson_path = "input/Al_Karama.geojson"
gdf = gpd.read_file(geojson_path)
print(f"Loaded boundary from: {geojson_path}")
print(f"Area: {gdf.iloc[0]['NAME_3']}, {gdf.iloc[0]['NAME_1']}")

# Convert to shapefile (ZenSVI requires shapefile input)
shp_path = "input/Al_Karama.shp"
gdf.to_file(shp_path)
print(f"Converted to shapefile: {shp_path}")

# Create output directory
output_dir = "data/svi_images"
os.makedirs(output_dir, exist_ok=True)

# Initialize the Mapillary downloader
downloader = MLYDownloader(mly_api_key=mly_api_key)

# Download SVI for the polygon boundary
print(f"\nDownloading SVI to: {output_dir}")
downloader.download_svi(
    dir_output=output_dir,
    input_shp_file=shp_path,
    update_pids=True
)

print("Download complete!")
