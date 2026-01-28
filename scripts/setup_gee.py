"""
Setup and test Google Earth Engine access.
"""

import subprocess
import sys

# Check if earthengine-api is installed
try:
    import ee
    print("✅ earthengine-api is installed")
except ImportError:
    print("Installing earthengine-api...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "earthengine-api"])
    import ee

print("\n" + "="*50)
print("GOOGLE EARTH ENGINE SETUP")
print("="*50)

# Try different initialization methods
def try_initialize():
    """Try various ways to initialize Earth Engine."""

    # Method 1: Try with user's project
    try:
        ee.Initialize(project='uobdubai')
        return True, "uobdubai project"
    except Exception as e:
        print(f"  uobdubai: {e}")

    # Method 2: Try with high-volume endpoint (no project needed)
    try:
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
        return True, "high-volume endpoint"
    except:
        pass

    # Method 3: Try legacy project
    try:
        ee.Initialize(project='earthengine-legacy')
        return True, "earthengine-legacy project"
    except:
        pass

    # Method 4: Try default
    try:
        ee.Initialize()
        return True, "default"
    except:
        pass

    return False, None

# First, check if we need to authenticate
print("\nChecking authentication...")
try:
    credentials, project = ee.data.get_credentials()
    if credentials is None:
        print("⚠️ No credentials found. Starting authentication...")
        ee.Authenticate()
except:
    print("⚠️ Need to authenticate. Starting...")
    ee.Authenticate()

# Now try to initialize
print("\nTrying to initialize...")
success, method = try_initialize()

if success:
    print(f"\n✅ Initialized successfully using {method}!")

    # Test with a simple query
    print("\n" + "="*50)
    print("TESTING CONNECTION...")
    print("="*50)

    try:
        # Test: Get Sentinel-2 image count for Al Karama area
        al_karama = ee.Geometry.Rectangle([55.29, 25.23, 55.32, 25.26])

        # Sentinel-2
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(al_karama) \
            .filterDate('2024-01-01', '2024-12-31')

        s2_count = s2.size().getInfo()
        print(f"\n✅ Sentinel-2 images for Al Karama (2024): {s2_count}")

        # Landsat 8
        l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
            .filterBounds(al_karama) \
            .filterDate('2024-01-01', '2024-12-31')

        l8_count = l8.size().getInfo()
        print(f"✅ Landsat 8 images for Al Karama (2024): {l8_count}")

        print("\n🎉 Google Earth Engine is working!")
        print("You can now run the satellite analysis script.")

    except Exception as e:
        print(f"\n⚠️ Connected but query failed: {e}")

else:
    print("\n❌ Could not initialize Earth Engine.")
    print("\nPlease try:")
    print("1. Visit: https://code.earthengine.google.com/")
    print("2. Sign in with your Google account")
    print("3. Accept the terms if prompted")
    print("4. Run this script again")
    print("\nOr enable the API at:")
    print("https://console.developers.google.com/apis/api/earthengine.googleapis.com/overview")
