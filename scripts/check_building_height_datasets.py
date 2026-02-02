"""
Check which building height datasets on GEE have data for Al Karama, Dubai.
Tests: Global Building Atlas (GBA), GHS-OBAT (UAE), UT-GLOBUS.
"""

import ee

ee.Initialize(project="uobdubai")

# Al Karama area
bbox = ee.Geometry.Rectangle([55.290, 25.230, 55.320, 25.255])
point = ee.Geometry.Point([55.305, 25.242])

print("=" * 60)
print("Checking building height datasets for Al Karama, Dubai")
print("=" * 60)

# --- 1. Global Building Atlas (GBA) ---
# Dubai (~25.24N, 55.30E) should be in tile e055_n30_e060_n25
# (covers 55-60E, 25-30N on a 5x5 degree grid)
print("\n[1] Global Building Atlas (GBA)")
gba_tile_ids = [
    "e055_n30_e060_n25",   # most likely tile
    "e050_n30_e055_n25",   # adjacent tile (in case boundary)
]
for tile_id in gba_tile_ids:
    asset_path = f"projects/sat-io/open-datasets/GLOBAL_BUILDING_ATLAS/{tile_id}"
    try:
        fc = ee.FeatureCollection(asset_path)
        filtered = fc.filterBounds(bbox)
        count = filtered.size().getInfo()
        print(f"  Tile {tile_id}: {count} buildings in Al Karama")
        if count > 0:
            sample = filtered.first().getInfo()
            props = sample.get("properties", {})
            print(f"  Sample properties: {list(props.keys())}")
            if "height" in props:
                print(f"  Sample height: {props['height']}m")
    except Exception as e:
        print(f"  Tile {tile_id}: NOT FOUND or error - {e}")

# --- 2. GHS-OBAT (UAE) ---
print("\n[2] GHS-OBAT (JRC, UAE)")
obat_path = "projects/sat-io/open-datasets/JRC/GHS-OBAT/GHS_OBAT_GPKG_ARE_E2020_R2024A_V1_0"
try:
    fc = ee.FeatureCollection(obat_path)
    filtered = fc.filterBounds(bbox)
    count = filtered.size().getInfo()
    print(f"  Buildings in Al Karama: {count}")
    if count > 0:
        sample = filtered.first().getInfo()
        props = sample.get("properties", {})
        print(f"  Properties: {list(props.keys())}")
        for key in ["height", "b_height", "avg_height", "built_h", "epoch", "function"]:
            if key in props:
                print(f"  {key}: {props[key]}")
except Exception as e:
    print(f"  NOT FOUND or error - {e}")

# --- 3. UT-GLOBUS (Dubai) ---
print("\n[3] UT-GLOBUS")
globus_names = ["dubai", "Dubai", "DUBAI", "abu_dhabi", "Abu_Dhabi"]
for name in globus_names:
    asset_path = f"projects/sat-io/open-datasets/UT-GLOBUS/{name}"
    try:
        fc = ee.FeatureCollection(asset_path)
        count = fc.size().getInfo()
        print(f"  '{name}': {count} buildings total")
        if count > 0:
            filtered = fc.filterBounds(bbox)
            local_count = filtered.size().getInfo()
            print(f"    In Al Karama: {local_count}")
    except Exception as e:
        print(f"  '{name}': NOT FOUND - {e}")

print("\n" + "=" * 60)
print("Done.")
