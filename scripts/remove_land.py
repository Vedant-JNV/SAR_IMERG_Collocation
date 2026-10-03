import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ==================================
# CONFIG
# ==================================

PARQUET_FILE = r"collocated_3.parquet"

LAND_SHP = r"ne_10m_land.shp"

LAT_COL = "lat"
LON_COL = "lon"

# ==================================
# LOAD DATA
# ==================================

print("Loading parquet...")

df = pd.read_parquet(PARQUET_FILE)

original_count = len(df)

print(f"Original points: {original_count:,}")

# ==================================
# CREATE POINTS
# ==================================

points = gpd.GeoSeries(
    [Point(lon, lat) for lon, lat in zip(df[LON_COL], df[LAT_COL])],
    crs="EPSG:4326"
)

# ==================================
# LOAD LAND POLYGONS
# ==================================

print("Loading land polygons...")

land = gpd.read_file(LAND_SHP)

# ==================================
# LAND TEST
# ==================================

print("Testing land/ocean...")

land_union = land.union_all()

is_land = points.within(land_union)

# ==================================
# KEEP OCEAN ONLY
# ==================================

df_ocean = df[~is_land].copy()

ocean_count = len(df_ocean)

print(f"Ocean points: {ocean_count:,}")
print(f"Removed land: {original_count - ocean_count:,}")

# ==================================
# OVERWRITE ORIGINAL FILE
# ==================================

print("Replacing original parquet...")

df_ocean.to_parquet(
    PARQUET_FILE,
    index=False
)

print("Done.")
print(f"Updated file: {PARQUET_FILE}")