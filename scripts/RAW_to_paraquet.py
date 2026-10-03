import os
import glob
import xml.etree.ElementTree as ET

import rasterio
import numpy as np
import pandas as pd
import h5py

from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree


# ==========================================================
# USER SETTINGS
# ==========================================================

SAFE_FOLDER = r"S1A_IW_GRDH_1SDV_20230614T012731_20230614T012756_048975_05E3BE_D434.SAFE"

IMERG_FILE = r"IMERG_1.HDF5"

OUTPUT_PARQUET = "collocated_raw_1.parquet"

STRIDE = 30


# ==========================================================
# FIND FILES
# ==========================================================

print("\n[1/8] Finding SAFE files...")

vv_tiff = glob.glob(
    os.path.join(
        SAFE_FOLDER,
        "measurement",
        "*vv*.tiff"
    )
)[0]

vh_tiff = glob.glob(
    os.path.join(
        SAFE_FOLDER,
        "measurement",
        "*vh*.tiff"
    )
)[0]

vv_xml = glob.glob(
    os.path.join(
        SAFE_FOLDER,
        "annotation",
        "*vv*.xml"
    )
)[0]

print("VV TIFF:", os.path.basename(vv_tiff))
print("VH TIFF:", os.path.basename(vh_tiff))


# ==========================================================
# READ TIFFS
# ==========================================================

print("\n[2/8] Reading sampled SAR data...")

with rasterio.open(vv_tiff) as src:

    vv = src.read(1)[::STRIDE, ::STRIDE]

    height = src.height
    width = src.width

with rasterio.open(vh_tiff) as src:

    vh = src.read(1)[::STRIDE, ::STRIDE]

print("Sampled shape:", vv.shape)


# ==========================================================
# SAMPLE PIXEL LOCATIONS
# ==========================================================

print("\n[3/8] Building sampled coordinates...")

sample_rows = np.arange(
    0,
    height,
    STRIDE
)

sample_cols = np.arange(
    0,
    width,
    STRIDE
)

rows, cols = np.meshgrid(
    sample_rows,
    sample_cols,
    indexing="ij"
)


# ==========================================================
# READ GEOLOCATION GRID
# ==========================================================

print("\n[4/8] Reading geolocation grid...")

tree = ET.parse(vv_xml)
root = tree.getroot()

geo_lines = []
geo_pixels = []

geo_lat = []
geo_lon = []

geo_inc = []

for elem in root.iter():

    if elem.tag.endswith("geolocationGridPoint"):

        vals = {}

        for child in elem:

            tag = child.tag.split("}")[-1]

            vals[tag] = child.text

        geo_lines.append(
            float(vals["line"])
        )

        geo_pixels.append(
            float(vals["pixel"])
        )

        geo_lat.append(
            float(vals["latitude"])
        )

        geo_lon.append(
            float(vals["longitude"])
        )

        geo_inc.append(
            float(vals["incidenceAngle"])
        )

geo_lines = np.array(geo_lines)
geo_pixels = np.array(geo_pixels)

geo_lat = np.array(geo_lat)
geo_lon = np.array(geo_lon)

geo_inc = np.array(geo_inc)

print("Grid points:", len(geo_lines))


# ==========================================================
# INTERPOLATE LAT/LON/ANGLE
# ==========================================================

print("\n[5/8] Interpolating geolocation...")

geo_points = np.column_stack(
    (
        geo_lines,
        geo_pixels
    )
)

sample_points = np.column_stack(
    (
        rows.ravel(),
        cols.ravel()
    )
)

lat_interp = LinearNDInterpolator(
    geo_points,
    geo_lat
)

lon_interp = LinearNDInterpolator(
    geo_points,
    geo_lon
)

inc_interp = LinearNDInterpolator(
    geo_points,
    geo_inc
)

lat = lat_interp(sample_points)
lon = lon_interp(sample_points)

incidence_angle = inc_interp(
    sample_points
)

print("Interpolation complete.")


# ==========================================================
# LOAD IMERG
# ==========================================================

print("\n[6/8] Loading IMERG...")

with h5py.File(
    IMERG_FILE,
    "r"
) as f:

    if "Grid/precipitation" in f:

        rain = f["Grid/precipitation"][:]

    elif "Grid/precipitationCal" in f:

        rain = f["Grid/precipitationCal"][:]

    else:

        raise Exception(
            "Could not find precipitation dataset"
        )

    imerg_lat = f["Grid/lat"][:]
    imerg_lon = f["Grid/lon"][:]

rain = np.squeeze(rain)

if rain.shape[0] == len(imerg_lon):

    rain = rain.T


# ==========================================================
# KD TREE COLLOCATION
# ==========================================================

print("\n[7/8] Collocating with IMERG...")

lon_grid, lat_grid = np.meshgrid(
    imerg_lon,
    imerg_lat
)

tree = cKDTree(
    np.column_stack(
        (
            lat_grid.ravel(),
            lon_grid.ravel()
        )
    )
)

dist, idx = tree.query(
    np.column_stack(
        (
            lat,
            lon
        )
    ),
    k=1
)

rain_values = rain.ravel()[idx]


# ==========================================================
# SAVE PARQUET
# ==========================================================

print("\n[8/8] Saving parquet...")

df = pd.DataFrame(
    {
        "lat": lat.astype(np.float32),
        "lon": lon.astype(np.float32),

        "VV_dB": vv.ravel().astype(
            np.float32
        ),

        "VH_dB": vh.ravel().astype(
            np.float32
        ),

        "incidence_angle":
            incidence_angle.astype(
                np.float32
            ),

        "rain_mm_hr":
            rain_values.astype(
                np.float32
            )
    }
)

df = df.dropna()

df.to_parquet(
    OUTPUT_PARQUET,
    index=False
)

print("\n================================")
print("DONE")
print("Rows:", len(df))
print("Output:", OUTPUT_PARQUET)
print("================================")