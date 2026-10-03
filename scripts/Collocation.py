import xarray as xr
import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# ==========================================================
# USER SETTINGS
# ==========================================================

SAR_NC = r"SAR_1.nc"

IMERG_FILE = r"IMERG_1.HDF5"

OUTPUT_PARQUET = "collocated_1.parquet"

STRIDE = 40


# ==========================================================
# STAGE 1 : LOAD SAR NETCDF
# ==========================================================

print("\n[1/7] Loading SAR NetCDF...")

ds = xr.open_dataset(
    SAR_NC,
    engine="netcdf4"
)

vv = ds["Sigma0_VV_db"].values
vh = ds["Sigma0_VH_db"].values

incidence_angle = ds["incidenceAngleFromEllipsoid"].values

sar_lat = ds["lat"].values
sar_lon = ds["lon"].values

rows, cols = np.indices(vv.shape)

rows = rows[::STRIDE, ::STRIDE]
cols = cols[::STRIDE, ::STRIDE]

vv = vv[::STRIDE, ::STRIDE]
vh = vh[::STRIDE, ::STRIDE]

incidence_angle = incidence_angle[::STRIDE, ::STRIDE]

lat = sar_lat[rows]
lon = sar_lon[cols]

print("SAR loaded.")
print("Sampled pixels:", vv.size)


# ==========================================================
# STAGE 2 : FLATTEN ARRAYS
# ==========================================================

print("\n[2/7] Flattening arrays...")

lat = lat.flatten()
lon = lon.flatten()

vv = vv.flatten()
vh = vh.flatten()

incidence_angle = incidence_angle.flatten()

print("Total sampled points:", len(vv))


# ==========================================================
# STAGE 3 : LOAD IMERG
# ==========================================================

print("\n[3/7] Loading IMERG...")

with h5py.File(IMERG_FILE, "r") as f:

    if "Grid/precipitation" in f:
        rain = f["Grid/precipitation"][:]
        print("\nUsing Grid/precipitation")

    elif "Grid/precipitationCal" in f:
        rain = f["Grid/precipitationCal"][:]
        print("\nUsing Grid/precipitationCal")

    else:
        raise Exception(
            "Could not find precipitation dataset."
        )

    imerg_lat = f["Grid/lat"][:]
    imerg_lon = f["Grid/lon"][:]

rain = np.squeeze(rain)

# IMERG sometimes comes as [lon,lat]
if rain.shape[0] == len(imerg_lon):
    rain = rain.T

print("Rain shape:", rain.shape)
print("IMERG loaded.")


# ==========================================================
# STAGE 4 : BUILD KD TREE
# ==========================================================

print("\n[4/7] Building KDTree...")

lon_grid, lat_grid = np.meshgrid(
    imerg_lon,
    imerg_lat
)

imerg_points = np.column_stack(
    (
        lat_grid.ravel(),
        lon_grid.ravel()
    )
)

tree = cKDTree(imerg_points)

print("IMERG cells:", len(imerg_points))


# ==========================================================
# STAGE 5 : COLLOCATION
# ==========================================================

print("\n[5/7] Collocating SAR -> IMERG...")

sar_points = np.column_stack(
    (
        lat,
        lon
    )
)

dist, idx = tree.query(
    sar_points,
    k=1
)

rain_flat = rain.ravel()

rain_values = rain_flat[idx]

print("Collocation complete.")


# ==========================================================
# STAGE 6 : FILTER INVALID DATA
# ==========================================================

print("\n[6/7] Filtering invalid samples...")

mask = (
    np.isfinite(vv)
    &
    np.isfinite(vh)
    &
    np.isfinite(incidence_angle)
    &
    np.isfinite(rain_values)
    &
    (rain_values >= 0)
)

lat = lat[mask]
lon = lon[mask]

vv = vv[mask]
vh = vh[mask]

incidence_angle = incidence_angle[mask]

rain_values = rain_values[mask]

print("Remaining samples:", len(vv))


# ==========================================================
# STAGE 7 : SAVE PARQUET
# ==========================================================

print("\n[7/7] Saving parquet...")

df = pd.DataFrame(
    {
        "lat": lat.astype(np.float32),
        "lon": lon.astype(np.float32),

        "VV_dB": vv.astype(np.float32),
        "VH_dB": vh.astype(np.float32),

        "incidence_angle": incidence_angle.astype(np.float32),

        "rain_mm_hr": rain_values.astype(np.float32)
    }
)

df.to_parquet(
    OUTPUT_PARQUET,
    index=False
)

print("\n========================================")
print("DONE")
print("Output file:", OUTPUT_PARQUET)
print("Rows:", len(df))
print("Columns:", list(df.columns))
print("========================================")