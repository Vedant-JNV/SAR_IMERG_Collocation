import xarray as xr
import h5py
import numpy as np
import pandas as pd

from scipy.spatial import cKDTree
from skimage.feature import graycomatrix
from skimage.feature import graycoprops


# ==========================================================
# USER SETTINGS
# ==========================================================

SAR_NC = r"SAR_1.nc"

IMERG_FILE = r"IMERG_1.HDF5"

OUTPUT_PARQUET = "collocated_1.parquet"

STRIDE = 1000

WINDOW = 11
HALF = WINDOW // 2
LEVELS = 32

ANGLES = [
    0,
    np.pi/4,
    np.pi/2,
    3*np.pi/4
]

# ==========================================================
# HELPERS
# ==========================================================

def quantize(img, vmin, vmax):

    q = (img - vmin) / (vmax - vmin)

    q = np.clip(q, 0, 1)

    q *= (LEVELS - 1)

    return q.astype(np.uint8)

def compute_glcm_features(patch, vmin, vmax):

    q = quantize(
        patch,
        vmin,
        vmax
    )

    glcm = graycomatrix(

        q,

        distances=[1],

        angles=ANGLES,

        levels=LEVELS,

        symmetric=True,

        normed=True

    )



    contrast = graycoprops(
        glcm,
        'contrast'
    ).mean()



    homogeneity = graycoprops(
        glcm,
        'homogeneity'
    ).mean()



    correlation = graycoprops(
        glcm,
        'correlation'
    ).mean()



    asm = graycoprops(
        glcm,
        'ASM'
    ).mean()



    ent = []


    for i in range(len(ANGLES)):

        P = glcm[:, :, 0, i]


        e = -np.sum(

            P * np.log2(

                P + 1e-12

            )

        )


        ent.append(e)



    entropy = np.mean(ent)


    return (

        contrast,

        homogeneity,

        asm,

        correlation,

        entropy

    )

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

VH_MIN = np.nanmin(vh)
VH_MAX = np.nanmax(vh)

VV_MIN = np.nanmin(vv)
VV_MAX = np.nanmax(vv)

incidence_angle = ds["incidenceAngleFromEllipsoid"].values

sar_lat = ds["lat"].values
sar_lon = ds["lon"].values

rows, cols = np.indices(vv.shape)

rows = rows[::STRIDE, ::STRIDE]
cols = cols[::STRIDE, ::STRIDE]

vv = vv[::STRIDE, ::STRIDE]
vh = vh[::STRIDE, ::STRIDE]

nrows, ncols = vh.shape

incidence_angle = incidence_angle[::STRIDE, ::STRIDE]

lat = sar_lat[rows]
lon = sar_lon[cols]

print("SAR loaded.")
print("Sampled pixels:", vv.size)

# ==========================================================
# PROCESS
# ==========================================================

results = []

count = 0


for r in range(

        HALF,

        nrows - HALF,

        STRIDE

):


    if r % 1000 == 0:
        print("Row", r)



    for c in range(

            HALF,

            ncols - HALF,

            STRIDE

    ):



        vh_patch = vh[

            r - HALF:r + HALF + 1,

            c - HALF:c + HALF + 1

        ]


        vv_patch = vv[

            r - HALF:r + HALF + 1,

            c - HALF:c + HALF + 1

        ]



        if not np.isfinite(vh_patch).all():
            continue


        if not np.isfinite(vv_patch).all():
            continue




        vh_features = compute_glcm_features(

            vh_patch,

            VH_MIN,

            VH_MAX

        )



        vv_features = compute_glcm_features(

            vv_patch,

            VV_MIN,

            VV_MAX

        )



        results.append([


            lat[r],

            lon[c],


            vv[r, c],

            vh[r, c],



            *vh_features,


            *vv_features


        ])



        count += 1

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

columns = [
'lat',
'lon',

'VV_dB',
'VH_dB',

'VH_contrast',
'VH_homogeneity',
'VH_ASM',
'VH_correlation',
'VH_entropy',

'VV_contrast',
'VV_homogeneity',
'VV_ASM',
'VV_correlation',
'VV_entropy',

'rain_mm_hr',
'incidence_angle'

]

results[0].append([rain_values.astype(np.float32)])
results[0].append([incidence_angle.astype(np.float32)])

df = pd.DataFrame(

    results,

    columns=columns

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
