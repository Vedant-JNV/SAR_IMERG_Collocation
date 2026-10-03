import pandas as pd
import numpy as np
import h5py

import matplotlib.pyplot as plt

from matplotlib.widgets import Slider
from matplotlib.colors import LogNorm

from skimage import exposure

# =====================================================
# FILES
# =====================================================

PARQUET = "collocated_1_icorrected.parquet"
IMERG_FILE = "IMERG_1.HDF5"
SAR_BAND = "VH_dB"

# =====================================================
# LOAD SAR
# =====================================================

print("Loading SAR...")

df = pd.read_parquet(PARQUET)

lat_min = df["lat"].min()
lat_max = df["lat"].max()

lon_min = df["lon"].min()
lon_max = df["lon"].max()

# =====================================================
# BUILD SAR GRID
# =====================================================

lats = np.sort(df["lat"].unique())
lons = np.sort(df["lon"].unique())

lat_idx = {v:i for i,v in enumerate(lats)}
lon_idx = {v:i for i,v in enumerate(lons)}

sar = np.full(
    (len(lats), len(lons)),
    np.nan,
    dtype=np.float32
)

for row in df.itertuples():

    sar[
        lat_idx[row.lat],
        lon_idx[row.lon]
    ] = getattr(row, SAR_BAND)

# =====================================================
# SAR STRETCH
# =====================================================

p2 = np.nanpercentile(sar,2)
p98 = np.nanpercentile(sar,98)

sar = np.clip(
    (sar-p2)/(p98-p2),
    0,
    1
)

sar = exposure.equalize_hist(
    np.nan_to_num(sar)
)

# =====================================================
# LOAD IMERG
# =====================================================

print("Loading IMERG...")

with h5py.File(IMERG_FILE,"r") as f:

    if "Grid/precipitation" in f:
        rain = f["Grid/precipitation"][:]

    else:
        rain = f["Grid/precipitationCal"][:]

    imerg_lat = f["Grid/lat"][:]
    imerg_lon = f["Grid/lon"][:]

rain = np.squeeze(rain)

if rain.shape[0] == len(imerg_lon):
    rain = rain.T

# =====================================================
# CROP TO SAR FOOTPRINT
# =====================================================

margin = 0.05

lat_mask = (
    (imerg_lat >= lat_min-margin)
    &
    (imerg_lat <= lat_max+margin)
)

lon_mask = (
    (imerg_lon >= lon_min-margin)
    &
    (imerg_lon <= lon_max+margin)
)

rain = rain[
    lat_mask,
    :
][:,
    lon_mask
]

imerg_lat = imerg_lat[lat_mask]
imerg_lon = imerg_lon[lon_mask]

# =====================================================
# CELL EDGES
# =====================================================

dlat = np.mean(np.diff(imerg_lat))
dlon = np.mean(np.diff(imerg_lon))

lat_edges = np.concatenate(
[
    [imerg_lat[0]-dlat/2],
    imerg_lat[:-1]+dlat/2,
    [imerg_lat[-1]+dlat/2]
])

lon_edges = np.concatenate(
[
    [imerg_lon[0]-dlon/2],
    imerg_lon[:-1]+dlon/2,
    [imerg_lon[-1]+dlon/2]
])

# =====================================================
# FIGURE
# =====================================================

fig, ax = plt.subplots(
    figsize=(14,10)
)

plt.subplots_adjust(
    bottom=0.15
)

# =====================================================
# SAR
# =====================================================

ax.imshow(
    sar,
    cmap="gray",
    origin="lower",
    extent=[
        lon_min,
        lon_max,
        lat_min,
        lat_max
    ]
)

# =====================================================
# INITIAL IMERG
# =====================================================

RAIN_THRESHOLD = 1.5

rain_display = rain.copy()

rain_display[
    rain_display < RAIN_THRESHOLD
] = np.nan

mesh = ax.pcolormesh(
    lon_edges,
    lat_edges,
    rain_display,

    cmap="turbo",

    norm=LogNorm(
        vmin=0.1,
        vmax=max(
            10,
            np.nanmax(rain)
        )
    ),

    alpha=0.45,

    shading="flat"
)

# =====================================================
# COLORBAR
# =====================================================

cbar = plt.colorbar(
    mesh,
    ax=ax
)

cbar.set_label(
    "IMERG Rainfall (mm/hr)"
)

# =====================================================
# SLIDERS
# =====================================================

alpha_ax = plt.axes(
    [0.15,0.07,0.7,0.02]
)

thr_ax = plt.axes(
    [0.15,0.03,0.7,0.02]
)

alpha_slider = Slider(
    alpha_ax,
    "Transparency",
    0,
    1,
    valinit=0.45
)

thr_slider = Slider(
    thr_ax,
    "Min Rain",
    0,
    20,
    valinit=1.5
)

# =====================================================
# UPDATE
# =====================================================

def update(val):

    mesh.set_alpha(
        alpha_slider.val
    )

    rain_new = rain.copy()

    rain_new[
        rain_new < thr_slider.val
    ] = np.nan

    mesh.set_array(
        rain_new.ravel()
    )

    fig.canvas.draw_idle()

alpha_slider.on_changed(update)
thr_slider.on_changed(update)

# =====================================================
# LABELS
# =====================================================

ax.set_title(
    f"{SAR_BAND} with IMERG Overlay"
)

ax.set_xlabel(
    "Longitude"
)

ax.set_ylabel(
    "Latitude"
)

plt.show()