import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


##################################################
# CONFIG
##################################################

FILE = "collocated_1.parquet"

SAR_VARIABLE = "VV_dB"

INITIAL_RAIN_THRESHOLD = 8
INITIAL_SAR_THRESHOLD = -16.1
INITIAL_ALPHA = 0.5


##################################################
# LOAD
##################################################

df = pd.read_parquet(FILE)

##################################################
# SAR VARIABLE
##################################################

if SAR_VARIABLE == "VH_minus_VV":
    sar_values = df["VH_dB"] - df["VV_dB"]

elif SAR_VARIABLE == "VV_minus_VH":
    sar_values = df["VV_dB"] - df["VH_dB"]

else:
    sar_values = df[SAR_VARIABLE]

thresholds = np.arange(df[SAR_VARIABLE].min(),df[SAR_VARIABLE].max()+0.5,0.5)
best_accuracy=0
for t in thresholds:

    rain_pred = df["VV_dB"] > t

    rain_true = df["rain_mm_hr"] > INITIAL_RAIN_THRESHOLD

    accuracy = (rain_pred == rain_true).mean()
    if accuracy>=best_accuracy:
        best_accuracy=accuracy
        INITIAL_SAR_THRESHOLD=t

##################################################
# GRID SETUP
##################################################

lons = np.sort(df.lon.unique())
lats = np.sort(df.lat.unique())

nx = len(lons)
ny = len(lats)

lon_idx = {v:i for i,v in enumerate(lons)}
lat_idx = {v:i for i,v in enumerate(lats)}

extent = (
    float(lons.min()),
    float(lons.max()),
    float(lats.min()),
    float(lats.max())
)


##################################################
# BUILD MASK FUNCTION
##################################################

def build_masks(rain_thr, sar_thr):

    rain_img = np.full((ny,nx), np.nan)
    sar_img = np.full((ny,nx), np.nan)

    rain_mask = (
            df["rain_mm_hr"] >= rain_thr
    ).astype(np.uint8)

    sar_mask = (
            sar_values >= sar_thr
    ).astype(np.uint8)


    for lon,lat,rain,sar in zip(

            df.lon,
            df.lat,
            rain_mask,
            sar_mask
    ):

        x = lon_idx[lon]
        y = lat_idx[lat]

        rain_img[y,x] = rain
        sar_img[y,x] = sar


    return rain_img,sar_img



##################################################
# INITIAL MASKS
##################################################

rain_img,sar_img = build_masks(

        INITIAL_RAIN_THRESHOLD,
        INITIAL_SAR_THRESHOLD
)


##################################################
# FIGURE
##################################################

fig,ax = plt.subplots(figsize=(12,8))

plt.subplots_adjust(bottom=0.25)



im1 = ax.imshow(

        sar_img,

        extent=extent,

        origin='lower',

        cmap='gray',

        vmin=0,
        vmax=1
)



im2 = ax.imshow(

        rain_img,

        extent=extent,

        origin='lower',

        cmap='Reds',

        alpha=INITIAL_ALPHA,

        vmin=0,
        vmax=1
)



ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")


title = ax.set_title(

f"{SAR_VARIABLE}"
)



##################################################
# SLIDERS
##################################################

ax_rain = plt.axes((0.15,0.13,0.70,0.03))
ax_sar = plt.axes((0.15,0.08,0.70,0.03))
ax_alpha = plt.axes((0.15,0.03,0.70,0.03))


slider_rain = Slider(

        ax_rain,

        'Rain',

        0,

        20,

        valinit=INITIAL_RAIN_THRESHOLD
)


slider_sar = Slider(

        ax_sar,

        'SAR',

        -30,

        10,

        valinit=INITIAL_SAR_THRESHOLD
)


slider_alpha = Slider(

        ax_alpha,

        'Alpha',

        0,

        1,

        valinit=INITIAL_ALPHA
)



##################################################
# UPDATE
##################################################

def update(_):


    rain_thr = slider_rain.val
    sar_thr = slider_sar.val
    alpha = slider_alpha.val


    rain_img,sar_img = build_masks(

            rain_thr,
            sar_thr
    )


    im1.set_data(sar_img)

    im2.set_data(rain_img)

    im2.set_alpha(alpha)



    title.set_text(

f"{SAR_VARIABLE} ≥ {sar_thr:.2f}     Rain ≥ {rain_thr:.2f}"
    )


    fig.canvas.draw_idle()



slider_rain.on_changed(update)
slider_sar.on_changed(update)
slider_alpha.on_changed(update)



plt.show()