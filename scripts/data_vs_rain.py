import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# CONFIG
# ==========================================================

FILE = "collocated_2.parquet"

RAIN_COL = "rain_mm_hr"
VV_COL   = "VV_dB"
VH_COL   = "VH_dB"

MIN_COUNT = 1000

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_parquet(FILE)

df = df.dropna(
    subset=[
        RAIN_COL,
        VV_COL,
        VH_COL
    ]
)

BINS = np.arange(
    df["rain_mm_hr"].min(),
    df["rain_mm_hr"].max()+0.5,
    0.5
)

# ==========================================================
# CREATE RAIN BINS
# ==========================================================

df["rain_bin"] = pd.cut(
    df[RAIN_COL],
    bins=BINS,
    include_lowest=True
)

# ==========================================================
# COMPUTE BIN STATS
# ==========================================================

stats = (
    df
    .groupby("rain_bin", observed=True)
    .agg(
        VV_mean=(VV_COL, "mean"),
        VH_mean=(VH_COL, "mean"),
        count=(RAIN_COL, "size")
    )
    .reset_index()
)

stats = stats[
    stats["count"] >= MIN_COUNT
].copy()

# ==========================================================
# LABELS
# ==========================================================

labels = [
    str(x)
    .replace("(", "")
    .replace("]", "")
    for x in stats["rain_bin"]
]

x = np.arange(len(stats))

# ==========================================================
# PLOT
# ==========================================================

fig, ax = plt.subplots(
    figsize=(14,7),
    constrained_layout=True
)

ax2 = ax.twinx()


# ==========================================================
# VV
# ==========================================================

vv_line = ax.plot(

    x,

    stats["VV_mean"],

    marker='o',

    markersize=6,

    linewidth=2.5,

    color='tab:blue',

    label='VV',
    
    alpha=0.7

)[0]


# ==========================================================
# VH
# ==========================================================

vh_line = ax2.plot(

    x,

    stats["VH_mean"],

    marker='s',

    markersize=6,

    linewidth=2.5,

    color='tab:red',

    label='VH',
    
    alpha=0.7

)[0]


# ==========================================================
# COUNT LABELS
# ==========================================================

for xi, yi, count in zip(

        x,

        stats["VV_mean"],

        stats["count"]
):

    ax.annotate(

        f'{count:,}',

        (xi, yi),

        xytext=(0, 10),

        textcoords='offset points',

        ha='center',

        fontsize=8,

        color='black'

    )


# ==========================================================
# X AXIS
# ==========================================================

ax.set_xticks(x)

ax.set_xticklabels(

    labels,

    rotation=45,

    ha='right'

)


# ==========================================================
# LABELS
# ==========================================================

ax.set_xlabel(

    'Rain Bin (mm/hr)',

    fontsize=12
)

ax.set_ylabel(

    'Mean VV (dB)',

    fontsize=12,

    color='tab:blue'
)

ax2.set_ylabel(

    'Mean VH (dB)',

    fontsize=12,

    color='tab:red'
)


ax.tick_params(

    axis='y',

    colors='tab:blue'
)

ax2.tick_params(

    axis='y',

    colors='tab:red'
)


# ==========================================================
# GRID
# ==========================================================

ax.grid(

    True,

    linestyle='--',

    linewidth=0.8,

    alpha=0.25

)


# ==========================================================
# COMBINED LEGEND
# ==========================================================

handles = [

    vv_line,

    vh_line

]

labels_leg = [

    h.get_label()

    for h in handles
]

ax.legend(

    handles,

    labels_leg,

    loc='upper left',

    frameon=True,

    fontsize=10

)


# ==========================================================
# TITLE
# ==========================================================

ax.set_title(

    'Mean VV and VH Response to Rainfall',

    fontsize=15,

    fontweight='bold',

    pad=15

)


# ==========================================================
# OPTIONAL Y ZOOM
# ==========================================================

vvmin = stats["VV_mean"].min()
vvmax = stats["VV_mean"].max()

pad = max(
    0.15,
    (vvmax-vvmin)*0.15
)

ax.set_ylim(

    vvmin-pad,

    vvmax+pad

)


vhmin = stats["VH_mean"].min()
vhmax = stats["VH_mean"].max()

pad = max(
    0.15,
    (vhmax-vhmin)*0.15
)

ax2.set_ylim(

    vhmin-pad,

    vhmax+pad

)


plt.show()

# ==========================================================
# PRINT TABLE
# ==========================================================

print("\nRain Bin Statistics\n")

print(
    stats.round(3).to_string(index=False)
)