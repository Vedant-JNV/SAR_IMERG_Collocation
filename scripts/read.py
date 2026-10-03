import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# SETTINGS
# ==========================================================

PARQUET_FILE = "collocated_1_icorrected.parquet"

# Increase to brighten image
BRIGHTNESS = 1.2

# Contrast stretch percentiles
LOW_PERCENTILE = 3
HIGH_PERCENTILE = 98

# Marker size
POINT_SIZE = 1

# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading parquet...\n")

df = pd.read_parquet(PARQUET_FILE)
print(df["VH_dB"].describe())

vv = df["VH_dB"].values

# ==========================================================
# PROFESSIONAL STRETCH
# ==========================================================

p_low = np.percentile(vv, LOW_PERCENTILE)
p_high = np.percentile(vv, HIGH_PERCENTILE)

vv_scaled = (vv - p_low) / (p_high - p_low)

vv_scaled = np.clip(vv_scaled, 0, 1)

# brightness control
vv_scaled = vv_scaled * BRIGHTNESS

vv_scaled = np.clip(vv_scaled, 0, 1)

# ==========================================================
# DISPLAY
# ==========================================================

fig, ax = plt.subplots(
    figsize=(10,10)
)

scatter = ax.scatter(
    df["lon"],
    df["lat"],
    c=vv_scaled,
    cmap="gray",
    s=POINT_SIZE,
    marker="s"
)

ax.set_title(
    "Sentinel-1 VH Backscatter",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.set_aspect("equal")

plt.tight_layout()

plt.show()