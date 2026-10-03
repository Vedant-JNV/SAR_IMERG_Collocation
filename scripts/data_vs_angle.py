import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Load data
# ==========================
df = pd.read_parquet("collocated_2.parquet")

# ==========================
# Incidence angle bins
# ==========================
bins = np.arange(29, 47, 1)

df["angle_bin"] = pd.cut(
    df["incidence_angle"],
    bins=bins,
    include_lowest=True
)

# ==========================
# Statistics per angle bin
# ==========================
stats = (
    df.groupby("angle_bin", observed=False)
      .agg(
          VV=("VV_dB", "mean"),
          VH=("VH_dB", "mean"),
          Rain=("rain_mm_hr", "mean"),
          Count=("VV_dB", "size")
      )
)

# VH - VV difference (dB)
stats["VH_minus_VV"] = stats["VH"] - stats["VV"]

# Bin centers
x = [b.mid for b in stats.index]

# ==========================
# Plot
# ==========================
fig, ax1 = plt.subplots(figsize=(11, 6))

# Left axis: SAR variables
l1 = ax1.plot(
    x, stats["VV"],
    marker="o",
    linewidth=2,
    label="VV"
)

l2 = ax1.plot(
    x, stats["VH"],
    marker="s",
    linewidth=2,
    label="VH"
)

l3 = ax1.plot(
    x, stats["VH_minus_VV"],
    marker="^",
    linestyle="--",
    linewidth=2,
    label="VH - VV"
)

ax1.set_xlabel("Incidence Angle (deg)")
ax1.set_ylabel("Backscatter / Difference (dB)")
ax1.grid(True, alpha=0.3)

# Right axis: Rain
ax2 = ax1.twinx()

l4 = ax2.plot(
    x, stats["Rain"],
    marker="D",
    linewidth=2,
    linestyle=":",
    label="Rain"
)

ax2.set_ylabel("Mean Rain Rate (mm/hr)")

# Combined legend
lines = l1 + l2 + l3 + l4
labels = [line.get_label() for line in lines]

ax1.legend(lines, labels, loc="best")

plt.title("VV, VH, VH−VV and Rain vs Incidence Angle")

plt.tight_layout()
plt.show()

# ==========================
# Print statistics
# ==========================
print(
    stats[
        ["VV", "VH", "VH_minus_VV", "Rain", "Count"]
    ].round(3)
)