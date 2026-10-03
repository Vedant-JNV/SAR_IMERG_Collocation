import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# CONFIG
# =====================================================

FILE = "collocated_2.parquet"

RAIN = "rain_mm_hr"
VV   = "VV_dB"
VH   = "VH_dB"

# =====================================================
# LOAD
# =====================================================

df = pd.read_parquet(FILE)

df["VH_minus_VV"] = df[VH] - df[VV]

vars_to_use = [
    RAIN,
    VV,
    VH,
    "VH_minus_VV"
]

corr = df[vars_to_use].corr(method="spearman")

print("\nspearman Correlation Matrix\n")
print(corr.round(4))

# =====================================================
# PLOT
# =====================================================

fig, ax = plt.subplots(figsize=(7,6))

im = ax.imshow(
    corr,
    vmin=-1,
    vmax=1
)

ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))

ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticklabels(corr.columns)

for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(
            j,
            i,
            f"{corr.iloc[i,j]:.2f}",
            ha="center",
            va="center",
            fontsize=11
        )

plt.colorbar(im, label="spearman Correlation")

plt.title("spearman Correlation Matrix")
plt.tight_layout()
plt.show()