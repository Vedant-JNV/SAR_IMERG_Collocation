import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_parquet("collocated_2.parquet")

bins = np.arange(
    df["VH_dB"].min(),
    df["VH_dB"].max()+0.5,
    0.5
)

df["VH_bin"] = pd.cut(df["VH_dB"], bins)

rain_mean = df.groupby("VH_bin")["rain_mm_hr"].mean()

plt.figure(figsize=(10,5))

rain_mean.plot(marker='o')

plt.ylabel("Mean rain_mm_hr (mm/hr)")
plt.xlabel("VH Bin")
plt.title("Mean rain_mm_hr vs VH")

plt.grid(True)
plt.show()