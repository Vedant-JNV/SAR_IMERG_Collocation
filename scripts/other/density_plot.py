import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_parquet("collocated_1_icorrected.parquet")

classes = [
    (0,0.5),
    (0.5,1),
    (1,5),
    (5,10),
    (10,20),
    (20,50)
]

plt.figure(figsize=(10,6))

for low,high in classes:

    subset = df[
        (df["rain_mm_hr"]>=low) &
        (df["rain_mm_hr"]<high)
    ]

    sns.kdeplot(
        subset["VH_dB"],
        label=f"{low}-{high}"
    )

plt.xlabel("VH/VV (dB)")
plt.ylabel("Density")
plt.title("VH/VV Distribution for Different rain_mm_hr Classes")
plt.legend()

plt.show()