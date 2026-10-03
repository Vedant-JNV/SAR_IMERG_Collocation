import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_parquet("collocated_2.parquet")
df2 = pd.read_parquet("collocated_3.parquet")

print(df["VH_dB"].describe(),"\n")
print(df2["VH_dB"].describe(),"\n")

print(df["VV_dB"].describe(),"\n")
print(df2["VV_dB"].describe(),"\n")

print(df["rain_mm_hr"].describe(),"\n")
print(df2["rain_mm_hr"].describe(),"\n")