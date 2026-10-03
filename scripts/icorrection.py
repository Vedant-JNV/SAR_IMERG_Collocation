import pandas as pd
from sklearn.linear_model import LinearRegression

# =========================
# INPUT / OUTPUT FILES
# =========================
input_file = "collocated_1.parquet"
output_file = "collocated_1_icorrected.parquet"

# =========================
# LOAD DATA
# =========================
df = pd.read_parquet(input_file)

# Use incidence angle directly (degrees)
X = df[["incidence_angle"]]

# =========================
# VV ANGLE CORRECTION
# =========================
vv_model = LinearRegression()
vv_model.fit(X, df["VV_dB"])

vv_expected = vv_model.predict(X)

# Replace original VV_dB with residuals
df["VV_dB"] = df["VV_dB"] - vv_expected

# =========================
# VH ANGLE CORRECTION
# =========================
vh_model = LinearRegression()
vh_model.fit(X, df["VH_dB"])

vh_expected = vh_model.predict(X)

# Replace original VH_dB with residuals
df["VH_dB"] = df["VH_dB"] - vh_expected

# =========================
# RESULTS
# =========================
print("\nVV slope:")
print(vv_model.coef_[0])

print("\nVH slope:")
print(vh_model.coef_[0])

# =========================
# SAVE
# =========================
df.to_parquet(output_file, index=False)

print(f"\nSaved angle-corrected dataset to: {output_file}")