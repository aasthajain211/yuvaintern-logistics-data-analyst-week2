import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ----------------------------------------------------
# STEP 1: SIMULATE DATA LOADING
# ----------------------------------------------------
np.random.seed(42)
data_size = 1000

raw_data = {
    "Shipment_ID": [f"SHP_{1000+i}" if i % 100 != 0 else None for i in range(data_size)],
    "Origin_Hub": np.random.choice(["London", "london", "Singapore", "Tokyo", None], data_size),
    "Destination_Hub": np.random.choice(["New York", "Rotterdam", "Shanghai"], data_size),
    "Planned_Duration_Days": np.random.randint(2, 30, data_size),
    "Actual_Duration_Days": np.random.exponential(scale=7, size=data_size),
    "Shipment_Weight_KG": np.random.normal(loc=5000, scale=2500, size=data_size),
    "Transport_Mode": np.random.choice(["Air", "AIR", "Ocean", "Road", "Rail"], data_size)
}

df = pd.DataFrame(raw_data)

# Injecting artificial defects manually
df.loc[10:20, "Actual_Duration_Days"] = np.nan
df.loc[30:35, "Shipment_Weight_KG"] = -999  # Structural system error
df.loc[40:45, "Shipment_Weight_KG"] = 75000  # Extreme outlier

print("--- Initial Raw Profile ---")
print(df.info())

# ----------------------------------------------------
# STEP 2: STRUCTURAL CLEANING & TEXT STANDARDISATION
# ----------------------------------------------------
df.dropna(subset=["Shipment_ID"], inplace=True)
df["Origin_Hub"] = df["Origin_Hub"].str.strip().str.title()
df["Transport_Mode"] = df["Transport_Mode"].str.strip().str.upper()
df.loc[df["Shipment_Weight_KG"] <= 0, "Shipment_Weight_KG"] = np.nan

# ----------------------------------------------------
# STEP 3: IMPUTING MISSING VALUES
# ----------------------------------------------------
df["Origin_Hub"].fillna("Unknown_Hub", inplace=True)
df["Actual_Duration_Days"] = df.groupby("Transport_Mode")["Actual_Duration_Days"].transform(lambda x: x.fillna(x.median()))
df["Shipment_Weight_KG"] = df.groupby("Transport_Mode")["Shipment_Weight_KG"].transform(lambda x: x.fillna(x.median()))

# ----------------------------------------------------
# STEP 4: OUTLIER TREATMENT (WINSORIZATION VIA IQR)
# ----------------------------------------------------
for col in ["Actual_Duration_Days", "Shipment_Weight_KG"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower_bound, upper_bound)

# ----------------------------------------------------
# STEP 5: FEATURE SCALING & NORMALIZATION
# ----------------------------------------------------
min_max_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

df["Scaled_Planned_Duration"] = min_max_scaler.fit_transform(df[["Planned_Duration_Days"]])
df["Normalized_Weight"] = standard_scaler.fit_transform(df[["Shipment_Weight_KG"]])

print("\n--- Cleaned & Preprocessed Profile ---")
print(df.head())

