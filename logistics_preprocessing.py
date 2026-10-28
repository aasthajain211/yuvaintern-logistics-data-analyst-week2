import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ----------------------------------------------------
# STEP 1: SIMULATE DATA LOADING (Synchronized with Schema)
# ----------------------------------------------------
np.random.seed(42)
data_size = 1000  # Representative sample from the 50,000 global records

raw_data = {
    "Shipment_ID": [f"SHP_{1000+i}" if i % 100 != 0 else None for i in range(data_size)],
    "Origin_Hub": np.random.choice(["London", "london", "Singapore", "Tokyo", None], data_size),
    "Destination_Hub": np.random.choice(["New York", "Rotterdam", "Shanghai"], data_size),
    "Planned_Duration_Days": np.random.randint(2, 30, data_size),
    "Actual_Duration_Days": np.random.exponential(scale=7, size=data_size),
    "Shipment_Weight_KG": np.random.normal(loc=5000, scale=2500, size=data_size),
    "Transport_Mode": np.random.choice(["Air", "AIR", "Ocean", "Road", "Rail"], data_size),
    "Delivery_Status": np.random.choice(["Delayed", "On-Time", "Cancelled"], size=data_size, p=[0.2, 0.7, 0.1])
}

df = pd.DataFrame(raw_data)

# Injecting artificial defects manually as per documentation schema
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
df["Origin_Hub"] = df["Origin_Hub"].fillna("Unknown_Hub")
df["Actual_Duration_Days"] = df.groupby("Transport_Mode")["Actual_Duration_Days"].transform(lambda x: x.fillna(x.median()))
df["Shipment_Weight_KG"] = df.groupby("Transport_Mode")["Shipment_Weight_KG"].transform(lambda x: x.fillna(x.median()))
df["Delivery_Status"] = df["Delivery_Status"].fillna("On-Time")

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
print(df[['Shipment_ID', 'Scaled_Planned_Duration', 'Normalized_Weight', 'Delivery_Status']].head())

# ----------------------------------------------------
# STEP 6: CONCRETE DOWNSTREAM LOGISTICS ANALYSIS (Predictive Modeling)
# ----------------------------------------------------
# Using cleaned data to predict whether a shipment will be delayed
df['Is_Delayed'] = (df['Delivery_Status'] == 'Delayed').astype(int)

X = df[['Scaled_Planned_Duration', 'Normalized_Weight', 'Actual_Duration_Days']]
y = df['Is_Delayed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

print(f"\n--- Downstream Logistics Analysis Results ---")
print(f"Logistic Regression Delay Prediction Accuracy using Cleaned Data: {accuracy * 100:.2f}%")
