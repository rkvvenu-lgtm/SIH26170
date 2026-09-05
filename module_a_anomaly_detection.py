import os
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. Load measurement dataset
# --------------------------------------------------

input_path = "../data/burn_in_measurements.csv"
output_dir = "../results"

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_path)

print("Dataset loaded successfully")
print("Number of components:", len(df))


# --------------------------------------------------
# 2. Select Module-A input features
# --------------------------------------------------

feature_columns = [
    "Iddq_0h_uA",
    "Iddq_24h_uA",
    "Iddq_96h_uA",
    "Leakage_0h_uA",
    "Leakage_24h_uA",
    "Leakage_96h_uA",
    "Delay_0h_ns",
    "Delay_24h_ns",
    "Delay_96h_ns"
]

X = df[feature_columns].copy()


# --------------------------------------------------
# 3. Standardize features
# --------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# --------------------------------------------------
# 4. Isolation Forest anomaly detection
# --------------------------------------------------

model = IsolationForest(
    n_estimators=200,
    contamination=0.15,
    random_state=42
)

model.fit(X_scaled)

df["Isolation_Prediction"] = model.predict(X_scaled)
df["Isolation_Score"] = -model.decision_function(X_scaled)

df["Isolation_Status"] = np.where(
    df["Isolation_Prediction"] == -1,
    "ANOMALY",
    "NORMAL"
)


# --------------------------------------------------
# 5. Lot-wise dynamic deviation
# --------------------------------------------------

dynamic_scores = []

for index, row in df.iterrows():

    lot_data = df[df["Lot_ID"] == row["Lot_ID"]]

    z_scores = []

    for column in feature_columns:

        lot_mean = lot_data[column].mean()
        lot_std = lot_data[column].std()

        if lot_std == 0 or pd.isna(lot_std):
            z_score = 0
        else:
            z_score = abs((row[column] - lot_mean) / lot_std)

        z_scores.append(z_score)

    maximum_z_score = max(z_scores)

    dynamic_scores.append(maximum_z_score)


df["Dynamic_Z_Score"] = dynamic_scores


# --------------------------------------------------
# 6. Dynamic anomaly status
# --------------------------------------------------

df["Dynamic_Status"] = np.where(
    df["Dynamic_Z_Score"] >= 3,
    "ANOMALY",
    "NORMAL"
)


# --------------------------------------------------
# 7. Combine both detection methods
# --------------------------------------------------

df["Module_A_Score"] = (
    0.5 * df["Isolation_Score"] +
    0.5 * (df["Dynamic_Z_Score"] / 5)
)

df["Module_A_Status"] = np.where(
    (df["Isolation_Status"] == "ANOMALY") |
    (df["Dynamic_Status"] == "ANOMALY"),
    "ANOMALY",
    "NORMAL"
)


# --------------------------------------------------
# 8. Add explanation
# --------------------------------------------------

def generate_explanation(row):

    if row["Module_A_Status"] == "NORMAL":
        return "Component behavior is within the expected lot distribution."

    if row["Dynamic_Z_Score"] >= 3:
        return (
            "Component shows significant deviation from its lot behavior. "
            "Dynamic outlier detected."
        )

    return "Isolation Forest detected abnormal component behavior."


df["Module_A_Explanation"] = df.apply(
    generate_explanation,
    axis=1
)


# --------------------------------------------------
# 9. Save Module-A results
# --------------------------------------------------

output_path = "../results/module_a_results.csv"

df.to_csv(output_path, index=False)

print("\nModule A completed successfully")
print("Output saved to:", output_path)

print("\nModule A status distribution:")
print(df["Module_A_Status"].value_counts())

print("\nSample results:")
print(
    df[
        [
            "Component_ID",
            "Lot_ID",
            "Module_A_Score",
            "Dynamic_Z_Score",
            "Module_A_Status"
        ]
    ].head(10)
)
