import os
import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Load Module-A and best-model Module-B results
# --------------------------------------------------

module_a_path = "../results/module_a_results.csv"

module_b_path = (
    "../results/best_model_module_b/"
    "best_model_module_b_results.csv"
)

module_a = pd.read_csv(module_a_path)
module_b = pd.read_csv(module_b_path)

print("Module-A and best-model Module-B loaded")


# --------------------------------------------------
# 2. Select required columns
# --------------------------------------------------

a_columns = [
    "Component_ID",
    "Lot_ID",
    "Module_A_Score",
    "Module_A_Status",
    "Module_A_Explanation"
]

b_columns = [
    "Component_ID",
    "Module_B_Status",
    "Module_B_Explanation"
]

module_a = module_a[a_columns]
module_b = module_b[b_columns]


# --------------------------------------------------
# 3. Merge both modules
# --------------------------------------------------

df = module_a.merge(
    module_b,
    on="Component_ID",
    how="inner"
)

print("Merged components:", len(df))


# --------------------------------------------------
# 4. Calculate risk score
# --------------------------------------------------

df["Anomaly_Risk"] = np.where(
    df["Module_A_Status"] == "ANOMALY",
    1.0,
    0.0
)

df["Drift_Risk"] = np.where(
    df["Module_B_Status"] == "EARLY_WARNING",
    1.0,
    0.0
)

df["Risk_Score"] = (
    0.5 * df["Anomaly_Risk"] +
    0.5 * df["Drift_Risk"]
)


# --------------------------------------------------
# 5. Final decision
# --------------------------------------------------

def final_decision(row):

    if (
        row["Module_A_Status"] == "ANOMALY"
        and row["Module_B_Status"] == "EARLY_WARNING"
    ):
        return "REJECT"

    elif (
        row["Module_A_Status"] == "ANOMALY"
        or row["Module_B_Status"] == "EARLY_WARNING"
    ):
        return "INVESTIGATE"

    return "PASS"


df["Final_Decision"] = df.apply(
    final_decision,
    axis=1
)


# --------------------------------------------------
# 6. Explainable decision
# --------------------------------------------------

def final_explanation(row):

    reasons = []

    if row["Module_A_Status"] == "ANOMALY":
        reasons.append(
            "Current measurements deviate from lot behavior"
        )

    if row["Module_B_Status"] == "EARLY_WARNING":
        reasons.append(
            "Predicted 168h drift exceeds the safe range"
        )

    if not reasons:
        return (
            "Component passed current and future "
            "behavior checks."
        )

    return "; ".join(reasons)


df["Final_Explanation"] = df.apply(
    final_explanation,
    axis=1
)


# --------------------------------------------------
# 7. Save final results
# --------------------------------------------------

output_dir = "../results/best_model_risk_fusion"
os.makedirs(output_dir, exist_ok=True)

output_path = (
    f"{output_dir}/final_decision_results.csv"
)

df.to_csv(output_path, index=False)


# --------------------------------------------------
# 8. Print results
# --------------------------------------------------

print("\n" + "=" * 50)
print("BEST-MODEL RISK FUSION COMPLETED")
print("=" * 50)

print("Output saved to:", output_path)

print("\nFinal decision distribution:")
print(df["Final_Decision"].value_counts())

print("\nSample results:")
print(
    df[
        [
            "Component_ID",
            "Module_A_Status",
            "Module_B_Status",
            "Risk_Score",
            "Final_Decision"
        ]
    ].head(10)
)
