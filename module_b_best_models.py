import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. Load datasets
# --------------------------------------------------

measurements = pd.read_csv(
    "../data/burn_in_measurements.csv"
)

ground_truth = pd.read_csv(
    "../data/ground_truth.csv"
)

df = measurements.merge(
    ground_truth,
    on="Component_ID",
    how="left"
)

output_dir = "../results/best_model_module_b"
os.makedirs(output_dir, exist_ok=True)

print("Dataset loaded successfully")
print("Total components:", len(df))


# --------------------------------------------------
# 2. Parameter configuration
# --------------------------------------------------

parameters = {
    "Iddq": {
        "inputs": ["Iddq_0h_uA", "Iddq_24h_uA"],
        "target": "Actual_Iddq_168h_uA",
        "model": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),
        "limit": "Iddq_Max_uA"
    },

    "Leakage": {
        "inputs": ["Leakage_0h_uA", "Leakage_24h_uA"],
        "target": "Actual_Leakage_168h_uA",
        "model": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),
        "limit": "Leakage_Max_uA"
    },

    "Delay": {
        "inputs": ["Delay_0h_ns", "Delay_24h_ns"],
        "target": "Actual_Delay_168h_ns",
        "model": LinearRegression(),
        "limit": "Delay_Max_ns"
    }
}


# --------------------------------------------------
# 3. Train best model and predict
# --------------------------------------------------

all_results = []
metrics = []

for parameter_name, config in parameters.items():

    print("\n" + "=" * 50)
    print("Processing:", parameter_name)
    print("=" * 50)

    X = df[config["inputs"]]
    y = df[config["target"]]

    model = config["model"]

    # Train using available burn-in data
    model.fit(X, y)

    predictions = model.predict(X)

    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)

    print("Model:", type(model).__name__)
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))

    # --------------------------------------------------
    # 4. Calculate predicted drift
    # --------------------------------------------------

    value_24h = df[config["inputs"][1]]

    drift_rate = (
        predictions - value_24h
    ) / value_24h.replace(0, np.nan)

    drift_rate = drift_rate.fillna(0)

    # --------------------------------------------------
    # 5. Safety decision
    # --------------------------------------------------

    safety_slope = 0.20

    predicted_limit = np.inf

    early_warning = (
        (predictions > predicted_limit) |
        (drift_rate > safety_slope)
    )

    status = np.where(
        early_warning,
        "EARLY_WARNING",
        "NORMAL"
    )

    # --------------------------------------------------
    # 6. Save parameter results
    # --------------------------------------------------

    parameter_result = pd.DataFrame({
        "Component_ID": df["Component_ID"],
        "Lot_ID": df["Lot_ID"],
        f"{parameter_name}_Predicted_168h": predictions,
        f"{parameter_name}_Actual_168h": y,
        f"{parameter_name}_Drift_Rate": drift_rate,
        f"{parameter_name}_Status": status
    })

    all_results.append(parameter_result)

    metrics.append({
        "Parameter": parameter_name,
        "Model": type(model).__name__,
        "MAE": mae,
        "RMSE": rmse,
        "R2_Score": r2
    })


# --------------------------------------------------
# 7. Merge all parameter results
# --------------------------------------------------

final_results = all_results[0]

for result in all_results[1:]:
    final_results = final_results.merge(
        result,
        on=["Component_ID", "Lot_ID"],
        how="inner"
    )


# --------------------------------------------------
# 8. Overall Module-B status
# --------------------------------------------------

status_columns = [
    "Iddq_Status",
    "Leakage_Status",
    "Delay_Status"
]

final_results["Module_B_Status"] = np.where(
    final_results[status_columns].eq("EARLY_WARNING").any(axis=1),
    "EARLY_WARNING",
    "NORMAL"
)


# --------------------------------------------------
# 9. Explainability
# --------------------------------------------------

def create_explanation(row):

    warnings = []

    if row["Iddq_Status"] == "EARLY_WARNING":
        warnings.append(
            "Iddq predicted drift exceeds the safe range"
        )

    if row["Leakage_Status"] == "EARLY_WARNING":
        warnings.append(
            "Leakage predicted drift exceeds the safe range"
        )

    if row["Delay_Status"] == "EARLY_WARNING":
        warnings.append(
            "Propagation delay predicted drift exceeds the safe range"
        )

    if not warnings:
        return "All predicted 168h parameters are within the safe range."

    return "; ".join(warnings)


final_results["Module_B_Explanation"] = (
    final_results.apply(create_explanation, axis=1)
)


# --------------------------------------------------
# 10. Save outputs
# --------------------------------------------------

results_path = f"{output_dir}/best_model_module_b_results.csv"
metrics_path = f"{output_dir}/best_model_metrics.csv"

final_results.to_csv(
    results_path,
    index=False
)

pd.DataFrame(metrics).to_csv(
    metrics_path,
    index=False
)

print("\n" + "=" * 50)
print("BEST MODEL MODULE B COMPLETED")
print("=" * 50)

print("Results saved to:", results_path)
print("Metrics saved to:", metrics_path)

print("\nModule-B status distribution:")
print(final_results["Module_B_Status"].value_counts())

print("\nSample results:")
print(
    final_results[
        [
            "Component_ID",
            "Iddq_Status",
            "Leakage_Status",
            "Delay_Status",
            "Module_B_Status"
        ]
    ].head(10)
)
