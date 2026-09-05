import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. Load datasets
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "model_comparison")

os.makedirs(OUTPUT_DIR, exist_ok=True)

measurements_path = os.path.join(
    DATA_DIR,
    "burn_in_measurements.csv"
)

ground_truth_path = os.path.join(
    DATA_DIR,
    "ground_truth.csv"
)

measurements = pd.read_csv(measurements_path)
ground_truth = pd.read_csv(ground_truth_path)

df = measurements.merge(
    ground_truth,
    on="Component_ID",
    how="left"
)

print("Dataset loaded successfully")
print("Total components:", len(df))


# --------------------------------------------------
# 2. Parameter configuration
# --------------------------------------------------

parameters = {
    "Iddq": {
        "input_columns": [
            "Iddq_0h_uA",
            "Iddq_24h_uA"
        ],
        "target_column": "Actual_Iddq_168h_uA",
        "unit": "µA"
    },

    "Leakage": {
        "input_columns": [
            "Leakage_0h_uA",
            "Leakage_24h_uA"
        ],
        "target_column": "Actual_Leakage_168h_uA",
        "unit": "µA"
    },

    "Delay": {
        "input_columns": [
            "Delay_0h_ns",
            "Delay_24h_ns"
        ],
        "target_column": "Actual_Delay_168h_ns",
        "unit": "ns"
    }
}


# --------------------------------------------------
# 3. Define models
# --------------------------------------------------

def create_models():

    return {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
    }


# --------------------------------------------------
# 4. Compare models for each parameter
# --------------------------------------------------

all_metrics = []
best_models = []

for parameter_name, config in parameters.items():

    print("\n" + "=" * 60)
    print("Parameter:", parameter_name)
    print("=" * 60)

    # Select input and target columns
    X = df[config["input_columns"]].copy()
    y = df[config["target_column"]].copy()

    # Remove missing values
    valid_data = pd.concat([X, y], axis=1).dropna()

    X = valid_data[config["input_columns"]]
    y = valid_data[config["target_column"]]

    if len(X) < 10:
        print(f"Not enough data for {parameter_name}")
        continue

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    models = create_models()
    parameter_predictions = {}

    for model_name, model in models.items():

        # Train model
        model.fit(X_train, y_train)

        # Predict 168h value
        y_pred = model.predict(X_test)

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)

        rmse = np.sqrt(
            mean_squared_error(y_test, y_pred)
        )

        r2 = r2_score(y_test, y_pred)

        all_metrics.append({
            "Parameter": parameter_name,
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2_Score": r2
        })

        parameter_predictions[model_name] = y_pred

        print(
            f"{model_name}: "
            f"MAE={mae:.4f}, "
            f"RMSE={rmse:.4f}, "
            f"R2={r2:.4f}"
        )

    # --------------------------------------------------
    # 5. Select best model based on minimum MAE
    # --------------------------------------------------

    parameter_metrics = pd.DataFrame([
        row for row in all_metrics
        if row["Parameter"] == parameter_name
    ])

    if parameter_metrics.empty:
        continue

    best_row = parameter_metrics.loc[
        parameter_metrics["MAE"].idxmin()
    ]

    best_models.append({
        "Parameter": parameter_name,
        "Best_Model": best_row["Model"],
        "Best_MAE": best_row["MAE"],
        "Best_RMSE": best_row["RMSE"],
        "Best_R2_Score": best_row["R2_Score"]
    })

    print("\nBest model:", best_row["Model"])

    # --------------------------------------------------
    # 6. Create model comparison graph
    # --------------------------------------------------

    x = np.arange(len(parameter_metrics))
    width = 0.25

    plt.figure(figsize=(10, 6))

    plt.bar(
        x - width,
        parameter_metrics["MAE"],
        width,
        label="MAE"
    )

    plt.bar(
        x,
        parameter_metrics["RMSE"],
        width,
        label="RMSE"
    )

    plt.bar(
        x + width,
        parameter_metrics["R2_Score"],
        width,
        label="R² Score"
    )

    plt.xticks(
        x,
        parameter_metrics["Model"],
        rotation=15
    )

    plt.title(
        f"{parameter_name} Model Performance Comparison"
    )

    plt.xlabel("Model")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    comparison_graph_path = os.path.join(
        OUTPUT_DIR,
        f"{parameter_name.lower()}_comparison.png"
    )

    plt.savefig(
        comparison_graph_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------
    # 7. Actual vs predicted graph
    # --------------------------------------------------

    plt.figure(figsize=(10, 6))

    for model_name, y_pred in parameter_predictions.items():

        plt.scatter(
            y_test,
            y_pred,
            alpha=0.5,
            label=model_name
        )

    min_value = min(y_test.min(), min(
        np.min(prediction)
        for prediction in parameter_predictions.values()
    ))

    max_value = max(y_test.max(), max(
        np.max(prediction)
        for prediction in parameter_predictions.values()
    ))

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        label="Ideal Prediction"
    )

    plt.title(
        f"Actual vs Predicted {parameter_name} at 168h"
    )

    plt.xlabel(
        f"Actual {parameter_name} ({config['unit']})"
    )

    plt.ylabel(
        f"Predicted {parameter_name} ({config['unit']})"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    prediction_graph_path = os.path.join(
        OUTPUT_DIR,
        f"{parameter_name.lower()}_actual_vs_predicted.png"
    )

    plt.savefig(
        prediction_graph_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# --------------------------------------------------
# 8. Save all results
# --------------------------------------------------

all_metrics_df = pd.DataFrame(all_metrics)
best_models_df = pd.DataFrame(best_models)

all_metrics_path = os.path.join(
    OUTPUT_DIR,
    "all_model_metrics.csv"
)

best_models_path = os.path.join(
    OUTPUT_DIR,
    "best_models.csv"
)

# Save results
all_metrics_df.to_csv(
    all_metrics_path,
    index=False
)

best_models_df.to_csv(
    best_models_path,
    index=False
)


# --------------------------------------------------
# 9. Final output
# --------------------------------------------------

print("\n" + "=" * 60)
print("MODEL COMPARISON COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nAll model metrics:")
print(all_metrics_df)

print("\nBest models:")
print(best_models_df)

print("\nResults saved inside:")
print(OUTPUT_DIR)
