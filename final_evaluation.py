import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# --------------------------------------------------
# 1. Load best-model final results
# --------------------------------------------------

results_path = (
    "../results/best_model_risk_fusion/"
    "final_decision_results.csv"
)

ground_truth_path = "../data/ground_truth.csv"

df = pd.read_csv(results_path)
ground_truth = pd.read_csv(ground_truth_path)

df = df.merge(
    ground_truth,
    on="Component_ID",
    how="left"
)

print("Best-model evaluation dataset loaded")
print("Total components:", len(df))


# --------------------------------------------------
# 2. Actual and predicted labels
# --------------------------------------------------

# True_Label:
# 0 = Healthy
# 1 = Latent defect
# 2 = Sudden failure

df["Actual_Defective"] = np.where(
    df["True_Label"] == 0,
    0,
    1
)

df["Predicted_Defective"] = np.where(
    df["Final_Decision"].isin(
        ["INVESTIGATE", "REJECT"]
    ),
    1,
    0
)


# --------------------------------------------------
# 3. Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    df["Actual_Defective"],
    df["Predicted_Defective"],
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix")
print(cm)

print("\nTrue Negative:", tn)
print("False Positive:", fp)
print("False Negative:", fn)
print("True Positive:", tp)


# --------------------------------------------------
# 4. Classification metrics
# --------------------------------------------------

accuracy = accuracy_score(
    df["Actual_Defective"],
    df["Predicted_Defective"]
)

precision = precision_score(
    df["Actual_Defective"],
    df["Predicted_Defective"],
    zero_division=0
)

recall = recall_score(
    df["Actual_Defective"],
    df["Predicted_Defective"],
    zero_division=0
)

f1 = f1_score(
    df["Actual_Defective"],
    df["Predicted_Defective"],
    zero_division=0
)

false_negative_rate = (
    fn / (fn + tp)
    if (fn + tp) > 0
    else 0
)


# --------------------------------------------------
# 5. Print metrics
# --------------------------------------------------

print("\nFinal Evaluation Metrics")
print("------------------------")

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("False Negative Rate:", round(false_negative_rate, 4))


# --------------------------------------------------
# 6. Decision distribution
# --------------------------------------------------

print("\nFinal Decision Distribution")
print(df["Final_Decision"].value_counts())


# --------------------------------------------------
# 7. Defect distribution
# --------------------------------------------------

print("\nActual Defect Distribution")
print(df["Actual_Defect_Type"].value_counts())


# --------------------------------------------------
# 8. Save evaluation report
# --------------------------------------------------

output_dir = "../results/best_model_evaluation"

os.makedirs(output_dir, exist_ok=True)

evaluation_report = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "False Negative Rate",
        "True Positive",
        "True Negative",
        "False Positive",
        "False Negative"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        false_negative_rate,
        tp,
        tn,
        fp,
        fn
    ]
})

report_path = (
    f"{output_dir}/final_evaluation_report.csv"
)

evaluation_report.to_csv(
    report_path,
    index=False
)


# --------------------------------------------------
# 9. Save evaluated component results
# --------------------------------------------------

evaluated_results_path = (
    f"{output_dir}/evaluated_components.csv"
)

df.to_csv(
    evaluated_results_path,
    index=False
)


# --------------------------------------------------
# 10. Final output
# --------------------------------------------------

print("\n" + "=" * 50)
print("BEST-MODEL FINAL EVALUATION COMPLETED")
print("=" * 50)

print("Report saved to:", report_path)
print("Component results saved to:", evaluated_results_path)
