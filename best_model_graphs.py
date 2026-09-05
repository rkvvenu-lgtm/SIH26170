import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


# --------------------------------------------------
# 1. Load final risk-fusion results
# --------------------------------------------------

fusion_path = (
    "../results/best_model_risk_fusion/"
    "final_decision_results.csv"
)

module_b_path = (
    "../results/best_model_module_b/"
    "best_model_module_b_results.csv"
)

ground_truth_path = "../data/ground_truth.csv"

fusion = pd.read_csv(fusion_path)
module_b = pd.read_csv(module_b_path)
ground_truth = pd.read_csv(ground_truth_path)

print("All files loaded successfully")


# --------------------------------------------------
# 2. Merge data
# --------------------------------------------------

df = fusion.merge(
    module_b,
    on=["Component_ID", "Lot_ID"],
    how="inner"
)

df = df.merge(
    ground_truth,
    on="Component_ID",
    how="left"
)

print("Total components:", len(df))


# --------------------------------------------------
# 3. Create output folder
# --------------------------------------------------

output_dir = "../results/best_model_graphs"

os.makedirs(output_dir, exist_ok=True)


# --------------------------------------------------
# 4. Actual and predicted labels
# --------------------------------------------------

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
# Graph 1: Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(
    df["Actual_Defective"],
    df["Predicted_Defective"],
    labels=[0, 1]
)

plt.figure(figsize=(6, 5))
plt.imshow(cm)

plt.title("Best-Model Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    [0, 1],
    ["Healthy", "Defective"]
)

plt.yticks(
    [0, 1],
    ["Healthy", "Defective"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()
plt.tight_layout()

plt.savefig(
    f"{output_dir}/confusion_matrix.png",
    dpi=300
)

plt.show()
plt.close()


# --------------------------------------------------
# Graph 2: Final Decision Distribution
# --------------------------------------------------

decision_counts = df["Final_Decision"].value_counts()

plt.figure(figsize=(7, 5))
decision_counts.plot(kind="bar")

plt.title("Best-Model Final Decision Distribution")
plt.xlabel("Final Decision")
plt.ylabel("Number of Components")

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    f"{output_dir}/final_decision_distribution.png",
    dpi=300
)

plt.show()
plt.close()


# --------------------------------------------------
# Graph 3: Risk Score Distribution
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["Risk_Score"],
    bins=20
)

plt.title("Best-Model Risk Score Distribution")
plt.xlabel("Risk Score")
plt.ylabel("Number of Components")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{output_dir}/risk_score_distribution.png",
    dpi=300
)

plt.show()
plt.close()


# --------------------------------------------------
# Graph 4: Actual vs Predicted Iddq
# --------------------------------------------------

sample_df = df.head(100)

plt.figure(figsize=(9, 5))

plt.plot(
    sample_df["Iddq_Actual_168h"].values,
    label="Actual Iddq at 168h"
)

plt.plot(
    sample_df["Iddq_Predicted_168h"].values,
    label="Predicted Iddq at 168h"
)

plt.title("Actual vs Predicted Iddq at 168h")
plt.xlabel("Component Index")
plt.ylabel("Iddq (µA)")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{output_dir}/iddq_actual_vs_predicted.png",
    dpi=300
)

plt.show()
plt.close()


# --------------------------------------------------
# Graph 5: Actual vs Predicted Leakage
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    sample_df["Leakage_Actual_168h"].values,
    label="Actual Leakage at 168h"
)

plt.plot(
    sample_df["Leakage_Predicted_168h"].values,
    label="Predicted Leakage at 168h"
)

plt.title("Actual vs Predicted Leakage at 168h")
plt.xlabel("Component Index")
plt.ylabel("Leakage (µA)")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{output_dir}/leakage_actual_vs_predicted.png",
    dpi=300
)

plt.show()
plt.close()


# --------------------------------------------------
# Graph 6: Actual vs Predicted Delay
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    sample_df["Delay_Actual_168h"].values,
    label="Actual Delay at 168h"
)

plt.plot(
    sample_df["Delay_Predicted_168h"].values,
    label="Predicted Delay at 168h"
)

plt.title("Actual vs Predicted Delay at 168h")
plt.xlabel("Component Index")
plt.ylabel("Delay (ns)")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{output_dir}/delay_actual_vs_predicted.png",
    dpi=300
)

plt.show()
plt.close()


# --------------------------------------------------
# 7. Save merged graph data
# --------------------------------------------------

df.to_csv(
    f"{output_dir}/graph_data.csv",
    index=False
)

print("\nAll best-model graphs generated successfully.")
print("Graphs saved inside:", output_dir)
