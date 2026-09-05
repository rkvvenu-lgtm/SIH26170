import os
import subprocess
import sys


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------
# Pipeline modules
# --------------------------------------------------

pipeline = [
    "module_a_anomaly_detection.py",
    "module_b_best_models.py",
    "model_comparison.py",
    "risk_fusion_best_models.py",
    "final_evaluation.py",
    "best_model_graphs.py"
]


# --------------------------------------------------
# Run each module automatically
# --------------------------------------------------

print("=" * 60)
print("AI-DRIVEN BURN-IN SCREENING SYSTEM")
print("AUTOMATED PIPELINE STARTED")
print("=" * 60)


for step, script in enumerate(pipeline, start=1):

    script_path = os.path.join(BASE_DIR, script)

    print("\n" + "-" * 60)
    print(f"STEP {step}: Running {script}")
    print("-" * 60)

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=BASE_DIR
    )

    if result.returncode != 0:

        print("\nERROR: Pipeline stopped.")
        print("Failed module:", script)

        sys.exit(1)

    print(f"STEP {step} COMPLETED SUCCESSFULLY")


# --------------------------------------------------
# Final output
# --------------------------------------------------

print("\n" + "=" * 60)
print("AUTOMATED PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nFinal outputs available in:")

print("../results/best_model_risk_fusion/")
print("../results/best_model_evaluation/")
print("../results/best_model_graphs/")
