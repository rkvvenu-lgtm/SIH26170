import os
import numpy as np
import pandas as pd

# ============================================================
# SIH26170 - FINAL SYNTHETIC BURN-IN DATASET GENERATOR
# ============================================================

np.random.seed(42)

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

NUM_COMPONENTS = 2000
NUM_LOTS = 20

# Burn-in measurement points
TIME_POINTS = [0, 24, 96, 168]

# Absolute datasheet limits
IDDQ_MAX_UA = 50.0
LEAKAGE_MAX_UA = 50.0
DELAY_MAX_NS = 30.0

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(DATA_DIR, exist_ok=True)

MEASUREMENTS_FILE = os.path.join(
    DATA_DIR,
    "burn_in_measurements.csv"
)

GROUND_TRUTH_FILE = os.path.join(
    DATA_DIR,
    "ground_truth.csv"
)

# ------------------------------------------------------------
# BASIC INFORMATION
# ------------------------------------------------------------

component_types = [
    "CMOS_LOGIC",
    "SRAM",
    "MICROCONTROLLER",
    "MIXED_SIGNAL_IC"
]

lot_ids = [
    f"LOT{i:02d}"
    for i in range(1, NUM_LOTS + 1)
]

# Manufacturing variation for each lot
lot_factors = {
    lot_id: np.random.uniform(0.92, 1.08)
    for lot_id in lot_ids
}

# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------

def positive(value, minimum=0.01):
    return max(float(value), minimum)


def rounded(value):
    return round(positive(value), 3)


# ------------------------------------------------------------
# DATA STORAGE
# ------------------------------------------------------------

measurement_records = []
ground_truth_records = []

# ============================================================
# GENERATE COMPONENT DATA
# ============================================================

for index in range(NUM_COMPONENTS):

    component_id = f"IC{index + 1:05d}"

    lot_id = lot_ids[index % NUM_LOTS]

    component_type = np.random.choice(
        component_types
    )

    temperature_c = np.random.normal(
        loc=125.0,
        scale=2.0
    )

    lot_factor = lot_factors[lot_id]

    # --------------------------------------------------------
    # Ground-truth class
    # --------------------------------------------------------
    # This is NOT given to the ML model.
    # It is used only for evaluation.

    random_value = np.random.random()

    if random_value < 0.85:
        defect_type = "HEALTHY"
        true_label = 0

    elif random_value < 0.95:
        defect_type = "LATENT_DEFECT"
        true_label = 1

    else:
        defect_type = "SUDDEN_FAILURE"
        true_label = 2

    # --------------------------------------------------------
    # Temperature effect
    # --------------------------------------------------------

    temperature_factor = 1.0 + (
        (temperature_c - 125.0) * 0.004
    )

    # --------------------------------------------------------
    # Initial measurements at 0h
    # --------------------------------------------------------

    base_iddq = (
        np.random.normal(10.0, 0.45)
        * lot_factor
        * temperature_factor
    )

    base_leakage = (
        np.random.normal(5.0, 0.25)
        * lot_factor
        * temperature_factor
    )

    base_delay = (
        np.random.normal(10.0, 0.35)
        * lot_factor
        * temperature_factor
    )

    # ========================================================
    # HEALTHY COMPONENT
    # ========================================================

    if defect_type == "HEALTHY":

        iddq_0h = base_iddq
        iddq_24h = iddq_0h + np.random.normal(0.30, 0.12)
        iddq_96h = iddq_24h + np.random.normal(0.25, 0.12)
        iddq_168h = iddq_96h + np.random.normal(0.30, 0.12)

        leakage_0h = base_leakage
        leakage_24h = leakage_0h + np.random.normal(0.15, 0.06)
        leakage_96h = leakage_24h + np.random.normal(0.20, 0.07)
        leakage_168h = leakage_96h + np.random.normal(0.25, 0.08)

        delay_0h = base_delay
        delay_24h = delay_0h + np.random.normal(0.05, 0.04)
        delay_96h = delay_24h + np.random.normal(0.08, 0.04)
        delay_168h = delay_96h + np.random.normal(0.10, 0.05)

    # ========================================================
    # LATENT DEFECT COMPONENT
    # ========================================================
    # Gradual abnormal drift.
    # Values are intentionally kept below absolute limits
    # in most cases, so static screening may miss them.

    elif defect_type == "LATENT_DEFECT":

        iddq_0h = base_iddq
        iddq_24h = iddq_0h + np.random.uniform(2.0, 4.0)
        iddq_96h = iddq_24h + np.random.uniform(5.0, 9.0)
        iddq_168h = iddq_96h + np.random.uniform(8.0, 13.0)

        leakage_0h = base_leakage
        leakage_24h = leakage_0h + np.random.uniform(1.0, 2.5)
        leakage_96h = leakage_24h + np.random.uniform(3.0, 6.5)
        leakage_168h = leakage_96h + np.random.uniform(5.0, 10.0)

        delay_0h = base_delay
        delay_24h = delay_0h + np.random.uniform(0.5, 1.2)
        delay_96h = delay_24h + np.random.uniform(1.2, 3.0)
        delay_168h = delay_96h + np.random.uniform(2.0, 4.5)

        # Keep latent defects mostly below datasheet limits
        iddq_168h = min(iddq_168h, 45.0)
        leakage_168h = min(leakage_168h, 45.0)
        delay_168h = min(delay_168h, 27.0)

    # ========================================================
    # SUDDEN FAILURE COMPONENT
    # ========================================================
    # Nearly normal initially, then sharp increase.

    else:

        iddq_0h = base_iddq
        iddq_24h = iddq_0h + np.random.normal(0.40, 0.15)
        iddq_96h = iddq_24h + np.random.normal(0.50, 0.20)
        iddq_168h = np.random.uniform(50.0, 70.0)

        leakage_0h = base_leakage
        leakage_24h = leakage_0h + np.random.normal(0.20, 0.08)
        leakage_96h = leakage_24h + np.random.normal(0.30, 0.10)
        leakage_168h = np.random.uniform(50.0, 75.0)

        delay_0h = base_delay
        delay_24h = delay_0h + np.random.normal(0.10, 0.05)
        delay_96h = delay_24h + np.random.normal(0.20, 0.08)
        delay_168h = np.random.uniform(35.0, 50.0)

    # --------------------------------------------------------
    # MEASUREMENT RECORD
    # --------------------------------------------------------
    # No defect label is included here.

    measurement_record = {
        "Component_ID": component_id,
        "Lot_ID": lot_id,
        "Component_Type": component_type,
        "Temperature_C": round(temperature_c, 2),

        "Iddq_0h_uA": rounded(iddq_0h),
        "Iddq_24h_uA": rounded(iddq_24h),
        "Iddq_96h_uA": rounded(iddq_96h),
        "Iddq_168h_uA": rounded(iddq_168h),

        "Leakage_0h_uA": rounded(leakage_0h),
        "Leakage_24h_uA": rounded(leakage_24h),
        "Leakage_96h_uA": rounded(leakage_96h),
        "Leakage_168h_uA": rounded(leakage_168h),

        "Delay_0h_ns": rounded(delay_0h),
        "Delay_24h_ns": rounded(delay_24h),
        "Delay_96h_ns": rounded(delay_96h),
        "Delay_168h_ns": rounded(delay_168h),

        # Static screening limits
        "Iddq_Max_Limit_uA": IDDQ_MAX_UA,
        "Leakage_Max_Limit_uA": LEAKAGE_MAX_UA,
        "Delay_Max_Limit_ns": DELAY_MAX_NS
    }

    measurement_records.append(
        measurement_record
    )

    # --------------------------------------------------------
    # GROUND-TRUTH RECORD
    # --------------------------------------------------------
    # Used only after prediction for evaluation.

    ground_truth_record = {
        "Component_ID": component_id,
        "Actual_Defect_Type": defect_type,
        "True_Label": true_label,

        # Actual future values for Module B evaluation
        "Actual_Iddq_168h_uA": rounded(iddq_168h),
        "Actual_Leakage_168h_uA": rounded(leakage_168h),
        "Actual_Delay_168h_ns": rounded(delay_168h)
    }

    ground_truth_records.append(
        ground_truth_record
    )

# ============================================================
# CREATE DATAFRAMES
# ============================================================

measurements_df = pd.DataFrame(
    measurement_records
)

ground_truth_df = pd.DataFrame(
    ground_truth_records
)

# Shuffle both files using the same component order
shuffle_order = np.random.permutation(
    len(measurements_df)
)

measurements_df = measurements_df.iloc[
    shuffle_order
].reset_index(drop=True)

ground_truth_df = ground_truth_df.iloc[
    shuffle_order
].reset_index(drop=True)

# ============================================================
# SAVE FILES
# ============================================================

measurements_df.to_csv(
    MEASUREMENTS_FILE,
    index=False
)

ground_truth_df.to_csv(
    GROUND_TRUTH_FILE,
    index=False
)

# ============================================================
# DISPLAY SUMMARY
# ============================================================

print("=" * 65)
print("SIH26170 FINAL BURN-IN DATASET CREATED")
print("=" * 65)

print(f"Measurements file : {MEASUREMENTS_FILE}")
print(f"Ground truth file : {GROUND_TRUTH_FILE}")

print(f"\nTotal components  : {len(measurements_df)}")
print(f"Measurement cols  : {len(measurements_df.columns)}")
print(f"Ground-truth cols : {len(ground_truth_df.columns)}")

print("\nGround-truth distribution:")
print(
    ground_truth_df["Actual_Defect_Type"].value_counts()
)

print("\nLot distribution:")
print(
    measurements_df["Lot_ID"].value_counts()
)

print("\nMeasurement columns:")
print(
    list(measurements_df.columns)
)

print("\nFirst five measurement rows:")
print(
    measurements_df.head()
)

print("\nFirst five ground-truth rows:")
print(
    ground_truth_df.head()
)

print("=" * 65)
print("DATASET GENERATION COMPLETED")
print("=" * 65)
