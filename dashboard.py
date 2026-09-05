import os

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI Burn-in Screening",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# 2. CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>

    /* Main application background */
    .stApp {
        background-color: #0e1117;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Main title */
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #a0aec0;
        margin-bottom: 1.5rem;
    }

    /* Section headings */
    .section-title {
        font-size: 1.35rem;
        font-weight: 650;
        color: #ffffff;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    /* Status box */
    .status-box {
        padding: 0.8rem 1rem;
        border-radius: 0.6rem;
        background-color: #dff5e3;
        border: 1px solid #a8d5b0;
        color: #166534;
        font-weight: 600;
    }

    /* KPI cards */
    div[data-testid="stMetric"] {
        background-color: #1f2937;
        border: 1px solid #374151;
        padding: 1rem;
        border-radius: 0.8rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
    }

    div[data-testid="stMetricLabel"] {
        color: #d1d5db !important;
    }

    div[data-testid="stMetricLabel"] p {
        color: #d1d5db !important;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    div[data-testid="stMetricValue"] div {
        color: #ffffff !important;
    }

    div[data-testid="stMetricDelta"] {
        color: #86efac !important;
    }

    div[data-testid="stMetricDelta"] svg {
        fill: #86efac !important;
    }

    /* General text */
    .stMarkdown p {
        color: #f3f4f6;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #ffffff !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #20212b;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {
        color: #f3f4f6 !important;
    }

    /* Selectbox and multiselect text */
    div[data-baseweb="select"] * {
        color: #f3f4f6 !important;
    }

    /* Input text */
    input {
        color: #ffffff !important;
    }

    /* Dataframe text */
    div[data-testid="stDataFrame"] {
        border: 1px solid #374151;
        border-radius: 0.5rem;
    }

    /* Info box */
    div[data-testid="stAlert"] {
        color: #ffffff;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# 3. LOAD DATA
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "..",
    "results"
)

FINAL_RESULTS_PATH = os.path.join(
    RESULTS_DIR,
    "best_model_risk_fusion",
    "final_decision_results.csv"
)

MODULE_B_PATH = os.path.join(
    RESULTS_DIR,
    "best_model_module_b",
    "best_model_module_b_results.csv"
)

if not os.path.exists(FINAL_RESULTS_PATH):

    st.error(
        "Final result file not found. "
        "Please run main.py first."
    )

    st.stop()

df = pd.read_csv(FINAL_RESULTS_PATH)


# ==================================================
# 4. LOAD MODULE B RESULTS
# ==================================================

if os.path.exists(MODULE_B_PATH):

    module_b_df = pd.read_csv(MODULE_B_PATH)

    if "Component_ID" in module_b_df.columns:

        extra_columns = [
            column
            for column in module_b_df.columns
            if column != "Component_ID"
            and column not in df.columns
        ]

        if extra_columns:

            df = df.merge(
                module_b_df[
                    ["Component_ID"] + extra_columns
                ],
                on="Component_ID",
                how="left"
            )


# ==================================================
# 5. HEADER
# ==================================================

st.markdown(
    '<div class="main-title">'
    '🔬 AI-Driven Component Burn-in Screening'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predictive Anomaly Detection and Drift Analysis'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="status-box">'
    '🟢 System Ready &nbsp; | &nbsp; AI Analysis Completed'
    '</div>',
    unsafe_allow_html=True
)

st.write("")


# ==================================================
# 6. SIDEBAR FILTERS
# ==================================================

st.sidebar.title("⚙️ Screening Controls")

st.sidebar.caption(
    "Filter and investigate component screening results."
)

decision_options = sorted(
    df["Final_Decision"].dropna().unique()
)

decision_filter = st.sidebar.multiselect(
    "Final Decision",
    options=decision_options,
    default=decision_options
)

lot_options = sorted(
    df["Lot_ID"].dropna().unique()
)

lot_filter = st.sidebar.multiselect(
    "Lot ID",
    options=lot_options,
    default=lot_options
)

risk_threshold = st.sidebar.slider(
    "Minimum Risk Score",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05
)

filtered_df = df[
    df["Final_Decision"].isin(decision_filter)
    & df["Lot_ID"].isin(lot_filter)
    & (df["Risk_Score"] >= risk_threshold)
]


# ==================================================
# 7. SCREENING OVERVIEW
# ==================================================

st.markdown(
    '<div class="section-title">📊 Screening Overview</div>',
    unsafe_allow_html=True
)

total_components = len(df)

pass_count = int(
    (df["Final_Decision"] == "PASS").sum()
)

investigate_count = int(
    (df["Final_Decision"] == "INVESTIGATE").sum()
)

reject_count = int(
    (df["Final_Decision"] == "REJECT").sum()
)

pass_percentage = (
    pass_count / total_components * 100
    if total_components > 0 else 0
)

investigate_percentage = (
    investigate_count / total_components * 100
    if total_components > 0 else 0
)

reject_percentage = (
    reject_count / total_components * 100
    if total_components > 0 else 0
)

average_risk = df["Risk_Score"].mean()

high_risk_count = int(
    (df["Risk_Score"] >= 0.60).sum()
)


# First row of KPI cards
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Components",
    f"{total_components:,}"
)

col2.metric(
    "PASS",
    f"{pass_count:,}",
    f"{pass_percentage:.1f}%"
)

col3.metric(
    "INVESTIGATE",
    f"{investigate_count:,}",
    f"{investigate_percentage:.1f}%"
)

col4.metric(
    "REJECT",
    f"{reject_count:,}",
    f"{reject_percentage:.1f}%"
)

st.write("")


# Second row of KPI cards
col5, col6, col7 = st.columns(3)

col5.metric(
    "Average Risk Score",
    f"{average_risk:.3f}"
)

col6.metric(
    "High-Risk Components",
    f"{high_risk_count:,}"
)

col7.metric(
    "Displayed Components",
    f"{len(filtered_df):,}"
)


# ==================================================
# 8. DECISION AND RISK ANALYSIS
# ==================================================

st.markdown(
    '<div class="section-title">📈 Decision and Risk Analysis</div>',
    unsafe_allow_html=True
)

chart_col1, chart_col2 = st.columns(2)


# Decision chart
with chart_col1:

    st.subheader("Final Decision Distribution")

    decision_counts = df["Final_Decision"].value_counts()

    fig1, ax1 = plt.subplots(figsize=(7, 5))

    ax1.pie(
        decision_counts.values,
        labels=decision_counts.index,
        autopct="%1.1f%%",
        startangle=90
    )

    ax1.set_title("Screening Decision Distribution")

    st.pyplot(fig1)
    plt.close(fig1)


# Risk chart
with chart_col2:

    st.subheader("Risk Score Distribution")

    fig2, ax2 = plt.subplots(figsize=(7, 5))

    ax2.hist(
        df["Risk_Score"].dropna(),
        bins=15
    )

    ax2.set_xlabel("Risk Score")
    ax2.set_ylabel("Number of Components")
    ax2.set_title("Component Risk Distribution")
    ax2.grid(axis="y", alpha=0.3)

    st.pyplot(fig2)
    plt.close(fig2)


# ==================================================
# 9. MODULE A ANOMALY ANALYSIS
# ==================================================

if "Module_A_Score" in df.columns:

    st.markdown(
        '<div class="section-title">'
        '🧠 Module A Anomaly Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    fig3, ax3 = plt.subplots(figsize=(10, 5))

    ax3.hist(
        df["Module_A_Score"].dropna(),
        bins=15
    )

    ax3.set_xlabel("Module A Anomaly Score")
    ax3.set_ylabel("Number of Components")
    ax3.set_title("Dynamic Anomaly Score Distribution")
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    st.pyplot(fig3)
    plt.close(fig3)


# ==================================================
# 10. LOT-WISE ANALYSIS
# ==================================================

st.markdown(
    '<div class="section-title">'
    '🏭 Lot-wise Screening Analysis'
    '</div>',
    unsafe_allow_html=True
)

lot_summary = (
    df.groupby("Lot_ID")
    .agg(
        Total_Components=("Component_ID", "count"),
        Average_Risk=("Risk_Score", "mean"),
        High_Risk=("Risk_Score", lambda x: (x >= 0.60).sum()),
        Reject_Count=(
            "Final_Decision",
            lambda x: (x == "REJECT").sum()
        ),
        Investigate_Count=(
            "Final_Decision",
            lambda x: (x == "INVESTIGATE").sum()
        )
    )
    .reset_index()
)

st.dataframe(
    lot_summary,
    use_container_width=True,
    hide_index=True
)

fig4, ax4 = plt.subplots(figsize=(12, 5))

ax4.bar(
    lot_summary["Lot_ID"].astype(str),
    lot_summary["Average_Risk"]
)

ax4.set_xlabel("Lot ID")
ax4.set_ylabel("Average Risk Score")
ax4.set_title("Average Risk Score by Lot")
ax4.tick_params(axis="x", rotation=45)
ax4.grid(axis="y", alpha=0.3)

plt.tight_layout()

st.pyplot(fig4)
plt.close(fig4)


# ==================================================
# 11. COMPONENT INVESTIGATION
# ==================================================

st.markdown(
    '<div class="section-title">'
    '🔎 Component Investigation'
    '</div>',
    unsafe_allow_html=True
)

component_ids = sorted(
    df["Component_ID"].astype(str).unique()
)

selected_component = st.selectbox(
    "Select Component ID",
    options=component_ids
)

selected_row = df[
    df["Component_ID"].astype(str) == selected_component
]

if not selected_row.empty:

    component = selected_row.iloc[0]

    st.subheader(
        f"Component Details: {selected_component}"
    )

    detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)

    detail_col1.metric(
        "Final Decision",
        str(component["Final_Decision"])
    )

    detail_col2.metric(
        "Risk Score",
        f"{component['Risk_Score']:.3f}"
    )

    detail_col3.metric(
        "Anomaly Risk",
        f"{component['Anomaly_Risk']:.3f}"
    )

    detail_col4.metric(
        "Drift Risk",
        f"{component['Drift_Risk']:.3f}"
    )

    st.write("")

    st.write(
        "**Module A Status:**",
        component.get("Module_A_Status", "N/A")
    )

    st.write(
        "**Module B Status:**",
        component.get("Module_B_Status", "N/A")
    )

    st.info(
        component.get(
            "Final_Explanation",
            "No explanation available."
        )
    )


# ==================================================
# 12. COMPONENT SCREENING TABLE
# ==================================================

st.markdown(
    '<div class="section-title">'
    '📋 Component Screening Results'
    '</div>',
    unsafe_allow_html=True
)

display_columns = [
    "Component_ID",
    "Lot_ID",
    "Module_A_Score",
    "Module_A_Status",
    "Module_B_Status",
    "Anomaly_Risk",
    "Drift_Risk",
    "Risk_Score",
    "Final_Decision",
    "Final_Explanation"
]

available_columns = [
    column
    for column in display_columns
    if column in filtered_df.columns
]

st.dataframe(
    filtered_df[available_columns],
    use_container_width=True,
    hide_index=True
)


# ==================================================
# 13. DOWNLOAD REPORT
# ==================================================

st.markdown(
    '<div class="section-title">'
    '📥 Export Screening Report'
    '</div>',
    unsafe_allow_html=True
)

csv_data = filtered_df.to_csv(index=False)

st.download_button(
    label="⬇️ Download Screening Report",
    data=csv_data,
    file_name="burn_in_screening_report.csv",
    mime="text/csv"
)


# ==================================================
# 14. FOOTER
# ==================================================

st.markdown("---")

st.caption(
    "AI-Driven Component Burn-in Screening System | "
    "Module A + Best Model Module B + Risk Fusion"
)
