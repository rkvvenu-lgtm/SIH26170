import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from datetime import datetime

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Burn-In AI Screening",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM STYLE
# --------------------------------------------------

st.markdown("""
<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], .main, .main .block-container { background-color: #0f172a !important; }
.stApp, .stApp p, .stApp span, .stApp label, .stApp div, .stApp li, h1, h2, h3, h4, h5, h6 { color: #f8fafc !important; }
section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div { background-color: #111827 !important; }
section[data-testid="stSidebar"] * { color: #f8fafc !important; }
.main-title { font-size: 36px; font-weight: 700; color: #f8fafc !important; margin-bottom: 5px; }
.subtitle { font-size: 16px; color: #cbd5e1 !important; margin-bottom: 25px; }
.section-title { font-size: 23px; font-weight: 650; color: #f8fafc !important; margin-top: 25px; margin-bottom: 12px; }
.info-box { padding: 18px; border-radius: 12px; background-color: #1e3a5f !important; border-left: 5px solid #38bdf8; color: #f8fafc !important; }
.info-box * { color: #f8fafc !important; }
[data-testid="stMetric"] { background-color: #1e293b !important; border: 1px solid #334155 !important; border-radius: 12px; padding: 15px; }
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: #f8fafc !important; }
.stButton > button, .stDownloadButton > button { background-color: #2563eb !important; color: #ffffff !important; border: 1px solid #3b82f6 !important; border-radius: 8px; font-weight: 600; }
div[data-baseweb="select"] > div, input, textarea { background-color: #1e293b !important; color: #f8fafc !important; border-color: #475569 !important; }
div[data-baseweb="select"] span, [data-testid="stFileUploader"] *, [data-testid="stCheckbox"] label, [data-testid="stRadio"] label { color: #f8fafc !important; }
[data-testid="stFileUploader"] { background-color: #1e293b !important; border: 1px solid #475569 !important; border-radius: 12px; padding: 12px; }
[data-testid="stAlert"] * { color: #f8fafc !important; }
[data-testid="stDataFrame"] { border: 1px solid #334155; border-radius: 8px; }
.stCaption, [data-testid="stCaptionContainer"] { color: #cbd5e1 !important; }
.footer { text-align: center; color: #94a3b8 !important; font-size: 13px; margin-top: 40px; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None

if "screening_done" not in st.session_state:
    st.session_state.screening_done = False

if "screening_result" not in st.session_state:
    st.session_state.screening_result = None

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 🔬 Burn-In AI")
    st.caption("Intelligent Component Screening")

    st.divider()

    page = st.radio(
        "Application Menu",
        [
            "🏠 Home",
            "📂 Upload Data",
            "⚙️ Screening",
            "📊 Results",
            "🔍 Investigation",
            "📥 Reports"
        ]
    )

    st.divider()

    st.info(
        "This application detects abnormal component behaviour "
        "and predicts possible burn-in drift."
    )

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">AI-Driven Burn-In Screening</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent anomaly detection and drift prediction for high-reliability components'
    '</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

if page == "🏠 Home":

    st.markdown("""
    <div class="info-box">
        <b>Welcome to the Burn-In Screening Application</b><br><br>
        Upload component measurement data, run AI-based screening,
        and identify components that require further investigation.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Application Workflow</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### 1️⃣")
        st.markdown("**Upload Data**")
        st.caption("Upload burn-in measurement CSV files.")

    with col2:
        st.markdown("### 2️⃣")
        st.markdown("**AI Screening**")
        st.caption("Detect anomalies and predict drift.")

    with col3:
        st.markdown("### 3️⃣")
        st.markdown("**Risk Analysis**")
        st.caption("Calculate component-level risk.")

    with col4:
        st.markdown("### 4️⃣")
        st.markdown("**Generate Report**")
        st.caption("Download screening results.")

    st.markdown(
        '<div class="section-title">Supported Measurements</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Measurement 1", "Iddq")

    with col2:
        st.metric("Measurement 2", "Leakage")

    with col3:
        st.metric("Measurement 3", "Propagation Delay")

# --------------------------------------------------
# UPLOAD DATA PAGE
# --------------------------------------------------

elif page == "📂 Upload Data":

    st.markdown(
        '<div class="section-title">Upload Burn-In Measurement Data</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload a CSV file containing component measurements "
        "at different burn-in time points."
    )

    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=["csv"],
        help="Upload your burn-in measurement dataset."
    )

    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file)

            st.session_state.uploaded_data = df
            st.session_state.screening_done = False
            st.session_state.screening_result = None

            st.success("CSV file uploaded successfully.")

            st.markdown("### Dataset Information")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Rows", df.shape[0])

            with col2:
                st.metric("Columns", df.shape[1])

            with col3:
                st.metric("Missing Values", int(df.isnull().sum().sum()))

            st.markdown("### Data Preview")
            st.dataframe(df.head(20), use_container_width=True)

            st.markdown("### Available Columns")
            st.write(list(df.columns))

        except Exception as error:
            st.error(f"Unable to read CSV file: {error}")

    elif st.session_state.uploaded_data is not None:
        st.success("Previously uploaded data is available.")

        st.dataframe(
            st.session_state.uploaded_data.head(20),
            use_container_width=True
        )

    else:
        st.info("Please upload a CSV file to continue.")

# --------------------------------------------------
# SCREENING PAGE
# --------------------------------------------------

elif page == "⚙️ Screening":

    st.markdown(
        '<div class="section-title">AI Screening Configuration</div>',
        unsafe_allow_html=True
    )

    if st.session_state.uploaded_data is None:
        st.warning("Please upload a CSV file before starting screening.")

    else:

        df = st.session_state.uploaded_data

        st.success(f"Dataset ready: {len(df)} components/records")

        col1, col2 = st.columns(2)

        with col1:
            component_type = st.selectbox(
                "Component Type",
                [
                    "Generic IC",
                    "Microcontroller",
                    "Memory Device",
                    "Analog IC",
                    "Power IC"
                ]
            )

        with col2:
            burn_in_duration = st.selectbox(
                "Burn-In Duration",
                [
                    "168 Hours",
                    "96 Hours",
                    "72 Hours",
                    "48 Hours",
                    "24 Hours"
                ]
            )

        st.markdown("### Screening Thresholds")

        col1, col2, col3 = st.columns(3)

        with col1:
            anomaly_threshold = st.slider(
                "Anomaly Threshold",
                0.0,
                1.0,
                0.50,
                0.05
            )

        with col2:
            drift_threshold = st.slider(
                "Drift Threshold",
                0.0,
                1.0,
                0.50,
                0.05
            )

        with col3:
            reject_threshold = st.slider(
                "Reject Risk Threshold",
                0.0,
                1.0,
                0.75,
                0.05
            )

        st.markdown("### Screening Modules")

        module_a = st.checkbox(
            "Module A — Dynamic Anomaly Detection",
            value=True
        )

        module_b = st.checkbox(
            "Module B — Drift Prediction",
            value=True
        )

        risk_fusion = st.checkbox(
            "Risk Fusion — Final Decision",
            value=True
        )

        if st.button(
            "🚀 Start AI Screening",
            type="primary",
            use_container_width=True
        ):

            with st.spinner("Running AI screening..."):

                result = df.copy()

                # Demo-compatible fallback screening.
                # Later this section will connect to trained models.

                numeric_columns = result.select_dtypes(
                    include=np.number
                ).columns

                if len(numeric_columns) > 0:

                    numeric_data = result[numeric_columns].copy()

                    normalized_data = (
                        numeric_data - numeric_data.min()
                    ) / (
                        numeric_data.max() - numeric_data.min()
                    ).replace(0, 1)

                    result["Anomaly_Risk"] = (
                        normalized_data.mean(axis=1)
                        .fillna(0.0)
                        .clip(0, 1)
                    )

                else:
                    result["Anomaly_Risk"] = 0.0

                result["Drift_Risk"] = result["Anomaly_Risk"]

                result["Risk_Score"] = (
                    0.5 * result["Anomaly_Risk"] +
                    0.5 * result["Drift_Risk"]
                )

                result["Final_Decision"] = np.select(
                    [
                        result["Risk_Score"] >= reject_threshold,
                        result["Risk_Score"] >= anomaly_threshold
                    ],
                    [
                        "REJECT",
                        "INVESTIGATE"
                    ],
                    default="PASS"
                )

                result["Final_Explanation"] = np.select(
                    [
                        result["Final_Decision"] == "REJECT",
                        result["Final_Decision"] == "INVESTIGATE"
                    ],
                    [
                        "High predicted risk. Immediate engineering review required.",
                        "Abnormal behaviour detected. Further investigation recommended."
                    ],
                    default="Component behaviour is within the screening range."
                )

                st.session_state.screening_result = result
                st.session_state.screening_done = True

            st.success("AI screening completed successfully.")

            st.info(
                "Current version uses a screening fallback. "
                "Your trained Module A, Module B and Risk Fusion models "
                "will be connected in the next step."
            )

# --------------------------------------------------
# RESULTS PAGE
# --------------------------------------------------

elif page == "📊 Results":

    st.markdown(
        '<div class="section-title">Screening Results</div>',
        unsafe_allow_html=True
    )

    result = st.session_state.screening_result

    if result is None:
        st.warning("No screening results available. Run screening first.")

    else:

        total = len(result)
        pass_count = int(
            (result["Final_Decision"] == "PASS").sum()
        )
        investigate_count = int(
            (result["Final_Decision"] == "INVESTIGATE").sum()
        )
        reject_count = int(
            (result["Final_Decision"] == "REJECT").sum()
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Components", total)

        with col2:
            st.metric("PASS", pass_count)

        with col3:
            st.metric("INVESTIGATE", investigate_count)

        with col4:
            st.metric("REJECT", reject_count)

        st.markdown("### Decision Distribution")

        decision_counts = (
            result["Final_Decision"]
            .value_counts()
            .reset_index()
        )

        decision_counts.columns = ["Decision", "Count"]

        fig = px.pie(
            decision_counts,
            names="Decision",
            values="Count",
            title="Component Screening Decisions"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Detailed Results")

        st.dataframe(
            result,
            use_container_width=True,
            height=450
        )

# --------------------------------------------------
# INVESTIGATION PAGE
# --------------------------------------------------

elif page == "🔍 Investigation":

    st.markdown(
        '<div class="section-title">Component Investigation</div>',
        unsafe_allow_html=True
    )

    result = st.session_state.screening_result

    if result is None:
        st.warning("Please run screening before investigating components.")

    else:

        selected_index = st.selectbox(
            "Select Component Record",
            result.index.tolist()
        )

        selected_row = result.loc[selected_index]

        st.markdown("### Component Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Risk Score",
                f"{selected_row['Risk_Score']:.3f}"
            )

        with col2:
            st.metric(
                "Anomaly Risk",
                f"{selected_row['Anomaly_Risk']:.3f}"
            )

        with col3:
            st.metric(
                "Drift Risk",
                f"{selected_row['Drift_Risk']:.3f}"
            )

        st.markdown("### Final Decision")

        decision = selected_row["Final_Decision"]

        if decision == "PASS":
            st.success("PASS — No significant abnormality detected.")

        elif decision == "INVESTIGATE":
            st.warning("INVESTIGATE — Further engineering analysis required.")

        else:
            st.error("REJECT — High-risk component detected.")

        st.markdown("### AI Explanation")

        st.info(selected_row["Final_Explanation"])

        st.markdown("### Selected Record")

        st.dataframe(
            selected_row.to_frame("Value"),
            use_container_width=True
        )

# --------------------------------------------------
# REPORTS PAGE
# --------------------------------------------------

elif page == "📥 Reports":

    st.markdown(
        '<div class="section-title">Download Screening Report</div>',
        unsafe_allow_html=True
    )

    result = st.session_state.screening_result

    if result is None:
        st.warning("No report available. Run screening first.")

    else:

        csv_data = result.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Results CSV",
            data=csv_data,
            file_name=f"burn_in_screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.success("Your screening report is ready.")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    '<div class="footer">'
    'AI-Driven Anomaly Detection in Component Burn-In & Screening<br>'
    'SIH26170 | Intelligent Burn-In Screening Prototype'
    '</div>',
    unsafe_allow_html=True
)
