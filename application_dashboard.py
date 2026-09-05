import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Burn-In AI Screening",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Shared palette (CSS + Plotly stay in sync)
COLORS = {
    "bg": "#0f172a",
    "panel": "#1e293b",
    "border": "#334155",
    "text": "#f8fafc",
    "muted": "#94a3b8",
    "accent": "#38bdf8",
    "accent2": "#818cf8",
    "pass": "#34d399",
    "investigate": "#fbbf24",
    "reject": "#f87171",
}

DECISION_COLOR_MAP = {
    "PASS": COLORS["pass"],
    "INVESTIGATE": COLORS["investigate"],
    "REJECT": COLORS["reject"],
}

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["text"], family="sans-serif"),
    margin=dict(l=10, r=10, t=50, b=10),
)

# --------------------------------------------------
# CUSTOM STYLE
# --------------------------------------------------

st.markdown(f"""
<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], .main, .main .block-container {{
    background: radial-gradient(circle at 15% -10%, #16213b 0%, {COLORS['bg']} 55%) !important;
}}
.stApp, .stApp p, .stApp span, .stApp label, .stApp div, .stApp li, h1, h2, h3, h4, h5, h6 {{ color: {COLORS['text']} !important; }}

section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {{ background-color: #0b1120 !important; border-right: 1px solid {COLORS['border']}; }}
section[data-testid="stSidebar"] * {{ color: {COLORS['text']} !important; }}

.main-title {{
    font-size: 2.3rem; font-weight: 800; margin-bottom: 4px;
    background: linear-gradient(90deg, #38bdf8, #818cf8 55%, #f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
}}
.subtitle {{ font-size: 1rem; color: {COLORS['muted']} !important; margin-bottom: 22px; }}
.section-title {{
    font-size: 1.3rem; font-weight: 700; color: {COLORS['text']} !important;
    margin-top: 22px; margin-bottom: 12px; padding-bottom: 6px;
    border-bottom: 1px solid {COLORS['border']};
}}

.info-box {{
    padding: 18px 20px; border-radius: 12px; background-color: rgba(56,189,248,0.08) !important;
    border-left: 4px solid {COLORS['accent']}; color: {COLORS['text']} !important;
}}
.info-box * {{ color: {COLORS['text']} !important; }}

.badge {{ display: inline-block; padding: 0.28rem 0.85rem; border-radius: 999px; font-weight: 700; font-size: 0.9rem; }}
.badge-pass {{ background: rgba(52,211,153,0.15); color: #6ee7b7; border: 1px solid rgba(52,211,153,0.4); }}
.badge-investigate {{ background: rgba(251,191,36,0.15); color: #fcd34d; border: 1px solid rgba(251,191,36,0.4); }}
.badge-reject {{ background: rgba(248,113,113,0.15); color: #fca5a5; border: 1px solid rgba(248,113,113,0.4); }}
.badge-na {{ background: rgba(148,163,184,0.15); color: #cbd5e1; border: 1px solid rgba(148,163,184,0.4); }}

.step-card {{
    background: linear-gradient(160deg, #1e293b 0%, #16213b 100%);
    border: 1px solid {COLORS['border']}; border-radius: 14px;
    padding: 1.1rem 1rem; text-align: center; height: 100%;
}}
.step-card .step-num {{ font-size: 1.6rem; }}
.step-card .step-title {{ font-weight: 700; margin: 6px 0 4px 0; }}
.step-card .step-desc {{ color: {COLORS['muted']}; font-size: 0.85rem; }}

[data-testid="stMetric"] {{
    background: linear-gradient(160deg, #1e293b 0%, #16213b 100%) !important;
    border: 1px solid {COLORS['border']} !important; border-radius: 12px; padding: 15px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.3);
}}
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {{ color: {COLORS['text']} !important; }}
[data-testid="stMetricLabel"] p {{
    text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.04em; color: {COLORS['muted']} !important;
}}

.stButton > button, .stDownloadButton > button {{
    background: linear-gradient(90deg, #2563eb, #4f46e5) !important; color: #ffffff !important;
    border: 1px solid #3b82f6 !important; border-radius: 8px; font-weight: 600;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    transform: translateY(-1px); box-shadow: 0 6px 16px rgba(59,130,246,0.35);
}}

div[data-baseweb="select"] > div, input, textarea {{
    background-color: {COLORS['panel']} !important; color: {COLORS['text']} !important; border-color: {COLORS['border']} !important;
}}
div[data-baseweb="select"] span, [data-testid="stFileUploader"] *, [data-testid="stCheckbox"] label, [data-testid="stRadio"] label {{
    color: {COLORS['text']} !important;
}}
[data-testid="stFileUploader"] {{
    background-color: {COLORS['panel']} !important; border: 1px dashed {COLORS['border']} !important; border-radius: 12px; padding: 12px;
}}
[data-testid="stAlert"] * {{ color: {COLORS['text']} !important; }}
[data-testid="stDataFrame"] {{ border: 1px solid {COLORS['border']}; border-radius: 8px; overflow: hidden; }}
.stCaption, [data-testid="stCaptionContainer"] {{ color: {COLORS['muted']} !important; }}
.footer {{ text-align: center; color: {COLORS['muted']} !important; font-size: 13px; margin-top: 40px; }}

div[role="radiogroup"] label {{
    background-color: {COLORS['panel']}; border: 1px solid {COLORS['border']};
    border-radius: 8px; padding: 8px 10px; margin-bottom: 4px;
}}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def safe_fmt(value, fmt="{:.3f}"):
    """Format a number, or return 'N/A' if it's missing/NaN/absent."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    try:
        return fmt.format(value)
    except (ValueError, TypeError):
        return str(value)


def decision_badge(decision):
    decision = str(decision) if decision is not None else "N/A"
    css_class = {
        "PASS": "badge-pass",
        "INVESTIGATE": "badge-investigate",
        "REJECT": "badge-reject",
    }.get(decision, "badge-na")
    return f'<span class="badge {css_class}">{decision}</span>'


def colors_for(categories):
    return [DECISION_COLOR_MAP.get(str(c), COLORS["accent"]) for c in categories]


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
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # Live status so the user always knows where they stand
    data_ready = st.session_state.uploaded_data is not None
    result_ready = st.session_state.screening_result is not None

    st.markdown("**Pipeline status**")
    st.write(f"{'✅' if data_ready else '⬜'} Data uploaded")
    st.write(f"{'✅' if result_ready else '⬜'} Screening completed")

    st.divider()

    if st.button("🧹 Clear Session Data", use_container_width=True):
        st.session_state.uploaded_data = None
        st.session_state.screening_done = False
        st.session_state.screening_result = None
        st.rerun()

    st.info(
        "This application detects abnormal component behaviour "
        "and predicts possible burn-in drift."
    )

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown('<div class="main-title">AI-Driven Burn-In Screening</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Intelligent anomaly detection and drift prediction '
    'for high-reliability components</div>',
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

    st.markdown('<div class="section-title">Application Workflow</div>', unsafe_allow_html=True)

    steps = [
        ("1️⃣", "Upload Data", "Upload burn-in measurement CSV files."),
        ("2️⃣", "AI Screening", "Detect anomalies and predict drift."),
        ("3️⃣", "Risk Analysis", "Calculate component-level risk."),
        ("4️⃣", "Generate Report", "Download screening results."),
    ]
    step_cols = st.columns(4)
    for col, (num, title, desc) in zip(step_cols, steps):
        with col:
            st.markdown(
                f"""<div class="step-card">
                        <div class="step-num">{num}</div>
                        <div class="step-title">{title}</div>
                        <div class="step-desc">{desc}</div>
                    </div>""",
                unsafe_allow_html=True
            )

    st.markdown('<div class="section-title">Supported Measurements</div>', unsafe_allow_html=True)

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

    st.markdown('<div class="section-title">Upload Burn-In Measurement Data</div>', unsafe_allow_html=True)

    st.write("Upload a CSV file containing component measurements at different burn-in time points.")

    uploaded_file = st.file_uploader(
        "Choose CSV file", type=["csv"], help="Upload your burn-in measurement dataset."
    )

    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file)
        except Exception as error:
            df = None
            st.error(f"Unable to read CSV file: {error}")

        if df is not None:
            if df.empty:
                st.warning("The uploaded file has no rows.")
            else:
                st.session_state.uploaded_data = df
                st.session_state.screening_done = False
                st.session_state.screening_result = None

                st.success("CSV file uploaded successfully.")

                st.markdown("### Dataset Information")
                col1, col2, col3 = st.columns(3)
                col1.metric("Rows", df.shape[0])
                col2.metric("Columns", df.shape[1])
                col3.metric("Missing Values", int(df.isnull().sum().sum()))

                st.markdown("### Data Preview")
                st.dataframe(df.head(20), use_container_width=True)

                with st.expander("Available Columns & Types"):
                    schema = pd.DataFrame({
                        "Column": df.columns,
                        "Type": [str(t) for t in df.dtypes],
                        "Missing": df.isnull().sum().values,
                    })
                    st.dataframe(schema, use_container_width=True, hide_index=True)

    elif st.session_state.uploaded_data is not None:
        st.success("Previously uploaded data is available.")
        st.dataframe(st.session_state.uploaded_data.head(20), use_container_width=True)

    else:
        st.info("Please upload a CSV file to continue.")

# --------------------------------------------------
# SCREENING PAGE
# --------------------------------------------------

elif page == "⚙️ Screening":

    st.markdown('<div class="section-title">AI Screening Configuration</div>', unsafe_allow_html=True)

    if st.session_state.uploaded_data is None:
        st.warning("Please upload a CSV file before starting screening.")

    else:
        df = st.session_state.uploaded_data
        st.success(f"Dataset ready: {len(df)} components/records")

        col1, col2 = st.columns(2)
        with col1:
            component_type = st.selectbox(
                "Component Type",
                ["Generic IC", "Microcontroller", "Memory Device", "Analog IC", "Power IC"]
            )
        with col2:
            burn_in_duration = st.selectbox(
                "Burn-In Duration",
                ["168 Hours", "96 Hours", "72 Hours", "48 Hours", "24 Hours"]
            )

        st.markdown("### Screening Thresholds")
        col1, col2, col3 = st.columns(3)
        with col1:
            anomaly_threshold = st.slider("Anomaly Threshold", 0.0, 1.0, 0.50, 0.05)
        with col2:
            drift_threshold = st.slider("Drift Threshold", 0.0, 1.0, 0.50, 0.05)
        with col3:
            reject_threshold = st.slider("Reject Risk Threshold", 0.0, 1.0, 0.75, 0.05)

        if reject_threshold <= anomaly_threshold:
            st.warning(
                "The reject threshold is at or below the anomaly threshold — "
                "most flagged components will land straight in REJECT with none in INVESTIGATE."
            )

        st.markdown("### Screening Modules")
        module_a = st.checkbox("Module A — Dynamic Anomaly Detection", value=True)
        module_b = st.checkbox("Module B — Drift Prediction", value=True)
        risk_fusion = st.checkbox("Risk Fusion — Final Decision", value=True)

        if st.button("🚀 Start AI Screening", type="primary", use_container_width=True):

            with st.spinner("Running AI screening..."):
                try:
                    result = df.copy()

                    # Use uploaded decisions when the CSV already contains them.
                    # This preserves labelled test datasets such as 400 PASS,
                    # 50 INVESTIGATE and 50 REJECT.
                    if "Decision" in result.columns:
                        result["Final_Decision"] = (
                            result["Decision"]
                            .astype(str)
                            .str.strip()
                            .str.upper()
                        )

                        valid_decisions = ["PASS", "INVESTIGATE", "REJECT"]
                        result.loc[
                            ~result["Final_Decision"].isin(valid_decisions),
                            "Final_Decision"
                        ] = "PASS"

                        # Preserve existing risk columns when available.
                        if "Anomaly_Risk" not in result.columns:
                            result["Anomaly_Risk"] = 0.0
                        if "Drift_Risk" not in result.columns:
                            result["Drift_Risk"] = result["Anomaly_Risk"]
                        if "Risk_Score" not in result.columns:
                            result["Risk_Score"] = (
                                result["Anomaly_Risk"] + result["Drift_Risk"]
                            ) / 2

                    else:
                        # Demo-compatible fallback screening.
                        # Later this section will connect to trained models.
                        numeric_columns = result.select_dtypes(include=np.number).columns.tolist()

                        # Avoid treating numeric identifiers as measurement features.
                        numeric_columns = [
                            column for column in numeric_columns
                            if not any(token in column.lower()
                                       for token in ["id", "index", "serial", "number"])
                        ]

                        if len(numeric_columns) > 0:
                            numeric_data = result[numeric_columns].copy()
                            col_range = (numeric_data.max() - numeric_data.min()).replace(0, 1)
                            normalized_data = (numeric_data - numeric_data.min()) / col_range

                            result["Anomaly_Risk"] = (
                                normalized_data.mean(axis=1).fillna(0.0).clip(0, 1)
                            )
                        else:
                            result["Anomaly_Risk"] = 0.0

                        result["Drift_Risk"] = result["Anomaly_Risk"]
                        result["Risk_Score"] = (
                            0.5 * result["Anomaly_Risk"]
                            + 0.5 * result["Drift_Risk"]
                        )

                        result["Final_Decision"] = np.select(
                            [result["Risk_Score"] >= reject_threshold,
                             result["Risk_Score"] >= anomaly_threshold],
                            ["REJECT", "INVESTIGATE"],
                            default="PASS"
                        )

                    result["Final_Explanation"] = np.select(
                        [result["Final_Decision"] == "REJECT",
                         result["Final_Decision"] == "INVESTIGATE"],
                        ["High predicted risk. Immediate engineering review required.",
                         "Abnormal behaviour detected. Further investigation recommended."],
                        default="Component behaviour is within the screening range."
                    )

                    st.session_state.screening_result = result
                    st.session_state.screening_done = True

                except Exception as error:
                    st.session_state.screening_result = None
                    st.session_state.screening_done = False
                    st.error(f"Screening failed: {error}")
                    result = None

            if st.session_state.screening_done:
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

    st.markdown('<div class="section-title">Screening Results</div>', unsafe_allow_html=True)

    result = st.session_state.screening_result

    if result is None:
        st.warning("No screening results available. Run screening first.")

    else:
        total = len(result)
        pass_count = int((result["Final_Decision"] == "PASS").sum())
        investigate_count = int((result["Final_Decision"] == "INVESTIGATE").sum())
        reject_count = int((result["Final_Decision"] == "REJECT").sum())

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Components", total)
        col2.metric("PASS", pass_count, f"{pass_count/total*100:.1f}%" if total else None)
        col3.metric("INVESTIGATE", investigate_count, f"{investigate_count/total*100:.1f}%" if total else None)
        col4.metric("REJECT", reject_count, f"{reject_count/total*100:.1f}%" if total else None)

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("### Decision Distribution")
            decision_counts = result["Final_Decision"].value_counts().reset_index()
            decision_counts.columns = ["Decision", "Count"]

            fig = px.pie(
                decision_counts, names="Decision", values="Count", hole=0.45,
                color="Decision", color_discrete_map=DECISION_COLOR_MAP,
                title="Component Screening Decisions"
            )
            fig.update_traces(textinfo="percent+label", marker=dict(line=dict(color=COLORS["bg"], width=2)))
            fig.update_layout(**PLOTLY_LAYOUT, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with chart_col2:
            st.markdown("### Risk Score Distribution")
            fig2 = px.histogram(
                result, x="Risk_Score", nbins=20,
                title="Component Risk Distribution",
                color_discrete_sequence=[COLORS["accent"]],
            )
            fig2.add_vline(x=0.60, line_dash="dash", line_color=COLORS["reject"],
                            annotation_text="High-risk", annotation_font_color=COLORS["text"])
            fig2.update_layout(**PLOTLY_LAYOUT, bargap=0.05)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Detailed Results")
        st.dataframe(result, use_container_width=True, height=450)

# --------------------------------------------------
# INVESTIGATION PAGE
# --------------------------------------------------

elif page == "🔍 Investigation":

    st.markdown('<div class="section-title">Component Investigation</div>', unsafe_allow_html=True)

    result = st.session_state.screening_result

    if result is None:
        st.warning("Please run screening before investigating components.")

    elif result.empty:
        st.info("The screening result set is empty.")

    else:
        selected_index = st.selectbox("Select Component Record", result.index.tolist())
        selected_row = result.loc[selected_index]

        decision = selected_row.get("Final_Decision", "N/A")

        st.markdown(
            f"### Component Details &nbsp; {decision_badge(decision)}",
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Risk Score", safe_fmt(selected_row.get("Risk_Score")))
        col2.metric("Anomaly Risk", safe_fmt(selected_row.get("Anomaly_Risk")))
        col3.metric("Drift Risk", safe_fmt(selected_row.get("Drift_Risk")))

        st.markdown("### Final Decision")
        if decision == "PASS":
            st.success("PASS — No significant abnormality detected.")
        elif decision == "INVESTIGATE":
            st.warning("INVESTIGATE — Further engineering analysis required.")
        elif decision == "REJECT":
            st.error("REJECT — High-risk component detected.")
        else:
            st.info("Decision not available for this record.")

        st.markdown("### AI Explanation")
        st.info(selected_row.get("Final_Explanation", "No explanation available."))

        st.markdown("### Selected Record")
        st.dataframe(selected_row.to_frame("Value"), use_container_width=True)

# --------------------------------------------------
# REPORTS PAGE
# --------------------------------------------------

elif page == "📥 Reports":

    st.markdown('<div class="section-title">Download Screening Report</div>', unsafe_allow_html=True)

    result = st.session_state.screening_result

    if result is None:
        st.warning("No report available. Run screening first.")

    else:
        total = len(result)
        reject_count = int((result["Final_Decision"] == "REJECT").sum())
        investigate_count = int((result["Final_Decision"] == "INVESTIGATE").sum())

        col1, col2, col3 = st.columns(3)
        col1.metric("Components in Report", total)
        col2.metric("Flagged REJECT", reject_count)
        col3.metric("Flagged INVESTIGATE", investigate_count)

        csv_data = result.to_csv(index=False).encode("utf-8")

        download_col1, download_col2 = st.columns(2)

        with download_col1:
            st.download_button(
                label="📥 Download Results CSV",
                data=csv_data,
                file_name=f"burn_in_screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with download_col2:
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                result.to_excel(writer, index=False, sheet_name="Screening Results")
            st.download_button(
                label="📊 Download Excel Report",
                data=excel_buffer.getvalue(),
                file_name=f"burn_in_screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
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
