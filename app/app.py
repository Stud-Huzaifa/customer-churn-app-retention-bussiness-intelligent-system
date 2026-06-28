"""Interactive Streamlit dashboard for batch customer churn prediction."""

from __future__ import annotations

import hashlib
import logging
import sys
from datetime import date
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import SAMPLE_INPUT_PATH
from src.monitoring import monitor_input_data, monitor_predictions
from src.predict import load_model, predict_customers
from src.preprocess import RAW_INPUT_COLUMNS
from src.utils import RISK_LEVELS, high_risk_customers, model_summary, risk_summary


LOGGER = logging.getLogger(__name__)
RISK_COLORS = ["#52B788", "#E0A43A", "#E06C75"]
CHART_TEXT = "#D9E4E0"
CHART_GRID = "#2B3833"
CHART_BACKGROUND = "#0F1412"


@st.cache_resource(show_spinner="Loading churn model...")
def get_model() -> Any:
    """Load one model instance for the lifetime of the Streamlit process."""
    return load_model()


@st.cache_data(show_spinner=False)
def read_csv_bytes(contents: bytes) -> pd.DataFrame:
    """Parse CSV bytes once per unique source."""
    return pd.read_csv(BytesIO(contents))


def preferred_columns(df: pd.DataFrame) -> list[str]:
    """Return a focused set of operational customer and prediction columns."""
    preferred = [
        "Customer_ID",
        "Name",
        "Location",
        "Total_Spending",
        "Website_Visits",
        "Customer_Support_Tickets",
        "Predicted_Churn",
        "Churn_Probability",
        "Risk_Level",
    ]
    return [column for column in preferred if column in df.columns]


def table_column_config() -> dict[str, Any]:
    """Return shared column formatting for prediction tables."""
    return {
        "Predicted_Churn": st.column_config.CheckboxColumn("Churn"),
        "Churn_Probability": st.column_config.ProgressColumn(
            "Churn Probability",
            min_value=0.0,
            max_value=1.0,
            format="%.3f",
        ),
        "Total_Spending": st.column_config.NumberColumn(
            "Total Spending",
            format="%.2f",
        ),
    }


def style_chart(chart: alt.Chart) -> alt.Chart:
    """Apply the dashboard's neutral chart styling."""
    return (
        chart.properties(background=CHART_BACKGROUND)
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=CHART_TEXT,
            titleColor=CHART_TEXT,
            gridColor=CHART_GRID,
            domainColor=CHART_GRID,
            tickColor=CHART_GRID,
        )
        .configure_legend(
            labelColor=CHART_TEXT,
            titleColor=CHART_TEXT,
            orient="top",
        )
    )


def risk_donut_chart(risk_data: pd.DataFrame) -> alt.Chart:
    """Build an interactive donut chart for customer risk levels."""
    selection = alt.selection_point(fields=["Risk Level"], bind="legend")
    chart = (
        alt.Chart(risk_data)
        .mark_arc(innerRadius=72, outerRadius=126, cornerRadius=3)
        .encode(
            theta=alt.Theta("Customers:Q"),
            color=alt.Color(
                "Risk Level:N",
                scale=alt.Scale(domain=list(RISK_LEVELS), range=RISK_COLORS),
                sort=list(RISK_LEVELS),
                title=None,
            ),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.3)),
            tooltip=[
                alt.Tooltip("Risk Level:N"),
                alt.Tooltip("Customers:Q", format=","),
            ],
        )
        .add_params(selection)
        .properties(height=315)
    )
    return style_chart(chart)


def probability_histogram(results: pd.DataFrame) -> alt.Chart:
    """Build an interactive distribution of churn probabilities."""
    selection = alt.selection_point(fields=["Risk_Level"], bind="legend")
    chart = (
        alt.Chart(results)
        .mark_bar(opacity=0.88)
        .encode(
            x=alt.X(
                "Churn_Probability:Q",
                bin=alt.Bin(step=0.05),
                title="Churn probability",
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y("count():Q", title="Customers"),
            color=alt.Color(
                "Risk_Level:N",
                scale=alt.Scale(domain=list(RISK_LEVELS), range=RISK_COLORS),
                sort=list(RISK_LEVELS),
                title=None,
            ),
            opacity=alt.condition(selection, alt.value(0.9), alt.value(0.2)),
            tooltip=[
                alt.Tooltip("count():Q", title="Customers", format=","),
                alt.Tooltip(
                    "Churn_Probability:Q",
                    title="Probability",
                    bin=alt.Bin(step=0.05),
                ),
            ],
        )
        .add_params(selection)
        .properties(height=315)
    )
    return style_chart(chart)


def engagement_chart(results: pd.DataFrame) -> alt.Chart:
    """Plot spending and website engagement for a representative sample."""
    sample_size = min(len(results), 5000)
    chart_data = results.sample(n=sample_size, random_state=42)
    tooltips = [
        alt.Tooltip("Churn_Probability:Q", format=".3f"),
        alt.Tooltip("Risk_Level:N"),
        alt.Tooltip("Total_Spending:Q", format=".2f"),
        alt.Tooltip("Website_Visits:Q", format=","),
    ]
    for column in ("Customer_ID", "Name", "Location"):
        if column in chart_data.columns:
            tooltips.insert(0, alt.Tooltip(f"{column}:N"))

    selection = alt.selection_point(fields=["Risk_Level"], bind="legend")
    chart = (
        alt.Chart(chart_data)
        .mark_circle(size=52, opacity=0.45)
        .encode(
            x=alt.X("Website_Visits:Q", title="Website visits"),
            y=alt.Y("Total_Spending:Q", title="Total spending"),
            color=alt.Color(
                "Risk_Level:N",
                scale=alt.Scale(domain=list(RISK_LEVELS), range=RISK_COLORS),
                sort=list(RISK_LEVELS),
                title=None,
            ),
            opacity=alt.condition(selection, alt.value(0.55), alt.value(0.07)),
            tooltip=tooltips,
        )
        .add_params(selection)
        .properties(height=350)
        .interactive()
    )
    return style_chart(chart)


def top_risk_chart(high_risk: pd.DataFrame, limit: int) -> alt.Chart:
    """Build a ranked chart of the highest-risk customers."""
    chart_data = high_risk.head(limit).copy()
    if "Customer_ID" in chart_data.columns:
        chart_data["Customer Label"] = chart_data["Customer_ID"].astype(str)
    else:
        chart_data["Customer Label"] = [
            f"Row {index + 1}" for index in range(len(chart_data))
        ]

    chart = (
        alt.Chart(chart_data)
        .mark_bar(color=RISK_COLORS[2], cornerRadiusEnd=3)
        .encode(
            x=alt.X(
                "Churn_Probability:Q",
                title="Churn probability",
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y(
                "Customer Label:N",
                sort="-x",
                title=None,
            ),
            tooltip=[
                alt.Tooltip("Customer Label:N", title="Customer"),
                alt.Tooltip("Churn_Probability:Q", format=".3f"),
            ],
        )
        .properties(height=max(220, limit * 24))
    )
    return style_chart(chart)


def filter_results(
    results: pd.DataFrame,
    query: str,
    selected_risks: list[str],
    probability_range: tuple[float, float],
    prediction_mode: str,
) -> pd.DataFrame:
    """Apply explorer filters to prediction results."""
    filtered = results.loc[
        results["Risk_Level"].isin(selected_risks)
        & results["Churn_Probability"].between(*probability_range)
    ].copy()

    if prediction_mode == "Likely Churn":
        filtered = filtered.loc[filtered["Predicted_Churn"] == 1]
    elif prediction_mode == "Likely Active":
        filtered = filtered.loc[filtered["Predicted_Churn"] == 0]

    query = query.strip().lower()
    searchable_columns = [
        column
        for column in ("Customer_ID", "Name", "Location")
        if column in filtered.columns
    ]
    if query and searchable_columns:
        matches = pd.Series(False, index=filtered.index)
        for column in searchable_columns:
            matches |= (
                filtered[column]
                .astype("string")
                .str.lower()
                .str.contains(query, regex=False, na=False)
            )
        filtered = filtered.loc[matches]

    return filtered


def render_customer_inspector(filtered: pd.DataFrame) -> None:
    """Render a compact detail view for one filtered customer."""
    if filtered.empty:
        return

    inspector_data = filtered.head(2000)

    def label_for(index: Any) -> str:
        row = inspector_data.loc[index]
        customer_id = row.get("Customer_ID", f"Row {index}")
        name = row.get("Name")
        return f"{customer_id} - {name}" if pd.notna(name) else str(customer_id)

    selected_index = st.selectbox(
        "Inspect customer",
        options=inspector_data.index,
        format_func=label_for,
    )
    customer = inspector_data.loc[selected_index]

    def numeric_value(column: str) -> float:
        value = pd.to_numeric(customer.get(column), errors="coerce")
        return 0.0 if pd.isna(value) else float(value)

    with st.container(border=True):
        st.markdown(f"**{label_for(selected_index)}**")
        detail_columns = st.columns(5)
        detail_columns[0].metric("Risk", str(customer["Risk_Level"]))
        detail_columns[1].metric(
            "Probability",
            f"{float(customer['Churn_Probability']):.1%}",
        )
        detail_columns[2].metric(
            "Total Spending",
            f"{numeric_value('Total_Spending'):.2f}",
        )
        detail_columns[3].metric(
            "Website Visits",
            f"{int(numeric_value('Website_Visits')):,}",
        )
        detail_columns[4].metric(
            "Support Tickets",
            f"{int(numeric_value('Customer_Support_Tickets')):,}",
        )


def data_quality_table(customers: pd.DataFrame) -> pd.DataFrame:
    """Summarize nulls, completeness, and data types by input column."""
    missing = customers.isna().sum()
    quality = pd.DataFrame(
        {
            "Column": customers.columns,
            "Type": customers.dtypes.astype(str).values,
            "Missing": missing.values,
            "Complete (%)": (
                (1 - missing.values / len(customers)) * 100
            ).round(2),
        }
    )
    return quality.sort_values("Complete (%)").reset_index(drop=True)


def brand_lockup() -> str:
    """Return the code-native product logo and wordmark."""
    return """
    <div class="brand-lockup">
        <div class="brand-mark" aria-hidden="true">
            <span class="brand-bar brand-bar-one"></span>
            <span class="brand-bar brand-bar-two"></span>
            <span class="brand-bar brand-bar-three"></span>
            <span class="brand-trend"></span>
        </div>
        <div>
            <div class="brand-name">RetentionIQ</div>
            <div class="brand-tagline">Customer intelligence</div>
        </div>
    </div>
    """


def render_dataset_strip(
    customers: pd.DataFrame,
    missing_cells: int,
    duplicate_customers: int,
) -> None:
    """Render compact input health statistics."""
    items = [
        ("Input rows", f"{len(customers):,}"),
        ("Columns", f"{len(customers.columns):,}"),
        ("Missing cells", f"{missing_cells:,}"),
        ("Duplicate customers", f"{duplicate_customers:,}"),
    ]
    item_html = "".join(
        (
            '<div class="dataset-stat">'
            f'<span class="dataset-label">{label}</span>'
            f'<strong class="dataset-value">{value}</strong>'
            "</div>"
        )
        for label, value in items
    )
    st.markdown(
        f'<div class="dataset-strip">{item_html}</div>',
        unsafe_allow_html=True,
    )


def render_kpi_strip(
    total_customers: int,
    predicted_churn: int,
    high_risk_count: int,
    high_risk_percentage: float,
    churn_rate: float,
    average_probability: float,
) -> None:
    """Render the primary prediction KPIs in a responsive product strip."""
    kpis = [
        ("Customers scored", f"{total_customers:,}", "teal"),
        ("Likely to churn", f"{predicted_churn:,}", "coral"),
        (
            f"High risk ({high_risk_percentage:.2f}%)",
            f"{high_risk_count:,}",
            "amber",
        ),
        ("Churn rate", f"{churn_rate:.2f}%", "coral"),
        ("Average probability", f"{average_probability:.2f}%", "green"),
    ]
    kpi_html = "".join(
        (
            f'<div class="kpi-item kpi-{accent}">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            "</div>"
        )
        for label, value, accent in kpis
    )
    st.markdown(
        f'<div class="kpi-grid">{kpi_html}</div>',
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Customer Churn Command Center",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #0F1412; }
    .block-container {
        max-width: 1500px;
        padding-top: 1.45rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3, p, label { letter-spacing: 0 !important; }
    h1 { color: #F3F7F5; font-size: 2.15rem !important; }
    h2, h3 { color: #E1EAE6; }
    [data-testid="stSidebar"] { background: #141B18; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #C0CDC8;
    }
    [data-testid="stMetric"] {
        background: #18201D;
        border: 1px solid #2B3833;
        border-top: 3px solid #45B8AF;
        border-radius: 6px;
        padding: 0.8rem 0.9rem;
        min-height: 110px;
    }
    [data-testid="stMetricLabel"] { color: #9FB0AA; }
    [data-testid="stMetricValue"] { color: #F0F5F3; }
    [data-testid="stFileUploaderDropzone"] {
        background: #18201D;
        border: 1px dashed #4C6B64;
        border-radius: 6px;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 6px;
        min-height: 2.6rem;
        font-weight: 600;
    }
    button[kind="primary"] {
        background: #2C958D;
        border-color: #2C958D;
    }
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 1rem;
        border-bottom: 1px solid #2B3833;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        height: 3rem;
        padding-left: 0.25rem;
        padding-right: 0.25rem;
    }
    .app-kicker {
        color: #5CC8BE;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0;
        margin-bottom: 0.25rem;
    }
    .app-subtitle {
        color: #9FB0AA;
        font-size: 1rem;
        margin-top: -0.6rem;
        margin-bottom: 1.4rem;
    }
    .model-status {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: #15271F;
        color: #8AD7A8;
        border: 1px solid #29563F;
        border-radius: 6px;
        padding: 0.65rem 0.75rem;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .status-dot {
        width: 0.55rem;
        height: 0.55rem;
        border-radius: 50%;
        background: #52B788;
        display: inline-block;
    }
    .section-rule {
        border-top: 1px solid #2B3833;
        margin: 1rem 0 1.25rem;
    }
    .brand-lockup {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        min-height: 44px;
        margin: 0.15rem 0 1.35rem;
    }
    .brand-mark {
        position: relative;
        width: 42px;
        height: 42px;
        flex: 0 0 42px;
        border: 1px solid #3A514B;
        border-radius: 7px;
        background: #18231F;
        overflow: hidden;
    }
    .brand-bar {
        position: absolute;
        bottom: 8px;
        width: 5px;
        border-radius: 2px 2px 0 0;
        background: #45B8AF;
    }
    .brand-bar-one { left: 9px; height: 10px; }
    .brand-bar-two { left: 18px; height: 17px; }
    .brand-bar-three { left: 27px; height: 25px; }
    .brand-trend {
        position: absolute;
        top: 11px;
        left: 8px;
        width: 23px;
        height: 2px;
        background: #E06C75;
        transform: rotate(-28deg);
        transform-origin: left center;
    }
    .brand-name {
        color: #F3F7F5;
        font-size: 1.05rem;
        font-weight: 750;
        line-height: 1.2;
    }
    .brand-tagline {
        color: #8FA29B;
        font-size: 0.76rem;
        line-height: 1.35;
    }
    .dataset-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        background: #151D1A;
        border-top: 1px solid #2B3833;
        border-bottom: 1px solid #2B3833;
        margin: 0.85rem 0 0.75rem;
    }
    .dataset-stat {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 0.75rem;
        min-height: 52px;
        padding: 0.9rem 1rem;
        border-right: 1px solid #2B3833;
    }
    .dataset-stat:last-child { border-right: 0; }
    .dataset-label {
        color: #8FA29B;
        font-size: 0.78rem;
    }
    .dataset-value {
        color: #E7EEEB;
        font-size: 1rem;
        font-weight: 700;
    }
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 0.8rem;
        margin: 0.4rem 0 1.25rem;
    }
    .kpi-item {
        position: relative;
        min-height: 108px;
        padding: 1rem 1rem 0.9rem;
        background: #18201D;
        border: 1px solid #2B3833;
        border-radius: 7px;
        overflow: hidden;
    }
    .kpi-item::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: #45B8AF;
    }
    .kpi-coral::before { background: #E06C75; }
    .kpi-amber::before { background: #E0A43A; }
    .kpi-green::before { background: #52B788; }
    .kpi-label {
        color: #91A39D;
        font-size: 0.77rem;
        line-height: 1.3;
        margin-bottom: 0.55rem;
    }
    .kpi-value {
        color: #F3F7F5;
        font-size: 1.65rem;
        font-weight: 720;
        line-height: 1.15;
    }
    .run-context {
        color: #8FA29B;
        font-size: 0.78rem;
        line-height: 1.6;
        padding: 0.2rem 0 0.35rem;
    }
    @media (max-width: 1000px) {
        .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .dataset-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .dataset-stat:nth-child(2) { border-right: 0; }
    }
    @media (max-width: 640px) {
        .kpi-grid { grid-template-columns: 1fr; }
        .dataset-strip { grid-template-columns: 1fr; }
        .dataset-stat { border-right: 0; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-kicker">RETENTIONIQ / RETENTION OPERATIONS</div>', unsafe_allow_html=True)
st.title("Churn Command Center")
st.markdown(
    '<div class="app-subtitle">Batch scoring, risk segmentation, and customer prioritization</div>',
    unsafe_allow_html=True,
)

try:
    model = get_model()
except (FileNotFoundError, TypeError, ValueError) as exc:
    st.error(f"The prediction model could not be loaded: {exc}")
    st.stop()

template_csv = pd.DataFrame(columns=RAW_INPUT_COLUMNS).to_csv(index=False)

with st.sidebar:
    st.markdown(brand_lockup(), unsafe_allow_html=True)
    st.markdown('<div class="app-kicker">PREDICTION RUN</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="model-status"><span class="status-dot"></span>Model ready</div>',
        unsafe_allow_html=True,
    )
    source_mode = st.segmented_control(
        "Data source",
        options=["Upload", "Sample"],
        default="Upload",
        width="stretch",
    )
    prediction_date = st.date_input(
        "Prediction date",
        value=date.today(),
        max_value=date.today(),
    )
    uploaded_file = None
    if source_mode == "Upload":
        uploaded_file = st.file_uploader("Customer CSV", type=["csv"])

    st.download_button(
        "Input template",
        data=template_csv,
        file_name="customer_churn_input_template.csv",
        mime="text/csv",
        icon=":material/download:",
        width="stretch",
    )

if source_mode == "Sample":
    file_contents = SAMPLE_INPUT_PATH.read_bytes()
    source_name = "Project sample"
elif uploaded_file is not None:
    file_contents = uploaded_file.getvalue()
    source_name = uploaded_file.name
else:
    st.info("Choose a customer CSV or switch to the project sample.")
    st.stop()

file_key = hashlib.sha256(file_contents).hexdigest()
result_key = f"{file_key}:{prediction_date.isoformat()}"

try:
    customers = read_csv_bytes(file_contents)
except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
    st.error(f"The selected CSV could not be read: {exc}")
    st.stop()

if customers.empty:
    st.error("The selected CSV contains no customer rows.")
    st.stop()

input_monitoring = monitor_input_data(customers)
if input_monitoring.missing_columns:
    st.error(
        f"Missing required columns: {', '.join(input_monitoring.missing_columns)}"
    )
    st.stop()

if not input_monitoring.values_valid:
    invalid_details = ", ".join(
        f"{column} ({count})"
        for column, count in input_monitoring.invalid_values_by_column.items()
    )
    st.error(f"Invalid input values detected: {invalid_details}")
    st.stop()

missing_cells = input_monitoring.missing_value_count
duplicate_customers = (
    int(customers["Customer_ID"].duplicated().sum())
    if "Customer_ID" in customers.columns
    else int(customers.duplicated().sum())
)

render_dataset_strip(customers, missing_cells, duplicate_customers)

with st.expander("Input Preview", expanded=False):
    st.dataframe(customers.head(25), hide_index=True, width="stretch")
    st.caption(f"{source_name} | showing up to 25 rows")

has_current_results = st.session_state.get("prediction_result_key") == result_key
button_label = "Refresh Prediction" if has_current_results else "Run Prediction"
if st.button(
    button_label,
    type="primary",
    icon=":material/analytics:",
    width="stretch",
):
    try:
        with st.spinner("Scoring customers..."):
            st.session_state["prediction_results"] = predict_customers(
                customers,
                reference_date=prediction_date,
                model=model,
            )
            st.session_state["prediction_result_key"] = result_key
            has_current_results = True
    except (TypeError, ValueError) as exc:
        st.error(f"Prediction could not be completed: {exc}")
    except Exception:
        LOGGER.exception("Unexpected prediction failure")
        st.error("Prediction failed unexpectedly. Check the application logs.")

if not has_current_results:
    st.stop()

results = st.session_state.get("prediction_results")
if not isinstance(results, pd.DataFrame):
    st.stop()

st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
summary = model_summary(results)
prediction_monitoring = monitor_predictions(results)
total_customers = int(summary["Total Customers"])
predicted_churn = int(summary["Predicted Churn"])
churn_rate = float(summary["Churn Rate (%)"])
average_probability = float(summary["Average Churn Probability (%)"])
high_risk_count = int((results["Risk_Level"] == "High Risk").sum())

render_kpi_strip(
    total_customers=total_customers,
    predicted_churn=predicted_churn,
    high_risk_count=high_risk_count,
    high_risk_percentage=prediction_monitoring.high_risk_percentage,
    churn_rate=churn_rate,
    average_probability=average_probability,
)

risk_data = risk_summary(results)
high_risk = high_risk_customers(results)

overview_tab, explorer_tab, high_risk_tab, monitoring_tab = st.tabs(
    ["Overview", "Customer Explorer", "High Risk", "Monitoring"]
)

with overview_tab:
    left_chart, right_chart = st.columns([1, 2])
    with left_chart:
        st.subheader("Risk Mix")
        st.altair_chart(
            risk_donut_chart(risk_data),
            width="stretch",
            theme=None,
        )
    with right_chart:
        st.subheader("Probability Distribution")
        st.altair_chart(
            probability_histogram(results),
            width="stretch",
            theme=None,
        )

    st.subheader("Engagement and Spending")
    st.altair_chart(
        engagement_chart(results),
        width="stretch",
        theme=None,
    )
    if len(results) > 5000:
        st.caption("Engagement chart displays a representative sample of 5,000 customers.")

with explorer_tab:
    st.subheader("Customer Explorer")
    filter_column_1, filter_column_2 = st.columns([2, 3])
    with filter_column_1:
        query = st.text_input(
            "Search",
            placeholder="Customer ID, name, or location",
            icon=":material/search:",
        )
    with filter_column_2:
        selected_risks = st.pills(
            "Risk levels",
            options=list(RISK_LEVELS),
            default=list(RISK_LEVELS),
            selection_mode="multi",
        )

    filter_column_3, filter_column_4 = st.columns([3, 2])
    with filter_column_3:
        probability_range = st.slider(
            "Probability range",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.01,
        )
    with filter_column_4:
        prediction_mode = st.segmented_control(
            "Prediction",
            options=["All", "Likely Churn", "Likely Active"],
            default="All",
            width="stretch",
        )

    filtered = filter_results(
        results,
        query=query,
        selected_risks=list(selected_risks or []),
        probability_range=probability_range,
        prediction_mode=str(prediction_mode),
    )

    sort_column, display_column = st.columns(2)
    with sort_column:
        sort_order = st.selectbox(
            "Sort by",
            options=[
                "Highest probability",
                "Lowest probability",
                "Highest spending",
            ],
        )
    with display_column:
        column_mode = st.segmented_control(
            "Columns",
            options=["Focused", "All"],
            default="Focused",
            width="stretch",
        )

    if sort_order == "Highest probability":
        filtered = filtered.sort_values("Churn_Probability", ascending=False)
    elif sort_order == "Lowest probability":
        filtered = filtered.sort_values("Churn_Probability")
    elif "Total_Spending" in filtered.columns:
        filtered = filtered.sort_values("Total_Spending", ascending=False)

    st.caption(f"{len(filtered):,} of {len(results):,} customers")
    visible_columns = (
        preferred_columns(filtered) if column_mode == "Focused" else list(filtered.columns)
    )
    st.dataframe(
        filtered.loc[:, visible_columns],
        hide_index=True,
        width="stretch",
        height=470,
        column_config=table_column_config(),
    )

    download_column, inspector_column = st.columns([1, 2])
    with download_column:
        st.download_button(
            "Download filtered results",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="filtered_churn_predictions.csv",
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
        )
    with inspector_column:
        render_customer_inspector(filtered)

with high_risk_tab:
    st.subheader("High-Risk Priority Queue")
    if high_risk.empty:
        st.info("No customers are currently classified as high risk.")
    else:
        control_column, export_column = st.columns([2, 1])
        with control_column:
            minimum_ranking = min(5, len(high_risk))
            maximum_ranking = min(30, len(high_risk))
            top_limit = st.slider(
                "Customers in ranking",
                min_value=minimum_ranking,
                max_value=maximum_ranking,
                value=min(15, len(high_risk)),
            )
        with export_column:
            st.download_button(
                "Download high-risk list",
                data=high_risk.to_csv(index=False).encode("utf-8"),
                file_name="high_risk_customers.csv",
                mime="text/csv",
                icon=":material/download:",
                width="stretch",
            )

        st.altair_chart(
            top_risk_chart(high_risk, top_limit),
            width="stretch",
            theme=None,
        )
        st.dataframe(
            high_risk.loc[:, preferred_columns(high_risk)],
            hide_index=True,
            width="stretch",
            height=470,
            column_config=table_column_config(),
        )

with monitoring_tab:
    st.subheader("Pipeline Monitoring")
    status_columns = st.columns(3)
    with status_columns[0]:
        st.success("Schema validation passed", icon=":material/check_circle:")
    with status_columns[1]:
        st.success("Value validation passed", icon=":material/check_circle:")
    with status_columns[2]:
        if missing_cells:
            st.warning(
                f"{missing_cells:,} missing values will be imputed",
                icon=":material/warning:",
            )
        else:
            st.success("No missing values", icon=":material/check_circle:")

    monitoring_metrics = st.columns(4)
    monitoring_metrics[0].metric(
        "Predictions Generated",
        f"{prediction_monitoring.prediction_count:,}",
    )
    monitoring_metrics[1].metric(
        "Predicted Churn",
        f"{prediction_monitoring.predicted_churn_percentage:.2f}%",
    )
    monitoring_metrics[2].metric(
        "High-Risk Share",
        f"{prediction_monitoring.high_risk_percentage:.2f}%",
    )
    monitoring_metrics[3].metric(
        "Average Probability",
        f"{prediction_monitoring.average_churn_probability:.2f}%",
    )

    st.subheader("Input Data Quality")
    quality = data_quality_table(customers)
    complete_rows = int(customers.notna().all(axis=1).sum())
    quality_metrics = st.columns(4)
    quality_metrics[0].metric("Complete Rows", f"{complete_rows:,}")
    quality_metrics[1].metric("Missing Cells", f"{missing_cells:,}")
    quality_metrics[2].metric("Duplicate Customers", f"{duplicate_customers:,}")
    quality_metrics[3].metric(
        "Locations",
        f"{customers['Location'].nunique(dropna=True):,}",
    )

    missing_quality = quality.loc[quality["Missing"] > 0]
    if not missing_quality.empty:
        missing_chart = (
            alt.Chart(missing_quality)
            .mark_bar(color=RISK_COLORS[1], cornerRadiusEnd=3)
            .encode(
                x=alt.X("Missing:Q", title="Missing values"),
                y=alt.Y("Column:N", sort="-x", title=None),
                tooltip=["Column:N", alt.Tooltip("Missing:Q", format=",")],
            )
            .properties(height=max(180, len(missing_quality) * 30))
        )
        st.altair_chart(
            style_chart(missing_chart),
            width="stretch",
            theme=None,
        )

    st.dataframe(
        quality,
        hide_index=True,
        width="stretch",
        column_config={
            "Complete (%)": st.column_config.ProgressColumn(
                "Complete",
                min_value=0,
                max_value=100,
                format="%.2f%%",
            )
        },
    )

with st.sidebar:
    st.divider()
    st.markdown(
        (
            '<div class="run-context">'
            f"Source: {escape(source_name)}<br>"
            f"Prediction date: {prediction_date.isoformat()}<br>"
            f"Customers scored: {total_customers:,}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    st.download_button(
        "All predictions",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="customer_churn_predictions.csv",
        mime="text/csv",
        icon=":material/download:",
        width="stretch",
    )
