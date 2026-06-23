import io
import zipfile
import datetime
import logging

import pandas as pd
import streamlit as st

from eda_engine import load_file, run_full_eda
import charts
from report_builder import build_html_report
from chat_assistant import answer_question, SUGGESTED_QUESTIONS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Auto-EDA",
    page_icon="icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Custom CSS for better UI
# ---------------------------------------------------------------------------
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #6366f1;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fc;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e5e7eb;
    }
    .stButton button {
        background-color: #6366f1;
        color: white;
        font-weight: 600;
    }
    .stButton button:hover {
        background-color: #4f46e5;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:

    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col2:
        st.image("icon.png", width=1000)

    st.markdown("---")
    st.write(
        "Upload a dataset and get an instant, automated exploratory data "
        "analysis — stats, missing values, outliers, distributions, "
        "correlations, and charts. Download everything as a report."
    )
    st.markdown("---")
    
    uploaded = st.file_uploader(
        "📁 Upload a file",
        type=["csv", "tsv", "txt", "xlsx", "xls", "json", "parquet"],
        help="Supported: CSV, TSV, TXT, Excel (xlsx/xls), JSON, Parquet",
    )
    
    st.markdown("---")
    
    # Configuration options
    with st.expander("⚙️ Configuration", expanded=False):
        top_n_cats = st.slider(
            "Top-N values per categorical column",
            min_value=5,
            max_value=30,
            value=10,
            help="Number of top categories to display for categorical columns"
        )
        
        outlier_method = st.selectbox(
            "Outlier detection method",
            options=["IQR", "Z-score", "Both"],
            index=2,
            help="Method to use for outlier detection"
        )
        
        correlation_method = st.selectbox(
            "Correlation method",
            options=["pearson", "spearman", "kendall"],
            index=0,
            help="Method to compute correlation"
        )
        
        max_sample_size = st.number_input(
            "Max sample size for normality test",
            min_value=100,
            max_value=10000,
            value=5000,
            step=500,
            help="Larger samples are subsampled for Shapiro-Wilk test"
        )
        
        show_advanced_stats = st.checkbox(
            "Show advanced statistics",
            value=True,
            help="Display skewness, kurtosis, and normality test results"
        )
    
    st.markdown("---")
    st.caption("🔒 Runs 100% locally. Your data never leaves your machine.")
    st.caption(f"🕒 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.markdown('<p class="main-header">Automated Exploratory Data Analysis</p>', unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Resolve data source: uploaded file or sample dataset
using_sample = uploaded is None and "sample_df" in st.session_state
has_data = uploaded is not None or using_sample

if not has_data:
    st.info("👈 Upload a CSV, Excel, JSON, TSV, or Parquet file from the sidebar to get started.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **What this app does automatically:**
        - 📋 Dataset overview (rows, columns, memory, duplicates)
        - 🧩 Column types & completeness
        - ❓ Missing value analysis (table + chart)
        - 🔢 Numeric summary statistics (mean, std, skew, kurtosis, normality test)
        - 🚨 Outlier detection (IQR method + Z-score method)
        """)
    with col2:
        st.markdown("""
        - 📈 Auto-generated distribution charts (histogram + boxplot per numeric column)
        - 🔤 Categorical breakdowns (top values + bar charts)
        - 🔗 Correlation heatmap + top correlated pairs
        - 💬 Chat assistant — ask questions about your data in plain English
        - 📥 One-click download: full HTML report + ZIP of all tables as CSV
        """)
    
    # Example data
    with st.expander("📊 Try with sample data", expanded=False):
        if st.button("Load Iris Dataset"):
            st.session_state["sample_df"] = pd.read_csv(
                "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
            )
            st.session_state["sample_name"] = "iris.csv"
            st.session_state["chat_history"] = []
            st.rerun()
        if st.button("Load Titanic Dataset"):
            st.session_state["sample_df"] = pd.read_csv(
                "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
            )
            st.session_state["sample_name"] = "titanic.csv"
            st.session_state["chat_history"] = []
            st.rerun()
    
    st.stop()

# Load dataset
dataset_name = uploaded.name if uploaded is not None else st.session_state.get("sample_name", "sample_dataset.csv")

if using_sample:
    df = st.session_state["sample_df"].copy()
    st.success(f"✅ Loaded sample **{dataset_name}** — {df.shape[0]:,} rows × {df.shape[1]:,} columns")
else:
    try:
        df = load_file(uploaded, uploaded.name)
    except Exception as e:
        st.error(f"❌ Could not read this file: {e}")
        logger.error(f"File loading error: {e}")
        st.stop()

    if df.empty:
        st.warning("⚠️ The uploaded file loaded but contains no rows.")
        st.stop()

    st.success(f"✅ Loaded **{dataset_name}** — {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    st.session_state["chat_history"] = []

# Data preview
with st.expander("🔍 Preview raw data (first 50 rows)", expanded=False):
    st.dataframe(df.head(50), use_container_width=True)
    
    # Quick data quality stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Cells", df.shape[0] * df.shape[1])
    with col2:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    with col3:
        st.metric("Data Types", len(df.dtypes.unique()))

# ---- Run EDA (cached so re-render on widget interaction is instant) ----
@st.cache_data(show_spinner=False)
def cached_eda(df_json, top_n, outlier_method, corr_method, max_sample_size):
    df_local = pd.read_json(io.StringIO(df_json), orient="split")
    # Store additional parameters in results for later use
    results = run_full_eda(
        df_local,
        outlier_method=outlier_method,
        correlation_method=corr_method,
        max_sample_size=max_sample_size
    )
    return results, df_local

with st.spinner("🔄 Running automated EDA... This may take a moment for large datasets."):
    try:
        # Convert datetime columns to string for JSON serialization
        df_json = df.to_json(orient="split", date_format="iso", default_handler=str)
        results, _ = cached_eda(
            df_json,
            top_n_cats,
            outlier_method,
            correlation_method,
            max_sample_size
        )
        
        # Update categorical summary with chosen top_n
        from eda_engine import get_categorical_summary
        results["categorical_summary"] = get_categorical_summary(
            df, results["overview"]["categorical_cols"], top_n=top_n_cats
        )
        
        # Store additional parameters
        results["params"] = {
            "top_n_cats": top_n_cats,
            "outlier_method": outlier_method,
            "correlation_method": correlation_method,
            "max_sample_size": max_sample_size
        }
        
    except Exception as e:
        st.error(f"❌ EDA failed: {e}")
        logger.error(f"EDA error: {e}", exc_info=True)
        st.stop()

ov = results["overview"]

# Check for potential issues
warnings = []
if ov["missing_pct"] > 20:
    warnings.append(f"⚠️ High missing values: {ov['missing_pct']}% of cells are missing")
if ov["duplicate_rows"] > ov["n_rows"] * 0.05:
    warnings.append(f"⚠️ {ov['duplicate_rows']:,} duplicate rows found ({ov['duplicate_pct']:.1f}% of data)")
if len(ov["numeric_cols"]) == 0:
    warnings.append("⚠️ No numeric columns found - correlation analysis will be limited")
if len(ov["categorical_cols"]) == 0:
    warnings.append("⚠️ No categorical columns found - categorical analysis will be limited")

if warnings:
    with st.expander("📢 Data Quality Warnings", expanded=True):
        for warning in warnings:
            st.warning(warning)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_missing, tab_numeric, tab_outliers, tab_cat, tab_corr, tab_chat, tab_download = st.tabs(
    ["📋 Overview", "❓ Missing", "🔢 Numeric", "🚨 Outliers", "🔤 Categorical", "🔗 Correlation", "💬 Chat", "📥 Download"]
)

# --- Overview ---
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📊 Rows", f"{ov['n_rows']:,}")
    c2.metric("📋 Columns", f"{ov['n_cols']:,}")
    c3.metric("❓ Missing cells", f"{ov['missing_pct']}%")
    c4.metric("🔄 Duplicate rows", f"{ov['duplicate_rows']:,}")

    c5, c6, c7 = st.columns(3)
    c5.metric("🔢 Numeric columns", len(ov["numeric_cols"]))
    c6.metric("🔤 Categorical columns", len(ov["categorical_cols"]))
    c7.metric("💾 Memory usage", f"{ov['memory_mb']} MB")

    if ov["likely_datetime_cols"]:
        st.info(f"🕒 Possible date/time columns detected: {', '.join(ov['likely_datetime_cols'])}")
        with st.expander("📈 Time series preview", expanded=False):
            date_col = st.selectbox("Date column", ov["likely_datetime_cols"], key="ts_date_col")
            value_candidates = ov["numeric_cols"] or [c for c in ov["columns"] if c != date_col]
            if value_candidates:
                value_col = st.selectbox("Value column", value_candidates, key="ts_value_col")
                fig = charts.time_series_plot(df, date_col, value_col)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="overview_timeseries")
                else:
                    st.caption("Could not build time series chart for the selected columns.")

    st.subheader("📋 Column Types & Completeness")
    st.dataframe(results["dtype_table"], use_container_width=True)
    
    # Column type distribution
    if len(ov["numeric_cols"]) > 0 and len(ov["categorical_cols"]) > 0:
        import plotly.express as px
        col_types = pd.DataFrame({
            "Type": ["Numeric", "Categorical", "Other"],
            "Count": [
                len(ov["numeric_cols"]),
                len(ov["categorical_cols"]),
                ov["n_cols"] - len(ov["numeric_cols"]) - len(ov["categorical_cols"])
            ]
        })
        fig = px.pie(col_types, values="Count", names="Type", title="Column Type Distribution", color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True, key="overview_col_types_pie")
with tab_missing:
    fig = charts.missing_values_bar(results["missing_summary"])
    if fig:
        st.plotly_chart(fig, use_container_width=True, key="missing_bar")
    st.dataframe(results["missing_summary"], use_container_width=True)
    
    # Visualize missing patterns
    if results["missing_summary"]["Missing Count"].sum() > 0 and len(ov["columns"]) > 1:
        with st.expander("🔍 Missing Value Patterns", expanded=False):
            st.caption("This shows where missing values occur in the dataset")
            missing_matrix = df.isna().astype(int)
            if missing_matrix.shape[0] > 1000:
                missing_matrix = missing_matrix.sample(1000, random_state=42)
            fig = charts.missing_pattern_heatmap(missing_matrix)
            if fig:
                st.plotly_chart(fig, use_container_width=True, key="missing_pattern_heatmap")
with tab_numeric:
    if ov["numeric_cols"]:
        st.subheader("📊 Summary Statistics")
        st.dataframe(results["numeric_summary"], use_container_width=True)

        # Additional stats if enabled
        if show_advanced_stats:
            with st.expander("📈 Advanced Statistics", expanded=False):
                st.caption("Skewness, kurtosis, and normality test results")
                st.dataframe(results["numeric_summary"][["Column", "Skew", "Kurtosis", "Normal? (p>0.05)"]], use_container_width=True)

        st.subheader("📉 Distributions")
        sel_col = st.selectbox("Choose a numeric column to inspect", ov["numeric_cols"])
        
        # Get the distribution data
        s = df[sel_col].dropna()
        if not s.empty:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean", f"{s.mean():.4f}")
            with col2:
                st.metric("Median", f"{s.median():.4f}")
            with col3:
                st.metric("Std Dev", f"{s.std():.4f}")
        
        c1, c2 = st.columns(2)
        with c1:
            f = charts.histogram(df, sel_col)
            if f:
                st.plotly_chart(f, use_container_width=True, key=f"numeric_hist_{sel_col}")
        with c2:
            f = charts.boxplot(df, sel_col)
            if f:
                st.plotly_chart(f, use_container_width=True, key=f"numeric_box_{sel_col}")
        
        # QQ plot for normality check
        with st.expander("📐 Normality Check (Q-Q Plot)", expanded=False):
            fig = charts.qq_plot(df, sel_col)
            if fig:
                st.plotly_chart(fig, use_container_width=True, key=f"numeric_qq_{sel_col}")
                st.caption("If points fall approximately along the diagonal line, the data is approximately normally distributed.")

        if len(ov["numeric_cols"]) >= 2:
            st.subheader("🔢 Scatter Matrix (first 5 numeric columns)")
            f = charts.scatter_matrix(df, ov["numeric_cols"])
            if f:
                st.plotly_chart(f, use_container_width=True, key="numeric_scatter_matrix")
            with st.expander("📊 Pairwise Comparison", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X axis", ov["numeric_cols"], key="x_pair")
                with col2:
                    y_col = st.selectbox("Y axis", ov["numeric_cols"], index=min(1, len(ov["numeric_cols"]) - 1), key="y_pair")
                if x_col != y_col:
                    f2 = charts.pairwise_scatter(df, x_col, y_col)
                    if f2:
                        st.plotly_chart(f2, use_container_width=True, key=f"numeric_pair_{x_col}_{y_col}")
                        corr = df[x_col].corr(df[y_col])
                        st.metric("Pearson Correlation", f"{corr:.4f}")
    else:
        st.info("ℹ️ No numeric columns found in this dataset.")

# --- Outliers ---
with tab_outliers:
    if not results["outliers"].empty:
        st.subheader("🚨 Outlier Summary")
        st.dataframe(results["outliers"], use_container_width=True)
        st.caption(
            "**IQR method**: flags points beyond 1.5×IQR from Q1/Q3. "
            "**Z-score method**: flags points more than 3 standard deviations from the mean."
        )
        
        # Show outlier distribution
        if len(ov["numeric_cols"]) > 0:
            with st.expander("📊 Outlier Distribution", expanded=False):
                outlier_counts = results["outliers"].melt(
                    id_vars=["Column"],
                    value_vars=["IQR Outliers", "Z-score Outliers (|z|>3)"],
                    var_name="Method",
                    value_name="Count"
                )
                import plotly.express as px
                fig = px.bar(outlier_counts, x="Column", y="Count", color="Method", 
                           title="Outlier Counts by Method",
                           barmode="group")
                st.plotly_chart(fig, use_container_width=True, key="outlier_dist_bar")
        if ov["numeric_cols"]:
            outlier_col = st.selectbox("View outliers for column", ov["numeric_cols"], key="outlier_col")
            s = df[outlier_col].dropna()
            if len(s) > 0:
                z_scores = (s - s.mean()) / s.std()
                outliers_z = s[abs(z_scores) > 3]
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                outliers_iqr = s[(s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)]
                
                st.write(f"**{outlier_col}** - {len(s):,} values")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("IQR Outliers", len(outliers_iqr), f"{100*len(outliers_iqr)/len(s):.1f}%")
                    st.write(f"Bounds: [{q1 - 1.5*iqr:.4f}, {q3 + 1.5*iqr:.4f}]")
                with col2:
                    st.metric("Z-score Outliers", len(outliers_z), f"{100*len(outliers_z)/len(s):.1f}%")
                    st.write(f"Z-score threshold: ±3")
    else:
        st.info("ℹ️ No numeric columns to check for outliers.")

# --- Categorical ---
with tab_cat:
    if ov["categorical_cols"]:
        st.subheader("🔤 Categorical Analysis")
        
        # Overview of categorical columns
        cat_overview = []
        for col in ov["categorical_cols"]:
            info = results["categorical_summary"][col]
            cat_overview.append({
                "Column": col,
                "Unique Values": info["unique"],
                "Missing": info["missing"],
                "Mode": info["mode"]
            })
        st.dataframe(pd.DataFrame(cat_overview), use_container_width=True)
        
        st.markdown("---")
        
        sel_cat = st.selectbox("Choose a categorical column to inspect", ov["categorical_cols"])
        info = results["categorical_summary"][sel_cat]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Unique values", info["unique"])
        c2.metric("Missing", info["missing"])
        c3.metric("Most common", str(info["mode"]))
        
        f = charts.bar_categorical(df, sel_cat, top_n=top_n_cats)
        if f:
            st.plotly_chart(f, use_container_width=True, key=f"cat_bar_{sel_cat}")
        
        # Show frequency table with percentages
        with st.expander("📊 Full Frequency Table", expanded=False):
            freq = df[sel_cat].value_counts(dropna=False).reset_index()
            freq.columns = [sel_cat, "Count"]
            freq["Percentage"] = (100 * freq["Count"] / len(df)).round(2)
            st.dataframe(freq, use_container_width=True)
    else:
        st.info("ℹ️ No categorical columns found in this dataset.")

# --- Correlation ---
with tab_corr:
    if not results["correlation"].empty:
        st.subheader("🔗 Correlation Analysis")
        
        # Correlation method indicator
        st.caption(f"Using **{correlation_method}** correlation method")
        
        f = charts.correlation_heatmap(results["correlation"])
        if f:
            st.plotly_chart(f, use_container_width=True, key="corr_heatmap")
        st.dataframe(results["top_correlations"], use_container_width=True)
        
        # Interactive pair exploration
        st.subheader("🔍 Explore a pair")
        c1, c2 = st.columns(2)
        with c1:
            x_col = st.selectbox("X axis", ov["numeric_cols"], key="x_corr")
        with c2:
            y_col = st.selectbox("Y axis", ov["numeric_cols"], index=min(1, len(ov["numeric_cols"]) - 1), key="y_corr")
        if x_col != y_col:
            # Use the pairwise scatter function that doesn't require statsmodels
            f2 = charts.pairwise_scatter(df, x_col, y_col)
            if f2:
                st.plotly_chart(f2, use_container_width=True, key=f"corr_pair_{x_col}_{y_col}")
                corr = df[x_col].corr(df[y_col], method=correlation_method)
                st.metric(f"{correlation_method.capitalize()} Correlation", f"{corr:.4f}")
        else:
            st.warning("Please select different columns for X and Y axes.")
        
        # Correlation threshold filter
        with st.expander("🔎 Correlation Filter", expanded=False):
            threshold = st.slider("Minimum correlation absolute value", 0.0, 1.0, 0.3, 0.05)
            filtered = results["top_correlations"][
                results["top_correlations"]["Correlation"].abs() >= threshold
            ]
            st.dataframe(filtered, use_container_width=True)
            
    else:
        st.info("ℹ️ Need at least 2 numeric columns to compute correlations.")

# --- Chat ---
with tab_chat:
    st.subheader("💬 Ask about your data")
    st.caption(
        "Local assistant — answers are generated from your EDA results. "
        "No API key needed; your data stays on your machine."
    )

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    st.markdown("**Quick questions:**")
    qcols = st.columns(2)
    for i, sq in enumerate(SUGGESTED_QUESTIONS):
        with qcols[i % 2]:
            if st.button(sq, key=f"suggest_{i}", use_container_width=True):
                reply = answer_question(sq, df, results)
                st.session_state["chat_history"].append({"role": "user", "content": sq})
                st.session_state["chat_history"].append({"role": "assistant", "content": reply})
                st.rerun()

    if user_msg := st.chat_input("Ask about missing values, correlations, outliers, a column…"):
        reply = answer_question(user_msg, df, results)
        st.session_state["chat_history"].append({"role": "user", "content": user_msg})
        st.session_state["chat_history"].append({"role": "assistant", "content": reply})
        st.rerun()

    if st.session_state["chat_history"]:
        if st.button("🗑️ Clear chat", type="secondary"):
            st.session_state["chat_history"] = []
            st.rerun()

# --- Download ---
with tab_download:
    st.subheader("📥 Export your results")
    st.write("Generate a full, self-contained HTML report (opens in any browser, includes all charts) "
             "or download every summary table as CSVs in a ZIP.")

    colA, colB = st.columns(2)

    with colA:
        if st.button("🧾 Generate HTML report", use_container_width=True, type="primary"):
            with st.spinner("Building report..."):
                figures = {
                    "missing_bar": charts.missing_values_bar(results["missing_summary"]),
                    "histograms": {c: charts.histogram(df, c) for c in ov["numeric_cols"]},
                    "boxplots": {c: charts.boxplot(df, c) for c in ov["numeric_cols"]},
                    "categorical_bars": {c: charts.bar_categorical(df, c, top_n=top_n_cats) for c in ov["categorical_cols"]},
                    "corr_heatmap": charts.correlation_heatmap(results["correlation"]) if not results["correlation"].empty else None,
                }
                html_str = build_html_report(
                    df, results, figures, 
                    filename=dataset_name,
                    include_advanced_stats=show_advanced_stats
                )
                st.session_state["html_report"] = html_str
            st.success("✅ Report ready — click below to download.")

        if "html_report" in st.session_state:
            st.download_button(
                "⬇️ Download HTML report",
                data=st.session_state["html_report"],
                file_name=f"EDA_report_{dataset_name.rsplit('.',1)[0]}_{datetime.date.today()}.html",
                mime="text/html",
                use_container_width=True,
            )

    with colB:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("dtype_summary.csv", results["dtype_table"].to_csv(index=False))
            zf.writestr("missing_values.csv", results["missing_summary"].to_csv(index=False))
            zf.writestr("numeric_summary.csv", results["numeric_summary"].to_csv(index=False))
            zf.writestr("outliers.csv", results["outliers"].to_csv(index=False))
            if not results["correlation"].empty:
                zf.writestr("correlation_matrix.csv", results["correlation"].to_csv())
                zf.writestr("top_correlations.csv", results["top_correlations"].to_csv(index=False))
            for col, info in results["categorical_summary"].items():
                safe = "".join(ch if ch.isalnum() else "_" for ch in col)
                zf.writestr(f"categorical_{safe}.csv", info["top_values"].to_csv(index=False))
        buf.seek(0)

        st.write("")  # spacing to align button vertically with left column
        st.download_button(
            "⬇️ Download all tables (ZIP of CSVs)",
            data=buf,
            file_name=f"EDA_tables_{dataset_name.rsplit('.',1)[0]}_{datetime.date.today()}.zip",
            mime="application/zip",
            use_container_width=True,
        )

    st.markdown("---")
    
    # Report preview option
    if "html_report" in st.session_state:
        with st.expander("👁️ Preview HTML Report", expanded=False):
            st.components.v1.html(st.session_state["html_report"], height=600, scrolling=True)
    
    st.caption(
        "💡 Tip: the HTML report is fully self-contained (charts included) — "
        "you can open it offline, email it, or print it to PDF from your browser (Ctrl/Cmd+P → Save as PDF)."
    )