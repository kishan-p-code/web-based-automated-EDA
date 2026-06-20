

import io
import zipfile
import datetime

import pandas as pd
import streamlit as st

from eda_engine import load_file, run_full_eda
import charts
from report_builder import build_html_report

st.set_page_config(page_title="Auto-EDA", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("📊 Auto-EDA")
st.sidebar.write(
    "Upload a dataset and get an instant, automated exploratory data "
    "analysis — stats, missing values, outliers, distributions, "
    "correlations, and charts. Download everything as a report."
)
st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader(
    "Upload a file",
    type=["csv", "tsv", "txt", "xlsx", "xls", "json", "parquet"],
    help="Supported: CSV, TSV, TXT, Excel (xlsx/xls), JSON, Parquet",
)
top_n_cats = st.sidebar.slider("Top-N values per categorical column", 5, 30, 10)
sample_note = st.sidebar.empty()

st.sidebar.markdown("---")
st.sidebar.caption("Runs 100% locally. Your data never leaves your machine.")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("Automated Exploratory Data Analysis")

if uploaded is None:
    st.info("👈 Upload a CSV, Excel, JSON, TSV, or Parquet file from the sidebar to get started.")
    st.markdown(
        """
        **What this app does automatically:**
        - 📋 Dataset overview (rows, columns, memory, duplicates)
        - 🧩 Column types & completeness
        - ❓ Missing value analysis (table + chart)
        - 🔢 Numeric summary statistics (mean, std, skew, kurtosis, normality test)
        - 🚨 Outlier detection (IQR method + Z-score method)
        - 📈 Auto-generated distribution charts (histogram + boxplot per numeric column)
        - 🔤 Categorical breakdowns (top values + bar charts)
        - 🔗 Correlation heatmap + top correlated pairs
        - 📥 One-click download: full HTML report + ZIP of all tables as CSV
        """
    )
    st.stop()

# ---- Load file ----
try:
    df = load_file(uploaded, uploaded.name)
except Exception as e:
    st.error(f"Could not read this file: {e}")
    st.stop()

if df.empty:
    st.warning("The uploaded file loaded but contains no rows.")
    st.stop()

st.success(f"Loaded **{uploaded.name}** — {df.shape[0]:,} rows × {df.shape[1]:,} columns")

with st.expander("🔍 Preview raw data (first 50 rows)"):
    st.dataframe(df.head(50), use_container_width=True)

# ---- Run EDA (cached so re-render on widget interaction is instant) ----
@st.cache_data(show_spinner=False)
def cached_eda(df_json, top_n):
    df_local = pd.read_json(io.StringIO(df_json), orient="split")
    return run_full_eda(df_local), df_local

with st.spinner("Running automated EDA..."):
    try:
        df_json = df.to_json(orient="split", date_format="iso")
        results, _ = cached_eda(df_json, top_n_cats)
        # re-run categorical with chosen top_n directly (cheap, not cached separately)
        from eda_engine import get_categorical_summary
        results["categorical_summary"] = get_categorical_summary(
            df, results["overview"]["categorical_cols"], top_n=top_n_cats
        )
    except Exception as e:
        st.error(f"EDA failed: {e}")
        st.stop()

ov = results["overview"]

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_missing, tab_numeric, tab_outliers, tab_cat, tab_corr, tab_download = st.tabs(
    ["📋 Overview", "❓ Missing", "🔢 Numeric", "🚨 Outliers", "🔤 Categorical", "🔗 Correlation", "📥 Download"]
)

# --- Overview ---
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{ov['n_rows']:,}")
    c2.metric("Columns", f"{ov['n_cols']:,}")
    c3.metric("Missing cells", f"{ov['missing_pct']}%")
    c4.metric("Duplicate rows", f"{ov['duplicate_rows']:,}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Numeric columns", len(ov["numeric_cols"]))
    c6.metric("Categorical columns", len(ov["categorical_cols"]))
    c7.metric("Memory usage", f"{ov['memory_mb']} MB")

    if ov["likely_datetime_cols"]:
        st.info(f"🕒 Possible date/time columns detected: {', '.join(ov['likely_datetime_cols'])}")

    st.subheader("Column Types & Completeness")
    st.dataframe(results["dtype_table"], use_container_width=True)

# --- Missing ---
with tab_missing:
    fig = charts.missing_values_bar(results["missing_summary"])
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("🎉 No missing values found in this dataset!")
    st.dataframe(results["missing_summary"], use_container_width=True)

# --- Numeric ---
with tab_numeric:
    if ov["numeric_cols"]:
        st.subheader("Summary Statistics")
        st.dataframe(results["numeric_summary"], use_container_width=True)

        st.subheader("Distributions")
        sel_col = st.selectbox("Choose a numeric column to inspect", ov["numeric_cols"])
        c1, c2 = st.columns(2)
        with c1:
            f = charts.histogram(df, sel_col)
            if f:
                st.plotly_chart(f, use_container_width=True)
        with c2:
            f = charts.boxplot(df, sel_col)
            if f:
                st.plotly_chart(f, use_container_width=True)

        if len(ov["numeric_cols"]) >= 2:
            st.subheader("Scatter Matrix (first 5 numeric columns)")
            f = charts.scatter_matrix(df, ov["numeric_cols"])
            if f:
                st.plotly_chart(f, use_container_width=True)
    else:
        st.info("No numeric columns found in this dataset.")

# --- Outliers ---
with tab_outliers:
    if not results["outliers"].empty:
        st.dataframe(results["outliers"], use_container_width=True)
        st.caption(
            "**IQR method**: flags points beyond 1.5×IQR from Q1/Q3. "
            "**Z-score method**: flags points more than 3 standard deviations from the mean."
        )
    else:
        st.info("No numeric columns to check for outliers.")

# --- Categorical ---
with tab_cat:
    if ov["categorical_cols"]:
        sel_cat = st.selectbox("Choose a categorical column to inspect", ov["categorical_cols"])
        info = results["categorical_summary"][sel_cat]
        c1, c2, c3 = st.columns(3)
        c1.metric("Unique values", info["unique"])
        c2.metric("Missing", info["missing"])
        c3.metric("Most common", str(info["mode"]))
        f = charts.bar_categorical(df, sel_cat, top_n=top_n_cats)
        if f:
            st.plotly_chart(f, use_container_width=True)
        st.dataframe(info["top_values"], use_container_width=True)
    else:
        st.info("No categorical columns found in this dataset.")

# --- Correlation ---
with tab_corr:
    if not results["correlation"].empty:
        f = charts.correlation_heatmap(results["correlation"])
        st.plotly_chart(f, use_container_width=True)
        st.subheader("Top correlated pairs")
        st.dataframe(results["top_correlations"], use_container_width=True)

        st.subheader("Explore a pair")
        c1, c2 = st.columns(2)
        with c1:
            x_col = st.selectbox("X axis", ov["numeric_cols"], key="x_pick")
        with c2:
            y_col = st.selectbox("Y axis", ov["numeric_cols"], index=min(1, len(ov["numeric_cols"]) - 1), key="y_pick")
        f2 = charts.pairwise_scatter(df, x_col, y_col)
        if f2:
            st.plotly_chart(f2, use_container_width=True)
    else:
        st.info("Need at least 2 numeric columns to compute correlations.")

# --- Download ---
with tab_download:
    st.subheader("📥 Export your results")
    st.write("Generate a full, self-contained HTML report (opens in any browser, includes all charts) "
             "or download every summary table as CSVs in a ZIP.")

    colA, colB = st.columns(2)

    with colA:
        if st.button("🧾 Generate HTML report", use_container_width=True):
            with st.spinner("Building report..."):
                figures = {
                    "missing_bar": charts.missing_values_bar(results["missing_summary"]),
                    "histograms": {c: charts.histogram(df, c) for c in ov["numeric_cols"]},
                    "boxplots": {c: charts.boxplot(df, c) for c in ov["numeric_cols"]},
                    "categorical_bars": {c: charts.bar_categorical(df, c, top_n=top_n_cats) for c in ov["categorical_cols"]},
                    "corr_heatmap": charts.correlation_heatmap(results["correlation"]) if not results["correlation"].empty else None,
                }
                html_str = build_html_report(df, results, figures, filename=uploaded.name)
                st.session_state["html_report"] = html_str
            st.success("Report ready — click below to download.")

        if "html_report" in st.session_state:
            st.download_button(
                "⬇️ Download HTML report",
                data=st.session_state["html_report"],
                file_name=f"EDA_report_{uploaded.name.rsplit('.',1)[0]}_{datetime.date.today()}.html",
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
            file_name=f"EDA_tables_{uploaded.name.rsplit('.',1)[0]}_{datetime.date.today()}.zip",
            mime="application/zip",
            use_container_width=True,
        )

    st.markdown("---")
    st.caption(
        "💡 Tip: the HTML report is fully self-contained (charts included) — "
        "you can open it offline, email it, or print it to PDF from your browser (Ctrl/Cmd+P → Save as PDF)."
    )
