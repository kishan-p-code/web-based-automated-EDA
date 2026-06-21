# 📊 Auto-EDA — Automated Exploratory Data Analysis Web App

Upload any dataset (CSV, TSV, Excel, JSON, Parquet) in your browser, and this
app automatically runs a full EDA pipeline — stats, missing values, outliers,
distributions, correlations, charts — and lets you **view everything
interactively** and **download a complete report**.

Runs **100% locally** on your machine. Your data never leaves your computer.

---

## 1. Setup (one time)

You need Python 3.9+ installed. Then, in this folder, run:

```bash
pip install -r requirements.txt
```

(If you're on a system that requires it: `pip install -r requirements.txt --break-system-packages`)

## 2. Run the app

```bash
streamlit run app.py
```

This will print a local URL, e.g.:

```
Local URL: http://localhost:8501
```

Open that URL in your browser. That's your "web app."

## 3. Use it

1. In the sidebar, click **"Upload a file"** and pick a `.csv`, `.tsv`, `.txt`,
   `.xlsx`, `.xls`, `.json`, or `.parquet` file.
2. The app automatically analyzes it and shows tabs:
   - **Overview** — shape, dtypes, memory, duplicates
   - **Missing** — missing value table + chart
   - **Numeric** — descriptive stats, histograms, boxplots, scatter matrix
   - **Outliers** — IQR and Z-score outlier detection per column
   - **Categorical** — top values, bar charts, unique counts
   - **Correlation** — heatmap + top correlated pairs + custom scatter
   - **Chat** — ask questions about your data in plain English (runs locally, no API key)
   - **Download** — generate and download:
     - a single self-contained **HTML report** (charts included, works offline)
     - a **ZIP of all summary tables as CSV**

### Turning the HTML report into a PDF
Open the downloaded `.html` file in any browser → press `Ctrl+P` (or `Cmd+P`
on Mac) → "Save as PDF". This gives you a polished PDF without needing any
extra PDF libraries.

---

## File structure

```
auto-eda-app/
├── app.py              # Streamlit UI — upload, tabs, download buttons
├── eda_engine.py        # Core analysis logic (pandas/numpy/scipy, no UI)
├── charts.py            # All Plotly chart builders
├── chat_assistant.py    # Local Q&A assistant for dataset questions
├── report_builder.py    # Builds the downloadable self-contained HTML report
├── requirements.txt
└── README.md
```

This separation means you can also reuse `eda_engine.py` + `charts.py` in a
notebook, CLI script, or a different UI framework (Flask, FastAPI, etc.) —
none of the analysis code depends on Streamlit.

## Supported file types
CSV, TSV, TXT (auto-delimiter-detect), XLSX, XLS, JSON, Parquet.

## Notes / known limits
- Very large files (millions of rows) will be slower in the browser-rendered
  charts; consider sampling first if you hit performance issues.
- The "likely datetime column" detector is a heuristic on a small sample —
  always double check it against the Overview tab before trusting it blindly.
- Normality test (Shapiro-Wilk) auto-samples to 5,000 rows for large columns
  for speed; this is standard practice since Shapiro-Wilk isn't reliable/fast
  for very large n anyway.
