"""
eda_engine.py
Core automated EDA logic. Pure pandas/numpy/scipy — no UI code here,
so it can be reused by the Streamlit app, a CLI, or tests.
"""

import io
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------------------------------
# 1. FILE LOADING (handles csv, tsv, xlsx, xls, json, parquet)
# --------------------------------------------------------------------------

def _reset_stream(uploaded_file):
    """Rewind file-like uploads so pandas can read from the start."""
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)


def load_file(uploaded_file, filename: str) -> pd.DataFrame:
    """
    Load a DataFrame from an uploaded file-like object based on its extension.
    `uploaded_file` must be a file-like object (has .read()) or a path.
    """
    name = filename.lower()
    _reset_stream(uploaded_file)

    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file, encoding_errors="replace")
        elif name.endswith(".tsv"):
            return pd.read_csv(uploaded_file, sep="\t", encoding_errors="replace")
        elif name.endswith(".xlsx"):
            return pd.read_excel(uploaded_file, engine="openpyxl")
        elif name.endswith(".xls"):
            try:
                return pd.read_excel(uploaded_file, engine="xlrd")
            except Exception:
                _reset_stream(uploaded_file)
                return pd.read_excel(uploaded_file, engine="openpyxl")
        elif name.endswith(".json"):
            _reset_stream(uploaded_file)
            data = json.load(uploaded_file) if hasattr(uploaded_file, "read") else json.loads(uploaded_file)
            try:
                return pd.json_normalize(data)
            except Exception:
                return pd.DataFrame(data)
        elif name.endswith(".parquet"):
            return pd.read_parquet(uploaded_file)
        elif name.endswith(".txt"):
            return pd.read_csv(uploaded_file, sep=None, engine="python")
        else:
            raise ValueError(f"Unsupported file type: {filename}")
    except Exception as e:
        raise ValueError(f"Error reading {filename}: {str(e)}")


# --------------------------------------------------------------------------
# 2. BASIC OVERVIEW
# --------------------------------------------------------------------------

def get_overview(df: pd.DataFrame) -> dict:
    """Get comprehensive dataset overview."""
    n_rows, n_cols = df.shape
    dup_count = int(df.duplicated().sum())
    total_cells = n_rows * n_cols
    missing_cells = int(df.isna().sum().sum())

    # Identify column types
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool", "str"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()

    # Try to auto-detect datetime-like object columns (heuristic)
    likely_datetime = []
    for col in categorical_cols:
        sample = df[col].dropna().astype(str).head(20)
        if len(sample) == 0:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(sample, errors="coerce", format=None)
            if parsed.notna().mean() > 0.8:
                likely_datetime.append(col)
        except Exception:
            pass

    mem_bytes = int(df.memory_usage(deep=True).sum())
    
    # Detect ID-like columns (high cardinality, mostly unique)
    id_like_cols = []
    for col in df.columns:
        if df[col].dtype in ["object", "str"]:
            unique_ratio = df[col].nunique(dropna=True) / max(1, len(df))
            if unique_ratio > 0.8 and len(df) > 100:
                id_like_cols.append(col)

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "duplicate_rows": dup_count,
        "duplicate_pct": round(100 * dup_count / n_rows, 2) if n_rows else 0,
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_pct": round(100 * missing_cells / total_cells, 2) if total_cells else 0,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "likely_datetime_cols": likely_datetime,
        "id_like_cols": id_like_cols,
        "memory_mb": round(mem_bytes / (1024 * 1024), 3),
        "columns": df.columns.tolist(),
    }


def get_dtype_table(df: pd.DataFrame) -> pd.DataFrame:
    """Get detailed column type information."""
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        n_missing = int(s.isna().sum())
        n_unique = int(s.nunique(dropna=True))
        dtype = str(s.dtype)
        
        # Determine if column might be an ID
        is_id = (n_unique > 0.8 * n and n > 100 and dtype in ["object", "str"])
        
        rows.append({
            "Column": col,
            "Dtype": dtype,
            "Non-Null": n - n_missing,
            "Missing": n_missing,
            "Missing %": round(100 * n_missing / n, 2) if n else 0,
            "Unique": n_unique,
            "Unique %": round(100 * n_unique / n, 2) if n else 0,
            "Potential ID": is_id,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. MISSING VALUES
# --------------------------------------------------------------------------

def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Get summary of missing values."""
    n = len(df)
    miss = df.isna().sum()
    out = pd.DataFrame({
        "Column": miss.index,
        "Missing Count": miss.values,
        "Missing %": (100 * miss.values / n).round(2) if n else 0,
    })
    out = out.sort_values("Missing Count", ascending=False).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------
# 4. NUMERIC SUMMARY (descriptive stats + skew/kurtosis + normality)
# --------------------------------------------------------------------------

def get_numeric_summary(df: pd.DataFrame, numeric_cols: list, max_sample_size: int = 5000) -> pd.DataFrame:
    """Get comprehensive numeric summary with statistical tests."""
    rows = []
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue
            
        desc = s.describe()
        skew = s.skew()
        kurt = s.kurtosis()
        
        # Normality test (Shapiro-Wilk) - sample if too large
        if len(s) > max_sample_size:
            sample = s.sample(max_sample_size, random_state=42)
        else:
            sample = s
        
        try:
            _, p_normal = stats.shapiro(sample)
        except Exception:
            p_normal = np.nan

        # Additional stats
        cv = (desc["std"] / desc["mean"] * 100) if desc["mean"] != 0 else np.nan
        range_val = desc["max"] - desc["min"]
        iqr = desc["75%"] - desc["25%"]
        
        rows.append({
            "Column": col,
            "Count": int(desc["count"]),
            "Mean": round(desc["mean"], 4),
            "Std": round(desc["std"], 4),
            "Min": round(desc["min"], 4),
            "25%": round(desc["25%"], 4),
            "Median": round(desc["50%"], 4),
            "75%": round(desc["75%"], 4),
            "Max": round(desc["max"], 4),
            "Range": round(range_val, 4),
            "IQR": round(iqr, 4),
            "CV (%)": round(cv, 2) if not np.isnan(cv) else np.nan,
            "Skew": round(skew, 3),
            "Kurtosis": round(kurt, 3),
            "Normal? (p>0.05)": "Yes" if (not np.isnan(p_normal) and p_normal > 0.05) else "No",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 5. OUTLIER DETECTION
# --------------------------------------------------------------------------

def detect_outliers(df: pd.DataFrame, numeric_cols: list, method: str = "Both") -> pd.DataFrame:
    """
    Detect outliers using IQR and/or Z-score methods.
    
    Parameters:
    - method: "IQR", "Z-score", or "Both"
    """
    rows = []
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) < 4:
            continue
        
        # IQR method
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_outliers = s[(s < lower) | (s > upper)]
        
        # Z-score method
        z = np.abs(stats.zscore(s))
        z_outliers = s[z > 3]
        
        row = {
            "Column": col,
            "IQR Outliers": len(iqr_outliers),
            "IQR Outlier %": round(100 * len(iqr_outliers) / len(s), 2),
            "Lower Bound": round(lower, 4),
            "Upper Bound": round(upper, 4),
            "Z-score Outliers (|z|>3)": len(z_outliers),
            "Z-score Outlier %": round(100 * len(z_outliers) / len(s), 2) if method in ["Z-score", "Both"] else 0,
        }
        
        if method == "IQR":
            row["Z-score Outliers (|z|>3)"] = 0
            row["Z-score Outlier %"] = 0
        elif method == "Z-score":
            row["IQR Outliers"] = 0
            row["IQR Outlier %"] = 0
        
        rows.append(row)
    
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 6. CATEGORICAL SUMMARY
# --------------------------------------------------------------------------

def get_categorical_summary(df: pd.DataFrame, categorical_cols: list, top_n: int = 10) -> dict:
    """Get summary for categorical columns."""
    result = {}
    n = len(df)
    for col in categorical_cols:
        s = df[col]
        vc = s.value_counts(dropna=True).head(top_n)
        vc_df = pd.DataFrame({
            "Value": vc.index.astype(str),
            "Count": vc.values,
            "Percent": (100 * vc.values / n).round(2) if n else 0,
        })
        
        # Additional metrics
        missing = int(s.isna().sum())
        unique = int(s.nunique(dropna=True))
        mode = s.mode().iloc[0] if not s.mode().empty else None
        mode_freq = s.value_counts(dropna=True).iloc[0] if not s.value_counts(dropna=True).empty else 0
        
        result[col] = {
            "unique": unique,
            "missing": missing,
            "mode": mode,
            "mode_freq": mode_freq,
            "mode_pct": (100 * mode_freq / (n - missing)) if (n - missing) > 0 else 0,
            "top_values": vc_df,
        }
    return result


# --------------------------------------------------------------------------
# 7. CORRELATION
# --------------------------------------------------------------------------

def get_correlation(df: pd.DataFrame, numeric_cols: list, method: str = "pearson") -> pd.DataFrame:
    """Compute correlation matrix."""
    if len(numeric_cols) < 2:
        return pd.DataFrame()
    return df[numeric_cols].corr(method=method).round(3)


def get_top_correlated_pairs(corr_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Get top correlated pairs from correlation matrix."""
    if corr_df.empty:
        return pd.DataFrame()
    
    pairs = []
    cols = corr_df.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], corr_df.iloc[i, j]))
    
    pairs_df = pd.DataFrame(pairs, columns=["Feature 1", "Feature 2", "Correlation"])
    pairs_df["Abs Correlation"] = pairs_df["Correlation"].abs()
    pairs_df = pairs_df.sort_values("Abs Correlation", ascending=False).drop(columns="Abs Correlation")
    return pairs_df.head(top_n).reset_index(drop=True)


# --------------------------------------------------------------------------
# 8. FULL REPORT DICT
# --------------------------------------------------------------------------

def run_full_eda(
    df: pd.DataFrame,
    outlier_method: str = "Both",
    correlation_method: str = "pearson",
    max_sample_size: int = 5000
) -> dict:
    """
    Run complete EDA and return all results as a dictionary.
    
    Parameters:
    - df: DataFrame to analyze
    - outlier_method: "IQR", "Z-score", or "Both"
    - correlation_method: "pearson", "spearman", or "kendall"
    - max_sample_size: Maximum sample size for normality test
    """
    overview = get_overview(df)
    numeric_cols = overview["numeric_cols"]
    categorical_cols = overview["categorical_cols"]

    return {
        "overview": overview,
        "dtype_table": get_dtype_table(df),
        "missing_summary": get_missing_summary(df),
        "numeric_summary": get_numeric_summary(df, numeric_cols, max_sample_size),
        "outliers": detect_outliers(df, numeric_cols, outlier_method),
        "categorical_summary": get_categorical_summary(df, categorical_cols),
        "correlation": get_correlation(df, numeric_cols, correlation_method),
        "top_correlations": get_top_correlated_pairs(get_correlation(df, numeric_cols, correlation_method)),
    }