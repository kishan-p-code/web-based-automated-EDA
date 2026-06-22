"""
chat_assistant.py
Local, rule-based chat assistant that answers questions about a dataset
using EDA results. No API key required — runs 100% on your machine.
"""

import re
from typing import Optional

import pandas as pd


SUGGESTED_QUESTIONS = [
    "Give me a summary of this dataset",
    "Which columns have missing values?",
    "What are the top correlated pairs?",
    "Which columns have the most outliers?",
    "Describe the numeric columns",
    "What categorical columns are in the data?",
    "Are there duplicate rows?",
    "Which column has the highest skewness?",
]


def _find_column(name: str, columns: list) -> Optional[str]:
    """Fuzzy-match a column name from user text."""
    name = name.strip().lower()
    if not name:
        return None
    for col in columns:
        if col.lower() == name:
            return col
    for col in columns:
        if name in col.lower() or col.lower() in name:
            return col
    return None


def _extract_column(text: str, columns: list) -> Optional[str]:
    """Try to pull a column name from natural-language text."""
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", text)
    for q in quoted:
        match = _find_column(q, columns)
        if match:
            return match

    for col in sorted(columns, key=len, reverse=True):
        if col.lower() in text.lower():
            return col

    after = re.search(
        r"(?:column|field|variable)\s+['\"]?(\w[\w\s-]*)['\"]?",
        text,
        re.IGNORECASE,
    )
    if after:
        return _find_column(after.group(1), columns)
    return None


def answer_question(question: str, df: pd.DataFrame, results: dict) -> str:
    """Answer a natural-language question using EDA results."""
    if not question or not question.strip():
        return "Ask me anything about your dataset — missing values, correlations, outliers, column stats, and more."

    q = question.strip().lower()
    ov = results["overview"]
    cols = ov["columns"]

    # --- Dataset summary ---
    if any(k in q for k in ("summary", "overview", "describe dataset", "tell me about")):
        lines = [
            f"**Dataset overview** ({ov['n_rows']:,} rows × {ov['n_cols']:,} columns)",
            f"- Missing cells: **{ov['missing_pct']}%** ({ov['missing_cells']:,} of {ov['total_cells']:,})",
            f"- Duplicate rows: **{ov['duplicate_rows']:,}** ({ov['duplicate_pct']}%)",
            f"- Numeric columns ({len(ov['numeric_cols'])}): {', '.join(ov['numeric_cols'][:8]) or 'none'}"
            + ("…" if len(ov['numeric_cols']) > 8 else ""),
            f"- Categorical columns ({len(ov['categorical_cols'])}): {', '.join(ov['categorical_cols'][:8]) or 'none'}"
            + ("…" if len(ov['categorical_cols']) > 8 else ""),
            f"- Memory usage: **{ov['memory_mb']} MB**",
        ]
        if ov["likely_datetime_cols"]:
            lines.append(f"- Possible date/time columns: {', '.join(ov['likely_datetime_cols'])}")
        if ov["id_like_cols"]:
            lines.append(f"- Potential ID columns: {', '.join(ov['id_like_cols'])}")
        return "\n".join(lines)

    # --- Missing values ---
    if any(k in q for k in ("missing", "null", "na ", "empty", "incomplete")):
        miss = results["missing_summary"]
        with_missing = miss[miss["Missing Count"] > 0]
        if with_missing.empty:
            return "No missing values were found in any column."
        lines = ["**Columns with missing values:**"]
        for _, row in with_missing.head(15).iterrows():
            lines.append(f"- `{row['Column']}`: {int(row['Missing Count']):,} ({row['Missing %']}%)")
        if len(with_missing) > 15:
            lines.append(f"… and {len(with_missing) - 15} more columns.")
        worst = with_missing.iloc[0]
        lines.append(
            f"\nWorst column: **`{worst['Column']}`** with {int(worst['Missing Count']):,} missing values ({worst['Missing %']}%)."
        )
        return "\n".join(lines)

    # --- Correlations ---
    if any(k in q for k in ("correlat", "relationship", "associated", "linear")):
        top = results["top_correlations"]
        if top.empty:
            return "Not enough numeric columns to compute correlations (need at least 2)."
        lines = ["**Top correlated feature pairs:**"]
        for _, row in top.head(8).iterrows():
            strength = "strong" if abs(row["Correlation"]) > 0.7 else "moderate" if abs(row["Correlation"]) > 0.4 else "weak"
            lines.append(
                f"- `{row['Feature 1']}` ↔ `{row['Feature 2']}`: **{row['Correlation']:.3f}** ({strength})"
            )
        best = top.iloc[0]
        return "\n".join(lines) + f"\n\nStrongest pair: **`{best['Feature 1']}`** and **`{best['Feature 2']}`** ({best['Correlation']:.3f})."

    # --- Outliers ---
    if any(k in q for k in ("outlier", "anomal", "extreme")):
        outliers = results["outliers"]
        if outliers.empty:
            return "No numeric columns available for outlier detection."
        ranked = outliers.sort_values("IQR Outliers", ascending=False)
        lines = ["**Outlier summary (IQR method):**"]
        for _, row in ranked.head(8).iterrows():
            lines.append(
                f"- `{row['Column']}`: {int(row['IQR Outliers'])} IQR outliers ({row['IQR Outlier %']}%), "
                f"{int(row['Z-score Outliers (|z|>3)'])} Z-score outliers"
            )
        worst = ranked.iloc[0]
        return "\n".join(lines) + f"\n\nMost outliers: **`{worst['Column']}`** ({int(worst['IQR Outliers'])} by IQR)."

    # --- Duplicates ---
    if any(k in q for k in ("duplicate", "duplicated", "repeated row")):
        n = ov["duplicate_rows"]
        if n == 0:
            return "No duplicate rows were found."
        return (
            f"Found **{n:,} duplicate rows** ({ov['duplicate_pct']}% of {ov['n_rows']:,} rows). "
            "Consider dropping duplicates if they are not intentional."
        )

    # --- Numeric columns ---
    if any(k in q for k in ("numeric", "number", "continuous", "quantitative")):
        if not ov["numeric_cols"]:
            return "No numeric columns were detected."
        ns = results["numeric_summary"]
        lines = [f"**{len(ov['numeric_cols'])} numeric columns:** {', '.join(ov['numeric_cols'])}"]
        if not ns.empty:
            lines.append("\n**Quick stats:**")
            for _, row in ns.head(10).iterrows():
                lines.append(
                    f"- `{row['Column']}`: mean={row['Mean']}, std={row['Std']}, "
                    f"range=[{row['Min']}, {row['Max']}], skew={row['Skew']}"
                )
        return "\n".join(lines)

    # --- Categorical columns ---
    if any(k in q for k in ("categorical", "category", "text column", "string column", "nominal")):
        if not ov["categorical_cols"]:
            return "No categorical columns were detected."
        lines = [f"**{len(ov['categorical_cols'])} categorical columns:**"]
        for col in ov["categorical_cols"][:12]:
            info = results["categorical_summary"].get(col, {})
            mode = info.get("mode", "—")
            unique = info.get("unique", "?")
            lines.append(f"- `{col}`: {unique} unique values, mode=`{mode}`")
        return "\n".join(lines)

    # --- Skewness ---
    if "skew" in q:
        ns = results["numeric_summary"]
        if ns.empty:
            return "No numeric columns for skewness analysis."
        ranked = ns.reindex(ns["Skew"].abs().sort_values(ascending=False).index)
        lines = ["**Skewness ranking (|skew|):**"]
        for _, row in ranked.head(8).iterrows():
            direction = "right-skewed" if row["Skew"] > 0.5 else "left-skewed" if row["Skew"] < -0.5 else "roughly symmetric"
            lines.append(f"- `{row['Column']}`: skew={row['Skew']} ({direction})")
        return "\n".join(lines)

    # --- Normality ---
    if any(k in q for k in ("normal", "normality", "gaussian", "distribution shape")):
        ns = results["numeric_summary"]
        if ns.empty:
            return "No numeric columns for normality analysis."
        normal_cols = ns[ns["Normal? (p>0.05)"] == "Yes"]["Column"].tolist()
        not_normal = ns[ns["Normal? (p>0.05)"] == "No"]["Column"].tolist()
        lines = []
        if normal_cols:
            lines.append(f"**Possibly normal** (Shapiro-Wilk p>0.05): {', '.join(f'`{c}`' for c in normal_cols)}")
        if not_normal:
            lines.append(f"**Likely non-normal**: {', '.join(f'`{c}`' for c in not_normal)}")
        return "\n".join(lines) if lines else "Could not assess normality."

    # --- Specific column ---
    col = _extract_column(question, cols)
    if col:
        if col in ov["numeric_cols"]:
            ns = results["numeric_summary"]
            row = ns[ns["Column"] == col]
            if not row.empty:
                r = row.iloc[0]
                return (
                    f"**`{col}`** (numeric)\n"
                    f"- Count: {int(r['Count']):,} | Missing: {int(df[col].isna().sum()):,}\n"
                    f"- Mean: {r['Mean']} | Median: {r['Median']} | Std: {r['Std']}\n"
                    f"- Min: {r['Min']} | Max: {r['Max']} | IQR: {r['IQR']}\n"
                    f"- Skew: {r['Skew']} | Kurtosis: {r['Kurtosis']} | Normal? {r['Normal? (p>0.05)']}"
                )
        if col in ov["categorical_cols"]:
            info = results["categorical_summary"].get(col, {})
            top = info.get("top_values")
            lines = [
                f"**`{col}`** (categorical)",
                f"- Unique values: {info.get('unique', '?')} | Missing: {info.get('missing', '?')}",
                f"- Mode: `{info.get('mode', '—')}` ({info.get('mode_pct', 0):.1f}% of non-null)",
            ]
            if top is not None and not top.empty:
                lines.append("\n**Top values:**")
                for _, r in top.head(5).iterrows():
                    lines.append(f"- `{r['Value']}`: {int(r['Count']):,} ({r['Percent']}%)")
            return "\n".join(lines)

    # --- Row/column counts ---
    if any(k in q for k in ("how many row", "row count", "number of row")):
        return f"The dataset has **{ov['n_rows']:,} rows** and **{ov['n_cols']:,} columns**."
    if any(k in q for k in ("how many col", "column count", "number of col")):
        return (
            f"The dataset has **{ov['n_cols']:,} columns**: "
            f"{len(ov['numeric_cols'])} numeric, {len(ov['categorical_cols'])} categorical."
        )

    # --- Help / fallback ---
    if any(k in q for k in ("help", "what can you", "what do you")):
        return (
            "I can answer questions about:\n"
            "- **Overview** — rows, columns, memory, duplicates\n"
            "- **Missing values** — which columns and how many\n"
            "- **Correlations** — strongest relationships between numeric features\n"
            "- **Outliers** — IQR and Z-score detection per column\n"
            "- **Column details** — ask about a specific column by name\n"
            "- **Distributions** — skewness, normality, numeric/categorical breakdowns\n\n"
            "Try: *Which columns have missing values?* or *Describe sepal_length*"
        )

    return (
        "I'm not sure how to answer that. Try asking about **missing values**, **correlations**, "
        "**outliers**, **duplicates**, or a **specific column name**. "
        "Type **help** to see what I can do."
    )
