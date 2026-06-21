"""
charts.py
All chart-building functions, using Plotly so charts are interactive
in Streamlit AND can be embedded as static images in the HTML report.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Try to import scipy for statistical functions
try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def missing_values_bar(missing_df: pd.DataFrame):
    """Create a bar chart of missing values by column."""
    d = missing_df[missing_df["Missing Count"] > 0]
    if d.empty:
        return None
    
    fig = px.bar(
        d, 
        x="Column", 
        y="Missing %", 
        text="Missing Count",
        title="Missing Values by Column",
        labels={"Missing %": "Missing (%)"},
        color_discrete_sequence=["#6366f1"]
    )
    fig.update_traces(
        textposition="outside",
        marker_color="#6366f1",
        hovertemplate="<b>%{x}</b><br>Missing: %{text:,}<br>Percentage: %{y:.1f}%<extra></extra>"
    )
    fig.update_layout(
        height=400, 
        xaxis_tickangle=-45,
        yaxis_title="Missing (%)",
        xaxis_title="",
        margin=dict(l=10, r=10, t=40, b=80),
        showlegend=False
    )
    return fig


def missing_pattern_heatmap(missing_matrix: pd.DataFrame):
    """Create a heatmap showing missing value patterns."""
    if missing_matrix.empty or missing_matrix.shape[1] < 2:
        return None
    
    fig = px.imshow(
        missing_matrix.T,
        color_continuous_scale=["#e5e7eb", "#6366f1"],
        title="Missing Value Pattern Heatmap",
        labels={"x": "Row Index", "y": "Column", "color": "Missing"},
        aspect="auto"
    )
    fig.update_layout(
        height=max(300, 30 * missing_matrix.shape[1]),
        margin=dict(l=10, r=10, t=40, b=60)
    )
    return fig


def correlation_heatmap(corr_df: pd.DataFrame):
    """Create a correlation heatmap with improved styling."""
    if corr_df.empty or len(corr_df) < 2:
        return None
    
    fig = px.imshow(
        corr_df,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, 
        zmax=1,
        title="Correlation Heatmap",
        aspect="auto"
    )
    fig.update_layout(
        height=max(400, 40 * len(corr_df)),
        margin=dict(l=10, r=10, t=40, b=80),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>"
    )
    return fig


def histogram(df: pd.DataFrame, col: str, bins: int = 40):
    """Create an interactive histogram with marginal boxplot."""
    s = df[col].dropna()
    if s.empty:
        return None
    
    fig = px.histogram(
        s, 
        x=col, 
        nbins=bins, 
        marginal="box", 
        title=f"Distribution of {col}",
        color_discrete_sequence=["#6366f1"]
    )
    fig.update_traces(
        marker_color="#6366f1",
        marker_line_color="#4f46e5",
        marker_line_width=0.5,
        opacity=0.8,
        hovertemplate="<b>%{x:.3f}</b><br>Count: %{y:,}<extra></extra>"
    )
    fig.update_layout(
        height=400,
        xaxis_title=col,
        yaxis_title="Frequency",
        margin=dict(l=10, r=10, t=40, b=40),
        showlegend=False
    )
    
    # Add mean and median lines
    mean_val = s.mean()
    median_val = s.median()
    fig.add_vline(x=mean_val, line_dash="dash", line_color="#ef4444", 
                  annotation_text=f"Mean: {mean_val:.3f}", annotation_position="top")
    fig.add_vline(x=median_val, line_dash="dot", line_color="#22c55e",
                  annotation_text=f"Median: {median_val:.3f}", annotation_position="bottom")
    
    return fig


def boxplot(df: pd.DataFrame, col: str):
    """Create an enhanced boxplot for outlier detection."""
    s = df[col].dropna()
    if s.empty:
        return None
    
    fig = px.box(
        s, 
        y=col, 
        title=f"Boxplot of {col}",
        points="outliers",
        color_discrete_sequence=["#6366f1"]
    )
    fig.update_traces(
        marker_color="#ef4444",
        marker_size=6,
        line_color="#4f46e5",
        hovertemplate="<b>%{y:.3f}</b><extra></extra>"
    )
    fig.update_layout(
        height=400,
        yaxis_title=col,
        xaxis_title="",
        margin=dict(l=10, r=10, t=40, b=40),
        showlegend=False
    )
    
    # Add statistical annotations
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    
    fig.add_hline(y=lower_fence, line_dash="dash", line_color="#f59e0b", 
                  annotation_text=f"Lower fence: {lower_fence:.3f}")
    fig.add_hline(y=upper_fence, line_dash="dash", line_color="#f59e0b",
                  annotation_text=f"Upper fence: {upper_fence:.3f}")
    
    return fig


def qq_plot(df: pd.DataFrame, col: str):
    """Create a Q-Q plot for normality check."""
    if not SCIPY_AVAILABLE:
        return None
    
    s = df[col].dropna()
    if s.empty or len(s) < 5:
        return None
    
    # Calculate theoretical quantiles
    n = len(s)
    theoretical = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))
    sample = np.sort(s)
    
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=theoretical, 
        y=sample,
        mode="markers",
        name="Data",
        marker=dict(color="#6366f1", size=4, opacity=0.7),
        hovertemplate="<b>Theoretical: %{x:.3f}</b><br>Sample: %{y:.3f}<extra></extra>"
    ))
    
    # Add diagonal reference line
    min_val = min(theoretical.min(), sample.min())
    max_val = max(theoretical.max(), sample.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="Normal Reference",
        line=dict(color="#ef4444", dash="dash", width=2)
    ))
    
    fig.update_layout(
        title=f"Q-Q Plot for {col}",
        xaxis_title="Theoretical Quantiles",
        yaxis_title=f"Sample Quantiles ({col})",
        height=400,
        margin=dict(l=10, r=10, t=40, b=40),
        legend=dict(x=0.02, y=0.98)
    )
    
    return fig


def bar_categorical(df: pd.DataFrame, col: str, top_n: int = 15):
    """Create an improved categorical bar chart."""
    vc = df[col].value_counts(dropna=True).head(top_n)
    if vc.empty:
        return None
    
    fig = px.bar(
        x=vc.values, 
        y=vc.index.astype(str), 
        orientation="h",
        title=f"Top {min(top_n, len(vc))} values — {col}",
        labels={"x": "Count", "y": col},
        color=vc.values,
        color_continuous_scale="Blues",
        text=vc.values
    )
    fig.update_traces(
        textposition="outside",
        marker_color="#10b981",
        hovertemplate="<b>%{y}</b><br>Count: %{x:,}<br>Percentage: %{customdata:.1f}%<extra></extra>",
        customdata=100 * vc.values / len(df)
    )
    fig.update_layout(
        height=max(350, 28 * len(vc)),
        yaxis={"categoryorder": "total ascending"},
        xaxis_title="Count",
        margin=dict(l=10, r=40, t=40, b=20),
        showlegend=False
    )
    return fig


def scatter_matrix(df: pd.DataFrame, numeric_cols: list, max_cols: int = 5):
    """Create a scatter matrix for numeric columns."""
    cols = numeric_cols[:max_cols]
    if len(cols) < 2:
        return None
    
    fig = px.scatter_matrix(
        df[cols],
        dimensions=cols,
        title="Scatter Matrix",
        color_discrete_sequence=["#6366f1"]
    )
    fig.update_traces(
        diagonal_visible=False,
        marker=dict(size=3, opacity=0.6, color="#6366f1"),
        hovertemplate="<b>%{x:.3f}</b> × <b>%{y:.3f}</b><extra></extra>"
    )
    fig.update_layout(height=700)
    return fig


def pairwise_scatter(df: pd.DataFrame, col_x: str, col_y: str):
    """Create a scatter plot for a pair of variables."""
    # Remove trendline to avoid statsmodels dependency
    fig = px.scatter(
        df,
        x=col_x,
        y=col_y,
        title=f"{col_y} vs {col_x}",
        opacity=0.6,
        color_discrete_sequence=["#6366f1"]
    )
    fig.update_traces(
        marker=dict(color="#6366f1", size=5),
        hovertemplate="<b>%{x:.3f}</b> × <b>%{y:.3f}</b><extra></extra>"
    )
    
    # Add correlation annotation
    corr = df[col_x].corr(df[col_y])
    if not np.isnan(corr):
        fig.add_annotation(
            x=0.95,
            y=0.95,
            xref="paper",
            yref="paper",
            text=f"Correlation: {corr:.3f}",
            showarrow=False,
            font=dict(size=12, color="#4f46e5"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#e5e7eb",
            borderwidth=1
        )
    
    fig.update_layout(height=450)
    return fig


def pairwise_scatter_with_trendline(df: pd.DataFrame, col_x: str, col_y: str):
    """Create a scatter plot with trendline for a pair of variables.
    This requires statsmodels to be installed."""
    try:
        import statsmodels.api as sm
        fig = px.scatter(
            df,
            x=col_x,
            y=col_y,
            trendline="ols",
            title=f"{col_y} vs {col_x}",
            opacity=0.6,
            color_discrete_sequence=["#6366f1"]
        )
        fig.update_traces(
            marker=dict(color="#6366f1", size=5),
            hovertemplate="<b>%{x:.3f}</b> × <b>%{y:.3f}</b><extra></extra>"
        )
        
        # Add correlation annotation
        corr = df[col_x].corr(df[col_y])
        if not np.isnan(corr):
            fig.add_annotation(
                x=0.95,
                y=0.95,
                xref="paper",
                yref="paper",
                text=f"Correlation: {corr:.3f}",
                showarrow=False,
                font=dict(size=12, color="#4f46e5"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#e5e7eb",
                borderwidth=1
            )
        
        fig.update_layout(height=450)
        return fig
    except ImportError:
        # Fallback to regular scatter without trendline
        return pairwise_scatter(df, col_x, col_y)


def time_series_plot(df: pd.DataFrame, date_col: str, value_col: str):
    """Create a time series plot for datetime data."""
    # Ensure date column is datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    
    if df.empty or value_col not in df.columns:
        return None
    
    fig = px.line(
        df,
        x=date_col,
        y=value_col,
        title=f"{value_col} over time",
        color_discrete_sequence=["#6366f1"]
    )
    fig.update_traces(
        line=dict(width=2),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>%{y:.3f}<extra></extra>"
    )
    fig.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title=value_col,
        margin=dict(l=10, r=10, t=40, b=40)
    )
    return fig


def multi_histogram(df: pd.DataFrame, numeric_cols: list, max_cols: int = 4):
    """Create a grid of histograms for multiple numeric columns."""
    cols = numeric_cols[:max_cols]
    if len(cols) < 1:
        return None
    
    n_cols = min(2, len(cols))
    n_rows = (len(cols) + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=cols,
        shared_yaxes=False
    )
    
    for i, col in enumerate(cols):
        row = i // n_cols + 1
        col_idx = i % n_cols + 1
        s = df[col].dropna()
        if not s.empty:
            fig.add_trace(
                go.Histogram(
                    x=s,
                    nbinsx=40,
                    name=col,
                    marker_color="#6366f1",
                    opacity=0.8
                ),
                row=row,
                col=col_idx
            )
    
    fig.update_layout(
        height=300 * n_rows,
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=40)
    )
    fig.update_xaxes(title_text="")
    return fig