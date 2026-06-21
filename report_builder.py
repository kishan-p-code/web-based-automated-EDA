"""
report_builder.py
Builds a single, self-contained, downloadable HTML report from the EDA
results dict + chart figures. No external dependencies at view-time —
Plotly JS is embedded inline so the file works offline / when emailed.
"""

import datetime
import pandas as pd
import plotly.io as pio


def _df_to_html(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Convert DataFrame to HTML table with styling."""
    if df is None or df.empty:
        return "<p class='muted'>No data available.</p>"
    
    # Truncate large DataFrames
    if len(df) > max_rows:
        df_display = df.head(max_rows)
        truncated_msg = f"<p class='muted'>Showing first {max_rows} rows of {len(df)} total.</p>"
    else:
        df_display = df
        truncated_msg = ""
    
    return truncated_msg + df_display.to_html(
        index=False, 
        classes="styled-table", 
        border=0,
        float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)
    )


def _fig_to_html(fig, include_js: bool = True) -> str:
    """Convert Plotly figure to HTML with optional JS inclusion."""
    if fig is None:
        return "<p class='muted'>Not enough data to generate plot.</p>"
    try:
        return pio.to_html(
            fig, 
            include_plotlyjs="inline" if include_js else False, 
            full_html=False,
            config={"displayModeBar": True, "responsive": True}
        )
    except Exception:
        return "<p class='muted'>Error rendering plot.</p>"


CSS = """
<style>
  :root { 
    --accent: #6366f1;
    --accent-hover: #4f46e5;
    --bg: #f8f9fc;
    --card: #ffffff;
    --text: #1f2430;
    --muted: #6b7280;
    --border: #e5e7eb;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
  }
  
  * { box-sizing: border-box; }
  
  body { 
    font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 0;
    line-height: 1.6;
  }
  
  .wrap { 
    max-width: 1200px; 
    margin: 0 auto; 
    padding: 32px 24px 80px; 
  }
  
  header { 
    padding: 28px 0 18px;
    border-bottom: 4px solid var(--accent);
    margin-bottom: 32px;
    background: linear-gradient(to right, var(--bg), #f0f2f8);
    border-radius: 12px;
    padding: 24px 32px;
  }
  
  header h1 { 
    margin: 0;
    font-size: 28px;
    font-weight: 700;
    color: var(--text);
  }
  
  header p { 
    color: var(--muted);
    margin: 6px 0 0;
    font-size: 14px;
  }
  
  h2 { 
    font-size: 22px;
    margin: 48px 0 16px;
    padding-bottom: 12px;
    border-bottom: 2px solid var(--border);
    color: #111827;
  }
  
  h3 { 
    font-size: 17px;
    margin: 24px 0 12px;
    color: #374151;
  }
  
  .grid { 
    display: grid; 
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); 
    gap: 14px; 
    margin: 16px 0 24px;
  }
  
  .stat-card { 
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 18px;
    transition: box-shadow 0.2s;
  }
  
  .stat-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
  }
  
  .stat-card .num { 
    font-size: 24px;
    font-weight: 700;
    color: var(--accent);
    line-height: 1.2;
  }
  
  .stat-card .label { 
    font-size: 12px;
    color: var(--muted);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  
  table.styled-table { 
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin: 12px 0 20px;
    background: var(--card);
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }
  
  table.styled-table th { 
    background: #f3f4f6;
    text-align: left;
    padding: 10px 12px;
    border-bottom: 2px solid var(--border);
    font-weight: 600;
    color: #374151;
  }
  
  table.styled-table td { 
    padding: 8px 12px;
    border-bottom: 1px solid #f0f1f3;
  }
  
  table.styled-table tr:hover td { 
    background: #f9fafb;
  }
  
  table.styled-table .highlight { 
    background: #eef2ff;
  }
  
  .muted { 
    color: var(--muted);
    font-style: italic;
  }
  
  .chip { 
    display: inline-block;
    background: #eef2ff;
    color: #4338ca;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 12px;
    margin: 2px 4px 2px 0;
    font-weight: 500;
  }
  
  .section { 
    margin-bottom: 12px;
  }
  
  .col-block { 
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 20px;
    transition: box-shadow 0.2s;
  }
  
  .col-block:hover {
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
  }
  
  .badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    margin-left: 8px;
  }
  
  .badge-success { background: #d1fae5; color: #065f46; }
  .badge-warning { background: #fef3c7; color: #92400e; }
  .badge-danger { background: #fee2e2; color: #991b1b; }
  
  footer { 
    text-align: center;
    color: var(--muted);
    font-size: 12px;
    margin-top: 60px;
    padding-top: 20px;
    border-top: 1px solid var(--border);
  }
  
  @media print {
    .col-block { page-break-inside: avoid; }
    .stat-card { break-inside: avoid; }
  }
  
  @media (max-width: 640px) {
    .grid { grid-template-columns: repeat(2, 1fr); }
    .wrap { padding: 16px 12px 60px; }
    header { padding: 16px; }
    table.styled-table { font-size: 11px; }
  }
</style>
"""


def build_html_report(
    df: pd.DataFrame,
    results: dict,
    figures: dict,
    filename: str = "dataset",
    include_advanced_stats: bool = True
) -> str:
    """
    Build a self-contained HTML report.
    
    Parameters:
    - df: Original DataFrame
    - results: EDA results dictionary
    - figures: Dictionary of Plotly figures
    - filename: Name of the dataset
    - include_advanced_stats: Whether to include advanced statistics
    """
    ov = results["overview"]
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # Stats cards
    stat_cards = f"""
    <div class="grid">
      <div class="stat-card">
        <div class="num">{ov['n_rows']:,}</div>
        <div class="label">📊 Rows</div>
      </div>
      <div class="stat-card">
        <div class="num">{ov['n_cols']:,}</div>
        <div class="label">📋 Columns</div>
      </div>
      <div class="stat-card">
        <div class="num">{ov['missing_pct']}%</div>
        <div class="label">❓ Missing Cells</div>
      </div>
      <div class="stat-card">
        <div class="num">{ov['duplicate_rows']:,}</div>
        <div class="label">🔄 Duplicate Rows</div>
      </div>
      <div class="stat-card">
        <div class="num">{len(ov['numeric_cols'])}</div>
        <div class="label">🔢 Numeric Cols</div>
      </div>
      <div class="stat-card">
        <div class="num">{len(ov['categorical_cols'])}</div>
        <div class="label">🔤 Categorical Cols</div>
      </div>
      <div class="stat-card">
        <div class="num">{ov['memory_mb']} MB</div>
        <div class="label">💾 Memory Usage</div>
      </div>
    </div>
    """

    # Column chips
    col_chips = "".join(f"<span class='chip'>{c}</span>" for c in ov["columns"])

    # Numeric distributions section
    numeric_html = ""
    for i, col in enumerate(ov["numeric_cols"]):
        hist_fig = figures["histograms"].get(col)
        box_fig = figures["boxplots"].get(col)
        
        # Get basic stats
        s = df[col].dropna()
        stats_line = ""
        if not s.empty:
            stats_line = f"""
            <div style="display:flex; gap:16px; flex-wrap:wrap; margin:4px 0 8px; font-size:13px; color:var(--muted);">
                <span>Mean: {s.mean():.3f}</span>
                <span>Median: {s.median():.3f}</span>
                <span>Std: {s.std():.3f}</span>
                <span>Min: {s.min():.3f}</span>
                <span>Max: {s.max():.3f}</span>
            </div>
            """
        
        numeric_html += f"""
        <div class="col-block">
            <h3>{col}</h3>
            {stats_line}
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px;">
                <div>{_fig_to_html(hist_fig, include_js=(i == 0))}</div>
                <div>{_fig_to_html(box_fig, include_js=False)}</div>
            </div>
        </div>
        """

    # Categorical sections
    cat_html = ""
    for i, (col, info) in enumerate(results["categorical_summary"].items()):
        fig = figures["categorical_bars"].get(col)
        cat_html += f"""
        <div class="col-block">
          <h3>{col} 
            <span class="chip">Unique: {info['unique']}</span>
            <span class="chip">Missing: {info['missing']}</span>
            <span class="chip">Mode: {info['mode']}</span>
          </h3>
          {_fig_to_html(fig, include_js=False)}
          {_df_to_html(info['top_values'])}
        </div>
        """

    # Correlation section
    corr_section = ""
    if not results["correlation"].empty:
        corr_section = f"""
        <h2>🔗 Correlation Analysis</h2>
        <div class="col-block">
            {_fig_to_html(figures.get('corr_heatmap'), include_js=False)}
        </div>
        <div class="col-block">
            <h3>Top correlated pairs</h3>
            {_df_to_html(results['top_correlations'])}
        </div>
        """

    # Advanced stats (numeric summary with skew/kurtosis)
    adv_stats = ""
    if include_advanced_stats and not results["numeric_summary"].empty:
        adv_stats = f"""
        <div class="col-block">
            <h3>📈 Advanced Statistics</h3>
            {_df_to_html(results['numeric_summary'][['Column', 'Skew', 'Kurtosis', 'Normal? (p>0.05)']])}
            <p class="muted" style="font-size:12px;">
                Skewness &gt; 1 or &lt; -1 indicates significant skewness.<br>
                Kurtosis &gt; 3 indicates heavy tails (leptokurtic).<br>
                Normality test: p &gt; 0.05 suggests the data may be normally distributed.
            </p>
        </div>
        """

    # Outliers
    outliers_html = ""
    if not results["outliers"].empty:
        outliers_html = f"""
        <h2>🚨 Outlier Detection</h2>
        <div class="col-block">
            {_df_to_html(results['outliers'])}
            <p class="muted" style="font-size:12px; margin-top:8px;">
                <strong>IQR method:</strong> Values beyond 1.5×IQR from Q1/Q3.<br>
                <strong>Z-score method:</strong> Values with |z| &gt; 3 (more than 3 standard deviations from mean).
            </p>
        </div>
        """

    # Build the full HTML
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>EDA Report — {filename}</title>
      {CSS}
    </head>
    <body>
      <div class="wrap">
        <header>
          <h1>📊 Automated EDA Report</h1>
          <p>
            <strong>Dataset:</strong> {filename} &nbsp;·&nbsp; 
            <strong>Generated:</strong> {now} &nbsp;·&nbsp; 
            <strong>Rows:</strong> {ov['n_rows']:,} &nbsp;·&nbsp; 
            <strong>Columns:</strong> {ov['n_cols']:,}
          </p>
        </header>

        <h2>📋 Dataset Overview</h2>
        {stat_cards}
        <div class="section" style="margin-top:8px;">{col_chips}</div>

        <h2>🧩 Column Types &amp; Completeness</h2>
        <div class="col-block">{_df_to_html(results['dtype_table'], max_rows=200)}</div>

        <h2>❓ Missing Values</h2>
        <div class="col-block">
            {_fig_to_html(figures.get('missing_bar'), include_js=False)}
            {_df_to_html(results['missing_summary'], max_rows=200)}
        </div>

        <h2>🔢 Numeric Summary</h2>
        <div class="col-block">
            {_df_to_html(results['numeric_summary'], max_rows=200)}
        </div>

        {adv_stats}

        {outliers_html}

        <h2>📊 Numeric Distributions</h2>
        {numeric_html if numeric_html else "<p class='muted'>No numeric columns found.</p>"}

        {corr_section}

        <h2>🔤 Categorical Breakdown</h2>
        {cat_html if cat_html else "<p class='muted'>No categorical columns found.</p>"}

        <footer>
          Generated by Auto-EDA App · {now} · 
          <span style="color:var(--accent);">Data processed locally</span>
        </footer>
      </div>
    </body>
    </html>
    """
    return html