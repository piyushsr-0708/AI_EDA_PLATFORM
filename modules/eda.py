"""Simple function-based EDA utilities using Plotly.

Functions accept a cleaned pandas DataFrame and return results/figures
suitable for integration into a Streamlit AutoEDA app. No Streamlit
dependencies or display logic are included — only data/figure creators
that return Plotly figures or pandas objects.
"""
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

__all__ = [
    "summary_statistics",
    "correlation_matrix_figure",
    "histogram_figures",
    "boxplot_figures",
    "missing_values_figure",
]


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    """Return list of numeric column names from the DataFrame."""
    return df.select_dtypes(include="number").columns.tolist()


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a summary statistics DataFrame for `df`.

    The returned DataFrame includes dtype, count, missing count/percent,
    unique, and the standard descriptive statistics (mean/std/min/quantiles/max)
    when applicable.

    Args:
        df: Cleaned pandas DataFrame.

    Returns:
        A pandas DataFrame indexed by column name with summary statistics.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    stats = pd.DataFrame(index=df.columns)
    stats["dtype"] = df.dtypes.astype(str)
    stats["count"] = df.count()
    stats["missing_count"] = df.isna().sum()
    stats["missing_pct"] = (stats["missing_count"] / len(df)) * 100
    stats["unique"] = df.nunique(dropna=True)

    # Merge numeric descriptive stats when available
    numeric_cols = _numeric_columns(df)
    if numeric_cols:
        num_desc = df[numeric_cols].describe().T
        # attach numeric stats into stats frame (only for numeric cols)
        for col in num_desc.index:
            for c in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
                stats.loc[col, c] = num_desc.loc[col, c]

    # Merge object-like descriptive stats where available (top/freq)
    try:
        obj_desc = df.select_dtypes(exclude="number").describe().T
        for col in obj_desc.index:
            for c in ["top", "freq"]:
                stats.loc[col, c] = obj_desc.loc[col].get(c)
    except (ValueError, TypeError):
        pass

    return stats


def correlation_matrix_figure(
    df: pd.DataFrame, method: str = "pearson", annotate: bool = True, return_matrix: bool = False
) -> go.Figure:
    """Create a Plotly heatmap figure of the correlation matrix for numeric cols.

    Args:
        df: pandas DataFrame.
        method: Correlation method passed to `DataFrame.corr()`.
        annotate: If True, show correlation values on the heatmap.
        return_matrix: If True, returns a tuple of (Figure, pandas DataFrame).

    Returns:
        A Plotly Figure with the correlation heatmap. If return_matrix is True,
        returns (Figure, DataFrame). If there are fewer than two numeric columns,
        returns an empty Figure (and None if return_matrix is True).
    """
    numeric = _numeric_columns(df)
    if len(numeric) < 2:
        fig = go.Figure()
        fig.update_layout(title_text="Not enough numeric columns for correlation")
        if return_matrix:
            return fig, None
        return fig

    corr = df[numeric].corr(method=method)
    # Use px.imshow for a readable annotated heatmap
    fig = px.imshow(
        corr,
        text_auto=annotate,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig.update_layout(title_text=f"Correlation matrix ({method})")
    
    if return_matrix:
        return fig, corr
    return fig


def histogram_figures(
    df: pd.DataFrame, columns: Optional[List[str]] = None, bins: int = 30
) -> Dict[str, go.Figure]:
    """Create histogram figures (Plotly) for numeric columns.

    Args:
        df: pandas DataFrame.
        columns: Optional list of columns to plot. If None, all numeric columns
            will be used.
        bins: Number of bins for histograms.

    Returns:
        A dict mapping column name -> Plotly Figure. 
        For highly skewed columns, an additional key `{col}_log` is included with a log-scale histogram.
    """
    numeric = _numeric_columns(df)
    if columns is None:
        cols = numeric
    else:
        cols = [c for c in columns if c in numeric]

    figs: Dict[str, go.Figure] = {}
    for col in cols:
        # Original scale histogram
        fig = px.histogram(df, x=col, nbins=bins, title=f"Histogram: {col} (Original Scale)")
        fig.update_layout(xaxis_title=col, yaxis_title="count")
        figs[col] = fig
        
        # Check skewness
        try:
            skew = df[col].skew()
            if abs(skew) > 1.0:
                # Log scale histogram for heavily skewed features
                import numpy as np
                # We need to handle zero or negative values gracefully before applying log, or let plotly handle it
                # Plotly's log_y=True makes the y-axis log-scale, but the user wants to deal with skewed features.
                # Usually it's the feature (x-axis) that is skewed.
                # However, Plotly px.histogram with log_x=True doesn't bin well for <= 0 values.
                # A common approach is to just use log_y to see the tail, OR transform the x data.
                # The user requested: "log-scale histogram".
                
                # To be safe with negative values, we shift the data if necessary, 
                # or just plot the histogram with a log scale on the y axis to see the distribution tails better.
                # Wait, "skewed feature" means the x-axis has a long tail. Setting log_y=True helps visualize the tail count.
                fig_log = px.histogram(df, x=col, nbins=bins, title=f"Histogram: {col} (Log Scale)", log_y=True)
                fig_log.update_layout(xaxis_title=col, yaxis_title="log(count)")
                figs[f"{col}_log"] = fig_log
        except Exception:
            pass
            
    return figs


def boxplot_figures(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, go.Figure]:
    """Create boxplot figures (Plotly) for numeric columns.

    Args:
        df: pandas DataFrame.
        columns: Optional list of columns to plot. If None, all numeric columns
            will be used.

    Returns:
        A dict mapping column name -> Plotly Figure.
    """
    numeric = _numeric_columns(df)
    if columns is None:
        cols = numeric
    else:
        cols = [c for c in columns if c in numeric]

    figs: Dict[str, go.Figure] = {}
    for col in cols:
        fig = px.box(df, y=col, points="outliers", title=f"Boxplot: {col}")
        fig.update_layout(yaxis_title=col)
        figs[col] = fig
    return figs


def missing_values_figure(df: pd.DataFrame, max_samples: int = 200) -> go.Figure:
    """Create a Plotly figure visualizing missing values.

    The figure contains two vertically stacked subplots:
    - Bar chart of missing counts per column (sorted)
    - Heatmap of missing indicator for a sample of rows (to show patterns)

    Args:
        df: pandas DataFrame.
        max_samples: Maximum number of rows to include in the missing-pattern
            heatmap to avoid overly large figures. If the DataFrame has fewer
            rows, all rows are used.

    Returns:
        A Plotly Figure with the missing-values visualizations.
    """
    miss_counts = df.isna().sum()
    miss_counts = miss_counts[miss_counts > 0].sort_values(ascending=False)

    if miss_counts.empty:
        return None

    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.12,
                        subplot_titles=("Missing values per column", "Missing value pattern (sample)"))

    # Bar chart of missing counts
    bar = go.Bar(x=miss_counts.index.tolist(), y=miss_counts.values.tolist(), marker_color="indianred")
    fig.add_trace(bar, row=1, col=1)
    fig.update_xaxes(tickangle=45, row=1, col=1)
    fig.update_yaxes(title_text="missing count", row=1, col=1)

    # Heatmap of missing pattern (transpose to show columns on y-axis)
    sample_n = min(len(df), max_samples)
    sample_df = df.sample(n=sample_n, random_state=42) if len(df) > sample_n else df
    indicator = sample_df.isna().astype(int).T
    indicator = indicator.loc[indicator.index.isin(miss_counts.index)]
    heat = go.Heatmap(
        z=indicator.values,
        x=indicator.columns.astype(str).tolist(),
        y=indicator.index.tolist(),
        colorscale=[[0, "#ffffff"], [1, "#636efa"]],
        showscale=True,
    )
    fig.add_trace(heat, row=2, col=1)
    fig.update_yaxes(autorange="reversed", row=2, col=1)
    fig.update_xaxes(title_text=f"rows (sample of {sample_n})", row=2, col=1)

    fig.update_layout(height=600, title_text="Missing value overview")
    return fig
