"""EDA Plot Exporter.

Exports EDA plots by regenerating them one at a time from the DataFrame,
so that no more than one Plotly figure is in memory at any moment.
"""
from pathlib import Path
import re
from typing import List, Tuple
import pandas as pd


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be a safe filename."""
    return re.sub(r"[^\w\-]", "_", name)


def export_eda_plots(
    plots_dir: Path,
    cleaned_df: pd.DataFrame,
    raw_df: pd.DataFrame = None,
) -> Tuple[List[str], List[str]]:
    """Export EDA plots by regenerating figures one at a time.

    This avoids holding all Plotly figures in memory simultaneously.
    Each figure is created, saved to disk, and immediately discarded.

    Args:
        plots_dir: Path to the run's plots/ directory.
        cleaned_df: The cleaned DataFrame (used for histograms, boxplots, correlation).
        raw_df: The raw DataFrame (used for missing values). Optional.

    Returns:
        Tuple of (List of exported plot filenames, List of exported matrix filenames).
    """
    # Lazy imports so the module loads fast
    from modules.eda import (
        missing_values_figure,
        correlation_matrix_figure,
        histogram_figures,
        boxplot_figures,
    )
    import gc

    exported_files = []
    matrix_files = []

    def _save_and_discard(fig, filename):
        """Write a single figure to disk, then immediately free it."""
        if fig is None:
            return
        try:
            filepath = plots_dir / filename
            fig.write_image(str(filepath))
            exported_files.append(filename)
        except Exception as e:
            print(f"Failed to export {filename}: {e}")
        finally:
            # Aggressively discard
            fig.data = []
            del fig
            gc.collect()

    # --- Missing values (from raw df) ---
    if raw_df is not None:
        mv_fig = missing_values_figure(raw_df)
        _save_and_discard(mv_fig, "missing_values.png")
        del mv_fig

    # --- Correlation heatmap ---
    numeric_cols = cleaned_df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) >= 2:
        corr_fig, corr_matrix = correlation_matrix_figure(cleaned_df, return_matrix=True)
        _save_and_discard(corr_fig, "correlation_heatmap.png")
        del corr_fig
        
        # Task 2: Export the correlation matrix to CSV and Markdown
        if corr_matrix is not None:
            csv_path = plots_dir / "correlation_matrix.csv"
            md_path = plots_dir / "correlation_matrix.md"
            
            corr_matrix.to_csv(csv_path)
            with open(md_path, "w") as f:
                f.write(corr_matrix.to_markdown())
                
            matrix_files.extend(["correlation_matrix.csv", "correlation_matrix.md"])

    # --- Histograms: one at a time ---
    if numeric_cols:
        for col in numeric_cols:
            figs = histogram_figures(cleaned_df, columns=[col])
            if col in figs:
                _save_and_discard(figs[col], f"histogram_{sanitize_filename(col)}.png")
            del figs
            gc.collect()

    # --- Boxplots: one at a time ---
    if numeric_cols:
        for col in numeric_cols:
            figs = boxplot_figures(cleaned_df, columns=[col])
            if col in figs:
                _save_and_discard(figs[col], f"boxplot_{sanitize_filename(col)}.png")
            del figs
            gc.collect()

    return exported_files, matrix_files
