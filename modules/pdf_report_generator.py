"""PDF Report Generation using ReportLab.

Generates professional text-based PDF reports for datasets and model training.
"""
import pandas as pd
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from datetime import datetime

def _create_pdf(filepath: str, title: str, sections: List[Dict[str, Any]], dataset_name: str = ""):
    """Internal helper to build a basic PDF."""
    doc = SimpleDocTemplate(filepath, pagesize=letter)
    styles = getSampleStyleSheet()
    Story = []
    
    # Title
    Story.append(Paragraph(title, styles['Title']))
    Story.append(Spacer(1, 8))
    
    # Dataset name if provided
    if dataset_name:
        Story.append(Paragraph(f"Dataset Name: {dataset_name}", styles['Normal']))
        
    # Generated timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    Story.append(Paragraph(f"Generated at: {timestamp}", styles['Normal']))
    Story.append(Spacer(1, 16))
    
    for section in sections:
        header = section.get('header', '')
        if header:
            Story.append(Paragraph(header, styles['Heading2']))
            Story.append(Spacer(1, 6))
        
        from reportlab.platypus import Table
        
        image_buffer = []
        def flush_images():
            if not image_buffer:
                return
            if len(image_buffer) == 1:
                Story.append(image_buffer[0])
            else:
                rows = []
                for i in range(0, len(image_buffer), 2):
                    row = image_buffer[i:i+2]
                    if len(row) == 1:
                        row.append("")
                    rows.append(row)
                Story.append(Table(rows))
            Story.append(Spacer(1, 4))
            image_buffer.clear()

        content = section.get('content', [])
        
        # Handle case where content itself is a DataFrame
        import pandas as _pd
        if isinstance(content, _pd.DataFrame):
            content = [content]
        
        for line in content:
            if isinstance(line, _pd.DataFrame):
                flush_images()
                from reportlab.platypus import Table, TableStyle
                from reportlab.lib import colors
                
                # Build table data: header row + data rows
                table_data = [list(line.columns)]
                for _, row in line.iterrows():
                    table_data.append([str(v) for v in row.values])
                
                t = Table(table_data)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ]))
                Story.append(t)
                Story.append(Spacer(1, 6))
            elif isinstance(line, str) and line.startswith("[IMAGE]"):
                img_path = line.replace("[IMAGE]", "").strip()
                try:
                    from reportlab.lib.units import inch
                    img = Image(img_path)
                    
                    aspect = img.imageWidth / float(img.imageHeight)
                    
                    # Max dimensions: 3.25 inches wide (to fit 2 columns), 3 inches tall
                    max_width = 3.25 * inch
                    max_height = 3.0 * inch
                    
                    if img.imageWidth > max_width or img.imageHeight > max_height:
                        if (max_width / aspect) <= max_height:
                            img.drawWidth = max_width
                            img.drawHeight = max_width / aspect
                        else:
                            img.drawHeight = max_height
                            img.drawWidth = max_height * aspect
                            
                    image_buffer.append(img)
                except Exception as e:
                    flush_images()
                    Story.append(Paragraph(f"*[Image not found or invalid: {img_path}]*", styles['Normal']))
                    Story.append(Spacer(1, 6))
            else:
                flush_images()
                Story.append(Paragraph(str(line), styles['Normal']))
                Story.append(Spacer(1, 4))
                
        flush_images()
            
        Story.append(Spacer(1, 8))
        
    doc.build(Story)

def generate_dataset_pdf(
    profile: Dict[str, Any], 
    cleaning_report: Dict[str, Any], 
    insights: Dict[str, List[str]], 
    output_path: str,
    dataset_name: str = "",
    plot_files: List[str] = None,
    matrix_files: List[str] = None
):
    """Generate the dataset and EDA PDF report."""
    sections = []
    
    # Dataset Summary
    summary_content = [
        f"Rows: {profile.get('rows', 0)}",
        f"Columns: {profile.get('columns', 0)}",
        f"Missing Values: {profile.get('total_missing', 0)}",
        f"Duplicates: {profile.get('duplicate_rows', 0)}"
    ]
    sections.append({"header": "Dataset Summary", "content": summary_content})
    
    # Cleaning Report
    cleaning_content = [
        f"Duplicates Removed: {cleaning_report.get('duplicates_removed', 0)}",
        f"Remaining Missing: {cleaning_report.get('remaining_missing', 0)}",
        f"Rows After Cleaning: {cleaning_report.get('rows_after_cleaning', 0)}"
    ]
    sections.append({"header": "Cleaning Report", "content": cleaning_content})
    
    # Business Insights
    insight_content = insights.get("dataset", []) + insights.get("eda", [])
    if insight_content:
        sections.append({"header": "Business Insights", "content": insight_content})
        
    # Embed plot images
    if plot_files:
        from pathlib import Path
        output_dir = Path(output_path).parent.parent / "plots"
        
        correlation_plots = []
        distribution_plots = []
        outlier_plots = []
        missing_value_plots = []
        other_plots = []
        
        for p in plot_files:
            img_path = str(output_dir / p)
            img_tag = f"[IMAGE]{img_path}"
            
            if "missing_values" in p:
                missing_value_plots.append(img_tag)
            elif "correlation" in p:
                correlation_plots.append(img_tag)
            elif "histogram" in p:
                distribution_plots.append(img_tag)
            elif "boxplot" in p:
                outlier_plots.append(img_tag)
            else:
                other_plots.append(img_tag)
                
    # Missing Value Analysis
    orig_missing = profile.get('total_missing', 0)
    rem_missing = cleaning_report.get('remaining_missing', 0)
    
    missing_text = []
    if orig_missing == 0:
        missing_text.append("Original Missing Values: 0")
        missing_text.append("Remaining Missing Values: 0")
        missing_text.append("No missing values were detected in the uploaded dataset.")
    else:
        missing_text.append("The chart below represents missing values detected in the original uploaded dataset before cleaning.")
        missing_text.append(f"Original Missing Values: {orig_missing}")
        missing_text.append(f"Remaining Missing Values After Cleaning: {rem_missing}")
        if rem_missing == 0:
            missing_text.append("All missing values were successfully handled during the cleaning process.")
        else:
            missing_text.append("Some missing values remain in the dataset and may affect predictive modeling.")
            
    missing_content = missing_text + missing_value_plots
    if missing_value_plots:
        missing_content.append("Figure: Missing values detected in the original dataset before cleaning.")
        
    sections.append({"header": "Missing Value Analysis", "content": missing_content})
        
    if plot_files:
        if correlation_plots:
            sections.append({"header": "Correlation Analysis", "content": correlation_plots})
            
            # Embed top correlations if exists
            top_corr_csv = output_dir / "top_correlations.csv"
            if top_corr_csv.exists():
                import pandas as pd
                try:
                    top_corrs_df = pd.read_csv(top_corr_csv)
                    if not top_corrs_df.empty:
                        # Format correlation values
                        top_corrs_df['Correlation'] = top_corrs_df['Correlation'].round(4)
                        sections.append({"header": "Top Correlations", "content": top_corrs_df})
                except Exception:
                    pass
        if distribution_plots:
            sections.append({"header": "Distribution Analysis", "content": distribution_plots})
        if outlier_plots:
            sections.append({"header": "Outlier Analysis", "content": outlier_plots})
        if other_plots:
            sections.append({"header": "Other Plots", "content": other_plots})
            
        # Continue listing exported files in an appendix section
        sections.append({"header": "Appendix: Exported Plot Files", "content": plot_files})
        
    # Matrix Files
    if matrix_files:
        sections.append({"header": "Correlation Matrix Export Files", "content": matrix_files})
        
    _create_pdf(str(output_path), "Dataset & EDA Report", sections, dataset_name)

def generate_training_pdf(
    problem_type: str, 
    model_results: Dict[str, Any], 
    insights: Dict[str, List[str]], 
    output_path: str,
    dataset_name: str = "",
    target_column: str = "",
    dataset_profile: Dict[str, Any] = None,
    metrics_df: pd.DataFrame = None,
    visualizations: List[str] = None,
    prediction_preview: pd.DataFrame = None,
    forecast_preview: pd.DataFrame = None
):
    """Generate the model training PDF report with a strict professional order."""
    sections = []
    
    # 1. Dataset Summary
    if dataset_profile:
        summary_lines = [
            f"Dataset Name: {dataset_name}",
            f"Rows: {dataset_profile.get('rows', 'N/A')}",
            f"Columns: {dataset_profile.get('columns', 'N/A')}",
            f"Target Column: {target_column}",
            f"Task Type: {problem_type.upper()}"
        ]
        sections.append({"header": "Dataset Summary", "content": summary_lines})
        
    # 2. Model Comparison Table
    if metrics_df is not None:
        sections.append({"header": "Model Comparison Table", "content": [metrics_df]})
        
    # 3. Best Model
    best_model = model_results.get("best_model_name", "Unknown")
    sections.append({"header": "Best Model", "content": [f"Selected Algorithm: {best_model}"]})
    
    # 4. Evaluation Metrics
    metrics = model_results.get("metrics", {})
    if best_model in metrics:
        best_metrics = metrics[best_model]
        def format_metric(val):
            if isinstance(val, (int, float)):
                if val == 0: return "0.0000"
                if abs(val) < 0.001: return f"{val:.4e}"
                return f"{val:.4f}"
            return str(val)
            
        metric_lines = [f"{k}: {format_metric(v)}" for k, v in best_metrics.items() if k != "interpretation"]
        
        if "interpretation" in best_metrics:
            metric_lines.append("")
            metric_lines.append(f"Interpretation: {best_metrics['interpretation']}")
            
        r2 = best_metrics.get("r2")
        rmse = best_metrics.get("rmse")
        if (r2 is not None and r2 >= 0.99) or (rmse is not None and rmse < 1e-5):
            metric_lines.append("⚠️ Near-perfect performance detected. Verify that target leakage is not present.")
            
        sections.append({"header": "Evaluation Metrics", "content": metric_lines})

    # 5. Visualizations
    if visualizations:
        viz_content = []
        for v_path in visualizations:
            viz_content.append(f"[IMAGE]{v_path}")
        sections.append({"header": "Visualizations", "content": viz_content})

    # 6. Feature Importance
    feature_importance = model_results.get("feature_importance")
    if feature_importance:
        import pandas as pd
        fi_df = pd.DataFrame(
            [{"Feature Name": feat, "Importance Score": round(score, 4)} 
             for feat, score in list(feature_importance.items())[:15]]
        )
        fi_content = [fi_df]
        fi_plot_path = model_results.get("feature_importance_plot")
        if fi_plot_path:
            fi_content.append(f"[IMAGE]{fi_plot_path}")
        sections.append({"header": "Feature Importance", "content": fi_content})

    # 7. Prediction Preview
    if prediction_preview is not None:
        sections.append({"header": "Prediction Preview", "content": [prediction_preview.head(10)]})

    # 8. Forecast Preview
    if forecast_preview is not None:
        sections.append({"header": "Forecast Preview", "content": [forecast_preview.head(10)]})
        
    _create_pdf(str(output_path), "Model Training Report", sections, dataset_name)


def generate_forecast_pdf(
    forecast_result: Dict[str, Any],
    output_path: str,
    dataset_name: str = "",
    target_column: str = "",
    date_column: str = "",
    forecast_plot_path: str = None
):
    """Generate the forecast PDF report.
    
    Args:
        forecast_result: Dictionary from forecasting.train_and_forecast().
        output_path: Path to save the PDF.
        dataset_name: Name of the dataset.
        target_column: Name of the target column.
        date_column: Name of the datetime column.
        forecast_plot_path: Path to forecast_plot.png.
    """
    sections = []
    
    # Forecast Configuration
    config_content = [
        f"Target Column: {target_column}",
        f"Date Column: {date_column}",
        f"Forecasting Engine: {forecast_result.get('engine', 'Unknown')}",
        f"Forecast Horizon: {forecast_result.get('horizon', 0)} periods",
        f"Frequency: {forecast_result.get('freq', 'Unknown')}",
    ]
    sections.append({"header": "Forecast Configuration", "content": config_content})
    
    # Forecast Insights
    insights = forecast_result.get("insights", {})
    insight_content = [
        f"Average Forecast: {insights.get('average_forecast', 'N/A')}",
        f"Forecast Range: {insights.get('forecast_range', 'N/A')}",
        f"Trend Direction: {insights.get('trend_direction', 'N/A')}",
        f"Trend Strength: {insights.get('trend_strength', 'N/A')}",
        f"Average Confidence Interval Width: {insights.get('avg_confidence_interval_width', 'N/A')}",
    ]
    sections.append({"header": "Forecast Insights", "content": insight_content})
    
    # Forecast Chart
    if forecast_plot_path:
        sections.append({
            "header": "Forecast Visualization", 
            "content": [f"[IMAGE]{forecast_plot_path}"]
        })
    
    # Forecast Statistics
    future_df = forecast_result.get("future_df")
    if future_df is not None and len(future_df) > 0:
        stats_content = [
            f"Forecast Start Date: {future_df['Date'].iloc[0].strftime('%Y-%m-%d') if hasattr(future_df['Date'].iloc[0], 'strftime') else str(future_df['Date'].iloc[0])}",
            f"Forecast End Date: {future_df['Date'].iloc[-1].strftime('%Y-%m-%d') if hasattr(future_df['Date'].iloc[-1], 'strftime') else str(future_df['Date'].iloc[-1])}",
            f"Total Forecast Points: {len(future_df)}",
            f"Minimum Forecast: {insights.get('min_forecast', 'N/A')}",
            f"Maximum Forecast: {insights.get('max_forecast', 'N/A')}",
            f"Average Forecast: {insights.get('average_forecast', 'N/A')}",
        ]
        sections.append({"header": "Forecast Statistics", "content": stats_content})
        
        # Forecast Sample
        sample_df = future_df[['Date', 'Forecast', 'Lower Bound', 'Upper Bound']].head(10).copy()
        # Format the numbers for cleaner display
        sample_df['Forecast'] = sample_df['Forecast'].round(4)
        sample_df['Lower Bound'] = sample_df['Lower Bound'].round(4)
        sample_df['Upper Bound'] = sample_df['Upper Bound'].round(4)
        sections.append({"header": "Forecast Sample", "content": sample_df})
    
    _create_pdf(str(output_path), "Forecast Report", sections, dataset_name)


def generate_prediction_pdf(
    dataset_name: str,
    prediction_summary: Dict[str, Any],
    sample_df: pd.DataFrame,
    output_path: str
):
    """Generate the prediction PDF report."""
    sections = []
    
    # Configuration Details
    config_content = [
        f"Prediction Dataset Source: {prediction_summary.get('source', 'Unknown')}",
        f"Total Predictions: {prediction_summary.get('count', 0)}",
        f"Prediction File Location: {prediction_summary.get('file_location', 'Unknown')}"
    ]
    sections.append({"header": "Prediction Configuration", "content": config_content})
    
    # Prediction Sample
    if not sample_df.empty:
        sections.append({"header": "Prediction Sample (First 10 Rows)", "content": sample_df})
        
    # Summary Statistics
    stats = prediction_summary.get('statistics')
    if stats:
        if "Class Distribution" in stats:
            # Classification
            dist_lines = []
            for cls, count in stats["Class Distribution"].items():
                dist_lines.append(f"Class {cls}: {count}")
            sections.append({"header": "Class Distribution", "content": dist_lines})
        else:
            # Regression
            stat_lines = [
                f"Minimum Prediction: {stats.get('min', 'N/A')}",
                f"Maximum Prediction: {stats.get('max', 'N/A')}",
                f"Mean Prediction: {stats.get('mean', 'N/A')}",
                f"Median Prediction: {stats.get('median', 'N/A')}",
                f"Standard Deviation: {stats.get('std', 'N/A')}",
            ]
            sections.append({"header": "Prediction Summary Statistics", "content": stat_lines})
            
    _create_pdf(str(output_path), "Prediction Report", sections, dataset_name)
