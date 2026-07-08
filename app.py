import re
from typing import NamedTuple
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
from modules.profiler import profile_dataset
from modules.cleaner import clean_dataset
from modules.task_detector import detect_task, get_task_metadata, assess_forecast_suitability
from modules.eda import (
    summary_statistics,
    correlation_matrix_figure,
    histogram_figures,
    boxplot_figures,
    missing_values_figure,
)
from modules.model_trainer import train_and_select_model
from modules.model_saver import save_model
from modules.report_generator import (
    generate_dataset_report,
    generate_model_report,
    save_report,
)
from modules.run_manager import create_run_folder, init_metadata, update_metadata
from modules.business_insights import generate_insights
from modules.pdf_report_generator import generate_dataset_pdf, generate_training_pdf, generate_forecast_pdf
from modules.plot_exporter import export_eda_plots

APP_VERSION = "2.1"

st.set_page_config(
    page_title="AI EDA Platform",
    page_icon="📊",
    layout="wide"
)

DATASET_DIR = Path("datasets")
DATASET_DIR.mkdir(exist_ok=True)

st.title("📊 AI EDA Platform")
st.markdown("Upload a dataset for automated analysis, model selection, and forecasting.")

st.sidebar.header("Navigation")
app_mode = st.sidebar.radio("Go to", ["Analysis Pipeline", "Run History Dashboard"])


# ─────────────────────────────────────────────────
# HELPER: Save a DataFrame to disk (lazy mkdir)
# ─────────────────────────────────────────────────
def _save_artifact_csv(df: pd.DataFrame, artifact_path: Path) -> bool:
    """Save a DataFrame to the artifact path. Returns True on success."""
    try:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(artifact_path, index=False)
        return artifact_path.exists() and not df.empty
    except Exception as e:
        st.warning(f"Failed to save {artifact_path.name}: {e}")
        return False


# ─────────────────────────────────────────────────
# HELPER: Upload a DataFrame to BigQuery with full
# verification and artifact version tracking
# ─────────────────────────────────────────────────
def _sync_artifact_to_bq(
    df: pd.DataFrame,
    local_path: Path,
    table_id: str,
    artifact_key: str,
    run_id: str,
    dataset_name: str,
    task_type: str,
    target_column: str,
    base_dir: Path
) -> dict:
    """Verify local file → verify df non-empty → upload → update metadata → UI.

    Returns a status dict suitable for embedding in metadata.json.
    """
    ts = datetime.utcnow().isoformat()

    # Pre-flight checks
    if not local_path.exists():
        st.warning(f"Skipping BQ upload for {artifact_key}: local file not found at {local_path}")
        return {"status": "Skipped", "reason": "local_file_missing", "table": table_id}

    if df.empty:
        st.warning(f"Skipping BQ upload for {artifact_key}: DataFrame is empty.")
        return {"status": "Skipped", "reason": "empty_dataframe", "table": table_id}

    try:
        import os
        from modules import bigquery_connector

        client = bigquery_connector.connect_bigquery()
        if client is None:
            return {"status": "Skipped", "reason": "no_client", "table": table_id}

        dataset_id = os.getenv("BIGQUERY_DATASET", "ai_eda_platform")
        res = bigquery_connector.append_dataframe(
            client, dataset_id, table_id, df,
            run_id=run_id,
            dataset_name=dataset_name,
            task_type=task_type,
            target_column=target_column,
        )

        if res.get("status") == "ok":
            status_obj = {
                "status": "Uploaded",
                "rows": len(df),
                "uploaded_at": ts,
                "table": table_id,
            }
            st.success(f"✓ {artifact_key} synced to BigQuery ({len(df)} rows)")
        else:
            status_obj = {
                "status": "Failed",
                "error": str(res.get("errors") or res.get("message", "unknown")),
                "uploaded_at": ts,
                "table": table_id,
            }
            st.warning(f"✗ {artifact_key} BQ upload returned: {res}")

        # Update metadata with the version object
        update_metadata(base_dir, {f"bq_{artifact_key}": status_obj})
        return status_obj

    except Exception as e:
        status_obj = {
            "status": "Failed",
            "error": str(e),
            "uploaded_at": ts,
            "table": table_id,
        }
        st.warning(f"✗ {artifact_key} BQ upload error: {e}")
        update_metadata(base_dir, {f"bq_{artifact_key}": status_obj})
        return status_obj


# ─────────────────────────────────────────────────
# RUN HISTORY DASHBOARD
# ─────────────────────────────────────────────────
if app_mode == "Run History Dashboard":
    st.title("📂 Run History Dashboard")
    st.markdown("Track all generated reports and models across past runs.")
    from modules.run_history import get_run_history
    history_df = get_run_history()

    if history_df.empty:
        st.info("No run history found. Upload a dataset to get started.")
    else:
        display_df = history_df.drop(columns=["_folder_path"], errors="ignore")
        st.dataframe(display_df, use_container_width=True)

        st.markdown("### Access Reports")
        st.markdown("Paths listed above are relative to the run's folder inside the `reports/` directory.")

    st.stop()


# ─────────────────────────────────────────────────
# SIDEBAR — Dataset Upload & Cloud Sync
# ─────────────────────────────────────────────────
st.sidebar.header("Dataset Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

st.sidebar.header("Analysis Mode")
analysis_mode = st.sidebar.radio(
    "Mode",
    ["Deterministic", "AI Agent"],
    index=0
)

if analysis_mode == "AI Agent":
    st.sidebar.header("AI Settings")
    api_key = st.sidebar.text_input("Gemini API Key", type="password")
    if api_key:
        st.session_state["gemini_api_key"] = api_key

st.sidebar.header("Cloud Sync")
cloud_sync = st.sidebar.checkbox(
    "Upload Results To BigQuery",
    value=False,
    help="When enabled, run metadata, dataset profile and training metrics will be uploaded to BigQuery.",
)


class ProcessedDataset(NamedTuple):
    df: pd.DataFrame
    profile: dict
    cleaned_df: pd.DataFrame
    cleaning_report: dict
    cleaned_profile: dict


@st.cache_data
def load_and_process(file_path: str) -> ProcessedDataset:
    df = pd.read_csv(file_path)
    profile = profile_dataset(df)
    cleaned_df, cleaning_report = clean_dataset(df, profile)
    cleaned_profile = profile_dataset(cleaned_df)
    return ProcessedDataset(df, profile, cleaned_df, cleaning_report, cleaned_profile)


# ─────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────
if uploaded_file is not None:

    file_path = DATASET_DIR / uploaded_file.name

    # Only write to disk if not already saved to avoid redundant writes on every rerun
    if not file_path.exists():
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    st.success(f"Dataset saved successfully: {uploaded_file.name}")


    # AI Agent mode: validate API key early
    if analysis_mode == "AI Agent":
        if not st.session_state.get("gemini_api_key"):
            st.warning("⚠️ Please enter your Gemini API Key in the sidebar to use AI Agent mode.")
            st.stop()

    # Both modes share the same preprocessing pipeline.
    # Branching happens only at Model Training below.
    data = load_and_process(str(file_path))
    df, profile, cleaned_df, cleaning_report, cleaned_profile = data

    # ── Create run folder & init artifacts registry ──
    if "current_dataset" not in st.session_state or st.session_state.current_dataset != uploaded_file.name:
        st.session_state.current_dataset = uploaded_file.name
        run_id, artifacts = create_run_folder(uploaded_file.name)
        st.session_state.run_id = run_id
        st.session_state.artifacts = artifacts
        # Backward-compat: keep run_paths as alias pointing to key directories
        st.session_state.run_paths = {
            "base": artifacts["base_dir"],
            "plots": artifacts["plots_dir"],
            "predictions": artifacts["predictions_csv"].parent,
            "models": artifacts["best_model_joblib"].parent,
            "forecasts": artifacts["forecast_csv"].parent,
            "feature_importance": artifacts["feature_importance_csv"].parent,
        }
        init_metadata(artifacts["base_dir"], run_id, uploaded_file.name, artifacts)
        st.session_state.plot_files = []
        st.session_state.matrix_files = []

        # ── STEP 1: Save uploaded dataset immediately ──
        _save_artifact_csv(df, artifacts["uploaded_dataset"])
        st.info(f"Uploaded dataset saved to: `{artifacts['uploaded_dataset']}`")

        # ── STEP 2: Save cleaned dataset immediately ──
        _save_artifact_csv(cleaned_df, artifacts["cleaned_dataset"])
        st.info(f"Cleaned dataset saved to: `{artifacts['cleaned_dataset']}`")

        # ── STEP 3: BQ sync for uploaded & cleaned datasets ──
        if cloud_sync:
            _sync_artifact_to_bq(
                df, artifacts["uploaded_dataset"], "uploaded_datasets", "uploaded_dataset",
                run_id, uploaded_file.name, "pending", "pending", artifacts["base_dir"]
            )
            _sync_artifact_to_bq(
                cleaned_df, artifacts["cleaned_dataset"], "cleaned_datasets", "cleaned_dataset",
                run_id, uploaded_file.name, "pending", "pending", artifacts["base_dir"]
            )

    # Convenience aliases for current run
    artifacts = st.session_state.artifacts
    run_paths = st.session_state.run_paths

    def save_plot_if_needed(fig, filename):
        if filename not in st.session_state.plot_files:
            try:
                plots_dir = artifacts["plots_dir"]
                plots_dir.mkdir(parents=True, exist_ok=True)
                filepath = plots_dir / filename
                fig.write_image(str(filepath))
                st.session_state.plot_files.append(filename)
            except Exception as e:
                print(f"Failed to save plot {filename}: {e}")

    # ─────────────────────────────────────────────────
    # DATASET PREVIEW & PROFILE
    # ─────────────────────────────────────────────────
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.subheader("Column Information")
    info_df = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str)
        }
    )
    st.dataframe(info_df)

    st.subheader("Dataset Profile")

    # Dataset overview metrics
    dp_col1, dp_col2, dp_col3, dp_col4 = st.columns(4)
    dp_col1.metric("Rows", profile.get("rows", df.shape[0]))
    dp_col2.metric("Columns", profile.get("columns", df.shape[1]))
    dp_col3.metric("Missing Values", profile.get("total_missing", int(df.isnull().sum().sum())))
    dp_col4.metric("Duplicate Rows", profile.get("duplicate_rows", int(df.duplicated().sum())))

    # Data type summary
    st.markdown("**Data Type Summary**")
    dtype_summary = pd.DataFrame(
        {
            "Type": ["Numerical", "Categorical", "Datetime"],
            "Count": [
                len(profile.get("numerical_columns", [])),
                len(profile.get("categorical_columns", [])),
                len(profile.get("datetime_cols", [])),
            ],
        }
    )
    st.table(dtype_summary)

    # Missing values summary (only show cols with > 0 missing)
    st.markdown("**Missing Values Summary**")
    missing_by_col = profile.get("missing_by_column", {})
    missing_rows = [(c, v) for c, v in missing_by_col.items() if v > 0]
    if not missing_rows:
        st.success("✓ No missing values detected.")
    else:
        missing_df = pd.DataFrame(missing_rows, columns=["Column", "Missing Values"]).sort_values(
            by="Missing Values", ascending=False
        )
        st.dataframe(missing_df)

    # Detected patterns in expandable sections
    st.markdown("**Detected Patterns**")
    with st.expander("Numeric-like Text Columns"):
        nl = profile.get("numeric_like_text", [])
        if nl:
            st.write(", ".join(nl))
        else:
            st.write("None detected.")

    with st.expander("Date Candidates"):
        dc = profile.get("date_candidates", [])
        if dc:
            st.write(", ".join(dc))
        else:
            st.write("None detected.")

    with st.expander("Binary Columns"):
        bc = profile.get("binary_columns", [])
        if bc:
            st.write(", ".join(bc))
        else:
            st.write("None detected.")

    with st.expander("Low Cardinality Columns"):
        lc = profile.get("low_cardinality", [])
        if lc:
            lf = pd.DataFrame(
                [(c, profile.get("unique_values", {}).get(c)) for c in lc],
                columns=["Column", "Unique Values"],
            )
            st.dataframe(lf)
        else:
            st.write("None detected.")

    with st.expander("High Cardinality Columns"):
        hc = profile.get("high_cardinality", [])
        if hc:
            hf = pd.DataFrame(
                [(c, profile.get("unique_values", {}).get(c)) for c in hc],
                columns=["Column", "Unique Values"],
            )
            st.dataframe(hf)
        else:
            st.write("None detected.")

    # Cleaning summary
    st.subheader("Cleaning Summary")
    cr = cleaning_report or {}
    ccol1, ccol2 = st.columns([2, 1])
    with ccol1:
        st.markdown(f"- **Duplicate rows removed:** {cr.get('duplicates_removed', 0)}")
        nc = cr.get("numeric_converted", [])
        st.markdown(f"- **Numeric conversions:** {', '.join(nc) if nc else 'None'}")
        dc = cr.get("date_converted", [])
        st.markdown(f"- **Date conversions:** {', '.join(dc) if dc else 'None'}")
        st.markdown(f"- **Remaining missing values:** {cr.get('remaining_missing', 0)}")
    with ccol2:
        st.metric("Rows After Cleaning", cr.get("rows_after_cleaning", len(cleaned_df)))

    # Cleaned dataset overview
    st.subheader("Cleaned Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", cleaned_profile.get("rows", cleaned_df.shape[0]))
    c2.metric("Columns", cleaned_profile.get("columns", cleaned_df.shape[1]))
    c3.metric("Missing Values", cleaned_profile.get("total_missing", int(cleaned_df.isnull().sum().sum())))
    c4.metric("Duplicate Rows", cleaned_profile.get("duplicate_rows", int(cleaned_df.duplicated().sum())))

    st.subheader("💡 Business Insights (Data & Structure)")
    base_insights = generate_insights(profile, cleaning_report)
    for insight in base_insights.get("dataset", []):
        st.info(f"📊 {insight}")
    for insight in base_insights.get("eda", []):
        st.info(f"🔍 {insight}")

    # ─────────────────────────────────────────────────
    # EDA ANALYSIS
    # ─────────────────────────────────────────────────
    st.subheader("EDA Analysis")

    try:
        st.markdown("Exploratory Data Analysis on the cleaned dataset.")

        with st.expander("Summary statistics", expanded=True):
            stats = summary_statistics(cleaned_df)
            st.dataframe(stats)

        with st.expander("Missing values visualization"):
            mv_fig = missing_values_figure(df)
            if mv_fig is not None:
                st.caption("Missing values shown on the raw dataset before cleaning.")
                st.plotly_chart(mv_fig, use_container_width=True)
                save_plot_if_needed(mv_fig, "missing_values.png")
            else:
                st.write("**Original Missing Values: 0**")
                st.write("**Remaining Missing Values: 0**")
                st.success("No missing values were detected in the uploaded dataset.")

        numeric_cols = cleaned_df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            st.info("No numeric columns available for correlation, histograms, or boxplots.")
        else:
            with st.expander("Correlation matrix"):
                corr_fig, corr_matrix = correlation_matrix_figure(cleaned_df, return_matrix=True)
                st.plotly_chart(corr_fig, use_container_width=True)

                if "correlation_heatmap.png" not in st.session_state.plot_files:
                    save_plot_if_needed(corr_fig, "correlation_heatmap.png")
                    if corr_matrix is not None:
                        plots_dir = artifacts["plots_dir"]
                        plots_dir.mkdir(parents=True, exist_ok=True)
                        csv_path = plots_dir / "correlation_matrix.csv"
                        md_path = plots_dir / "correlation_matrix.md"
                        corr_matrix.to_csv(csv_path)
                        with open(md_path, "w") as f:
                            f.write(corr_matrix.to_markdown())
                        st.session_state.matrix_files.extend(["correlation_matrix.csv", "correlation_matrix.md"])

                        # Save Top Correlations
                        stacked = corr_matrix.stack().reset_index()
                        stacked.columns = ['Feature A', 'Feature B', 'Correlation']
                        stacked = stacked[stacked['Feature A'] < stacked['Feature B']]

                        top_pos = stacked[stacked['Correlation'] > 0].sort_values(by='Correlation', ascending=False).head(5)
                        top_neg = stacked[stacked['Correlation'] < 0].sort_values(by='Correlation', ascending=True).head(5)

                        top_corrs = pd.concat([top_pos, top_neg])
                        top_corrs_csv = plots_dir / "top_correlations.csv"
                        top_corrs.to_csv(top_corrs_csv, index=False)

                if corr_matrix is not None:
                    # Display top correlations in UI
                    stacked = corr_matrix.stack().reset_index()
                    stacked.columns = ['Feature A', 'Feature B', 'Correlation']
                    stacked = stacked[stacked['Feature A'] < stacked['Feature B']]

                    top_pos = stacked[stacked['Correlation'] > 0].sort_values(by='Correlation', ascending=False).head(5)
                    top_neg = stacked[stacked['Correlation'] < 0].sort_values(by='Correlation', ascending=True).head(5)

                    if not top_pos.empty or not top_neg.empty:
                        st.subheader("Top Correlations")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Top Positive Correlations**")
                            st.dataframe(top_pos)
                        with col2:
                            st.markdown("**Top Negative Correlations**")
                            st.dataframe(top_neg)

            from modules.plot_exporter import sanitize_filename
            with st.expander("Histograms"):
                for col_name, fig in histogram_figures(cleaned_df).items():
                    st.plotly_chart(fig, use_container_width=True)
                    save_plot_if_needed(fig, f"histogram_{sanitize_filename(col_name)}.png")

            with st.expander("Boxplots"):
                for col_name, fig in boxplot_figures(cleaned_df).items():
                    st.plotly_chart(fig, use_container_width=True)
                    save_plot_if_needed(fig, f"boxplot_{sanitize_filename(col_name)}.png")

    except Exception as e:
        st.error(f"EDA failed: {e}")

    # ─────────────────────────────────────────────────
    # TARGET SELECTION & TASK DETECTION
    # ─────────────────────────────────────────────────
    cols = list(cleaned_df.columns)
    date_candidates = set(cleaned_profile.get("date_candidates", []) + cleaned_profile.get("datetime_cols", []))
    default_index = 0
    for i, c in enumerate(cols):
        if c not in date_candidates:
            default_index = i
            break
    target_column = st.selectbox("Select Target Column", cols, index=default_index)

    problem_type = detect_task(cleaned_df, cleaned_profile, target_column)

    st.subheader("Detected Problem Type")
    st.success(problem_type.upper())

    # Task Detection Transparency
    task_meta = get_task_metadata(cleaned_df, cleaned_profile, target_column, problem_type)
    st.caption(f"**Confidence:** {task_meta['confidence']} — {task_meta['reason']}")

    # Forecast Suitability Assessment
    forecast_suit = assess_forecast_suitability(cleaned_df, target_column, cleaned_profile)
    if forecast_suit["suitability"] != "Low":
        st.subheader("📈 Forecast Suitability")
        suit_color = "success" if forecast_suit["suitability"] == "High" else "info"
        getattr(st, suit_color)(f"Suitability: {forecast_suit['suitability']}")
        st.caption(forecast_suit["reason"])

    # ─────────────────────────────────────────────────
    # MODEL TRAINING
    # ─────────────────────────────────────────────────
    st.subheader("Model Training")
    st.markdown("Train simple baseline models for the detected task.")

    if problem_type == "unknown":
        st.warning("Could not detect a supported problem type. Please select a different target column.")
    elif problem_type == "forecasting":
        st.info("Task identified as Forecasting. Please use the Forecasting Engine below.")
    elif analysis_mode == "AI Agent":
        # ─── AI AGENT MODEL TRAINING ───
        st.info("🤖 **AI Agent Mode**: The AI will autonomously select and train models.")
        if st.button("🤖 Train Models (AI Agent)", type="primary"):
            try:
                plot_files = st.session_state.plot_files
                matrix_files = st.session_state.matrix_files
                
                # Save the cleaned dataset for the AI agent to consume
                cleaned_path = str(artifacts["cleaned_dataset"])
                _save_artifact_csv(cleaned_df, artifacts["cleaned_dataset"])

                with st.spinner("🤖 AI Agent is selecting and training models..."):
                    from modules.ai_agent.orchestrator import run_ai_pipeline
                    ai_result = run_ai_pipeline(
                        api_key=st.session_state["gemini_api_key"],
                        cleaned_file_path=cleaned_path,
                        target_column=target_column,
                        problem_type=problem_type,
                        profile=cleaned_profile,
                        artifacts=artifacts,
                        cloud_sync=cloud_sync
                    )

                    if ai_result.get("success"):
                        st.session_state.temp_result = ai_result.get("raw_result")
                        st.session_state.temp_ai_summary = ai_result.get("ai_summary", "")
                    else:
                        st.error(f"AI Agent failed: {ai_result.get('error', 'Unknown error')}")

            except Exception as e:
                st.error(f"AI Agent training failed: {e}")
                import traceback
                traceback.print_exc()

    else:
        if st.button("Train Models"):
            try:
                plot_files = st.session_state.plot_files
                matrix_files = st.session_state.matrix_files

                with st.spinner("Training models — this may take a moment..."):
                    from modules.model_trainer import train_and_select_model
                    result = train_and_select_model(
                        cleaned_df, target_column, problem_type
                    )
                    st.session_state.temp_result = result
            except Exception as e:
                st.error(f"Deterministic training failed: {e}")
                import traceback
                traceback.print_exc()

    if 'temp_result' in st.session_state and st.session_state.temp_result is not None:
        result = st.session_state.temp_result
        del st.session_state.temp_result
        # Apply mode prefix to post-training artifacts to prevent overwriting
        prefix = "ai_" if analysis_mode == "AI Agent" else "det_"
        post_training_keys = [
            "metrics_csv", "feature_importance_csv", "feature_importance_png",
            "predictions_csv", "prediction_report_pdf", "forecast_csv",
            "forecast_report_pdf", "training_report_pdf", "best_model_joblib",
            "feature_schema_json", "model_report_json"
        ]
        for key in post_training_keys:
            if key in st.session_state.artifacts:
                orig_path = st.session_state.artifacts[key]
                base_name = orig_path.name
                if base_name.startswith("ai_"):
                    base_name = base_name[3:]
                elif base_name.startswith("det_"):
                    base_name = base_name[4:]
                
                st.session_state.artifacts[key] = orig_path.with_name(prefix + base_name)
        
        artifacts = st.session_state.artifacts
        
        best_model = result["best_model"]
        best_name = result["best_model_name"]
        metrics = result["metrics"]

        st.markdown("**Best model**")
        st.write(best_name)

        # Build comparison table for candidate metrics
        metrics_rows = []
        if isinstance(metrics, dict):
            for model_name, metric in metrics.items():
                if not isinstance(metric, dict):
                    continue
                if problem_type == "regression":
                    metrics_rows.append({
                        "Model": model_name,
                        "R²": metric.get("r2"),
                        "MAE": metric.get("mae"),
                        "RMSE": metric.get("rmse"),
                        "MAPE": metric.get("mape"),
                        "Training Time (s)": metric.get("Training Time (s)")
                    })
                else:
                    metrics_rows.append({
                        "Model": model_name,
                        "Accuracy": metric.get("accuracy"),
                        "Precision": metric.get("precision"),
                        "Recall": metric.get("recall"),
                        "F1": metric.get("f1"),
                        "ROC-AUC": metric.get("roc_auc"),
                        "Training Time (s)": metric.get("Training Time (s)")
                    })

        metrics_df = pd.DataFrame(metrics_rows)
        if not metrics_df.empty:
            metrics_df = metrics_df.round(4)
            st.markdown("**Model Comparison Table**")
            st.dataframe(metrics_df)
            # Save metrics CSV to artifact registry path
            metrics_saved = _save_artifact_csv(metrics_df, artifacts["metrics_csv"])
            if metrics_saved:
                st.success(f"Metrics CSV saved to: `{artifacts['metrics_csv']}`")
            result["metrics_df"] = metrics_df

        # Best model summary cards
        if best_name in metrics and isinstance(metrics[best_name], dict):
            best_metrics = metrics[best_name]
        else:
            best_metrics = {}
        st.markdown("**Best Model Summary**")
        bm_cols = st.columns(3)
        bm_cols[0].metric("Best Model", best_name)
        if problem_type == "regression":
            bm_cols[1].metric("R²", round(best_metrics.get("r2", 0), 4) if best_metrics.get("r2") is not None else "N/A")
            bm_cols[2].metric("RMSE", round(best_metrics.get("rmse", 0), 4) if best_metrics.get("rmse") is not None else "N/A")
        else:
            bm_cols[1].metric("F1", round(best_metrics.get("f1", 0), 4) if best_metrics.get("f1") is not None else "N/A")
            bm_cols[2].metric("ROC-AUC", round(best_metrics.get("roc_auc", 0), 4) if best_metrics.get("roc_auc") is not None else "N/A")

        # Save best model
        try:
            model_file_name = "best_model"
            import importlib, sys
            import modules.model_saver
            importlib.reload(modules.model_saver)
            feature_schema = {
                "target_column": target_column,
                "problem_type": problem_type,
                "feature_columns": result["feature_columns"]
            }
            models_dir = artifacts["best_model_joblib"].parent
            saved_path = modules.model_saver.save_model(
                best_model,
                model_file_name,
                output_dir=models_dir,
                feature_schema=feature_schema
            )
            st.success(f"Best model saved to: {saved_path}")
        except Exception as e:
            st.error(f"Failed to save model: {e}")

        st.markdown("**💡 Model Insights**")
        full_insights = generate_insights(profile, cleaning_report, result)
        for insight in full_insights.get("model", []):
            st.success(f"🤖 {insight}")

        # ── Generate Visualizations ──
        import modules.model_visualizer as mv
        viz_paths = []
        st.markdown("**Model Visualizations**")

        try:
            plots_dir = artifacts["plots_dir"]
            plots_dir.mkdir(parents=True, exist_ok=True)
            if problem_type == "regression":
                res_plots = mv.generate_regression_plots(result["y_test"], result["y_pred"], result.get("feature_importance"), str(plots_dir))
                for p, fig in res_plots:
                    viz_paths.append(p)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                res_plots = mv.generate_classification_plots(result["y_test"], result["y_pred"], result.get("y_score"), result.get("feature_importance"), str(plots_dir))
                for p, fig in res_plots:
                    viz_paths.append(p)
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Visualization generation failed: {e}")

        # ── Feature Importance ──
        fi_data = result.get("feature_importance")
        if fi_data:
            st.markdown("**📊 Top Feature Importances**")
            fi_dir = artifacts["feature_importance_csv"].parent
            fi_dir.mkdir(parents=True, exist_ok=True)

            # Save CSV
            fi_df = pd.DataFrame(list(fi_data.items()), columns=["Feature", "Importance"])
            fi_saved = _save_artifact_csv(fi_df, artifacts["feature_importance_csv"])
            if fi_saved:
                st.success(f"Feature importance CSV saved to: `{artifacts['feature_importance_csv']}`")

            # Generate plot using model_visualizer (saves to fi_dir)
            p, fig = mv.generate_feature_importance_plot(fi_data, str(fi_dir))
            if p:
                viz_paths.append(p)
                result["feature_importance_plot"] = p
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

        # ── Generate Reports ──
        try:
            dataset_report = generate_dataset_report(profile, cleaning_report)
            model_report = generate_model_report(problem_type, best_name, metrics, result.get("feature_importance"))

            dataset_report_path = save_report(dataset_report, "dataset_report", reports_dir=artifacts["base_dir"])
            model_report_path = save_report(model_report, "model_report", reports_dir=artifacts["base_dir"])

            # Generate PDF Reports
            eda_pdf_path = artifacts["eda_report_pdf"]
            eda_pdf_path.parent.mkdir(parents=True, exist_ok=True)
            generate_dataset_pdf(
                profile, cleaning_report, full_insights, eda_pdf_path,
                dataset_name=uploaded_file.name,
                plot_files=plot_files,
                matrix_files=matrix_files
            )

            training_pdf_path = artifacts["training_report_pdf"]
            training_pdf_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_profile_info = {
                "rows": int(profile.get("rows", 0)),
                "columns": int(profile.get("columns", 0)),
            }
            generate_training_pdf(
                problem_type, result, full_insights, training_pdf_path,
                dataset_name=uploaded_file.name,
                target_column=target_column,
                dataset_profile=dataset_profile_info,
                metrics_df=metrics_df,
                visualizations=viz_paths
            )

            st.session_state.training_result = result
            st.session_state.full_insights = full_insights
            st.session_state.target_column = target_column
            st.session_state.problem_type = problem_type

            st.success(f"Dataset JSON report saved to: {dataset_report_path}")
            st.success(f"Model JSON report saved to: {model_report_path}")
            st.success(f"EDA PDF report saved to: {eda_pdf_path}")
            st.success(f"Training PDF report saved to: {training_pdf_path}")

            # ── Update metadata immediately after training ──
            update_metadata(artifacts["base_dir"], {
                "task_type": problem_type,
                "target_column": target_column,
                "best_model": best_name,
                "eda_pdf": "reports/eda_report.pdf",
                "training_pdf": "reports/training_report.pdf"
            })

            # ── Event-driven BQ sync for training artifacts ──
            if cloud_sync:
                try:
                    import os
                    from modules import bigquery_connector
                    ts = datetime.utcnow().isoformat()

                    try:
                        client = bigquery_connector.connect_bigquery()
                    except Exception as e:
                        st.warning(f"BigQuery client not available: {e}")
                        client = None

                    if client is not None:
                        dataset_id = os.getenv("BIGQUERY_DATASET", "ai_eda_platform")

                        # 1. Run Metadata
                        run_meta_row = {
                            "run_id": st.session_state.run_id,
                            "dataset_name": uploaded_file.name,
                            "target_column": target_column,
                            "task_type": problem_type,
                            "best_model": best_name,
                            "run_timestamp": ts,
                            "app_version": APP_VERSION,
                        }
                        bigquery_connector.upload_run_metadata(client, dataset_id, run_meta_row)

                        # 2. Dataset Profile
                        profile_row = {
                            "run_id": st.session_state.run_id,
                            "dataset_name": uploaded_file.name,
                            "task_type": problem_type,
                            "target_column": target_column,
                            "rows": int(profile.get("rows", 0)),
                            "columns": int(profile.get("columns", 0)),
                            "missing_values": int(profile.get("total_missing", 0)),
                            "duplicate_rows": int(profile.get("duplicate_rows", 0)),
                            "timestamp": ts,
                            "app_version": APP_VERSION,
                        }
                        bigquery_connector.upload_dataset_profile(client, dataset_id, profile_row)

                        # 3. Training Metrics
                        bm = metrics.get(best_name, {}) if isinstance(metrics, dict) else {}
                        tm_row = {
                            "run_id": st.session_state.run_id,
                            "task_type": problem_type,
                            "target_column": target_column,
                            "best_model": best_name,
                            "accuracy": float(bm.get("accuracy")) if bm.get("accuracy") is not None else None,
                            "f1": float(bm.get("f1")) if bm.get("f1") is not None else None,
                            "precision": float(bm.get("precision")) if bm.get("precision") is not None else None,
                            "recall": float(bm.get("recall")) if bm.get("recall") is not None else None,
                            "r2": float(bm.get("r2")) if bm.get("r2") is not None else None,
                            "rmse": float(bm.get("rmse")) if bm.get("rmse") is not None else None,
                            "mae": float(bm.get("mae")) if bm.get("mae") is not None else None,
                            "timestamp": ts,
                            "app_version": APP_VERSION,
                        }
                        bigquery_connector.upload_training_metrics(client, dataset_id, tm_row)
                        st.success("✓ Run metadata, profile & training metrics synced to BigQuery.")

                        # 4. Metrics CSV
                        if metrics_saved:
                            _sync_artifact_to_bq(
                                metrics_df, artifacts["metrics_csv"], "metrics", "metrics",
                                st.session_state.run_id, uploaded_file.name, problem_type, target_column,
                                artifacts["base_dir"]
                            )

                        # 5. Feature Importance CSV
                        if fi_data and fi_saved:
                            _sync_artifact_to_bq(
                                fi_df, artifacts["feature_importance_csv"], "feature_importance", "feature_importance",
                                st.session_state.run_id, uploaded_file.name, problem_type, target_column,
                                artifacts["base_dir"]
                            )

                except Exception as e:
                    st.warning(f"BigQuery sync error: {e}")

        except Exception as e:
            st.error(f"Failed to generate or save reports: {e}")




    # ---------------------------------------------------------
    # PREDICTION ENGINE UI
    # ---------------------------------------------------------
    if "artifacts" in st.session_state:
        models_dir = artifacts["best_model_joblib"].parent
        model_path = artifacts["best_model_joblib"]
        schema_path = artifacts["feature_schema_json"]

        if model_path.exists() and schema_path.exists():
            st.markdown("---")
            st.header("🔮 Prediction Engine")
            st.markdown("Use the trained model to generate predictions on new or existing data.")

            pred_source = st.radio("Select Prediction Data Source", ["Use Current Cleaned Dataset", "Upload New Dataset"])

            pred_df = None
            if pred_source == "Use Current Cleaned Dataset":
                pred_df = cleaned_df
            else:
                pred_file = st.file_uploader("Upload Prediction CSV", type=["csv"], key="pred_uploader")
                if pred_file is not None:
                    pred_df = pd.read_csv(pred_file)

            if pred_df is not None:
                if st.button("Generate Predictions", type="primary"):
                    try:
                        from modules.predictor import load_predictor, prepare_prediction_data, generate_predictions, PredictionValidationError

                        model, schema = load_predictor(models_dir)
                        prepared_df = prepare_prediction_data(pred_df, schema)
                        predictions = generate_predictions(model, prepared_df)

                        pred_source_label = "Current Cleaned Dataset" if pred_source == "Use Current Cleaned Dataset" else "Uploaded Prediction Dataset"

                        st.success(f"Successfully generated {len(predictions)} predictions!")
                        st.info(f"Prediction Dataset Source: {pred_source_label}")

                        # Merge predictions back for display
                        result_df = prepared_df.copy()
                        result_df["Prediction"] = predictions

                        st.dataframe(result_df.head())

                        # Save predictions to artifact registry path
                        pred_csv_path = artifacts["predictions_csv"]
                        pred_saved = _save_artifact_csv(result_df, pred_csv_path)
                        if pred_saved:
                            st.success(f"Predictions saved to: `{pred_csv_path}`")

                        # Regenerate PDF with prediction summary if we have the training state
                        if "training_result" in st.session_state:
                            from modules.pdf_report_generator import generate_training_pdf, generate_prediction_pdf
                            import numpy as np

                            # Compute prediction statistics
                            pred_stats = {}
                            problem = st.session_state.problem_type
                            if problem == "classification":
                                class_counts = pd.Series(predictions).value_counts().to_dict()
                                pred_stats["Class Distribution"] = {str(k): int(v) for k, v in class_counts.items()}
                            else:
                                pred_arr = np.array(predictions, dtype=float)
                                pred_stats = {
                                    "min": round(float(pred_arr.min()), 4),
                                    "max": round(float(pred_arr.max()), 4),
                                    "mean": round(float(pred_arr.mean()), 4),
                                    "median": round(float(np.median(pred_arr)), 4),
                                    "std": round(float(pred_arr.std()), 4),
                                }

                            pred_summary = {
                                "source": pred_source_label,
                                "count": len(predictions),
                                "file_location": str(pred_csv_path),
                                "statistics": pred_stats
                            }

                            # Display prediction statistics in UI
                            st.subheader("Prediction Summary Statistics")
                            if problem == "classification":
                                st.markdown("**Class Distribution**")
                                for cls, count in pred_stats["Class Distribution"].items():
                                    st.write(f"Class {cls}: {count}")
                            else:
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Min", pred_stats["min"])
                                col1.metric("Max", pred_stats["max"])
                                col2.metric("Mean", pred_stats["mean"])
                                col2.metric("Median", pred_stats["median"])
                                col3.metric("Std Dev", pred_stats["std"])

                            # Generate dedicated prediction PDF
                            pred_pdf_path = artifacts["prediction_report_pdf"]
                            pred_pdf_path.parent.mkdir(parents=True, exist_ok=True)
                            sample_for_pdf = result_df.head(10)
                            generate_prediction_pdf(
                                dataset_name=st.session_state.current_dataset,
                                prediction_summary=pred_summary,
                                sample_df=sample_for_pdf,
                                output_path=str(pred_pdf_path)
                            )
                            st.success(f"Prediction PDF report saved to: `{pred_pdf_path}`")

                            # Update model JSON report
                            import json
                            model_json_path = artifacts["base_dir"] / "model_report.json"
                            if model_json_path.exists():
                                with open(model_json_path, "r", encoding="utf-8") as f:
                                    model_rep = json.load(f)
                                model_rep["prediction_summary"] = pred_summary
                                with open(model_json_path, "w", encoding="utf-8") as f:
                                    json.dump(model_rep, f, indent=2, ensure_ascii=False)
                                st.info("Model JSON report has been updated with the Prediction Summary.")

                            # ── Update metadata immediately ──
                            update_metadata(artifacts["base_dir"], {
                                "prediction_count": len(predictions),
                                "prediction_pdf": "predictions/prediction_report.pdf"
                            })

                            # ── Event-driven BQ sync for predictions ──
                            if cloud_sync and pred_saved:
                                _sync_artifact_to_bq(
                                    result_df, pred_csv_path, "predictions", "predictions",
                                    st.session_state.run_id, st.session_state.current_dataset,
                                    st.session_state.problem_type, st.session_state.target_column,
                                    artifacts["base_dir"]
                                )

                    except PredictionValidationError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

    # ---------------------------------------------------------
    # FORECASTING ENGINE UI
    # ---------------------------------------------------------
    if "artifacts" in st.session_state:
        forecast_suit = assess_forecast_suitability(cleaned_df, target_column, cleaned_profile)

        if forecast_suit["suitability"] in ("High", "Medium"):
            st.markdown("---")
            st.header("📈 Forecasting Engine")
            st.markdown("Generate time-series forecasts using the detected datetime column.")

            date_column = forecast_suit.get("date_column", "")
            st.info(f"Date column: **{date_column}** | Target: **{target_column}**")

            col_h, col_f = st.columns(2)
            with col_h:
                horizon = st.selectbox("Forecast Horizon (periods)", [7, 30, 90], index=1)
            with col_f:
                freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}
                freq_label = st.selectbox("Frequency", list(freq_map.keys()))
                freq = freq_map[freq_label]

            if st.button("Run Forecast", type="primary"):
                try:
                    from modules.forecasting import prepare_time_series, train_and_forecast, generate_forecast_plot

                    with st.spinner("Preparing time series and training forecast model..."):
                        ts_df = prepare_time_series(cleaned_df, date_column, target_column, freq=freq)
                        forecast_result = train_and_forecast(ts_df, horizon=horizon, freq=freq)

                    st.success(f"Forecast generated using {forecast_result['engine']}!")

                    # Display insights
                    insights = forecast_result["insights"]
                    ic1, ic2, ic3 = st.columns(3)
                    ic1.metric("Average Forecast", insights["average_forecast"])
                    ic2.metric("Forecast Range", insights["forecast_range"])
                    ic3.metric("Trend Direction", insights["trend_direction"])

                    ic4, ic5, ic6 = st.columns(3)
                    ic4.metric("Trend Strength", insights.get("trend_strength", "N/A"))
                    ic5.metric("Avg CI Width", insights.get("avg_confidence_interval_width", "N/A"))
                    ic6.metric("Min Forecast", insights.get("min_forecast", "N/A"))

                    # Forecast Statistics Table
                    st.subheader("Forecast Statistics")
                    stats_data = {
                        "Statistic": ["Minimum Forecast", "Maximum Forecast", "Average Forecast", "Trend Strength", "Avg Confidence Interval Width"],
                        "Value": [
                            str(insights.get("min_forecast", "N/A")),
                            str(insights.get("max_forecast", "N/A")),
                            str(insights["average_forecast"]),
                            f"{insights.get('trend_strength', 'N/A')} ({insights['trend_direction']})",
                            str(insights.get("avg_confidence_interval_width", "N/A"))
                        ]
                    }
                    st.table(pd.DataFrame(stats_data))

                    # Display future forecast table sample
                    st.subheader("Forecast Sample")
                    sample_df = forecast_result["future_df"][['Date', 'Forecast', 'Lower Bound', 'Upper Bound']].head(10).copy()
                    st.dataframe(sample_df)

                    # Generate plot
                    forecasts_dir = artifacts["forecast_csv"].parent
                    forecasts_dir.mkdir(parents=True, exist_ok=True)
                    plot_path = generate_forecast_plot(
                        ts_df, forecast_result, forecasts_dir, target_column
                    )

                    # Display plot
                    import plotly.graph_objects as go
                    future_df = forecast_result["future_df"]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ts_df['ds'], y=ts_df['y'], mode='lines', name='Historical', line=dict(color='#2196F3')))
                    fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Forecast'], mode='lines', name='Forecast', line=dict(color='#FF5722', dash='dash')))
                    fig.add_trace(go.Scatter(
                        x=pd.concat([future_df['Date'], future_df['Date'][::-1]]),
                        y=pd.concat([future_df['Upper Bound'], future_df['Lower Bound'][::-1]]),
                        fill='toself', fillcolor='rgba(255,87,34,0.15)',
                        line=dict(color='rgba(255,87,34,0)'), showlegend=True, name='Confidence Interval'
                    ))
                    fig.update_layout(title=f"Forecast: {target_column}", xaxis_title="Date", yaxis_title=target_column, template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)

                    # Save forecast CSV to artifact registry path
                    forecast_csv_path = artifacts["forecast_csv"]
                    forecast_saved = _save_artifact_csv(future_df, forecast_csv_path)
                    if forecast_saved:
                        st.success(f"Forecast CSV saved to: `{forecast_csv_path}`")
                    st.success(f"Forecast plot saved to: `{plot_path}`")

                    # Generate forecast PDF
                    forecast_pdf_path = artifacts["forecast_report_pdf"]
                    forecast_pdf_path.parent.mkdir(parents=True, exist_ok=True)
                    generate_forecast_pdf(
                        forecast_result,
                        str(forecast_pdf_path),
                        dataset_name=uploaded_file.name,
                        target_column=target_column,
                        date_column=date_column,
                        forecast_plot_path=plot_path
                    )
                    st.success(f"Forecast PDF report saved to: `{forecast_pdf_path}`")

                    # ── Update metadata immediately ──
                    update_metadata(artifacts["base_dir"], {
                        "task_type": "forecasting",
                        "target_column": target_column,
                        "forecast_enabled": True,
                        "forecast_horizon": horizon,
                        "forecast_pdf": "forecasts/forecast_report.pdf"
                    })

                    # ── Event-driven BQ sync for forecast ──
                    if cloud_sync and forecast_saved:
                        _sync_artifact_to_bq(
                            future_df, forecast_csv_path, "forecasts", "forecast",
                            st.session_state.run_id, uploaded_file.name,
                            "forecasting", target_column,
                            artifacts["base_dir"]
                        )

                except Exception as e:
                    st.error(f"Forecasting failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        elif forecast_suit["suitability"] == "Low":
            st.markdown("---")
            st.subheader("📈 Forecasting Engine")
            st.warning(
                f"**Forecasting is unavailable for this dataset.**\n\n"
                f"Reason: {forecast_suit.get('reason', 'No valid datetime column was detected.')}\n\n"
                f"To enable forecasting, ensure your dataset contains a column with datetime values "
                f"and that the selected target column is numeric."
            )
