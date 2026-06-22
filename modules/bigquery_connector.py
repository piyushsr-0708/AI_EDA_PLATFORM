"""BigQuery Connector Module (Stub).

Provides architectural stubs for future Google BigQuery integration.
No cloud dependencies are required. All functions return informative
NotImplemented messages until the integration is activated.

Future integration will require:
    - google-cloud-bigquery
    - google-auth
    - A GCP project with BigQuery API enabled
"""
from typing import Dict, Any, Optional
import pandas as pd


_NOT_IMPLEMENTED_MSG = (
    "NotImplemented: BigQuery cloud integration is pending. "
    "This stub is an architectural placeholder. "
    "No cloud calls are made at this time."
)


def connect_bigquery(project_id: str = None, credentials_path: str = None) -> Dict[str, Any]:
    """Establish a connection to Google BigQuery.
    
    Args:
        project_id: GCP project ID.
        credentials_path: Path to service account JSON key file.
        
    Returns:
        Dictionary with connection status.
    """
    return {
        "status": "not_implemented",
        "message": _NOT_IMPLEMENTED_MSG,
        "project_id": project_id
    }


def upload_dataset(
    df: pd.DataFrame,
    table_id: str,
    dataset_name: str = "",
    if_exists: str = "replace"
) -> Dict[str, Any]:
    """Upload a dataset DataFrame to a BigQuery table.
    
    Args:
        df: The DataFrame to upload.
        table_id: Fully qualified BigQuery table ID (project.dataset.table).
        dataset_name: Human-readable dataset name for logging.
        if_exists: Behavior when table exists ('replace', 'append', 'fail').
        
    Returns:
        Dictionary with upload status.
    """
    return {
        "status": "not_implemented",
        "message": _NOT_IMPLEMENTED_MSG,
        "table_id": table_id,
        "row_count": len(df) if df is not None else 0
    }


def upload_predictions(
    predictions_df: pd.DataFrame,
    table_id: str,
    run_id: str = ""
) -> Dict[str, Any]:
    """Upload prediction results to a BigQuery table.
    
    Args:
        predictions_df: DataFrame containing predictions.
        table_id: Fully qualified BigQuery table ID.
        run_id: The run identifier for traceability.
        
    Returns:
        Dictionary with upload status.
    """
    return {
        "status": "not_implemented",
        "message": _NOT_IMPLEMENTED_MSG,
        "table_id": table_id,
        "run_id": run_id,
        "row_count": len(predictions_df) if predictions_df is not None else 0
    }


def upload_forecasts(
    forecast_df: pd.DataFrame,
    table_id: str,
    run_id: str = ""
) -> Dict[str, Any]:
    """Upload forecast results to a BigQuery table.
    
    Args:
        forecast_df: DataFrame containing forecast data.
        table_id: Fully qualified BigQuery table ID.
        run_id: The run identifier for traceability.
        
    Returns:
        Dictionary with upload status.
    """
    return {
        "status": "not_implemented",
        "message": _NOT_IMPLEMENTED_MSG,
        "table_id": table_id,
        "run_id": run_id,
        "row_count": len(forecast_df) if forecast_df is not None else 0
    }


def upload_metrics(
    metrics: Dict[str, Any],
    table_id: str,
    run_id: str = ""
) -> Dict[str, Any]:
    """Upload model training metrics to a BigQuery table.
    
    Args:
        metrics: Dictionary of model metrics (R2, RMSE, F1, etc.).
        table_id: Fully qualified BigQuery table ID.
        run_id: The run identifier for traceability.
        
    Returns:
        Dictionary with upload status.
    """
    return {
        "status": "not_implemented",
        "message": _NOT_IMPLEMENTED_MSG,
        "table_id": table_id,
        "run_id": run_id,
        "metric_keys": list(metrics.keys()) if metrics else []
    }
