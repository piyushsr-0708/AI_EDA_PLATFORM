"""BigQuery Connector Module.

Provides convenience helpers to upload run metadata, dataset profiles, and
training metrics to Google BigQuery. Functions are defensive and will raise
or return structured results. The app integrates these helpers but will
never allow upload failures to abort core workflows.

Configuration:
- `GOOGLE_APPLICATION_CREDENTIALS` environment variable is honored by the
  google-cloud-bigquery client. Optionally set `GCP_PROJECT_ID` and
  `BIGQUERY_DATASET` in the environment (or via a .env file).
"""
from typing import Dict, Any, Optional, List
import os
from datetime import datetime

try:
    from google.cloud import bigquery
    from google.api_core.exceptions import NotFound
except Exception:
    bigquery = None  # type: ignore
    NotFound = Exception  # fallback


DEFAULT_DATASET = os.getenv("BIGQUERY_DATASET", "ai_eda_platform")
DEFAULT_PROJECT = os.getenv("GCP_PROJECT_ID")


def _is_available() -> bool:
    return bigquery is not None


def connect_bigquery(project_id: Optional[str] = None, credentials_path: Optional[str] = None):
    """Create and return a BigQuery client.

    Args:
        project_id: Optional GCP project id. Falls back to env var.
        credentials_path: Optional credentials path — respected by client if
            the environment variable `GOOGLE_APPLICATION_CREDENTIALS` is set.

    Returns:
        google.cloud.bigquery.Client instance.

    Raises:
        RuntimeError if the google-cloud-bigquery package is not installed.
    """
    if not _is_available():
        raise RuntimeError("google-cloud-bigquery is not available. Install the package to enable BigQuery uploads.")

    proj = project_id or DEFAULT_PROJECT
    # if a credentials path is provided, set env var so the client will use it
    if credentials_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)

    client = bigquery.Client(project=proj) if proj else bigquery.Client()
    return client


def create_table_if_not_exists(client, dataset_id: str, table_id: str, schema: List[bigquery.SchemaField]):
    """Ensure a dataset and table exist with the provided schema.

    Args:
        client: BigQuery client
        dataset_id: Dataset name (not fully-qualified)
        table_id: Table name (not fully-qualified)
        schema: List of bigquery.SchemaField
    """
    dataset_ref = client.dataset(dataset_id)
    try:
        client.get_dataset(dataset_ref)
    except NotFound:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = os.getenv("BIGQUERY_LOCATION", "US")
        client.create_dataset(dataset, exists_ok=True)

    table_ref = dataset_ref.table(table_id)
    try:
        client.get_table(table_ref)
    except NotFound:
        table = bigquery.Table(table_ref, schema=schema)
        client.create_table(table)


def _insert_rows(client, dataset_id: str, table_id: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Insert rows into a BigQuery table and return a status dict."""
    table_ref = f"{client.project}.{dataset_id}.{table_id}"
    errors = client.insert_rows_json(table_ref, rows)
    if errors:
        return {"status": "error", "errors": errors}
    return {"status": "ok", "inserted": len(rows)}


def upload_run_metadata(client, dataset_id: str, row: Dict[str, Any]):
    """Upload a single run_metadata row.

    Expected keys in `row`: run_id, dataset_name, target_column, task_type, best_model, run_timestamp
    """
    schema = [
        bigquery.SchemaField("run_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("dataset_name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("target_column", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("task_type", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("best_model", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("run_timestamp", "TIMESTAMP", mode="NULLABLE"),
    ]
    create_table_if_not_exists(client, dataset_id, "run_metadata", schema)
    return _insert_rows(client, dataset_id, "run_metadata", [row])


def upload_dataset_profile(client, dataset_id: str, row: Dict[str, Any]):
    """Upload a single dataset_profiles row.

    Expected keys: run_id, dataset_name, rows, columns, missing_values, duplicate_rows, timestamp
    """
    schema = [
        bigquery.SchemaField("run_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("dataset_name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("rows", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("columns", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("missing_values", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("duplicate_rows", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("timestamp", "TIMESTAMP", mode="NULLABLE"),
    ]
    create_table_if_not_exists(client, dataset_id, "dataset_profiles", schema)
    return _insert_rows(client, dataset_id, "dataset_profiles", [row])


def upload_training_metrics(client, dataset_id: str, row: Dict[str, Any]):
    """Upload a single training_metrics row.

    Expected keys: run_id, task_type, target_column, best_model, accuracy, f1, precision, recall, r2, rmse, mae, timestamp
    """
    schema = [
        bigquery.SchemaField("run_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("task_type", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("target_column", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("best_model", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("accuracy", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("f1", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("precision", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("recall", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("r2", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("rmse", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("mae", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("timestamp", "TIMESTAMP", mode="NULLABLE"),
    ]
    create_table_if_not_exists(client, dataset_id, "training_metrics", schema)
    return _insert_rows(client, dataset_id, "training_metrics", [row])

