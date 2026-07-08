"""BigQuery Connector Module.

Provides convenience helpers to upload run metadata, dataset profiles, and
training metrics to Google BigQuery. Functions are defensive and will raise
or return structured results. The app integrates these helpers but will
never allow upload failures to abort core workflows.

Configuration:
`GOOGLE_APPLICATION_CREDENTIALS` environment variable is honored by the
  google-cloud-bigquery client. Optionally set `GCP_PROJECT_ID` and
  `BIGQUERY_DATASET` in the environment (or via a .env file).
"""
from typing import Dict, Any, Optional, List, Sequence
import hashlib
import os
from datetime import datetime
import json
from pathlib import Path

import pandas as pd

try:
    from google.cloud import bigquery
    from google.api_core.exceptions import NotFound
except Exception:
    bigquery = None  # type: ignore
    NotFound = Exception  # fallback

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    from google.oauth2 import service_account
except Exception:
    service_account = None


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


    # Resolve credentials according to priority:
    # 1) GOOGLE_APPLICATION_CREDENTIALS env var
    # 2) .env file (loaded into env)
    # 3) credentials/service_account.json in project root

    # If a credentials_path argument is provided, prefer that.
    cred_path = None
    if credentials_path:
        cred_path = Path(credentials_path)

    # 1) env var
    if not cred_path:
        env_cred = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if env_cred:
            p = Path(env_cred)
            if p.exists():
                cred_path = p

    # 2) .env file
    if not cred_path and load_dotenv is not None:
        # load .env from current working directory (project root)
        load_dotenv()
        env_cred = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if env_cred:
            p = Path(env_cred)
            if p.exists():
                cred_path = p

    # 3) credentials/service_account.json inside project root
    if not cred_path:
        candidate = Path.cwd() / "credentials" / "service_account.json"
        if candidate.exists():
            cred_path = candidate

    if not cred_path:
        raise RuntimeError(
            "No service account credentials found. Set the GOOGLE_APPLICATION_CREDENTIALS env var, add it to a .env file, or place credentials/service_account.json in the project root."
        )

    if service_account is None:
        raise RuntimeError("google.oauth2.service_account is required but not installed. Install 'google-auth' to enable explicit credentials handling.")

    # Load project id preference
    proj = project_id or os.getenv("GCP_PROJECT_ID") or DEFAULT_PROJECT

    # Attempt to extract project_id from the credentials JSON if still missing
    if not proj:
        try:
            with open(cred_path, "r", encoding="utf-8") as f:
                j = json.load(f)
            proj = j.get("project_id")
        except Exception:
            proj = None

    if not proj:
        raise RuntimeError(
            "GCP project id not found. Set GCP_PROJECT_ID env var or include 'project_id' in the service account JSON."
        )

    # Create credentials and client explicitly
    try:
        creds = service_account.Credentials.from_service_account_file(str(cred_path))
    except Exception as e:
        raise RuntimeError(f"Failed to load service account credentials from {cred_path}: {e}")

    try:
        client = bigquery.Client(project=proj, credentials=creds)
    except Exception as e:
        raise RuntimeError(f"Failed to construct BigQuery client for project '{proj}': {e}")

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


def _normalize_row_values(row: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {}
    for key, value in row.items():
        if isinstance(value, datetime):
            normalized[key] = value.isoformat()
        elif isinstance(value, (int, float, str, bool)) or value is None:
            normalized[key] = value
        else:
            normalized[key] = str(value)
    return normalized


def _delete_by_run_id(client, dataset_id: str, table_id: str, run_id: str):
    """Delete existing records for a given run_id to ensure idempotency."""
    if not run_id:
        return
    query = f"DELETE FROM `{client.project}.{dataset_id}.{table_id}` WHERE run_id = '{run_id}'"
    try:
        # We use a short timeout as this is a simple deletion
        client.query(query).result(timeout=30)
    except Exception as e:
        # If the table doesn't exist or deletion fails, we just continue
        pass

def _insert_rows(client, dataset_id: str, table_id: str, rows: List[Dict[str, Any]], row_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """Insert rows into a BigQuery table and return a status dict."""
    if not rows:
        return {"status": "ok", "inserted": 0}
    table_ref = f"{client.project}.{dataset_id}.{table_id}"
    normalized_rows = [_normalize_row_values(row) for row in rows]
    
    # Try deleting the run_id first if present in the first row
    first_row_run_id = normalized_rows[0].get("run_id")
    if first_row_run_id:
        _delete_by_run_id(client, dataset_id, table_id, first_row_run_id)
        
    try:
        errors = client.insert_rows_json(table_ref, normalized_rows, row_ids=row_ids)
    except TypeError:
        # Fallback when insert_rows_json does not support row_ids in older client versions.
        errors = client.insert_rows_json(table_ref, normalized_rows)
    if errors:
        return {"status": "error", "errors": errors}
    return {"status": "ok", "inserted": len(rows)}


def _dataframe_to_dict_rows(df, keys: Sequence[str]) -> List[Dict[str, Any]]:
    rows = []
    for _, row in df.iterrows():
        mapped = {key: row.get(key, None) for key in keys}
        rows.append(_normalize_row_values(mapped))
    return rows


def _schema_from_dataframe(df: pd.DataFrame) -> List[bigquery.SchemaField]:
    schema: List[bigquery.SchemaField] = []
    for column in df.columns:
        dtype = df[column].dtype
        if pd.api.types.is_integer_dtype(dtype):
            field_type = "INT64"
        elif pd.api.types.is_float_dtype(dtype):
            field_type = "FLOAT64"
        elif pd.api.types.is_bool_dtype(dtype):
            field_type = "BOOL"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            field_type = "TIMESTAMP"
        else:
            field_type = "STRING"
        # Ensure fallback types are strings for safety against complex objects
        if field_type == "STRING" and column in df.columns:
            df[column] = df[column].astype(str)
        schema.append(bigquery.SchemaField(column, field_type, mode="NULLABLE"))
    return schema


def _hash_record(record: Dict[str, Any]) -> str:
    serialized = json.dumps(record, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


import re

def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize DataFrame column names for BigQuery compatibility."""
    df_bq = df.copy()
    new_cols = []
    for col in df_bq.columns:
        c = str(col)
        c = c.replace("R²", "r2")
        c = c.replace("%", "pct")
        c = c.lower()
        c = c.replace(" ", "_")
        c = c.replace("(", "").replace(")", "")
        c = re.sub(r'[^a-z0-9_]', '', c)
        c = re.sub(r'_+', '_', c)
        c = c.strip('_')
        new_cols.append(c)
    df_bq.columns = new_cols
    return df_bq


def append_dataframe(client, dataset_id: str, table_id: str, df: pd.DataFrame, 
                     run_id: str = None, dataset_name: str = None, 
                     task_type: str = None, target_column: str = None):
    """Append rows from a DataFrame into BigQuery with retries and metadata injection."""
    if df.empty:
        return {"status": "ok", "inserted": 0}
    
    df_bq = sanitize_column_names(df)
    
    # Inject mandatory metadata columns
    if run_id is not None: df_bq.insert(0, "run_id", run_id)
    if dataset_name is not None: df_bq.insert(1, "dataset_name", dataset_name)
    if task_type is not None: df_bq.insert(2, "task_type", task_type)
    if target_column is not None: df_bq.insert(3, "target_column", target_column)
    
    # Add created_at and app_version unconditionally
    from datetime import datetime
    df_bq["created_at"] = datetime.utcnow().isoformat()
    df_bq["app_version"] = "2.1"
    
    schema = _schema_from_dataframe(df_bq)
    create_table_if_not_exists(client, dataset_id, table_id, schema)
    rows = df_bq.to_dict(orient="records")
    unique_rows = []
    row_ids = []
    hashes = set()
    for row in rows:
        h = _hash_record(row)
        if h not in hashes:
            hashes.add(h)
            unique_rows.append(row)
            row_ids.append(h)
            
    import time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return _insert_rows(client, dataset_id, table_id, unique_rows, row_ids)
        except Exception as e:
            err_msg = str(e).lower()
            if "timeout" in err_msg or "connection reset" in err_msg or "quota" in err_msg or "unavailable" in err_msg or "rate limit" in err_msg:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
            return {"status": "error", "message": str(e)}


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
        bigquery.SchemaField("task_type", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("target_column", "STRING", mode="NULLABLE"),
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

