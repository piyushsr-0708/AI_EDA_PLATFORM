import pandas as pd

def sync_to_bigquery(dataset_path: str, metrics_path: str, run_id: str, dataset_name: str, task_type: str, target_column: str) -> dict:
    """Syncs artifacts to BigQuery. Returns status."""
    try:
        import os
        from modules import bigquery_connector
        client = bigquery_connector.connect_bigquery()
        if not client:
            return {"error": "Failed to connect to BigQuery."}
            
        dataset_id = os.getenv("BIGQUERY_DATASET", "ai_eda_platform")
        
        # Sync dataset
        df = pd.read_csv(dataset_path)
        res1 = bigquery_connector.append_dataframe(
            client, dataset_id, "cleaned_datasets", df, run_id, dataset_name, task_type, target_column
        )
        
        # Sync metrics
        metrics_df = pd.read_csv(metrics_path)
        res2 = bigquery_connector.append_dataframe(
            client, dataset_id, "metrics", metrics_df, run_id, dataset_name, task_type, target_column
        )
        
        return {"status": "success", "dataset_upload": res1, "metrics_upload": res2}
    except Exception as e:
        return {"error": str(e)}
