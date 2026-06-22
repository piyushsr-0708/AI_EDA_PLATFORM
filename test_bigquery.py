from google.cloud import bigquery
import pandas as pd
import os

# ------------------------------------------------------------------
# Credentials
# ------------------------------------------------------------------

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    "credentials/service_account.json"
)

# ------------------------------------------------------------------
# BigQuery Client
# ------------------------------------------------------------------

client = bigquery.Client()

print(f"Connected to Project: {client.project}")

# ------------------------------------------------------------------
# Test Data
# ------------------------------------------------------------------

df = pd.DataFrame({
    "test_col": [1, 2, 3],
    "message": ["Hello", "BigQuery", "Success"]
})

# ------------------------------------------------------------------
# Target Table
# ------------------------------------------------------------------

table_id = (
    "csv-bigquery-project-497608.ai_eda_platform.test_table"
)

print(f"Uploading to: {table_id}")

# ------------------------------------------------------------------
# Upload
# ------------------------------------------------------------------

job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE"
)

job = client.load_table_from_dataframe(
    df,
    table_id,
    job_config=job_config
)

job.result()

# ------------------------------------------------------------------
# Verification
# ------------------------------------------------------------------

table = client.get_table(table_id)

print("\nUpload Successful!")
print(f"Rows Loaded: {table.num_rows}")
print(f"Columns: {len(table.schema)}")
print(f"Table: {table.table_id}")