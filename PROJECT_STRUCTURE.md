AI_EDA_PLATFORM/ — Project structure and module descriptions

Root files
- `app.py`: Streamlit application entrypoint and UI orchestration.
- `requirements.txt`: Python dependency manifest used to install packages.
- `README.md`: Project overview and usage instructions.
- `LICENSE`: MIT license for the repository.
- `.env.example`: Template environment variables for optional BigQuery integration.

Top-level folders
- `modules/`: Core application modules. Key modules:
  - `profiler.py`: Builds a dataset profile (rows, columns, missing, types, cardinality, etc.).
  - `cleaner.py`: Cleans datasets (duplicate removal, numeric/date conversions, imputation) and returns a cleaning report.
  - `eda.py`: Plotly-based EDA utilities (histograms, boxplots, correlation heatmaps, missing values visualization).
  - `model_trainer.py`: Simple training pipelines using scikit-learn and automatic preprocessing.
  - `model_saver.py`: Persist/load models using `joblib` and manage feature schema JSON.
  - `pdf_report_generator.py`: Build PDF reports using ReportLab.
  - `predictor.py`: Prepare and validate data for predictions and run the trained models.
  - `forecasting.py`: Time-series forecasting using Prophet with statsmodels fallback.
  - `bigquery_connector.py`: Stub for future BigQuery integration (no cloud calls currently).
  - `plot_exporter.py`: Export Plotly figures to disk for inclusion in reports.
  - `run_manager.py` & `run_history.py`: Create run folders and list past runs.

- `reports/`: Timestamped run folders (created when performing an analysis). Each run contains exports, PDFs, and a `metadata.json` file.
- `datasets/`: Uploaded CSV files (ignored by .gitignore).
- `models/`: Saved model artifacts and feature schema files (joblib + JSON).
- `credentials/`: Local credentials (e.g., `service_account.json`) — MUST NOT be committed.
- `forecasts/`: Forecast outputs saved by the forecasting engine.

Notes
- The code is organized to keep display logic (`app.py`) separated from processing logic (`modules/`).
- BigQuery support is currently a stub; production use will require adding the `google-cloud-bigquery` client and appropriate credentials management.
