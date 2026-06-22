# AI EDA Platform

AI EDA Platform is a lightweight Streamlit application that performs automated exploratory data analysis (EDA), dataset profiling, cleaning, simple model training, predictions, and time-series forecasting. It is designed as a modular, easy-to-extend codebase for data practitioners who want a reproducible and reportable data exploration workflow.

## Features

- Upload CSV datasets and generate a structured dataset profile
- Automated cleaning: duplicate removal, numeric/date conversions, missing value imputation
- Visual EDA: missing-value charts, histograms, boxplots, and correlation heatmaps (Plotly)
- Simple model training (scikit-learn) with automatic preprocessing pipelines
- Prediction engine to run trained models on new datasets
- Time-series forecasting using Prophet (with Holt-Winters fallback)
- PDF report generation (ReportLab) and run history management
- BigQuery integration: stubbed connector (coming soon)

## Architecture Overview

- `app.py` — Streamlit application entry point and UI
- `modules/` — Core application logic separated into small helper modules (profiling, cleaning, EDA, modeling, reports)
- `reports/` — Timestamped run folders containing PDFs, exported plots, and JSON reports
- `datasets/` — Uploaded datasets (excluded from the repository `.gitignore`)
- `models/` — Persisted model artifacts generated during training

## Folder Structure

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for a detailed layout and descriptions.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/<your-org>/ai-eda-platform.git
cd ai-eda-platform
```

2. Create and activate a Python virtual environment (recommended):

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. (Optional) Provide Google Cloud credentials if you plan to enable BigQuery integration later. See `.env.example`.

## Running the Application

```bash
streamlit run app.py
```

Open the Streamlit UI in your browser (usually at `http://localhost:8501`).

## Sample Workflow

1. Upload a CSV dataset using the sidebar uploader.
2. Review the Dataset Overview, Data Type Summary, and Detected Patterns.
3. Inspect Cleaning Summary and run EDA visualizations.
4. Select a target column and use the Model Training panel to train baseline models.
5. Use the Prediction Engine to score new data or the Forecasting Engine for time-series forecasts.
6. Generated PDFs, exported plots, and JSON reports are saved under `reports/<dataset>_<timestamp>/`.

## Generated Reports

- `eda_report.pdf` — Dataset profile, EDA figures, and business insights
- `training_report.pdf` — Model training summary, metrics, and feature importance
- `prediction_report.pdf` — Prediction summaries and sample outputs

## Prediction Engine

Load a trained model (saved into the run's `models/` folder) and run predictions on the current cleaned dataset or a new CSV. The predictor enforces a strict feature schema to maintain consistency with training.

## Forecasting Engine

The forecasting workflow looks for a date column and a numeric target. Prophet is used by default; if Prophet is unavailable or fails, a Holt-Winters fallback is attempted.

## Run History Dashboard

The app provides a Run History dashboard that lists previous runs, generated PDFs, and metadata stored in each run folder.

## BigQuery Integration (Coming Soon)

A stubbed `modules/bigquery_connector.py` exists as a placeholder. When enabled, the application will support uploading datasets, predictions, and metrics to BigQuery.

## Future Enhancements

- User authentication and role-based access to reports
- Interactive column-level drill-down and profiling
- Expandable model zoo with hyperparameter tuning and cross-validation
- CI checks and automated PDF validation

## License

This project is released under the MIT License — see [LICENSE](LICENSE).

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.
