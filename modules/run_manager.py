"""Run Manager module for generating and managing timestamped run folders."""

from datetime import datetime
from pathlib import Path
import re
from typing import Dict
import os

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Reports root can be relocated outside the repository by setting the
# `AI_EDA_REPORTS_DIR` environment variable. When unset, default to
# a safe external path to avoid triggering Streamlit file-watchers inside
# the project directory.
REPORTS_ROOT = Path(
    os.getenv(
        "AI_EDA_REPORTS_DIR",
        "D:/AI_EDA_PLATFORM_REPORTS",
    )
)

def create_run_folder(dataset_name: str) -> Dict[str, Path]:
    """Create a unique run folder with subdirectories.
    
    Args:
        dataset_name: Original name of the uploaded dataset file.
        
    Returns:
        Tuple containing (run_id: str, artifacts: Dict[str, Path])
    """
    # Sanitize dataset name
    safe_name = Path(dataset_name).stem
    safe_name = re.sub(r"[^\w\-]", "_", safe_name)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    run_folder_name = f"{safe_name}_{timestamp}"
    # Use configurable reports root for runtime artifacts to avoid
    # polluting the repository workspace and triggering file watchers.
    base_path = REPORTS_ROOT / run_folder_name
    
    # Define the centralized artifacts registry
    artifacts = {
        "base_dir": base_path,
        "uploaded_dataset": base_path / "uploaded" / "uploaded_dataset.csv",
        "cleaned_dataset": base_path / "cleaned" / "cleaned_dataset.csv",
        "metrics_csv": base_path / "metrics" / "metrics.csv",
        "feature_importance_csv": base_path / "feature_importance" / "feature_importance.csv",
        "feature_importance_png": base_path / "feature_importance" / "feature_importance.png",
        "predictions_csv": base_path / "predictions" / "predictions.csv",
        "prediction_report_pdf": base_path / "predictions" / "prediction_report.pdf",
        "forecast_csv": base_path / "forecasts" / "forecast.csv",
        "forecast_report_pdf": base_path / "forecasts" / "forecast_report.pdf",
        "eda_report_pdf": base_path / "reports" / "eda_report.pdf",
        "training_report_pdf": base_path / "reports" / "training_report.pdf",
        "best_model_joblib": base_path / "models" / "best_model.joblib",
        "feature_schema_json": base_path / "models" / "feature_schema.json",
        "plots_dir": base_path / "plots",
        "metadata_json": base_path / "metadata.json",
        "dataset_report_json": base_path / "dataset_report.json",
        "model_report_json": base_path / "model_report.json"
    }
        
    return run_folder_name, artifacts


def init_metadata(run_folder_path: Path, run_id: str, dataset_name: str, artifacts: Dict[str, Path]) -> None:
    """Initialize a metadata.json file in the run folder."""
    import json
    metadata = {
        "run_id": run_id,
        "dataset_name": dataset_name,
        "run_timestamp": datetime.now().isoformat(),
        "task_type": None,
        "target_column": None,
        "best_model": None,
        "prediction_count": 0,
        "forecast_enabled": False,
        "forecast_horizon": None,
        "eda_pdf": None,
        "training_pdf": None,
        "prediction_pdf": None,
        "forecast_pdf": None,
        "artifacts": {k: str(v) for k, v in artifacts.items()}
    }
    
    # Ensure base directory exists before writing metadata
    run_folder_path.mkdir(parents=True, exist_ok=True)
    
    metadata_path = run_folder_path / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def update_metadata(run_folder_path: Path, updates: Dict) -> None:
    """Update specific fields in metadata.json."""
    import json
    metadata_path = run_folder_path / "metadata.json"
    
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
        metadata.update(updates)
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
