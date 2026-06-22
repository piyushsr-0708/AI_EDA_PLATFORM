"""Run Manager module for generating and managing timestamped run folders."""

from datetime import datetime
from pathlib import Path
import re
from typing import Dict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

def create_run_folder(dataset_name: str) -> Dict[str, Path]:
    """Create a unique run folder with subdirectories.
    
    Args:
        dataset_name: Original name of the uploaded dataset file.
        
    Returns:
        Dictionary of created Path objects for each required subdirectory.
    """
    # Sanitize dataset name
    safe_name = Path(dataset_name).stem
    safe_name = re.sub(r"[^\w\-]", "_", safe_name)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    run_folder_name = f"{safe_name}_{timestamp}"
    base_path = _PROJECT_ROOT / "reports" / run_folder_name
    
    # Define subdirectories
    paths = {
        "base": base_path,
        "plots": base_path / "plots",
        "predictions": base_path / "predictions",
        "models": base_path / "models",
        "forecasts": base_path / "forecasts",
        "feature_importance": base_path / "feature_importance",
    }
    
    # Create all directories
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
        
    return run_folder_name, paths


def init_metadata(run_folder_path: Path, run_id: str, dataset_name: str) -> None:
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
        "forecast_pdf": None
    }
    
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
