import json
from pathlib import Path
from typing import List, Dict
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_run_history() -> pd.DataFrame:
    """Scans the reports directory and returns a DataFrame of run history based on metadata.json."""
    reports_dir = _PROJECT_ROOT / "reports"
    if not reports_dir.exists():
        return pd.DataFrame()

    runs = []
    for run_folder in reports_dir.iterdir():
        if not run_folder.is_dir():
            continue
            
        metadata_path = run_folder / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    
                run_id = metadata.get("run_id", run_folder.name)
                # Format timestamp for better readability if possible
                ts = metadata.get("run_timestamp", "")
                if ts:
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(ts)
                        ts = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        pass
                        
                runs.append({
                    "Run ID": run_id,
                    "Dataset": metadata.get("dataset_name", "Unknown"),
                    "Task": metadata.get("task_type") or "Pending",
                    "Model": metadata.get("best_model") or "Pending",
                    "Timestamp": ts,
                    "EDA PDF": metadata.get("eda_pdf"),
                    "Training PDF": metadata.get("training_pdf"),
                    "Prediction PDF": metadata.get("prediction_pdf"),
                    "Forecast PDF": metadata.get("forecast_pdf"),
                    "_folder_path": str(run_folder)  # Keep for internal use
                })
            except Exception as e:
                print(f"Failed to read metadata for {run_folder.name}: {e}")
                continue

    if runs:
        df = pd.DataFrame(runs)
        df = df.sort_values(by="Timestamp", ascending=False).reset_index(drop=True)
        return df
    
    return pd.DataFrame()
