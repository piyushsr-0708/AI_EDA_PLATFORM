"""Report generation helpers.

Functions to build simple dataset and model reports and persist them as JSON
files under the `reports/` directory. Designed to be small, dependency-light,
and robust to common input types from the profiling and cleaning utilities.
"""

from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any, Dict, Union

import numpy as np
import pandas as pd

__all__ = [
    "generate_dataset_report",
    "generate_model_report",
    "save_report"
]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _safe_serialize(obj: Any) -> Any:
    """Convert common Python/Pandas/Numpy objects into JSON-serializable forms."""

    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, datetime):
        return obj.isoformat()

    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    if isinstance(obj, dict):
        return {
            k: _safe_serialize(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [
            _safe_serialize(i)
            for i in obj
        ]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, pd.DataFrame):
        try:
            return _safe_serialize(
                obj.to_dict(orient="list")
            )
        except (TypeError, ValueError):
            return _safe_serialize(
                obj.to_dict()
            )

    if isinstance(obj, pd.Series):
        return _safe_serialize(
            obj.to_dict()
        )

    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return type(obj).__name__


def generate_dataset_report(
    profile: Any,
    cleaning_report: Any
) -> Dict[str, Any]:
    """Generate a dataset report."""

    report: Dict[str, Any] = {
        "report_version": "1.0",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat()
    }

    try:
        report["profile_summary"] = _safe_serialize(
            profile
        )
    except Exception as e:
        report["profile_summary"] = {
            "error": f"Could not serialize profile: {e}"
        }

    try:
        report["cleaning_summary"] = _safe_serialize(
            cleaning_report
        )
    except Exception as e:
        report["cleaning_summary"] = {
            "error": f"Could not serialize cleaning_report: {e}"
        }

    return report


def generate_model_report(
    problem_type: str,
    best_model_name: str,
    metrics: Any,
    feature_importance: Any = None
) -> Dict[str, Any]:
    """Generate a model report."""

    if not problem_type or not isinstance(
        problem_type,
        str
    ):
        raise ValueError(
            "problem_type must be a non-empty string"
        )

    if not best_model_name or not isinstance(
        best_model_name,
        str
    ):
        raise ValueError(
            "best_model_name must be a non-empty string"
        )

    report: Dict[str, Any] = {
        "report_version": "1.0",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "problem_type": problem_type,
        "best_model_name": best_model_name,
    }

    if feature_importance:
        report["feature_importance"] = _safe_serialize(feature_importance)

    try:
        report["metrics"] = _safe_serialize(
            metrics
        )
    except Exception as e:
        report["metrics"] = {
            "error": f"Could not serialize metrics: {e}"
        }

    try:
        if (
            isinstance(metrics, dict)
            and best_model_name in metrics
        ):
            report["best_model_metrics"] = (
                _safe_serialize(
                    metrics[best_model_name]
                )
            )
    except Exception:
        report["best_model_metrics"] = None

    return report


def save_report(
    report_data: Dict[str, Any],
    report_name: str,
    reports_dir: Union[str, Path] = "reports"
) -> str:
    """Save report as JSON and return file path."""

    if (
        not report_name
        or not isinstance(report_name, str)
    ):
        raise ValueError(
            "report_name must be a non-empty string"
        )

    safe_name = Path(report_name).name

    if safe_name != report_name:
        raise ValueError(
            "report_name must not contain path separators"
        )

    reports_dir_path = Path(reports_dir)
    if reports_dir_path.is_absolute():
        reports_path = reports_dir_path.resolve()
    else:
        reports_path = (_PROJECT_ROOT / reports_dir_path).resolve()

    reports_path.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path = reports_path / f"{report_name}.json"

    serializable = _safe_serialize(
        report_data
    )

    try:
        with open(
            file_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                serializable,
                f,
                indent=2,
                ensure_ascii=False
            )

    except Exception as e:
        raise OSError(
            f"Failed to save report to {file_path}: {e}"
        ) from e

    return file_path.as_posix()