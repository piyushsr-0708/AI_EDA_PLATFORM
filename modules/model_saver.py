"""Utilities to save and load trained models using joblib.

Provides simple function-based API to persist sklearn pipelines or other
picklable Python objects into the project's `models/` directory.
"""
import warnings
from pathlib import Path
from typing import Any, Union

import joblib

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _models_dir_path(models_dir: str = "models") -> Path:
    """Return the Path to the models directory, creating it if needed."""
    path = (_PROJECT_ROOT / models_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_model(model: Any, model_name: str, output_dir: Union[str, Path] = None, feature_schema: dict = None) -> str:
    """Save a trained model to the `models/` directory (or specified output_dir) using joblib.

    Args:
        model: A picklable Python object (e.g., sklearn Pipeline) to save.
        model_name: Short name for the model file (without extension).
        output_dir: Optional custom output directory.
        feature_schema: Optional dictionary describing the model features to be saved as JSON.

    Returns:
        The POSIX string path to the saved file.

    Raises:
        ValueError: if `model_name` is empty, invalid, or contains path traversal sequences.
        OSError: if saving fails due to filesystem issues.
    """
    if not model_name or not isinstance(model_name, str):
        raise ValueError("`model_name` must be a non-empty string")

    safe_name = Path(model_name).name
    if safe_name != model_name:
        raise ValueError("`model_name` must not contain path separators or traversal sequences")

    if output_dir is not None:
        models_path = Path(output_dir).resolve()
        models_path.mkdir(parents=True, exist_ok=True)
    else:
        models_path = _models_dir_path("models")
        
    file_path = models_path / f"{model_name}.joblib"

    if file_path.exists():
        warnings.warn(f"Overwriting existing model at {file_path}")

    try:
        joblib.dump(model, file_path)
    except Exception as e:
        raise OSError(f"Failed to save model to {file_path}: {e}")
        
    if feature_schema is not None:
        import json
        schema_path = models_path / "feature_schema.json"
        try:
            with open(schema_path, "w") as f:
                json.dump(feature_schema, f, indent=4)
        except Exception as e:
            warnings.warn(f"Failed to save feature schema to {schema_path}: {e}")
            
    return file_path.as_posix()


def load_model(model_path: Union[str, Path]):
    """Load a model from disk using joblib.

    Args:
        model_path: Path to the saved model file. Can be a string or Path.

    Returns:
        The loaded Python object.

    Raises:
        ValueError: If the path escapes the allowed models directory.
        FileNotFoundError: If the model file does not exist.
        OSError: If loading fails.
    """
    path = Path(model_path).resolve()
    # Allow loading from any directory within the project
    allowed_dir = _PROJECT_ROOT.resolve()
    if not str(path).startswith(str(allowed_dir)):
        raise ValueError(f"Model path escapes the allowed project directory: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    try:
        return joblib.load(path)
    except Exception as e:
        raise OSError(f"Failed to load model from {path}: {e}")

