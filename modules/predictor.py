"""Prediction Engine Module.

Provides robust validation and execution for generating predictions
from trained models, adhering to strict schema validation.
"""
import json
from pathlib import Path
import pandas as pd
from typing import Tuple, Any

from modules.model_saver import load_model as ms_load_model

class PredictionValidationError(Exception):
    """Exception raised when a dataset fails schema validation."""
    pass

def load_predictor(models_dir: Path, model_name: str = "best_model") -> Tuple[Any, dict]:
    """Load the trained model and its associated feature schema.
    
    Args:
        models_dir: Path to the models directory of a specific run.
        model_name: Name of the model file without extension.
        
    Returns:
        Tuple of (Loaded Model Pipeline, Feature Schema Dict).
    """
    model_path = models_dir / f"{model_name}.joblib"
    schema_path = models_dir / "feature_schema.json"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not schema_path.exists():
        raise FileNotFoundError(f"Feature schema not found: {schema_path}. Re-train the model to generate it.")
        
    model = ms_load_model(model_path)
    with open(schema_path, "r") as f:
        schema = json.load(f)
        
    return model, schema

def prepare_prediction_data(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Validate and prepare the DataFrame for prediction.
    
    Implements strict Validation Policy:
    1. Rejects if missing required columns.
    2. Ignores extra columns.
    3. Reorders to match training schema exactly.
    
    Args:
        df: The raw input DataFrame.
        schema: The feature schema dictionary.
        
    Returns:
        A precisely formatted DataFrame ready for prediction.
        
    Raises:
        PredictionValidationError: If any required feature columns are missing.
    """
    required_cols = schema.get("feature_columns", [])
    
    if not required_cols:
        raise ValueError("Feature schema is missing 'feature_columns'.")
        
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        missing_str = "\n* ".join(missing_cols)
        raise PredictionValidationError(
            f"Prediction file is missing required columns:\n\n* {missing_str}\n\n"
            "Please upload a dataset containing all training features used during model training."
        )
        
    # Ignore extra columns and strictly restore the original column order
    return df[required_cols].copy()

def generate_predictions(model, prepared_df: pd.DataFrame) -> pd.Series:
    """Generate predictions using the prepared DataFrame.
    
    Args:
        model: The trained scikit-learn Pipeline.
        prepared_df: The validated feature DataFrame.
        
    Returns:
        A pandas Series containing the predictions.
    """
    predictions = model.predict(prepared_df)
    return pd.Series(predictions, index=prepared_df.index, name="Prediction")
