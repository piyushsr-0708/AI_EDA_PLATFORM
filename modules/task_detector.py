import pandas as pd

def detect_task(df, profile, target_column):
    target = df[target_column]

    all_date_cols = set(profile.get('date_candidates', []) + profile.get('datetime_cols', []))

    if target_column in all_date_cols:
        return 'unknown'

    non_target_cols = [c for c in df.columns if c != target_column]
    if (
        len(all_date_cols) >= 1
        and target_column not in all_date_cols
        and pd.api.types.is_numeric_dtype(target)
        and all(c in all_date_cols for c in non_target_cols)
    ):
        return 'forecasting'

    unique_ratio = target.nunique() / len(target)
    if target.dtype == 'object' and target_column not in profile.get('high_cardinality', []):
        return 'classification'
    if pd.api.types.is_numeric_dtype(target):
        if target.nunique() <= 20:
            return 'classification'
        return 'regression'

    return 'unknown'


def get_task_metadata(df, profile, target_column, detected_task):
    """Return task detection metadata for UI and reporting only.
    
    Does NOT alter the behavior of detect_task(). This function is purely
    informational and should never be used in training or prediction logic.
    
    Args:
        df: The cleaned DataFrame.
        profile: The dataset profile dictionary.
        target_column: The selected target column name.
        detected_task: The string returned by detect_task().
        
    Returns:
        Dictionary with 'confidence' and 'reason' keys.
    """
    target = df[target_column]
    nunique = target.nunique()
    n_rows = len(target)
    
    if detected_task == "classification":
        if target.dtype == "object":
            reason = f"Target column '{target_column}' is categorical with {nunique} unique values."
            confidence = "High"
        elif nunique <= 5:
            reason = f"Target column '{target_column}' contains only {nunique} unique numeric values, treated as class labels."
            confidence = "High"
        elif nunique <= 20:
            reason = f"Target column '{target_column}' contains {nunique} unique values and was classified using the low-cardinality rule (≤20 unique)."
            confidence = "Medium"
        else:
            reason = f"Target column '{target_column}' was classified as categorical."
            confidence = "Medium"
            
    elif detected_task == "regression":
        reason = f"Target column '{target_column}' is numeric with {nunique} unique values out of {n_rows} rows."
        if nunique / n_rows > 0.5:
            confidence = "High"
        else:
            confidence = "Medium"
            
    elif detected_task == "forecasting":
        reason = f"Target column '{target_column}' is numeric and the dataset structure indicates a time-series."
        confidence = "Medium"
        
    else:
        reason = f"Could not determine a supported task type for target '{target_column}'."
        confidence = "Low"
    
    return {
        "confidence": confidence,
        "reason": reason
    }


def assess_forecast_suitability(df, target_column, profile):
    """Assess whether the dataset is suitable for time-series forecasting.
    
    Separate from detect_task(). Evaluates datetime columns, numeric targets,
    and chronological observation counts.
    
    Args:
        df: The cleaned DataFrame.
        target_column: The selected target column name.
        profile: The dataset profile dictionary.
        
    Returns:
        Dictionary with 'suitability' ('High', 'Medium', 'Low') and 'reason'.
    """
    target = df[target_column]
    
    # Check if target is numeric
    if not pd.api.types.is_numeric_dtype(target):
        return {
            "suitability": "Low",
            "reason": f"Target column '{target_column}' is not numeric. Forecasting requires a numeric target."
        }
    
    # Find datetime columns
    all_date_cols = list(set(
        profile.get('date_candidates', []) + profile.get('datetime_cols', [])
    ))
    # Exclude target from datetime candidates
    all_date_cols = [c for c in all_date_cols if c != target_column]
    
    if not all_date_cols:
        return {
            "suitability": "Low",
            "reason": "No datetime column detected in the dataset. Forecasting requires at least one datetime column."
        }
    
    # Pick the best datetime column (most unique chronological values)
    best_date_col = None
    best_unique_count = 0
    
    for col in all_date_cols:
        try:
            dt_series = pd.to_datetime(df[col], errors='coerce')
            unique_count = dt_series.dropna().nunique()
            if unique_count > best_unique_count:
                best_unique_count = unique_count
                best_date_col = col
        except Exception:
            continue
    
    if best_date_col is None or best_unique_count < 10:
        return {
            "suitability": "Low",
            "reason": "Datetime information exists but does not appear to represent a continuous time series."
        }
    
    # Assess suitability based on observation count
    reasons = [
        f"Datetime column detected: {best_date_col}",
        f"Numeric target selected: {target_column}",
        f"{best_unique_count} chronological observations available"
    ]
    reason_str = ". ".join(reasons) + "."
    
    if best_unique_count >= 100:
        return {"suitability": "High", "reason": reason_str, "date_column": best_date_col}
    elif best_unique_count >= 30:
        return {"suitability": "Medium", "reason": reason_str, "date_column": best_date_col}
    else:
        return {"suitability": "Low", "reason": reason_str, "date_column": best_date_col}