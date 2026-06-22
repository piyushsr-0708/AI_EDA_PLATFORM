"""Business Insights Generator.

Rule-based generation of human-readable insights from dataset profiles,
cleaning reports, and model training metrics.
"""

from typing import Dict, Any, List

def generate_insights(
    profile: Dict[str, Any],
    cleaning_report: Dict[str, Any],
    model_results: Dict[str, Any] = None
) -> Dict[str, List[str]]:
    """Generate business insights based on the provided reports.
    
    Args:
        profile: The dataset profile dictionary.
        cleaning_report: The cleaning report dictionary.
        model_results: The model training results dictionary (optional).
        
    Returns:
        A dictionary with categories 'dataset', 'eda', and 'model' containing lists of insight strings.
    """
    insights = {
        "dataset": [],
        "eda": [],
        "model": []
    }
    
    # --- Dataset & Cleaning Insights ---
    rows = profile.get("rows", 0)
    cols = profile.get("columns", 0)
    if rows and cols:
        insights["dataset"].append(f"The dataset contains {rows} rows and {cols} columns.")
        
    duplicates_removed = cleaning_report.get("duplicates_removed", 0)
    if duplicates_removed > 0:
        insights["dataset"].append(f"Removed {duplicates_removed} duplicate rows to improve data quality.")
        
    total_missing = profile.get("total_missing", 0)
    if total_missing > 0:
        insights["dataset"].append(f"Detected {total_missing} missing values initially.")
        if cleaning_report.get("remaining_missing", 1) == 0:
             insights["dataset"].append("All missing values were successfully imputed.")
    else:
        insights["dataset"].append("The dataset is completely free of missing values.")
        
    high_card = profile.get("high_cardinality", [])
    if high_card:
        cols_str = ", ".join(high_card[:3]) + ("..." if len(high_card) > 3 else "")
        insights["dataset"].append(f"High-cardinality columns detected (many unique values): {cols_str}.")

    # --- EDA / Structure Insights ---
    num_cols = len(profile.get("numerical_columns", []))
    cat_cols = len(profile.get("categorical_columns", []))
    insights["eda"].append(f"Feature breakdown: {num_cols} numerical features and {cat_cols} categorical features.")
    
    binary_cols = profile.get("binary_columns", [])
    if binary_cols:
        cols_str = ", ".join(binary_cols[:3]) + ("..." if len(binary_cols) > 3 else "")
        insights["eda"].append(f"Binary/Boolean features identified: {cols_str}.")
        
    date_cols = cleaning_report.get("date_converted", [])
    if date_cols:
        cols_str = ", ".join(date_cols)
        insights["eda"].append(f"Automatically parsed datetime features: {cols_str}.")

    numeric_converted = cleaning_report.get("numeric_converted", [])
    if numeric_converted:
        cols_str = ", ".join(numeric_converted[:3])
        insights["eda"].append(f"Cleaned and converted text columns to numeric: {cols_str}.")

    # --- Model Insights ---
    if model_results:
        best_name = model_results.get("best_model_name", "Unknown Model")
        insights["model"].append(f"The {best_name} algorithm achieved the highest overall performance.")
        
        metrics = model_results.get("metrics", {})
        if best_name in metrics:
            best_metrics = metrics[best_name]
            
            # Classification metrics
            if "accuracy" in best_metrics:
                acc = best_metrics["accuracy"]
                insights["model"].append(f"The model successfully predicts the correct category {acc*100:.1f}% of the time.")
            if "f1_score" in best_metrics:
                f1 = best_metrics["f1_score"]
                insights["model"].append(f"F1 Score (balance of precision and recall) is {f1:.3f}.")
                
            # Regression metrics
            if "r2" in best_metrics:
                r2 = best_metrics["r2"]
                insights["model"].append(f"The model explains {r2*100:.1f}% of the variance in the target variable.")
            if "rmse" in best_metrics:
                rmse = best_metrics["rmse"]
                insights["model"].append(f"The Root Mean Squared Error (RMSE) is {rmse:.3f}, indicating the average prediction error.")

    return insights
