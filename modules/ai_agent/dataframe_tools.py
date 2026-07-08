import pandas as pd

def inspect_dataset(file_path: str) -> dict:
    """Reads a CSV file and returns its schema, row count, column count, and missing values."""
    try:
        df = pd.read_csv(file_path)
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "columns_list": list(df.columns),
            "dtypes": {k: str(v) for k, v in df.dtypes.items()},
            "missing_values": int(df.isnull().sum().sum())
        }
    except Exception as e:
        return {"error": str(e)}

def clean_dataset(file_path: str, output_path: str, target_column: str) -> dict:
    """Cleans a dataset (drops missing target rows, imputes others) and saves to output_path."""
    try:
        df = pd.read_csv(file_path)
        
        if target_column and target_column in df.columns:
            df = df.dropna(subset=[target_column])
            
        numeric_cols = df.select_dtypes(include=["number"]).columns
        categorical_cols = df.select_dtypes(exclude=["number"]).columns
        
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        if len(categorical_cols) > 0:
            modes = df[categorical_cols].mode()
            if not modes.empty:
                df[categorical_cols] = df[categorical_cols].fillna(modes.iloc[0])
            else:
                df[categorical_cols] = df[categorical_cols].fillna("Unknown")
            
        df.to_csv(output_path, index=False)
        return {"status": "success", "rows_after_cleaning": len(df)}
    except Exception as e:
        return {"error": str(e)}

def get_summary_statistics(file_path: str) -> dict:
    """Returns summary statistics for the dataset."""
    try:
        df = pd.read_csv(file_path)
        stats = df.describe(include='all').to_dict()
        clean_stats = {}
        for col, col_stats in stats.items():
            clean_stats[str(col)] = {}
            for stat_name, val in col_stats.items():
                if pd.isna(val):
                    clean_stats[str(col)][str(stat_name)] = None
                elif hasattr(val, 'item'):
                    clean_stats[str(col)][str(stat_name)] = val.item()
                else:
                    clean_stats[str(col)][str(stat_name)] = str(val) if not isinstance(val, (int, float, bool, str)) else val
        return clean_stats
    except Exception as e:
        return {"error": str(e)}
