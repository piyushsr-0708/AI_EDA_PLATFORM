import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def generate_visualizations(cleaned_file_path: str, plots_dir: str, target_column: str) -> dict:
    """Generates standard EDA plots (correlation matrix, target distribution) and saves them to plots_dir."""
    try:
        df = pd.read_csv(cleaned_file_path)
        out_dir = Path(plots_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        generated = []
        
        # Target dist
        plt.figure()
        if pd.api.types.is_numeric_dtype(df[target_column]):
            sns.histplot(df[target_column], kde=True)
        else:
            sns.countplot(x=df[target_column])
        p1 = out_dir / "target_distribution.png"
        plt.savefig(p1)
        plt.close()
        generated.append("target_distribution.png")
        
        # Correlation matrix
        numeric_df = df.select_dtypes(include=["number"])
        if len(numeric_df.columns) > 1:
            plt.figure(figsize=(10,8))
            sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm')
            p2 = out_dir / "correlation_matrix.png"
            plt.savefig(p2)
            plt.close()
            generated.append("correlation_matrix.png")
            
        return {"status": "success", "generated_plots": generated}
    except Exception as e:
        return {"error": str(e)}
