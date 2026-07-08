import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

def _save_plot(fig, filename, output_dir):
    try:
        out_path = Path(output_dir) / filename
        fig.write_image(str(out_path))
        return str(out_path)
    except Exception as e:
        print(f"Failed to save {filename}: {e}")
        return None

def generate_regression_plots(y_true, y_pred, feature_importance, output_dir):
    """Generate plots for regression models and return a list of (image_path, fig)."""
    results = []
    
    # 1. Predicted vs Actual
    fig_pv = px.scatter(
        x=y_true, y=y_pred, 
        labels={'x': 'Actual Values', 'y': 'Predicted Values'},
        title='Predicted vs Actual'
    )
    # Add identity line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    fig_pv.add_shape(
        type="line", line=dict(dash="dash"),
        x0=min_val, y0=min_val, x1=max_val, y1=max_val
    )
    p = _save_plot(fig_pv, "predicted_vs_actual.png", output_dir)
    if p: results.append((p, fig_pv))

    # 2. Residual Plot
    residuals = np.array(y_true) - np.array(y_pred)
    fig_res = px.scatter(
        x=y_pred, y=residuals,
        labels={'x': 'Predicted Values', 'y': 'Residuals'},
        title='Residual Plot'
    )
    fig_res.add_hline(y=0, line_dash="dash", line_color="red")
    p = _save_plot(fig_res, "residuals.png", output_dir)
    if p: results.append((p, fig_res))
    
    return results

def generate_classification_plots(y_true, y_pred, y_score, feature_importance, output_dir):
    """Generate plots for classification models and return a list of (image_path, fig)."""
    results = []
    
    # 1. Confusion Matrix
    try:
        cm = confusion_matrix(y_true, y_pred)
        # Sort classes just in case
        classes = sorted(list(set(y_true).union(set(y_pred))))
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale='Blues',
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=[str(c) for c in classes],
            y=[str(c) for c in classes],
            title="Confusion Matrix"
        )
        p = _save_plot(fig_cm, "confusion_matrix.png", output_dir)
        if p: results.append((p, fig_cm))
    except Exception as e:
        print(f"Confusion matrix error: {e}")

    # Binary Classification Curves
    if len(set(y_true)) == 2 and y_score is not None:
        try:
            # Handle y_score shape
            y_score_arr = np.array(y_score)
            if len(y_score_arr.shape) == 2 and y_score_arr.shape[1] == 2:
                scores = y_score_arr[:, 1]
            elif len(y_score_arr.shape) == 1:
                scores = y_score_arr
            else:
                scores = None
                
            if scores is not None:
                # 2. ROC Curve
                fpr, tpr, _ = roc_curve(y_true, scores)
                roc_auc = auc(fpr, tpr)
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC curve (area = {roc_auc:.2f})', mode='lines'))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random'))
                fig_roc.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
                p = _save_plot(fig_roc, "roc_curve.png", output_dir)
                if p: results.append((p, fig_roc))
                
                # 3. Precision-Recall Curve
                precision, recall, _ = precision_recall_curve(y_true, scores)
                pr_auc = auc(recall, precision)
                
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(x=recall, y=precision, name=f'PR curve (area = {pr_auc:.2f})', mode='lines'))
                fig_pr.update_layout(title='Precision-Recall Curve', xaxis_title='Recall', yaxis_title='Precision')
                p = _save_plot(fig_pr, "precision_recall_curve.png", output_dir)
                if p: results.append((p, fig_pr))
        except Exception as e:
            print(f"Curve error: {e}")
            
    return results

def generate_feature_importance_plot(feature_importance, output_dir):
    """Generate Feature Importance Plot and return (path, fig)."""
    if not feature_importance:
        return None, None
        
    try:
        fi_df = pd.DataFrame(list(feature_importance.items()), columns=["Feature", "Importance"])
        top_n = fi_df.head(15).sort_values(by="Importance", ascending=True)
        fig_fi = px.bar(
            top_n, x="Importance", y="Feature",
            orientation='h', title="Top 15 Feature Importances"
        )
        fig_fi.update_layout(margin=dict(l=150))
        p = _save_plot(fig_fi, "feature_importance.png", output_dir)
        return p, fig_fi
    except Exception as e:
        print(f"Feature importance plot error: {e}")
        return None, None
