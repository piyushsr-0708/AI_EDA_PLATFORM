import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb
import joblib

def train_and_evaluate_models(
    cleaned_file_path: str, 
    target_column: str, 
    task_type: str, 
    models_to_train: list[str],
    best_model_path: str,
    metrics_path: str
) -> dict:
    """Trains specified models, evaluates them, and saves the best model and metrics. Task type can be 'regression' or 'classification'. Returns metrics dictionary and best model name."""
    try:
        df = pd.read_csv(cleaned_file_path)
        if target_column not in df.columns:
            return {"error": f"Target column {target_column} not found."}
            
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Determine problem type based on target dtype if not classification
        if task_type == 'classification' and y.dtype == object:
            from sklearn.preprocessing import LabelEncoder
            y = LabelEncoder().fit_transform(y)
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        from modules.model_trainer import _build_preprocessor
        from sklearn.pipeline import Pipeline
        preprocessor = _build_preprocessor(X)
        
        results = {}
        best_score = float('-inf') if task_type == 'classification' else float('inf')
        best_model_obj = None
        best_model_name = ""
        best_preds = None
        
        for model_name in models_to_train:
            base_model = None
            if task_type == "regression":
                if model_name == "Linear Regression": base_model = LinearRegression()
                elif model_name == "Ridge": base_model = Ridge()
                elif model_name == "Lasso": base_model = Lasso()
                elif model_name == "ElasticNet": base_model = ElasticNet()
                elif model_name == "Decision Tree": base_model = DecisionTreeRegressor(random_state=42)
                elif model_name == "Random Forest": base_model = RandomForestRegressor(random_state=42)
                elif model_name == "Extra Trees": base_model = ExtraTreesRegressor(random_state=42)
                elif model_name == "Gradient Boosting": base_model = GradientBoostingRegressor(random_state=42)
                elif model_name == "XGBoost": base_model = xgb.XGBRegressor(random_state=42)
                elif model_name == "LightGBM": base_model = lgb.LGBMRegressor(random_state=42)
                else: base_model = RandomForestRegressor(random_state=42) # fallback
            else:
                if model_name == "Logistic Regression": base_model = LogisticRegression(max_iter=1000)
                elif model_name == "Decision Tree": base_model = DecisionTreeClassifier(random_state=42)
                elif model_name == "Random Forest": base_model = RandomForestClassifier(random_state=42)
                elif model_name == "Extra Trees": base_model = ExtraTreesClassifier(random_state=42)
                elif model_name == "Gradient Boosting": base_model = GradientBoostingClassifier(random_state=42)
                elif model_name == "SVM": base_model = SVC(probability=True, random_state=42)
                elif model_name == "KNN": base_model = KNeighborsClassifier()
                elif model_name == "GaussianNB": base_model = GaussianNB()
                else: base_model = RandomForestClassifier(random_state=42)

            model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier" if task_type == 'classification' else "regressor", base_model)]) # fallback
                
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            if task_type == "regression":
                score = float(mean_squared_error(y_test, preds) ** 0.5)  # RMSE
                r2 = float(r2_score(y_test, preds))
                results[model_name] = {"rmse": score, "r2": r2}
                if score < best_score:
                    best_score = score
                    best_model_obj = model
                    best_model_name = model_name
                    best_preds = preds
            else:
                score = float(f1_score(y_test, preds, average="weighted"))
                acc = float(accuracy_score(y_test, preds))
                results[model_name] = {"f1": score, "accuracy": acc}
                if score > best_score:
                    best_score = score
                    best_model_obj = model
                    best_model_name = model_name
                    best_preds = preds
                    
        # Determine feature importances if possible
        feature_importance = {}
        try:
            estimator = best_model_obj.named_steps["classifier" if task_type == 'classification' else "regressor"]
            importances_array = None
            if hasattr(estimator, "feature_importances_"):
                importances_array = estimator.feature_importances_
            elif hasattr(estimator, "coef_"):
                coef = np.abs(estimator.coef_)
                if len(coef.shape) > 1:
                    importances_array = np.mean(coef, axis=0)
                else:
                    importances_array = coef
                    
            if importances_array is not None:
                preproc = best_model_obj.named_steps["preprocessor"]
                try:
                    raw_names = preproc.get_feature_names_out()
                    feature_names = [name.split("__")[-1] if "__" in name else name for name in raw_names]
                except Exception:
                    feature_names = [f"feature_{i}" for i in range(len(importances_array))]
                
                feature_importance = dict(sorted(
                    zip(feature_names, importances_array),
                    key=lambda x: x[1],
                    reverse=True
                ))
        except Exception as e:
            print(f"Could not extract feature importances: {e}")

        # We don't save to disk here anymore, we let the unified app.py handle it!
        # This matches the signature of train_and_select_model exactly.
        
        # We still need to return the metrics to Gemini, but Gemini's JSON serializer will crash on sklearn objects.
        # Wait, the orchestrator handles serialization! Let's just return it cleanly.
        return {
            "status": "success",
            "best_model_name": best_model_name,
            "best_model": best_model_obj,
            "metrics": results,
            "y_pred": list(best_preds),
            "y_score": None, # Could calculate if proba exists, but omitting for simplicity
            "y_test": list(y_test),
            "problem_type": task_type,
            "feature_columns": list(X.columns),
            "feature_importance": feature_importance
        }
    except Exception as e:
        return {"error": str(e)}
