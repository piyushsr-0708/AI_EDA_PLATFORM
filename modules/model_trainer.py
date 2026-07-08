"""Simple model training utilities using scikit-learn.

Functions train a small set of models for regression or classification,
automatically encode categorical features, evaluate models, and select the
best model. Designed to be simple and modular for integration into apps.

API:
    train_and_select_model(df, target_column, problem_type, test_size=0.2, random_state=42)

Returns:
    best_model: fitted sklearn estimator (Pipeline)
    best_name: string name of the selected model
    results: dict mapping model name -> metrics dict

Supported models:
    Regression: LinearRegression, RandomForestRegressor
    Classification: LogisticRegression, RandomForestClassifier

Only scikit-learn is used.
"""
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    Lasso,
    ElasticNet,
)
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


def _build_preprocessor(X: pd.DataFrame):
    """Build a ColumnTransformer that scales numeric and encodes categorical cols.

    Args:
        X: DataFrame of features.

    Returns:
        transformer: a fitted ColumnTransformer (not fitted here) ready for Pipeline.
    """
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=50),
            ),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    if not transformers:
        # No features? Create passthrough
        return ColumnTransformer([], remainder="passthrough")

    return ColumnTransformer(transformers, remainder="drop")


def _get_models(problem_type: str):
    """Return a dict of name->estimator objects (unfitted) for the problem type."""
    if problem_type.lower() == "regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(random_state=42),
            "Lasso": Lasso(random_state=42),
            "ElasticNet": ElasticNet(random_state=42),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(random_state=42),
            "Extra Trees": ExtraTreesRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        }
        try:
            from xgboost import XGBRegressor
            models["XGBoost"] = XGBRegressor(random_state=42, verbosity=0)
        except Exception:
            pass
        try:
            from lightgbm import LGBMRegressor
            models["LightGBM"] = LGBMRegressor(random_state=42)
        except Exception:
            pass
        return models
    elif problem_type.lower() == "classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Extra Trees": ExtraTreesClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "KNN": KNeighborsClassifier(),
            "SVM": SVC(probability=True),
            "GaussianNB": GaussianNB(),
        }
    else:
        raise ValueError(f"Unsupported problem_type: {problem_type}")


def _evaluate_regression(y_true, y_pred):
    r2 = float(metrics.r2_score(y_true, y_pred))
    mae = float(metrics.mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(metrics.mean_squared_error(y_true, y_pred)))
    try:
        mape = float(metrics.mean_absolute_percentage_error(y_true, y_pred))
    except Exception:
        mape = None
    
    if r2 < 0.30:
        interp = "Weak predictive power"
    elif r2 < 0.70:
        interp = "Moderate predictive power"
    else:
        interp = "Strong predictive power"
        
    result = {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "interpretation": interp
    }
    if mape is not None:
        result["mape"] = mape
    return result


def _evaluate_classification(y_true, y_pred, y_score=None, average: str = "weighted"):
    acc = float(metrics.accuracy_score(y_true, y_pred))
    
    if acc < 0.60:
        interp = "Weak performance"
    elif acc < 0.80:
        interp = "Moderate performance"
    else:
        interp = "Strong performance"
        
    result = {
        "accuracy": acc,
        "f1": float(metrics.f1_score(y_true, y_pred, average=average)),
        "precision": float(metrics.precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(metrics.recall_score(y_true, y_pred, average=average, zero_division=0)),
        "interpretation": interp
    }
    if y_score is not None:
        try:
            if hasattr(y_score, "shape") and len(y_score.shape) == 2:
                if y_score.shape[1] == 2:
                    roc_auc = metrics.roc_auc_score(y_true, y_score[:, 1])
                else:
                    roc_auc = metrics.roc_auc_score(y_true, y_score, multi_class="ovr", average=average)
            else:
                roc_auc = metrics.roc_auc_score(y_true, y_score)
            result["roc_auc"] = float(roc_auc)
        except Exception:
            pass
    return result


def train_and_select_model(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict:
    """Train models for the given problem and select the best.

    Process:
        - Split the data into train/test
        - Build preprocessing pipeline (automatic encoding)
        - Train each candidate model inside a Pipeline(preprocessor, estimator)
        - Evaluate on test set
        - Select best model (R2 for regression, F1 for classification)
        - Retrain best model on the full dataset before returning

    Args:
        df: Cleaned pandas DataFrame containing features and target.
        target_column: Name of the target column in `df`.
        problem_type: One of 'regression' or 'classification'.
        test_size: Fraction of data to hold out for testing.
        random_state: Random seed for splits and models.

    Returns a standardized dictionary with the following keys:
        - best_model_name: name of the selected model
        - best_model: fitted sklearn Pipeline (preprocessor + estimator) trained on full data
        - metrics: dict mapping model name -> metrics dict (evaluated on test set)
        - y_pred: predictions on the held-out test set made by the trained candidate
        - y_test: true target values for the held-out test set (unaltered original values)
        - problem_type: the provided problem_type string
    """
    if target_column not in df.columns:
        raise ValueError("target_column not found in DataFrame")

    if len(df) < 10:
        raise ValueError(f"Dataset too small for training: {len(df)} rows")

    X = df.drop(columns=[target_column]).select_dtypes(exclude=["datetime64[ns]", "datetimetz"])
    y = df[target_column]

    # Split before any target encoding to prevent leakage
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=(y if problem_type.lower() == "classification" else None)
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    # Keep a copy of the original y_test values (before any encoding)
    y_test_original = y_test.copy()

    # Encode target for classification after split to prevent leakage
    label_encoder = None
    n_classes = None
    if problem_type.lower() == "classification":
        if y_train.dtype == object or not np.issubdtype(y_train.dtype, np.number):
            label_encoder = LabelEncoder()
            y_train = pd.Series(label_encoder.fit_transform(y_train), index=y_train.index)
            y_test = pd.Series(label_encoder.transform(y_test), index=y_test.index)
        n_classes = y_train.nunique()

    models = _get_models(problem_type)

    results: Dict[str, Dict] = {}
    fitted_pipelines = {}

    for name, estimator in models.items():
        pipeline = Pipeline([("preproc", _build_preprocessor(X_train)), ("est", estimator)])
        try:
            start_time = time.time()
            pipeline.fit(X_train, y_train)
            fit_time = time.time() - start_time
            
            y_pred = pipeline.predict(X_test)
            y_score = None
            if problem_type.lower() == "classification":
                if hasattr(pipeline, "predict_proba"):
                    try:
                        y_score = pipeline.predict_proba(X_test)
                    except Exception:
                        y_score = None
                elif hasattr(pipeline, "decision_function"):
                    try:
                        y_score = pipeline.decision_function(X_test)
                    except Exception:
                        y_score = None

            if problem_type.lower() == "regression":
                metric = _evaluate_regression(y_test, y_pred)
            else:
                average = "binary" if n_classes == 2 else "weighted"
                metric = _evaluate_classification(y_test, y_pred, y_score=y_score, average=average)
            
            metric["Training Time (s)"] = round(fit_time, 4)
            results[name] = metric
            fitted_pipelines[name] = pipeline
        except Exception as e:
            # record failure
            results[name] = {"error": str(e)}

    # Select best model
    best_name = None
    best_score = None
    for name, metric in results.items():
        if "error" in metric:
            continue
        if problem_type.lower() == "regression":
            score = metric.get("r2")
        else:
            score = metric.get("f1")

        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_name = name

    if best_name is None:
        raise RuntimeError("No model trained successfully")

    if problem_type.lower() == "regression" and best_score < 0:
        results["__warning__"] = {"warning": "All models have negative R2; check data quality"}

    # Prepare y_pred and y_test for the selected best candidate (from the pipeline
    # that was trained on the train split).
    best_candidate_pipeline = fitted_pipelines.get(best_name)
    if best_candidate_pipeline is None:
        raise RuntimeError("Best model pipeline not available for predictions")

    y_pred_test = best_candidate_pipeline.predict(X_test)
    
    y_score_test = None
    if problem_type.lower() == "classification":
        if hasattr(best_candidate_pipeline, "predict_proba"):
            try:
                y_score_test = best_candidate_pipeline.predict_proba(X_test)
            except Exception:
                pass
        elif hasattr(best_candidate_pipeline, "decision_function"):
            try:
                y_score_test = best_candidate_pipeline.decision_function(X_test)
            except Exception:
                pass
        
        if y_score_test is not None:
            if isinstance(y_score_test, np.ndarray):
                y_score_test = y_score_test.tolist()

    # Retrain best model on full dataset using a fresh estimator
    y_full = y
    if label_encoder is not None:
        y_full = pd.Series(label_encoder.fit_transform(y), index=y.index)
    best_pipeline = Pipeline([("preproc", _build_preprocessor(X)), ("est", clone(models[best_name]))])
    
    print("\n--- FINAL VALIDATION BEFORE FITTING ---")
    print(f"target_column: {target_column}")
    print(f"X columns: {X.columns.tolist()}")
    print("---------------------------------------\n")
    
    best_pipeline.fit(X, y_full)

    # Extract feature importances for tree-based models or coefficients for linear models
    feature_importance = None
    try:
        estimator = best_pipeline.named_steps["est"]
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
            preprocessor = best_pipeline.named_steps["preproc"]
            # Get transformed feature names
            try:
                raw_names = preprocessor.get_feature_names_out()
                feature_names = [name.split("__")[-1] if "__" in name else name for name in raw_names]
            except Exception:
                feature_names = [f"feature_{i}" for i in range(len(importances_array))]
            
            fi_dict = dict(sorted(
                zip(feature_names, importances_array),
                key=lambda x: x[1],
                reverse=True
            ))
            feature_importance = fi_dict
    except Exception as e:
        print(f"Could not extract feature importances: {e}")

    # Standardized return structure
    return {
        "best_model_name": best_name,
        "best_model": best_pipeline,
        "metrics": results,
        "y_pred": list(y_pred_test),
        "y_score": y_score_test,
        "y_test": list(y_test_original),
        "problem_type": problem_type.lower(),
        "feature_columns": X.columns.tolist(),
        "feature_importance": feature_importance
    }


if __name__ == "__main__":
    # small smoke test when run directly (not required)
    import sklearn.datasets as ds

    X, y = ds.fetch_california_housing(return_X_y=True)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y
    result = train_and_select_model(df, "target", "regression")
    print(result["best_model_name"], result["metrics"])
