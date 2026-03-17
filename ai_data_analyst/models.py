"""
models.py — Complete ML model system with auto-detection, training,
evaluation, model comparison, and best-model recommendation.
"""

import warnings
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    r2_score, mean_squared_error, mean_absolute_error,
    silhouette_score
)
from sklearn.impute import SimpleImputer

from utils import (
    print_section, print_success, print_error, print_warning, print_info,
    bold, green, red, yellow, cyan, dim, magenta,
    prompt, pick_column, detect_column_types, detect_problem_type,
    timestamped_filename, ensure_output_dir, press_enter
)


# ─────────────────────────────────────────────────────────────────────────────
# Model registries
# ─────────────────────────────────────────────────────────────────────────────

REGRESSION_MODELS = {
    "Linear Regression":           LinearRegression,
    "Ridge Regression":            Ridge,
    "Lasso Regression":            Lasso,
    "Decision Tree Regressor":     DecisionTreeRegressor,
    "Random Forest Regressor":     RandomForestRegressor,
    "Gradient Boosting Regressor": GradientBoostingRegressor,
}

CLASSIFICATION_MODELS = {
    "Logistic Regression":            LogisticRegression,
    "Decision Tree Classifier":       DecisionTreeClassifier,
    "Random Forest Classifier":       RandomForestClassifier,
    "Gradient Boosting Classifier":   GradientBoostingClassifier,
}

# Accuracy mode → hyperparameter presets
ACCURACY_PRESETS = {
    "fast": {
        "RandomForestRegressor":     {"n_estimators": 50,  "max_depth": 5,  "random_state": 42},
        "RandomForestClassifier":    {"n_estimators": 50,  "max_depth": 5,  "random_state": 42},
        "GradientBoostingRegressor": {"n_estimators": 50,  "max_depth": 3,  "random_state": 42},
        "GradientBoostingClassifier":{"n_estimators": 50,  "max_depth": 3,  "random_state": 42},
        "DecisionTreeRegressor":     {"max_depth": 5,      "random_state": 42},
        "DecisionTreeClassifier":    {"max_depth": 5,      "random_state": 42},
        "LogisticRegression":        {"max_iter": 200,     "random_state": 42},
        "Ridge":                     {"alpha": 1.0},
        "Lasso":                     {"alpha": 1.0,        "max_iter": 1000},
    },
    "balanced": {
        "RandomForestRegressor":     {"n_estimators": 100, "max_depth": 10, "random_state": 42},
        "RandomForestClassifier":    {"n_estimators": 100, "max_depth": 10, "random_state": 42},
        "GradientBoostingRegressor": {"n_estimators": 100, "max_depth": 4,  "random_state": 42},
        "GradientBoostingClassifier":{"n_estimators": 100, "max_depth": 4,  "random_state": 42},
        "DecisionTreeRegressor":     {"max_depth": 10,     "random_state": 42},
        "DecisionTreeClassifier":    {"max_depth": 10,     "random_state": 42},
        "LogisticRegression":        {"max_iter": 500,     "random_state": 42},
        "Ridge":                     {"alpha": 0.5},
        "Lasso":                     {"alpha": 0.5,        "max_iter": 2000},
    },
    "high_accuracy": {
        "RandomForestRegressor":     {"n_estimators": 300, "max_depth": None,"random_state": 42},
        "RandomForestClassifier":    {"n_estimators": 300, "max_depth": None,"random_state": 42},
        "GradientBoostingRegressor": {"n_estimators": 300, "max_depth": 5,  "random_state": 42, "learning_rate": 0.05},
        "GradientBoostingClassifier":{"n_estimators": 300, "max_depth": 5,  "random_state": 42, "learning_rate": 0.05},
        "DecisionTreeRegressor":     {"max_depth": None,   "random_state": 42},
        "DecisionTreeClassifier":    {"max_depth": None,   "random_state": 42},
        "LogisticRegression":        {"max_iter": 2000,    "random_state": 42, "C": 0.1},
        "Ridge":                     {"alpha": 0.1},
        "Lasso":                     {"alpha": 0.1,        "max_iter": 5000},
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Data preprocessor
# ─────────────────────────────────────────────────────────────────────────────

class DataPreprocessor:
    """Handles NaN imputation, encoding, and scaling."""

    def __init__(self):
        self._num_imputer  = SimpleImputer(strategy="median")
        self._cat_imputer  = SimpleImputer(strategy="most_frequent")
        self._encoders: Dict[str, LabelEncoder] = {}
        self._scaler       = RobustScaler()
        self._num_cols: List[str] = []
        self._cat_cols: List[str] = []
        self._fitted       = False

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()
        self._num_cols = X.select_dtypes(include=np.number).columns.tolist()
        self._cat_cols = X.select_dtypes(include="object").columns.tolist()

        if self._num_cols:
            X[self._num_cols] = self._num_imputer.fit_transform(X[self._num_cols])
        if self._cat_cols:
            X[self._cat_cols] = self._cat_imputer.fit_transform(X[self._cat_cols])
            for col in self._cat_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self._encoders[col] = le

        self._fitted = True
        return self._scaler.fit_transform(X)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()
        if self._num_cols:
            X[self._num_cols] = self._num_imputer.transform(X[self._num_cols])
        if self._cat_cols:
            X[self._cat_cols] = self._cat_imputer.transform(X[self._cat_cols])
            for col in self._cat_cols:
                if col in self._encoders:
                    X[col] = self._encoders[col].transform(X[col].astype(str))
        return self._scaler.transform(X)


# ─────────────────────────────────────────────────────────────────────────────
# Model Trainer
# ─────────────────────────────────────────────────────────────────────────────

class ModelTrainer:
    """
    Auto-detects problem type, trains all relevant models,
    evaluates them, and recommends the best one.
    """

    def __init__(self, df: pd.DataFrame, model_dir: str = "outputs/models"):
        self.df        = df
        self.model_dir = ensure_output_dir(model_dir)
        self.col_types = detect_column_types(df)
        self.results: Dict[str, Dict] = {}
        self.best_model_name: str = ""
        self.best_model: Any = None
        self.preprocessor = DataPreprocessor()
        self.label_encoder: Optional[LabelEncoder] = None
        self.target_col: str = ""
        self.problem_type: str = ""
        self.accuracy_mode: str = "balanced"

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """Interactive ML training flow."""
        print_section("ML MODEL TRAINING")

        # Step 1: Pick target column or clustering
        self._pick_problem_setup()

        # Step 2: Choose accuracy mode
        self._pick_accuracy_mode()

        # Step 3: Train & evaluate
        if self.problem_type == "clustering":
            self._run_clustering()
        elif self.problem_type == "regression":
            self._run_regression()
        else:
            self._run_classification()

        # Step 4: Show comparison & recommend
        self._show_results()

        # Step 5: Save best model
        if self.best_model and confirm_save():
            self._save_best_model()

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #

    def _pick_problem_setup(self) -> None:
        print("\n  Problem type options:")
        print("    1. Auto-detect from target column")
        print("    2. Regression")
        print("    3. Classification")
        print("    4. Clustering (KMeans, no target needed)")

        choice = prompt("Choose", "1")

        if choice == "4":
            self.problem_type = "clustering"
            return

        self.target_col = pick_column(self.df, "Select target column")
        if not self.target_col:
            print_warning("No target selected. Defaulting to clustering.")
            self.problem_type = "clustering"
            return

        if choice == "1":
            self.problem_type = detect_problem_type(self.df, self.target_col)
            print_info(f"Auto-detected problem type: {bold(self.problem_type)}")
        elif choice == "2":
            self.problem_type = "regression"
        elif choice == "3":
            self.problem_type = "classification"

    def _pick_accuracy_mode(self) -> None:
        print("\n  Accuracy / speed trade-off:")
        print(f"    1. {green('Fast')}          — quick results, lower accuracy")
        print(f"    2. {yellow('Balanced')}      — good balance {dim('(default)')}")
        print(f"    3. {red('High Accuracy')} — best results, slower training")

        mode_map = {"1": "fast", "2": "balanced", "3": "high_accuracy"}
        choice   = prompt("Choose", "2")
        self.accuracy_mode = mode_map.get(choice, "balanced")
        print_info(f"Accuracy mode: {bold(self.accuracy_mode)}")

    # ------------------------------------------------------------------ #
    # Feature / target preparation
    # ------------------------------------------------------------------ #

    def _prepare_xy(self) -> Tuple[np.ndarray, np.ndarray]:
        drop_cols = [self.target_col]
        X_df = self.df.drop(columns=drop_cols, errors="ignore")
        X_df = X_df.select_dtypes(include=[np.number, "object"])
        y    = self.df[self.target_col].copy()

        # Encode classification target if needed
        if self.problem_type == "classification" and y.dtype == object:
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y.astype(str))
        else:
            y = pd.to_numeric(y, errors="coerce").fillna(y.median() if y.dtype != object else 0)

        X_arr = self.preprocessor.fit_transform(X_df)
        return X_arr, np.array(y)

    # ------------------------------------------------------------------ #
    # Regression
    # ------------------------------------------------------------------ #

    def _run_regression(self) -> None:
        print_info("Training regression models …")
        X, y = self._prepare_xy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        preset = ACCURACY_PRESETS[self.accuracy_mode]

        for name, cls in REGRESSION_MODELS.items():
            cls_name = cls.__name__
            params   = preset.get(cls_name, {})
            try:
                model = cls(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                r2   = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae  = mean_absolute_error(y_test, y_pred)

                cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

                self.results[name] = {
                    "model":    model,
                    "r2":       round(r2, 4),
                    "rmse":     round(rmse, 4),
                    "mae":      round(mae, 4),
                    "cv_mean":  round(cv_scores.mean(), 4),
                    "cv_std":   round(cv_scores.std(), 4),
                    "primary":  r2,
                }
                bar = green("█") * int(max(0, r2) * 20) + dim("░") * (20 - int(max(0, r2) * 20))
                print(f"  {name:<36} R²={green(f'{r2:.4f}'):<14}  {bar}")
            except Exception as exc:
                print_warning(f"  {name} failed: {exc}")

        if self.results:
            self.best_model_name = max(
                self.results, key=lambda k: self.results[k]["primary"]
            )
            self.best_model = self.results[self.best_model_name]["model"]

    # ------------------------------------------------------------------ #
    # Classification
    # ------------------------------------------------------------------ #

    def _run_classification(self) -> None:
        print_info("Training classification models …")
        X, y = self._prepare_xy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        preset = ACCURACY_PRESETS[self.accuracy_mode]

        for name, cls in CLASSIFICATION_MODELS.items():
            cls_name = cls.__name__
            params   = preset.get(cls_name, {})
            try:
                model = cls(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                acc      = accuracy_score(y_test, y_pred)
                cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

                self.results[name] = {
                    "model":    model,
                    "accuracy": round(acc, 4),
                    "cv_mean":  round(cv_scores.mean(), 4),
                    "cv_std":   round(cv_scores.std(), 4),
                    "report":   classification_report(y_test, y_pred),
                    "primary":  acc,
                }
                bar = green("█") * int(acc * 20) + dim("░") * (20 - int(acc * 20))
                print(f"  {name:<36} Acc={green(f'{acc:.4f}'):<14}  {bar}")
            except Exception as exc:
                print_warning(f"  {name} failed: {exc}")

        if self.results:
            self.best_model_name = max(
                self.results, key=lambda k: self.results[k]["primary"]
            )
            self.best_model = self.results[self.best_model_name]["model"]

    # ------------------------------------------------------------------ #
    # Clustering
    # ------------------------------------------------------------------ #

    def _run_clustering(self) -> None:
        print_info("Running KMeans clustering …")
        num_df = self.df.select_dtypes(include=np.number).dropna()

        if num_df.empty:
            print_error("No numeric columns available for clustering.")
            return

        scaler = RobustScaler()
        X      = scaler.fit_transform(num_df)

        # Find best K using elbow + silhouette
        k_range = range(2, min(11, len(num_df) // 5 + 2))
        best_score = -1
        best_k     = 3

        print(f"\n  {'K':<6} {'Silhouette':<16} {'Inertia'}")
        print(f"  {'─'*6} {'─'*16} {'─'*12}")

        for k in k_range:
            try:
                km    = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(X)
                score  = silhouette_score(X, labels)
                print(f"  {k:<6} {score:<16.4f} {km.inertia_:.2f}")
                if score > best_score:
                    best_score = score
                    best_k     = k
            except Exception:
                pass

        # Train with best K
        best_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels  = best_km.fit_predict(X)

        self.results["KMeans"] = {
            "model":       best_km,
            "best_k":      best_k,
            "silhouette":  round(best_score, 4),
            "inertia":     round(best_km.inertia_, 2),
            "primary":     best_score,
        }
        self.best_model_name = "KMeans"
        self.best_model      = best_km

        # Add cluster labels to df
        idx_map = num_df.index
        self.df.loc[idx_map, "cluster"] = labels.astype(int)
        print_success(f"Cluster labels added as column 'cluster'")

    # ------------------------------------------------------------------ #
    # Results display
    # ------------------------------------------------------------------ #

    def _show_results(self) -> None:
        if not self.results:
            print_error("No results to display.")
            return

        print_section("MODEL COMPARISON RESULTS")

        if self.problem_type == "regression":
            self._print_regression_table()
        elif self.problem_type == "classification":
            self._print_classification_table()
        else:
            self._print_clustering_table()

        # Best model recommendation
        print_section("BEST MODEL RECOMMENDATION")
        self._print_recommendation()

    def _print_regression_table(self) -> None:
        print(f"\n  {'Model':<36} {'R²':>8} {'RMSE':>10} {'MAE':>10} {'CV R²':>10}")
        print(f"  {'─'*36} {'─'*8} {'─'*10} {'─'*10} {'─'*10}")
        for name, res in sorted(self.results.items(),
                                key=lambda x: x[1]["r2"], reverse=True):
            marker = green(" ★") if name == self.best_model_name else "  "
            r2_str = green(f'{res["r2"]:>8.4f}') if name == self.best_model_name else f'{res["r2"]:>8.4f}'
            print(
                f"{marker} {name:<34} "
                f"{r2_str} "
                f"{res['rmse']:>10.4f} "
                f"{res['mae']:>10.4f} "
                f"{res['cv_mean']:>8.4f}±{res['cv_std']:.3f}"
            )

    def _print_classification_table(self) -> None:
        print(f"\n  {'Model':<36} {'Accuracy':>10} {'CV Accuracy':>12}")
        print(f"  {'─'*36} {'─'*10} {'─'*12}")
        for name, res in sorted(self.results.items(),
                                key=lambda x: x[1]["accuracy"], reverse=True):
            marker = green(" ★") if name == self.best_model_name else "  "
            acc_str = green(f'{res["accuracy"]:>10.4f}') if name == self.best_model_name else f'{res["accuracy"]:>10.4f}'
            print(
                f"{marker} {name:<34} "
                f"{acc_str} "
                f"{res['cv_mean']:>8.4f}±{res['cv_std']:.3f}"
            )

        # Detailed report for best model
        if self.best_model_name in self.results:
            print(f"\n  {bold('Classification Report — ' + self.best_model_name)}")
            print(self.results[self.best_model_name].get("report", ""))

    def _print_clustering_table(self) -> None:
        res = self.results.get("KMeans", {})
        print(f"\n  {bold('KMeans Clustering Results')}")
        print(f"  Best K         : {green(str(res.get('best_k', '?')))}")
        print(f"  Silhouette     : {green(str(res.get('silhouette', '?')))}")
        print(f"  Inertia        : {res.get('inertia', '?')}")

    def _print_recommendation(self) -> None:
        if not self.best_model_name:
            return

        res = self.results[self.best_model_name]
        print(f"\n  {bold(cyan('🏆 Best Model'))} : {green(bold(self.best_model_name))}")

        if self.problem_type == "regression":
            r2 = res["r2"]
            print(f"  R² Score       : {green(str(r2))}")
            print(f"  RMSE           : {res['rmse']}")
            print(f"  CV R² (mean)   : {res['cv_mean']} ± {res['cv_std']}")

            # Explain selection
            if r2 >= 0.9:
                quality = green("Excellent fit (≥ 0.90)")
            elif r2 >= 0.75:
                quality = yellow("Good fit (≥ 0.75)")
            elif r2 >= 0.5:
                quality = yellow("Moderate fit (≥ 0.50)")
            else:
                quality = red("Weak fit (< 0.50) — consider feature engineering")

            print(f"\n  {bold('Why this model?')}")
            print(f"  → Highest R² score among all trained models.")
            print(f"  → Model quality: {quality}")
            print(f"  → Accuracy mode used: {bold(self.accuracy_mode)}")

        elif self.problem_type == "classification":
            acc = res["accuracy"]
            print(f"  Accuracy       : {green(str(acc))}")
            print(f"  CV Accuracy    : {res['cv_mean']} ± {res['cv_std']}")
            if acc >= 0.95:
                quality = green("Outstanding (≥ 95%)")
            elif acc >= 0.85:
                quality = green("Strong (≥ 85%)")
            elif acc >= 0.70:
                quality = yellow("Acceptable (≥ 70%)")
            else:
                quality = red("Needs improvement (< 70%)")
            print(f"\n  {bold('Why this model?')}")
            print(f"  → Highest accuracy among all trained classifiers.")
            print(f"  → Performance: {quality}")

        else:
            print(f"  Silhouette     : {green(str(res.get('silhouette')))}")
            print(f"  Best K         : {res.get('best_k')}")
            print(f"\n  {bold('Why this model?')}")
            print(f"  → Best silhouette score indicates most natural cluster separation.")

    # ------------------------------------------------------------------ #
    # Save model
    # ------------------------------------------------------------------ #

    def _save_best_model(self) -> None:
        if not self.best_model:
            return
        filename = timestamped_filename(
            f"{self.best_model_name.replace(' ', '_')}", "pkl"
        )
        path = self.model_dir / filename
        bundle = {
            "model":          self.best_model,
            "model_name":     self.best_model_name,
            "problem_type":   self.problem_type,
            "accuracy_mode":  self.accuracy_mode,
            "metrics":        self.results.get(self.best_model_name, {}),
            "preprocessor":   self.preprocessor,
            "label_encoder":  self.label_encoder,
            "target_col":     self.target_col,
        }
        with open(path, "wb") as f:
            pickle.dump(bundle, f)
        print_success(f"Best model saved → {path}")


def confirm_save() -> bool:
    ans = input(f"\n  {bold('Save best model to disk?')} {dim('[y/N]')}: ").strip().lower()
    return ans in ("y", "yes")
