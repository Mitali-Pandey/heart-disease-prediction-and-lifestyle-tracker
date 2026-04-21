"""
Multi-Model Training and Comparison Module
Trains and compares: Logistic Regression, Random Forest, XGBoost, SVM, Neural Network

Reliability features:
- Scaling inside Pipeline (correct CV — no leakage)
- Model selection by 5-fold CV F1 on training data only (not test-set argmax)
- Sigmoid probability calibration (CalibratedClassifierCV) on training data
- Bootstrap 95% CIs for test-set ROC-AUC and F1
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

MODEL_ORDER = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "SVM",
    "Neural Network",
]


def _bootstrap_auc_f1_ci(y_true, y_pred_proba, n_bootstrap=600, random_state=42):
    """Percentile 95% CIs for ROC-AUC and F1 (threshold 0.5) on held-out test."""
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_pred_proba)
    n = len(y_true)
    aucs, f1s = [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
        y_hat = (ys >= 0.5).astype(int)
        f1s.append(f1_score(yt, y_hat, zero_division=0))
    if not aucs:
        return (np.nan, np.nan), (np.nan, np.nan)
    return (
        (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))),
        (float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))),
    )


class ModelTrainer:
    """Train and compare multiple ML models with calibrated probabilities."""

    def __init__(self, X, y, test_size=0.2, random_state=42):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.cv_selection = {}
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        self.scaler = StandardScaler()
        self.scaler.fit(self.X_train)
        self._skf = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=random_state
        )

    def _make_base_estimator(self, model_name):
        """Unfitted estimator; scaling is inside Pipeline where needed."""
        rs = self.random_state
        if model_name == "Logistic Regression":
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(random_state=rs, max_iter=2000)),
                ]
            )
        if model_name == "Random Forest":
            return Pipeline(
                [
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=100,
                            random_state=rs,
                            n_jobs=-1,
                        ),
                    )
                ]
            )
        if model_name == "XGBoost":
            return Pipeline(
                [
                    (
                        "clf",
                        xgb.XGBClassifier(
                            random_state=rs,
                            eval_metric="logloss",
                            n_jobs=1,
                        ),
                    )
                ]
            )
        if model_name == "SVM":
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", SVC(probability=True, random_state=rs)),
                ]
            )
        if model_name == "Neural Network":
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        MLPClassifier(
                            hidden_layer_sizes=(100, 50),
                            random_state=rs,
                            max_iter=1000,
                        ),
                    ),
                ]
            )
        raise ValueError(f"Unknown model: {model_name}")

    def train_all_models(self):
        """CV for selection (train only), then fit sigmoid-calibrated models on full train."""
        self.models = {}
        self.cv_selection = {}

        for name in MODEL_ORDER:
            base = self._make_base_estimator(name)
            scores = cross_val_score(
                clone(base),
                self.X_train,
                self.y_train,
                cv=self._skf,
                scoring="f1",
                n_jobs=-1,
            )
            self.cv_selection[name] = {
                "mean": float(scores.mean()),
                "std": float(scores.std()),
            }

        for name in MODEL_ORDER:
            base = self._make_base_estimator(name)
            cal = CalibratedClassifierCV(
                estimator=base, method="sigmoid", cv=3
            )
            cal.fit(self.X_train, self.y_train)
            self.models[name] = cal

        print("All models trained successfully (calibrated).")

    def evaluate_model(self, model, model_name):
        """Evaluate on held-out test (reporting only). Uses raw features (Pipeline inside model)."""
        X_test = self.X_test
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)

        (auc_lo, auc_hi), (f1_lo, f1_hi) = _bootstrap_auc_f1_ci(
            self.y_test, y_pred_proba, n_bootstrap=600, random_state=self.random_state
        )

        cv = self.cv_selection[model_name]
        cm = confusion_matrix(self.y_test, y_pred)
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)

        return {
            "model_name": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "cv_f1_mean": cv["mean"],
            "cv_f1_std": cv["std"],
            "auc_ci_low": auc_lo,
            "auc_ci_high": auc_hi,
            "f1_ci_low": f1_lo,
            "f1_ci_high": f1_hi,
            "confusion_matrix": cm,
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
        }

    def evaluate_all_models(self):
        for model_name, model in self.models.items():
            self.results[model_name] = self.evaluate_model(model, model_name)
        return self.results

    def get_best_model(self, metric="cv_f1_mean"):
        """Best model by CV F1 on training folds (default), not by test-set tuning."""
        if not self.results:
            self.evaluate_all_models()
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x][metric])
        return (
            best_model_name,
            self.models[best_model_name],
            self.results[best_model_name],
        )

    def plot_roc_curves(self):
        plt.figure(figsize=(10, 8))
        for model_name in MODEL_ORDER:
            if model_name not in self.results:
                continue
            results = self.results[model_name]
            plt.plot(
                results["fpr"],
                results["tpr"],
                label=f"{model_name} (AUC = {results['roc_auc']:.3f})",
            )
        plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title(
            "ROC Curves (calibrated probabilities, held-out test)",
            fontsize=16,
            fontweight="bold",
        )
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        return plt

    def plot_confusion_matrices(self):
        ordered = [n for n in MODEL_ORDER if n in self.results]
        n_models = len(ordered)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
        if n_models == 1:
            axes = [axes]
        for idx, model_name in enumerate(ordered):
            results = self.results[model_name]
            cm = results["confusion_matrix"]
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx])
            axes[idx].set_title(
                f'{model_name}\nAccuracy: {results["accuracy"]:.3f}'
            )
            axes[idx].set_ylabel("True Label")
            axes[idx].set_xlabel("Predicted Label")
        plt.tight_layout()
        return plt

    def get_comparison_dataframe(self):
        rows = []
        for model_name in MODEL_ORDER:
            if model_name not in self.results:
                continue
            r = self.results[model_name]
            rows.append(
                {
                    "Model": model_name,
                    "CV F1 (select)": r["cv_f1_mean"],
                    "CV F1 std": r["cv_f1_std"],
                    "Test Accuracy": r["accuracy"],
                    "Test Precision": r["precision"],
                    "Test Recall": r["recall"],
                    "Test F1": r["f1_score"],
                    "F1 95% CI": f"{r['f1_ci_low']:.3f}–{r['f1_ci_high']:.3f}",
                    "Test ROC-AUC": r["roc_auc"],
                    "AUC 95% CI": f"{r['auc_ci_low']:.3f}–{r['auc_ci_high']:.3f}",
                }
            )
        df = pd.DataFrame(rows)
        return df.sort_values("CV F1 (select)", ascending=False)

    def save_model(self, model, model_name, filepath):
        joblib.dump(model, filepath)
        print(f"{model_name} saved to {filepath}")

    def save_scaler(self, filepath):
        joblib.dump(self.scaler, filepath)
        print(f"Scaler saved to {filepath}")

    def load_model(self, filepath):
        return joblib.load(filepath)

    def predict(self, model_name, X, scaled=True):
        """Predict using stored model. `scaled` ignored — pipelines handle scaling."""
        model = self.models[model_name]
        return model.predict(X), model.predict_proba(X)
