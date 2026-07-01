# models.py
# Classifier definitions and cross-validation training.

import pandas as pd
from scipy import sparse

from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import N_SPLITS, RANDOM_STATE


def _to_dense(X):
    return X.toarray() if sparse.issparse(X) else X


def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_split=10, min_samples_leaf=5,
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting":      GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        "AdaBoost":               AdaBoostClassifier(random_state=RANDOM_STATE),
        "KNN":                    KNeighborsClassifier(n_neighbors=7),
        "Gaussian Naive Bayes":   GaussianNB(),
        "SVC": SVC(
            probability=True, class_weight="balanced", random_state=RANDOM_STATE
        ),
    }


def train_and_evaluate(X, y, preprocessor):
    cv     = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    models = get_models()

    all_metrics     = []
    all_predictions = {}

    for model_name, classifier in models.items():
        print(f"Training: {model_name}")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("to_dense",     FunctionTransformer(_to_dense, accept_sparse=True)),
            ("classifier",   classifier),
        ])

        fold_metrics     = []
        fold_predictions = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            pipeline.fit(X_train, y_train)
            y_pred  = pipeline.predict(X_val)
            y_proba = pipeline.predict_proba(X_val)

            fold_metrics.append({
                "model":             model_name,
                "fold":              fold,
                "accuracy":          accuracy_score(y_val, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_val, y_pred),
                "f1_macro":          f1_score(y_val, y_pred, average="macro"),
                "auc_macro":         roc_auc_score(
                    y_val, y_proba, multi_class="ovr",
                    average="macro", labels=pipeline.classes_
                ),
            })

            fold_df  = pd.DataFrame({"true": y_val.values, "pred": y_pred})
            proba_df = pd.DataFrame(
                y_proba, columns=[f"proba_{c}" for c in pipeline.classes_]
            )
            fold_df = pd.concat([fold_df.reset_index(drop=True), proba_df], axis=1)
            fold_df["model"]     = model_name
            fold_df["fold"]      = fold
            fold_df["max_proba"] = proba_df.max(axis=1)
            fold_df["correct"]   = fold_df["true"] == fold_df["pred"]

            fold_predictions.append(fold_df)

        all_metrics.append(pd.DataFrame(fold_metrics))
        all_predictions[model_name] = pd.concat(fold_predictions, ignore_index=True)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    return metrics_df, all_predictions


def summarise_metrics(metrics_df):
    return (
        metrics_df
        .groupby("model")[["accuracy", "balanced_accuracy", "f1_macro", "auc_macro"]]
        .mean()
        .sort_values("f1_macro", ascending=False)
        .round(3)
    )