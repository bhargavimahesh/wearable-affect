"""Leave-one-subject-out (LOSO) evaluation."""

from collections.abc import Callable

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

ID_COLUMNS = ["subject_id", "start_s", "target"]
METRICS = ["macro_f1", "balanced_accuracy", "auroc"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All columns except identifiers and the target, i.e. what the model may see."""
    return [c for c in df.columns if c not in ID_COLUMNS]


def loso_predict(df: pd.DataFrame, make_model: Callable) -> pd.DataFrame:
    """Stress probability for every window, each predicted by a model that never saw its subject."""
    X = df[feature_columns(df)]
    y = df["target"]
    groups = df["subject_id"]

    folds = []
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        model = make_model()  # a fresh, untrained model for every fold
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        fold = df.iloc[test_idx][ID_COLUMNS].copy()
        fold["prob"] = model.predict_proba(X.iloc[test_idx])[:, 1]  # probability of stress
        folds.append(fold)
    return pd.concat(folds, ignore_index=True)


def score_predictions(preds: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Metrics per subject from window-level predictions, in numeric subject order."""
    rows = []
    for subject_id, group in preds.groupby("subject_id"):
        y_pred = (group["prob"] >= threshold).astype(int)
        rows.append(
            {
                "subject_id": subject_id,
                "macro_f1": f1_score(group["target"], y_pred, average="macro", zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(group["target"], y_pred),
                "auroc": roc_auc_score(group["target"], group["prob"]),
            }
        )
    results = pd.DataFrame(rows)
    return results.sort_values("subject_id", key=lambda s: s.str[1:].astype(int), ignore_index=True)


def loso_evaluate(df: pd.DataFrame, make_model: Callable, threshold: float = 0.5) -> pd.DataFrame:
    """Per-subject metrics for one model under leave-one-subject-out evaluation."""
    return score_predictions(loso_predict(df, make_model), threshold)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of each metric across left-out subjects."""
    return results[METRICS].agg(["mean", "std"]).T