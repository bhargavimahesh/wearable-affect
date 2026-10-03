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


def loso_evaluate(df: pd.DataFrame, make_model: Callable, threshold: float = 0.5) -> pd.DataFrame:
    """Train on all subjects but one, test on the left-out subject; repeat for every subject."""
    X = df[feature_columns(df)]
    y = df["target"]
    groups = df["subject_id"]

    rows = []
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        model = make_model()  # a fresh, untrained model for every fold
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        y_true = y.iloc[test_idx]
        y_prob = model.predict_proba(X.iloc[test_idx])[:, 1]  # probability of stress
        y_pred = (y_prob >= threshold).astype(int)

        rows.append(
            {
                "subject_id": groups.iloc[test_idx[0]],
                "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "auroc": roc_auc_score(y_true, y_prob),
            }
        )
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of each metric across left-out subjects."""
    return results[METRICS].agg(["mean", "std"]).T