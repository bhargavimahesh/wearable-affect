"""Personalization from a short calm calibration period at the start of each recording."""

from collections.abc import Callable

import numpy as np
import pandas as pd

from wearable_affect.evaluation import feature_columns

WINDOW_S = 60


def split_calibration(df: pd.DataFrame, minutes: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each subject's windows into calibration (first `minutes` of the recording) and the rest."""
    first_start = df.groupby("subject_id")["start_s"].transform("min")
    is_calibration = df["start_s"] + WINDOW_S <= first_start + minutes * 60

    calibration, rest = df[is_calibration], df[~is_calibration]
    if (calibration["target"] == 1).any():
        raise ValueError("Calibration windows contain stress; calibration must be calm only")
    missing = set(df["subject_id"]) - set(calibration["subject_id"])
    if missing:
        raise ValueError(f"No calibration windows for {sorted(missing)}")
    return calibration.reset_index(drop=True), rest.reset_index(drop=True)


def subtract_baseline(df: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    """Express every feature relative to the subject's own calibration mean."""
    cols = feature_columns(df)
    baseline = calibration.groupby("subject_id")[cols].mean()
    out = df.copy()
    out[cols] = df[cols] - baseline.loc[df["subject_id"]].to_numpy()
    return out


def loso_predict_personalised(
    df: pd.DataFrame,
    make_model: Callable,
    minutes: int,
    normalise: bool,
    quantile: float = 0.95,
) -> pd.DataFrame:
    """LOSO predictions on non-calibration windows, plus a personal threshold per subject.

    Calibration windows are never scored. They are used only to (a) express features relative to
    the subject's calm baseline (if `normalise`) and (b) set a personal threshold: the `quantile`
    of the model's stress probabilities on that subject's calm calibration windows.
    """
    calibration, rest = split_calibration(df, minutes)
    if normalise:
        rest = subtract_baseline(rest, calibration)
        calibration = subtract_baseline(calibration, calibration)
    cols = feature_columns(rest)

    folds = []
    for subject in rest["subject_id"].unique():
        train = rest[rest["subject_id"] != subject]
        model = make_model()  # fresh model per fold
        model.fit(train[cols], train["target"])

        test = rest[rest["subject_id"] == subject]
        calm = calibration[calibration["subject_id"] == subject]

        fold = test[["subject_id", "start_s", "target"]].copy()
        fold["prob"] = model.predict_proba(test[cols])[:, 1]
        fold["personal_threshold"] = np.quantile(model.predict_proba(calm[cols])[:, 1], quantile)
        folds.append(fold)
    return pd.concat(folds, ignore_index=True)