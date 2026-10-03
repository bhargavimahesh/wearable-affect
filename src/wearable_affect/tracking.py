"""Log leave-one-subject-out experiments to MLflow."""

import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import mlflow
import pandas as pd

from wearable_affect.data import PROJECT_ROOT
from wearable_affect.evaluation import METRICS, feature_columns, loso_predict, score_predictions

TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
ARTIFACT_DIR = PROJECT_ROOT / "mlruns"


def _git_info() -> dict[str, str]:
    """Current commit hash, and whether there are uncommitted changes."""
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": str(bool(git("status", "--porcelain", "--", "src", "pyproject.toml", "uv.lock"))),
    }


def _simple_params(model) -> dict:
    """The model's settings that are plain numbers, strings, booleans or None."""
    return {
        k: v for k, v in model.get_params().items()
        if isinstance(v, (int, float, str, bool)) or v is None
    }


def run_loso_experiment(
    run_name: str,
    features: pd.DataFrame,
    make_model: Callable,
    params: dict | None = None,
    experiment: str = "wesad-stress",
) -> pd.DataFrame:
    """Run LOSO evaluation for one model and record everything about it in MLflow."""
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=ARTIFACT_DIR.as_uri())
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(_git_info())
        mlflow.log_params({
            "model_factory": make_model.__name__,
            "n_windows": len(features),
            "n_subjects": features["subject_id"].nunique(),
            "n_features": len(feature_columns(features)),
            **(params or {}),
        })
        mlflow.log_params({f"model.{k}": v for k, v in _simple_params(make_model()).items()})

        preds = loso_predict(features, make_model)
        results = score_predictions(preds)

        for metric in METRICS:
            mlflow.log_metric(f"{metric}_mean", results[metric].mean())
            mlflow.log_metric(f"{metric}_std", results[metric].std())

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            results.to_csv(tmp / "per_subject.csv", index=False)
            preds.to_csv(tmp / "predictions.csv", index=False)
            mlflow.log_artifacts(str(tmp))

    return results