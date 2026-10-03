import numpy as np
import pandas as pd

from wearable_affect.models import make_dummy, make_logreg
from wearable_affect.evaluation import feature_columns, loso_evaluate, loso_predict

def fake_feature_table(informative: bool) -> pd.DataFrame:
    """4 subjects x 20 windows; one feature either separates the classes or is pure noise."""
    rng = np.random.default_rng(0)
    rows = []
    for s in range(4):
        for i in range(20):
            target = int(i < 6)  # 30% stress, like WESAD
            signal = 3.0 * target if informative else 0.0
            rows.append(
                {
                    "subject_id": f"S{s}",
                    "start_s": i * 30,
                    "target": target,
                    "f1": signal + rng.normal(0, 0.1),
                    "f2": rng.normal(),
                }
            )
    return pd.DataFrame(rows)


def test_identifiers_and_target_are_never_features():
    df = fake_feature_table(informative=True)
    assert feature_columns(df) == ["f1", "f2"]


def test_one_result_row_per_subject():
    results = loso_evaluate(fake_feature_table(informative=True), make_logreg)
    assert sorted(results["subject_id"]) == ["S0", "S1", "S2", "S3"]


def test_informative_feature_is_learned():
    results = loso_evaluate(fake_feature_table(informative=True), make_logreg)
    assert results["auroc"].min() > 0.95


def test_majority_baseline_scores_at_chance():
    results = loso_evaluate(fake_feature_table(informative=True), make_dummy)
    assert np.allclose(results["balanced_accuracy"], 0.5)
    assert np.allclose(results["auroc"], 0.5)


def test_every_window_gets_exactly_one_prediction():
    df = fake_feature_table(informative=True)
    preds = loso_predict(df, make_logreg)
    assert len(preds) == len(df)
    assert not preds.duplicated(["subject_id", "start_s"]).any()