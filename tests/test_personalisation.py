import numpy as np
import pandas as pd
import pytest

from wearable_affect.models import make_logreg
from wearable_affect.personalisation import loso_predict_personalised, split_calibration, subtract_baseline


def fake_table(offsets: dict[str, float]) -> pd.DataFrame:
    """Per subject: 20 calm windows, then 10 stress windows (30 s apart).

    Stress raises `f` by 1 above the subject's own level; levels differ by `offsets`.
    """
    rng = np.random.default_rng(0)
    rows = []
    for subject, offset in offsets.items():
        for i in range(30):
            target = int(i >= 20)
            rows.append({"subject_id": subject, "start_s": 100 + i * 30, "target": target,
                         "f": offset + target + rng.normal(0, 0.1)})
    return pd.DataFrame(rows)


def test_calibration_takes_first_minutes_of_each_subject():
    df = fake_table({"S1": 0.0, "S2": 0.0})
    calibration, rest = split_calibration(df, minutes=2)
    assert list(calibration.groupby("subject_id").size()) == [3, 3]  # starts at 0, 30, 60 s
    assert len(calibration) + len(rest) == len(df)


def test_calibration_must_be_calm():
    df = fake_table({"S1": 0.0})
    with pytest.raises(ValueError):
        split_calibration(df, minutes=15)  # 15 min reaches into the stress windows


def test_subtract_baseline_removes_personal_level():
    df = fake_table({"S1": 0.0, "S2": 5.0})
    calibration, rest = split_calibration(df, minutes=5)
    normalised = subtract_baseline(rest, calibration)
    calm_means = normalised[normalised["target"] == 0].groupby("subject_id")["f"].mean()
    assert np.allclose(calm_means, 0, atol=0.1)  # both subjects' calm level is now ~0


def test_normalisation_helps_shifted_subject():
    # S4's calm level equals everyone else's stress level, like S7 in WESAD.
    df = fake_table({"S1": 0.0, "S2": 0.0, "S3": 0.0, "S4": 1.0})
    raw = loso_predict_personalised(df, make_logreg, minutes=5, normalise=False)
    norm = loso_predict_personalised(df, make_logreg, minutes=5, normalise=True)

    def s4_accuracy(preds):
        s4 = preds[preds["subject_id"] == "S4"]
        return ((s4["prob"] >= 0.5) == s4["target"]).mean()

    assert s4_accuracy(norm) > s4_accuracy(raw)


def test_each_subject_gets_one_personal_threshold():
    df = fake_table({"S1": 0.0, "S2": 0.0, "S3": 1.0})
    preds = loso_predict_personalised(df, make_logreg, minutes=5, normalise=False)
    assert (preds.groupby("subject_id")["personal_threshold"].nunique() == 1).all()