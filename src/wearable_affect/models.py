"""Model factories: each call returns a fresh, untrained model."""

from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_dummy():
    """Always predicts the majority class (non-stress). The floor any real model must beat."""
    return DummyClassifier(strategy="most_frequent")


def make_logreg():
    """Linear model: fill missing values, scale features, then logistic regression."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )


def make_lightgbm():
    """Gradient-boosted trees; handles missing values natively."""
    return LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        class_weight="balanced",
        random_state=0,
        verbose=-1,
    )