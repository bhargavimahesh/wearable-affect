"""Create a stand-in detector trained on synthetic data, for CI.

The real model is derived from WESAD and never stored in GitHub. This one has the same structure
(feature names, pipeline, file format) so CI can test that the image and API work.

Usage: python scripts/make_ci_model.py OUTPUT_PATH
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from wearable_affect.detector import StressDetector
from wearable_affect.features import features_from_signals
from wearable_affect.models import make_logreg
from wearable_affect.synthetic import synthetic_signals

names = list(features_from_signals(synthetic_signals(60)).keys())
rng = np.random.default_rng(0)
model = make_logreg().fit(pd.DataFrame(rng.normal(size=(40, len(names))), columns=names), [0, 1] * 20)

detector = StressDetector(
    model=model,
    feature_names=names,
    metadata={"model": "ci-stand-in (synthetic data, not for use)", "sklearn_version": sklearn.__version__},
)
out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
detector.save(out)
print(f"Stand-in model saved to {out}")