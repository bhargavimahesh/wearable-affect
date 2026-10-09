# Decisions

Design decisions in pipeline order, each with its reasoning. Dates show when a decision was made, where recorded.
Results and analyses are in [FINDINGS.md](FINDINGS.md).

**Contents:** 1. Environment and code structure · 2. Data and labels · 3. Windowing · 4. Features ·
5. Evaluation and models · 6. Experiment tracking · 7. Foundation models · 8. Label validity ·
9. Personalisation · 10. Probability calibration · 11. Deployable detector · 12. API · 13. Docker ·
14. CI · Lessons · Open questions

---

## 1. Environment and code structure

### Use uv for environment management (2026-09-27)
- **Decision:** One isolated Python environment per project, managed with uv.
- **Why:** Keeps this project's packages separate from my research setup, and records exact versions in
  pyproject.toml / uv.lock so the environment can be recreated on any machine (including a cloud server later).
- **Alternatives:** conda, or plain pip + venv.
- **Consequence:** Always run code with the project's .venv, or via `uv run`.

### Runtime vs development dependencies (2026-09-27)
- **Decision:** Jupyter, ipykernel, matplotlib and pytest are dev dependencies.
- **Why:** The production container should only include what the code needs to run.
- **Consequence:** Any library the code imports directly is declared explicitly (e.g. scikit-learn), even if
  another package already installs it.
- See also section 13: PyTorch moved to an `fm` group and LightGBM to `dev`.

### Reusable code lives in an installable package (src/wearable_affect) (2026-09-28)
- **Why:** Better than keeping code in notebooks and repeatedly copying it from one notebook to another for
  training, testing, etc. Changes may become untraceable and mistakes may get carried forward.

---

## 2. Data and labels

### Wrist signals only (BVP, EDA, ACC, TEMP) (2026-09-28)
- **Why:** Simulates a wearable product.
- **Alternatives:** Add chest ECG and respiration, which would likely improve accuracy but wouldn't match a
  real deployment.
- **Consequence:** Results are lower than chest-based numbers in the literature; that's expected.

### Binary classification: stress vs non-stress (2026-09-29)
- **Decision:** Stress is label 2. Non-stress is baseline (1) plus amusement (3). Everything else is dropped:
  transitions (0), meditation (4), and the "ignore" labels (5–7).
- **Why:** Matches the binary task defined in the original WESAD paper, so results can be directly compared.
- **Possible extension:** Include meditation as non-stress too.

---

## 3. Windowing

### 60-second windows, starting every 30 seconds (50% overlap) (2026-09-29)
- **Why:** Trade-off between sufficient information in a sample and sufficient inputs for training. EDA and
  heart rate variability change slowly, so very short windows don't contain enough of the response; about
  60 seconds is also common in the stress literature. The 30 s step gives more windows from limited data.
- Overlapping windows are strongly correlated, but this is harmless because we use LOSO.
- **Consequence:** ~1015 windows, ~30% stress, ~67 per subject.

### Single-condition ("pure") windows only (2026-09-29)
- **Why:** Windows spanning two conditions have no honest single label, so they are dropped. Each window
  belongs to a single condition: clean data.

### Windows are model-agnostic; each model gets an adapter (2026-09-29)
- **Decision:** Windowing knows nothing about models. A model with special input needs (e.g. PaPaGei: ~10 s
  PPG at 125 Hz) converts the 60 s window itself (resample, split, combine embeddings).
- **Why:** Every model is evaluated on exactly the same windows, labels and splits — a fair comparison.
  New models only need a new adapter.

---

## 4. Features

### NeuroKit2 vs own feature extraction: NeuroKit2 (2026-09-30)
- **Decision:** Simple handcrafted features first (EDA, TEMP, ACC, HR/HRV), using NeuroKit2 for PPG peaks and
  EDA decomposition (validated methods).
- **Why:** With simple features and low sampling frequencies, NeuroKit2 might be overkill and adds overhead as
  a dependency. But since we don't want to change later, we use NeuroKit2 because it has validated methods.

### Features computed per window, identically in training and serving (2026-09-30)
- **Why:** In production there will be one window at a time instead of the whole recording. Preprocessing
  whole recordings for training would mean the model is trained on differently prepared data than it sees
  in production. This is called training/serving skew.

### EDA: no cleaning filter; smoothmedian decomposition (2026-09-30, confirmed 2026-10-01)
- **Why:** NeuroKit2's EDA filter skips itself at ≤ 6 Hz (we have 4 Hz). The default high-pass decomposition
  (0.05 Hz) is unreliable on 60 s windows; smoothmedian works on short segments.

### Missing heart features are NaN, not imputed in the feature table (2026-10-01/02)
- **Why:** Wrist BVP is often noisy. When a window has too few detected beats, we don't know the heart rate,
  so we return NaN; a made-up value is worse than admitting it's missing. LightGBM handles NaN natively;
  logistic regression imputes inside its pipeline.

### SCRs counted with an absolute 0.01 µS threshold (2026-10-02)
- **Issue:** There were more SCR peaks in non-stress than in stress: ~40 SCRs per minute in non-stress
  (one every 1.5 seconds), fewer in the stress phase.
- **Cause:** NeuroKit2's SCR detector uses `amplitude_min=0.1` by default, a threshold *relative* to the
  largest peak in the window (anything above 10% of it). In a non-stress window, the largest peak can be a
  sensor artifact, so small noise produces many "peaks". Sensor noise was counted as SCRs in calm windows,
  reversing the stress effect.
- **Decision:** Absolute threshold of 0.01 µS.
- **Consequence:** SCR rates now ~7/min (non-stress) vs ~20/min (stress). Threshold chosen on physiological
  grounds, never by model score.

---

## 5. Evaluation and models

### Leave-one-subject-out (LOSO) evaluation
- **Why:** Avoids data leakage by design (see Lessons).

### Model factories, not model objects (`make_model`, not a fitted model) (2026-10-03)
- **Decision:** Evaluation receives a function that creates a fresh model (`make_model`); each fold gets
  its own object.
- **Why:** In LOSO, we want to be sure each fold's model is created afresh. Reusing one object can leak state
  between folds, and Python variables refer to the same object rather than copying it. The factory is also
  the single place a model's configuration is defined, so production uses exactly the evaluated settings.
- **Alternatives:** Pass a model object and copy it inside the loop with `sklearn.base.clone(model)`, which
  creates a fresh, unfitted copy with the same settings. But `make_model` is more explicit and also works
  for objects that scikit-learn's `clone` doesn't support.

### Baseline models: dummy, logistic regression, LightGBM (2026-10-03)
- **Dummy** (always "non-stress"): the floor. For our class balance, it gets macro F1 ≈ 0.41, balanced
  accuracy 0.50 and AUROC 0.50. Any model that doesn't clearly beat this has learned nothing.
- **Logistic regression:** simple, fast, interpretable. If it performs as well as LightGBM, the simpler
  model wins.
- **LightGBM:** the stronger baseline, which handles nonlinear effects and missing values natively.

### Keep window-level predictions, score separately (2026-10-03)
- **Decision:** `loso_predict` returns a probability for every window; `score_predictions` computes metrics
  from them.
- **Why:** Enables error analysis (e.g. inspecting one subject's probabilities) and re-scoring at different
  thresholds without retraining.

### Preprocessing inside the model pipeline (2026-10-03)
- **Why:** Imputer and scaler learn from training subjects only in each fold. Scaling the whole dataset first
  would let the test subject influence preprocessing (leakage).

### class_weight="balanced" (2026-10-03)
- **Why:** Mistakes on the rarer stress class count more during training.

### Fixed hyperparameters, no tuning on LOSO results (2026-10-03)
- **Why:** Choosing settings by best LOSO score uses the test subjects to decide, inflating results. Proper
  tuning needs nested cross-validation.

---

## 6. Experiment tracking

### MLflow for experiment tracking, with a local SQLite store
- **Decision:** Every LOSO evaluation runs through `run_loso_experiment`, which logs to MLflow. Records go in
  `mlflow.db` (SQLite) and artifact files in `mlruns/`, both in the project root.
- **Why:** Many variants are coming (personalisation, foundation models). Without tracking, results end up
  scattered across notebook outputs, and no number can be traced back to the code and settings that
  produced it.
- **Alternatives:** Manual results tables; Weights & Biases (hosted, needs an account).
- **Consequence:** `mlflow.db` and `mlruns/` are gitignored (generated output). Results worth keeping
  publicly go into FINDINGS.md. MLflow is a dev dependency (see Lessons).

### Each run logs parameters, model settings, metrics, per-subject results and predictions
- **Why:** Mean/std metrics are for comparing runs; per-subject results and window-level predictions allow
  error analysis (like the S14 investigation) on any past run without retraining.
- **Consequence:** A `features` parameter labels the feature-set version (e.g. `neurokit_v1`), so runs can be
  filtered by feature version.

### Artifact location fixed to the project root
- **Why:** By default MLflow saves artifacts relative to wherever the code runs, so notebook runs would put
  them in `notebooks/mlruns`. Same principle as DEFAULT_DATA_DIR: file locations must never depend on the
  working directory.

### Runs are tagged with the Git commit; code is committed before experiments
- **Why:** The commit hash identifies exactly what ran. `git_dirty` only checks `src/` and the dependency
  files, because notebooks change every time they run.
- **Working rule:** Commit code before running experiments, so each run's commit hash identifies exactly
  what ran.

---

## 7. Foundation models

### Why foundation models when the baseline already works well (AUROC 0.97) and personalisation is the problem?
- The foundation model is evaluated for average performance, label efficiency (fewer training subjects),
  robustness (high-motion windows), and EDA non-responders. A null result on average is a valid outcome.
- A statement like "PaPaGei matched but didn't beat handcrafted features on WESAD; it helped when training
  data was scarce and on high-motion windows, and here's why" is valuable. (Example framing written before the
  experiments; label efficiency and high-motion windows were not tested.)

### Swap one modality at a time
- **Why:** Keeping everything else identical (windows, splits, other features, models) means any change in
  performance comes from that foundation model alone. PaPaGei (BVP) first, then UME (EDA).
- **Status:** PaPaGei done; UME not yet.

### Check pretraining data for WESAD before using a foundation model
- **Why:** If WESAD was in the pretraining data, the model has seen our test subjects, and LOSO results would
  be optimistic.
- **Status:** Checked for PaPaGei: pretrained on VitalDB, MIMIC-III and MESA, not WESAD.

### PaPaGei or Pulse-PPG? PaPaGei first
- PaPaGei is trained on clinical datasets (VitalDB, MIMIC-III and MESA: typically clinical-grade finger PPG,
  125 Hz; ours is 64 Hz wrist PPG), so we have to handle dataset domain shift.
- Pulse-PPG might be a better fit, as it is pretrained on field data and might be more robust to motion.
- But Pulse-PPG requires 4-minute segments. Options: add history to the 60 s windows (may lead to impure
  labels), feed only 60 s, or redo the baseline models with 4-min windows (would leave very few inputs).
- We could use both: clinical-pretrained vs field-pretrained foundation models on wrist PPG.
- **Decision:** We continue with PaPaGei.

### PaPaGei: pretrained weights only
- **Why:** The PaPaGei repository is archived, and pretraining is not needed; we only use the released
  weights to compute embeddings.

---

## 8. Label validity

### Manipulation check (self-reports)
- **Why:** S14's EDA did not vary much (SCL change ~0.05 µS) but HRV features did. Self-reports showed that
  S14's valence was lowest and arousal highest during the stress phase. But did the stress manipulation work
  on everyone?
- **Criterion (fixed before looking):** the stress manipulation "worked" for a person if their self-reported
  arousal (SAM) after the TSST was higher than after baseline.
- **Result:** All 15 subjects reported higher arousal (change +1 to +7) and mostly lower valence. Labels are
  valid at the level of self-report.
- Subjective and electrodermal responses dissociate for some people: S17 reported the strongest stress
  (arousal 2 → 9, valence 7 → 1) but SCL dropped (1.28 → 1.05 µS); S14 reported strong stress (arousal 2 → 7)
  with almost no SCL change (0.29 → 0.34 µS).
- Weakest self-reported responders (S6, S9: +1) were not the hardest to classify (AUROC 0.92, 1.00).
  Self-reported intensity does not predict physiological detectability.

---

## 9. Personalisation

### Design: two approaches, both using only calm calibration windows
- **Approach 1, baseline subtraction:** subtract the person's calm mean from each feature.
- **Approach 2, personal threshold:** after training, the model scores the test person's calibration windows;
  their threshold is set at the 95th percentile of those calm probabilities ("flag stress when a probability
  is higher than 95% of your calm moments").
- **Variants:** 2 normalisation options × 2 threshold options × 3 calibration lengths (2, 5, 10 minutes) ×
  2 models = 24 runs.
- **Features:** handcrafted only; PaPaGei not used, as it did not yield a consistent improvement.

### Calibration = first minutes of each recording; never scored
- **Why:** Matches a real product (the first minutes after putting the device on) and needs no stress labels
  from the new user. Calibration windows are excluded from scoring, so generic and personalised models are
  evaluated on exactly the same windows. The code checks that calibration windows are calm and raises an
  error otherwise.

### Baseline normalisation by subtraction, per feature
- **Why:** Expresses each feature as a change from the person's own calm state, removing person-specific
  offsets (e.g. S7's high SCL). Subtraction, not division: some features can be zero or negative (slopes,
  SCR counts), so division could mean dividing by zero. Per feature, so a cardiac response still counts when
  EDA doesn't respond. Training subjects are normalised the same way, so the model learns on the same kind of
  features it is tested on.

### Success criterion fixed before running
- **Decision:** A personalised variant counts as an improvement if its mean macro F1 is higher than the
  generic model's at the same calibration length, AND more subjects improve than get worse (ties excluded).
- **Why:** The best average can hide losses for many people (10-min logistic regression had the highest mean
  but failed the per-subject test).

### Personal threshold from calm data: rejected
- **Why:** Worse than the fixed 0.5 threshold in all 12 comparisons. Quiet sitting doesn't represent all
  non-stress states, so the threshold ends up too low. Not tuned further, to avoid searching variants until
  one works.

### Deployed configuration: logistic regression + baseline normalisation, 5 min, threshold 0.5
- **Why:** Highest mean F1 among variants meeting the criterion (0.896, 9 better / 5 worse); 5 minutes is a
  reasonable onboarding request; logistic regression is simpler, interpretable, and likely better calibrated
  than LightGBM.
- **Caveat:** Chosen after seeing results among 12 variants on the same 15 subjects, so its score is somewhat
  optimistic; an independent dataset would be needed for an unbiased estimate.

---

## 10. Probability calibration

### No probability recalibration
- **Why:** The chosen logistic regression is well calibrated on LOSO predictions (Brier 0.076).

---

## 11. Deployable detector

### Final detector trained on all 15 subjects
- **Why:** LOSO estimates how the procedure performs for a new person; the deployed model should learn from
  all available data.

### One feature function (features_from_signals) for training and serving
- **Why:** Prevents training/serving skew. A test runs the same recording through both paths and requires
  identical probabilities.

### Stateless detector: baselines are computed and returned, not stored
- **Why:** The service needs no memory of users, can run as many copies as needed, and keeps no personal
  data. Where baselines live is decided in the API step.

### Detector saved with joblib, including feature order and training metadata
- **Why:** Model, feature order and threshold travel together. The scikit-learn version is recorded and
  checked at load time, because saved models are only reliable with the same version.

---

## 12. API

### API: /health, /calibrate, /predict with FastAPI
- **Why:** Typed request validation (Pydantic) and automatic interactive documentation.

### Stateless API: the client keeps the baseline and sends it with each prediction
- **Why:** The server stores no personal data and no user state; any number of copies can answer any request.

### Validate all input at the boundary
- **Why:** A model given malformed data usually returns plausible nonsense rather than crashing. Structural
  problems → 422; valid but unacceptable requests → 400, always with a clear message.

### Missing features reported in every prediction
- **Why:** A first signal-quality indicator; e.g. a flat pulse (lost skin contact) means no heart features,
  which the client should know.

### Model location from the MODEL_PATH environment variable
- **Why:** The same code runs on a laptop and in a container without changes.

---

## 13. Docker

### Docker image: python:3.11-slim, uv with the same version as development
- **Why:** Runs identically on the laptop and in the cloud; uv reads uv.lock exactly as it was written.

### Serving image contains runtime dependencies only
- **Decision:** PyTorch moved to an `fm` dependency group and LightGBM to `dev` (with a lazy import in
  `make_lightgbm`, so importing `models.py` doesn't require it). The image installs with --no-default-groups:
  no dev tools, no PyTorch, no LightGBM. Locally, default-groups keeps everything.
- **Why:** The deployed detector (logistic regression) uses neither; the CUDA build of PyTorch alone is
  several GB. Verified with an isolated install of runtime dependencies only.
- **Result:** Image 280 MB compressed / 1.16 GB on disk.

### Dependency layer before code layer
- **Why:** Code changes rebuild in seconds, because the slow dependency layer is reused.

### Container runs as a non-root user
- **Why:** Limits what an attacker could do if the service were ever compromised.

### Model baked into the image; image kept private
- **Why:** Simple and reproducible for now. The model is derived from WESAD, so the image must not be
  published. How CI builds get the model: see section 14.

### Container built and tested in GitHub Codespaces
- **Why:** Docker can't be installed on the development laptop (managed computer). Codespaces provides a Linux
  machine with Docker in the browser; GitHub Actions automates the same build (section 14).

### Smoke-test script for a running API (scripts/smoke_test_api.py)
- **Why:** One command checks health, calibration and prediction, and that the API's answer equals the
  detector's. Reusable for the container, CI and the cloud deployment.

---

## 14. CI

### CI with GitHub Actions: tests, then image build and smoke test, on every push and pull request
- **Why:** Problems surface within minutes on a clean machine, not weeks later on someone else's. The Docker
  job only runs if the tests pass.

### CI uses a stand-in model trained on synthetic data
- **Why:** The real model is derived from WESAD and must not be stored in GitHub. CI checks the software
  (code, dependencies, image, API); the model's quality is established by LOSO evaluation. The real model
  enters only the deployed image, from private storage.

### PyTorch-dependent tests skip when the `fm` group isn't installed
- **Why:** CI has no GPU, and installing CUDA PyTorch on every run would be slow; the deployed service
  doesn't use it.

---

# Lessons

## Environment and tooling
- When a package is missing even after installing, first verify which Python is running:
  `import sys; print(sys.executable)`.
- Open the project folder itself in VS Code, otherwise it won't find .venv.
- Close all notebooks before installing a package, or commit everything first.
- On Windows, stop notebook kernels before installing packages; files in use can't be replaced, leaving
  half-installed packages. If the environment acts strangely, rebuild it from uv.lock.
- Python variables refer to objects; appending the same object 15 times gives 15 references to one object.
- MLflow goes in the dev group because it's used for running experiments, not for serving predictions.
- MLflow 3 lists classic runs under 'Training runs'. When a web UI seems empty, check the server log to see
  whether data was actually requested. http://127.0.0.1:5000/#/experiments/1/runs
- GitHub only finds workflows in .github/workflows/ (dot, plural); with no workflow file, the Actions tab shows
  template suggestions instead of runs.

## Data and physiology
- EDA rises during the stress condition, which is the classic SNS response. But EDA doesn't drop back
  immediately after the stress condition ends; it recovers slowly. So a window just after stress can still
  look stressed, even though its label says otherwise.
- The absolute EDA level differs a lot between people. One person's "calm" can be higher than another's
  "stressed" — a strong argument for normalising per person.
- The sampling rate of the labels is 700 Hz.
- S14's best threshold is 0.05, which means something about their physiology is very calm (not much change:
  SCL mean 0.29 → 0.34).

## Windowing and design
- Better to define windows in *seconds*, not samples. Each signal has a different sampling rate.
- Integers are better (whole seconds instead of minutes).
- 60 s windows (64 Hz) are for analysis. It does not matter what PaPaGei expects (10 s segments, 125 Hz);
  an input adapter is needed later. Only ensure that the segment lengths are multiples of each other and
  that there is no large information loss.

## Features
- Validating features against domain knowledge before modelling caught a bug that tests alone would have
  missed (SCR detection).

## Evaluation and metrics
- LOSO avoids data leakage by design.
- Looking at models that work badly for every subject tells you about that subject's data quality.
- AUROC is the chance that the model gives a higher stress score to a randomly picked stressed window than to
  a randomly picked calm window (orders or ranks the pair correctly). A low threshold catches almost all
  stress but raises many false alarms; a high threshold does the opposite. Plotting these two rates for every
  threshold traces a curve, the ROC curve. AUROC measures ranking quality, independent of any threshold.
  F1 and balanced accuracy measure the actual yes/no decisions at one chosen threshold.
- High AUROC with low F1 means the model ranks correctly but the threshold is wrong for that person.
- Averages hide failures: always look at per-subject results.
- When an optimum lands on the edge of a search grid, the true optimum is probably beyond it.
- Compare variants only on identical test windows (here: at the same calibration length).

## Experiment practice
- Write success criteria and predictions down before running an experiment; it protects against
  reinterpreting results afterwards, and makes confirmed predictions meaningful.
- The best mean is not the most reliable choice; check how many individuals improve.

## Personalisation
- A personalisation method rests on its assumptions: if the calibration minutes are not representative,
  normalisation makes things worse for that person.!!
- Calm-only data can describe a person's baseline, but not where their decision threshold should be.!!
- The first 5 minutes of each recording (calm baseline) were used in two tests:
  - Baseline subtraction: the mean of each feature over the 9 calibration windows was subtracted from all
    later windows. This improved mean macro F1 (logistic regression: 0.839 → 0.896), though not for every
    subject (9 better, 5 worse).
  - Personal threshold: the model's predicted probabilities on the calibration windows were used to set a
    person-specific threshold (their 95th percentile). This performed worse than the fixed 0.5 threshold in
    all 12 comparisons, and was therefore rejected.
  - Only baseline subtraction was retained; the deployed model uses a fixed 0.5 threshold for everyone.

## Probability calibration
- Probability calibration checks whether the model's predicted probabilities can be taken at face value.
  E.g. for 0.8: collect all windows predicted around 0.8 and check their true labels. If about 80% are stress,
  the model is calibrated at that level; if fewer, it is overconfident; if more, underconfident. Repeating
  this for every probability range gives the reliability diagram.
- Calibration is separate from ranking (AUROC) and from decisions at a threshold (F1): a model can rank windows
  well and still give probabilities that are too high or too low.
- Training and (output) calibration are not the same. Training is learning weights for each feature;
  calibration describes the quality of the output and whether output probabilities can be taken at face
  value. A trained model is not always well calibrated. Here, the logistic regression model is already well
  calibrated, without needing an extra step such as Platt scaling or isotonic regression.
- The 5-minute personal calibration is something else again: it measures a person's calm baseline, not the
  model's probabilities. So in this project, "calibration" can mean the person's baseline (input), the
  quality of the probabilities (output), or a correction step for those probabilities.

---

# Open questions
- What kinds of personalisation for production?
- How to know the threshold for a new subject? (Partly answered: a calm-based personal threshold was tested
  and rejected; the deployed model uses 0.5 for everyone — see section 9.)
