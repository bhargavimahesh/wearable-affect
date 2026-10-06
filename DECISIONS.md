# Decisions

## 2026-09-27: 
- Use uv for environment management
- **Decision:** One isolated Python environment per project, managed with uv.
- **Why:** Keeps this project's packages separate from my research setup, and 
- records exact versions in pyproject.toml / uv.lock so the environment can be
  recreated on any machine (including a cloud server later).
- **Alternatives:** conda, or plain pip + venv.
- **Consequence:** Always run code with the project's .venv, or via `uv run`.

- Runtime vs development dependencies
- **Decision:** Jupyter, ipykernel, matplotlib and pytest are dev dependencies.
- **Why:** The production container should only include what the code needs to run.
- **Consequence:** Any library the code imports directly is declared explicitly (e.g. scikit-learn),
  even if another package already installs it.

## 2026-09-28
- Using wrist signals only (BVP, EDA, ACC, TEMP) to simulate a wearable product
- Alternative was to add chest ECG and respiration - which would likely improve 
  accuracy but wouldn't match a real deployment.
- Results are lower than chest-based numbers in the literature; that's expected.

- Reusable code lives in an installable package (src/wearable_affect)
- Better than having on notebooks and repeatedly copying from one notebook to
  other for training, testing, etc. Changes may become intraceable and mistakes 
  may get carried forward.

## 2026-09-29
- Binary classification of stress: stress vs. non-stress. 
  Stress is label 2. 
  Non-stress is baseline (1) plus amusement (3). 
  Everything else is dropped: transitions (0), meditation (4), and the "ignore" labels (5–7).
- Why: matches the binary task defined in the original WESAD paper, so results can be directly compared.
- Extension: including meditation as non-stress too.

- Window length: 60-second windows, starting every 30 seconds (50% overlap)
  Trade-off between sufficient information in a sample vs sufficient inputs for training.
  EDA and heart rate variability change slowly, so very short windows don't 
  contain enough of the response; about 60 seconds is also common in the stress literature.
  The 30 s step gives more windows from limited data; 
- Windows spanning two conditions have no honest single label, so they are dropped.
- Overlapping windows are strongly correlated, but harmless as we use LOSO.
- **Consequence:** ~1015 windows, ~30% stress, ~67 per subject.
- Windows belong to a single condition; "pure" windows. Clean data.

- Windows are model-agnostic; each model gets an adapter
- **Decision:** Windowing knows nothing about models. A model with special input needs (e.g. PaPaGei:
  ~10 s PPG at 125 Hz) converts the 60 s window itself (resample, split, combine embeddings).
- **Why:** Every model is evaluated on exactly the same windows, labels and splits — a fair
  comparison. New models only need a new adapter.


## 2026-09-30
- Neurokit vs. own feature extraction = NeuroKit2
- Simple handcrafted features first (EDA, TEMP, ACC, HR/HRV) 
- NeuroKit2 for PPG peaks and EDA decomposition (validated methods)
- Why: We are working with simple features and lower frequences, NeuroKit might be an overkill and creates an overhead 
  as a dependency. But since we do not want to change later, we could use NeuroKit2 as it has validated methods.

- EDA: no cleaning filter; smoothmedian decomposition
- **Why:** NeuroKit2's EDA filter skips itself at ≤ 6 Hz (we have 4 Hz). The default high-pass
  decomposition (0.05 Hz) is unreliable on 60 s windows; smoothmedian works on short segments.

- Signal preprocessing needs to be performed on each segment.
- Why: because in production, there will be one window at a time instead of th ewhole recording. 
- This is a problem beecause model would be trained on differently prepared data.
- This is called training skew.

## 01.10.2026
- Wrist BVP is often noisy. When a window has too few detected beats, we return NaN
- Smoothmedian EDA decomposition, because a high-pass filter is unreliable on 60 s windows.
- Missing heart features are NaN, not imputed

## 02.10.2026
- EDA: there were more SCR peaks in non-stress than stress
- SCR detection now uses an absolute 0.01 µS threshold. NeuroKit2's default relative threshold counted sensor noise as SCRs   in calm windows and reversed the stress effect
- Issue: 40 SCRs per minute in non-stress, meaning one SCR every 1.5 seconds. Stress phase had lesser.
- Neurokit2's SCR detector's default peak detection threshold is amplitude_min=0.1 (anything above 10% of the largest peak in that window). In non-stress window, largest peak could be sensor artifact. small noises = more peaks. 
- SCRs counted with an absolute 0.01 µS threshold
- **Why:** NeuroKit2's default threshold is relative to the largest peak in the window, so in calm
  windows sensor noise was counted as SCRs (~40/min), reversing the stress effect.
- **Consequence:** SCR rates now ~7/min (non-stress) vs ~20/min (stress). Threshold chosen on
  physiological grounds, never by model score.

- Missing heart features are NaN, not imputed in the feature table
- **Why:** Too few detected beats means we don't know the heart rate; a made-up value is worse than
  admitting it's missing. LightGBM handles NaN natively; logistic regression imputes inside its
  pipeline.

## 03.10.2026
- Model factories, not model objects (make_model instead of fit)
- Evaluation receives a function that creates a fresh model (`make_model`).
- Why: Reusing one object can leak state between folds, and Python variables refer to the same object rather than copying it
- Why the function instead of model object? Because in LOSO setting, we want to be sure that the model is created afresh.
- The factory is also the single place a model's configuration is defined, so production uses exactly the evaluated settings.
- Each fold gets its own object.
- Alternative: scikit-learn offers another approach: pass a model object and copy it inside the loop with sklearn.base.clone(model), which creates a fresh, unfitted copy with the same settings. But make_model is more explicit and is applicable to objects that are not supported by scikit clone.


- Baseline models: dummy, logistic regression, lightGBM 
- Dummy (always "non-stress"): the floor. For our class balance, it gets macro F1 ≈ 0.41, balanced accuracy 0.50, and AUROC 0.50. Any model that doesn't clearly beat this has learned nothing.
- Logistic regression: simple, fast, interpretable. If it performs as well as LightGBM, the simpler model wins.
- LightGBM: the stronger baseline, which handles nonlinear effects and missing values natively.

- Keep window-level predictions, score separately
- **Decision:** `loso_predict` returns a probability for every window; `score_predictions` computes
  metrics from them.
- **Why:** Enables error analysis (e.g. inspecting one subject's probabilities) and re-scoring at
  different thresholds without retraining.

- Preprocessing inside the model pipeline
- **Why:** Imputer and scaler learn from training subjects only in each fold. Scaling the whole
  dataset first would let the test subject influence preprocessing (leakage).

- class_weight="balanced"
- **Why:** Mistakes on the rarer stress class count more during training.

- Fixed hyperparameters, no tuning on LOSO results
- **Why:** Choosing settings by best LOSO score uses the test subjects to decide, inflating results.
  Proper tuning needs nested cross-validation.


- MLflow for experiment tracking, with a local SQLite store
- **Decision:** Every LOSO evaluation runs through `run_loso_experiment`, which logs to MLflow.
  Records go in `mlflow.db` (SQLite) and artifact files in `mlruns/`, both in the project root.
- **Why:** Many variants are coming (personalisation, foundation models). Without tracking, results
  end up scattered across notebook outputs, and no number can be traced back to the code and
  settings that produced it.
- **Alternatives:** Manual results tables; Weights & Biases (hosted, needs an account).
- **Consequence:** `mlflow.db` and `mlruns/` are gitignored (generated output). Results worth
  keeping publicly go into FINDINGS.md.

- Each run logs parameters, model settings, metrics, per-subject results and predictions
- **Why:** Mean/std metrics are for comparing runs; per-subject results and window-level
  predictions allow error analysis (like the S14 investigation) on any past run without retraining.
- **Consequence:** A `features` parameter labels the feature-set version (e.g. `neurokit_v1`),
  so runs can be filtered by feature version.

- Artifact location fixed to the project root
- **Why:** By default MLflow saves artifacts relative to wherever the code runs, so notebook runs
  would put them in `notebooks/mlruns`. Same principle as DEFAULT_DATA_DIR: file locations must
  never depend on the working directory.

- Runs are tagged with the Git commit; code is committed before experiments
- **Why:** The commit hash identifies exactly what ran. `git_dirty` only checks `src/` and the
  dependency files, because notebooks change every time they run.
- **Working rule:** Commit code before running experiments, so each run's commit hash identifies
  exactly what ran.

### Why foundation models when baseline models already work well (AUROC=0.97), and personalization is the problem instead?
- Foundation model evaluated for average performance, label efficiency (fewer training subjects), robustness (high-motion windows), and EDA non-responders. A null result on average is a valid outcome.
- "PaPaGei matched but didn't beat handcrafted features on WESAD; it helped when training data was scarce and on high-motion windows, and here's why"... is valuable

### PaPaGei or Pulse-PPG?
- PaPaGei is trained on clinical datasets, so we need to handle dataset domain shift (VitalDB, MIMIC-III, and MESA,
typically clinical-grade finger PPG, 125 Hz; Ours 64 Hz)
- Pulse-PPG might be a better fit as it is pre-trained on field data, might be more robust for motion
- But it requires 4-minute segments (add history to 60-sec outputs - may lead to impure labels, feed only 60 sec, or redo baseline models with 4-min segments - will lead to very few inputs)
- We could use both: clinical-pretrained vs. field-pretrained foundation models on wrist PPG
- We continue with PaPaGEi

### Papager weights only
- papagei is archived; pretraining not needed

### Manipulation check (self-reports)
- the stress manipulation "worked" for a person if their self-reported arousal was higher after the TSST than after baseline
- Why: S14's EDA did not vary much (0.3) but HRV features did. Self-reports show that the valence was the lowest and arousal was the highest during Stress phase. But did stress manipulation work on everyone?
- Criterion fixed before looking: stress manipulation "worked" if self-reported arousal (SAM)
  after TSST > after baseline.
- Result: all 15 subjects reported higher arousal (change +1 to +7) and mostly lower valence.
  Labels are valid at the level of self-report.
- Subjective and electrodermal responses dissociate for some people: S17 reported the strongest
  stress (arousal 2 -> 9, valence 7 -> 1) but SCL dropped (1.28 -> 1.05 µS); S14 reported strong
  stress (arousal 2 -> 7) with almost no SCL change (0.29 -> 0.34 µS).
- Weakest self-reported responders (S6, S9: +1) were not the hardest to classify (AUROC 0.92,
  1.00). Self-reported intensity does not predict physiological detectability.

### Personalization approaches
- Two approaches, both using only the calm windows 
- Approach 1: Baseline subtraction (not division, may end up dividing by zero), Each feature is normalised separately.
- Approach 2: After training, the model scores the test person's calibration windows. Their threshold is set at the 95th percentile of those calm probabilities: "flag stress when a probability is higher than 95% of your calm moments."
- Success criterion: a personalised variant counts as an improvement if its mean macro F1 is higher than the generic model's at the same calibration length, and more subjects improve than get worse (ties excluded).
- Not using PAPAGei as it did not yield consistent improvement 
- Variants: 2 normalisation options × 2 threshold options × 3 calibration lengths (2, 5, 10 minutes) × 2 models = 24 runs.

## Personalisation

### Calibration = first minutes of each recording; never scored
- **Why:** Matches a real product (the first minutes after putting the device on) and needs no
  stress labels from the new user. Calibration windows are excluded from scoring, so generic and
  personalised models are evaluated on exactly the same windows. The code checks that
  calibration windows are calm and raises an error otherwise.

### Baseline normalisation by subtraction, per feature
- **Why:** Expresses each feature as a change from the person's own calm state, removing
  person-specific offsets (e.g. S7's high SCL). Subtraction, not division: some features can be
  zero or negative (slopes, SCR counts). Per feature, so a cardiac response still counts when EDA
  doesn't respond. Training subjects are normalised the same way, so the model learns on the
  same kind of features it is tested on.

### Success criterion fixed before running
- **Decision:** Mean macro F1 higher than generic at the same calibration length, AND more
  subjects improve than get worse.
- **Why:** The best average can hide losses for many people (10-min logistic regression had the
  highest mean but failed the per-subject test).

### Personal threshold from calm data: rejected
- **Why:** Worse than the fixed 0.5 threshold in all 12 comparisons. Quiet sitting doesn't
  represent all non-stress states, so the threshold ends up too low. Not tuned further, to avoid
  searching variants until one works.

### Deployed configuration: logistic regression + baseline normalisation, 5 min, threshold 0.5
- **Why:** Highest mean F1 among variants meeting the criterion (0.896, 9 better / 5 worse);
  5 minutes is a reasonable onboarding request; logistic regression is simpler, interpretable,
  and likely better calibrated than LightGBM.
- **Caveat:** Chosen after seeing results among 12 variants on the same 15 subjects, so its score
  is somewhat optimistic; an independent dataset would be needed for an unbiased estimate.

## Calibration
- No probability recalibration: the chosen logistic regression is well calibrated on LOSO predictions (Brier 0.076)

## Detector
### Final detector trained on all 15 subjects
- **Why:** LOSO estimates how the procedure performs for a new person; the deployed model should
  learn from all available data.

### One feature function (features_from_signals) for training and serving
- **Why:** Prevents training/serving skew. A test runs the same recording through both paths and
  requires identical probabilities.

### Stateless detector: baselines are computed and returned, not stored
- **Why:** The service needs no memory of users, can run as many copies as needed, and keeps no
  personal data. Where baselines live is decided in the API step.

### Detector saved with joblib, including feature order and training metadata
- **Why:** Model, feature order and threshold travel together. The scikit-learn version is recorded
  and checked at load time, because saved models are only reliable with the same version.


# Lessons
- When a package is missing even after installing, first verify which python is running. `import sys; print(sys.executable)`
- Open the project folder itself in VS Code, otherwise it won't find .venv.
- EDA: It rises during the stress condition, which is the classic SNS response. But EDA doesn't 
  drop back immediately after the stress condition ends; it recovers slowly. So a window just 
  after stress can still look stressed, even though its label says otherwise. 
  Second, the absolute EDA level differs a lot between people. 
  One person's "calm" can be higher than another's "stressed" - strong argument for normalising per person
- Sampling rate of Labels is 700. 
- Better to define Windows in *seconds*, not samples. Each signal has different Fs.
- Integers are better (seconds instead of minutes)
- 60-sec Windows (64 Hz) are for analysis. It does not matter what the PaPaGei expects (10-sec segments, 125 Hz). 
  An input adapter is needed later. Only need to ensure that the segments are multiples of the other and not a huge info loss.
- Close all notebooks before installing a package or commit everything.
- Validating features against domain knowledge before modelling caught a bug that tests alone would have missed. (SCR detection)
- LOSO avoids data leakage by design.
- On Windows, stop notebook kernels before installing packages; files in use can't be replaced,
  leaving half-installed packages. If the environment acts strangely, rebuild it from uv.lock.
- Python variables refer to objects; appending the same object 15 times gives 15 references
  to one object.
- By looking at models that work badly for every subject tells about the subject's data quality
- AUROC is the chance that the model gives a higher stress score to a randomly picked stressed window than to a randomly picked calm window (ordered or ranked the pair correctly). So when AUROC is high and F1 is low for a subject, it could mean that the probability threshold is wrong for the person. A low threshold catches almost all stress but raises many false alarms; a high threshold does the opposite. Plotting these two rates for every threshold traces a curve, the ROC curve. AUROC measures ranking quality, independent of any threshold. F1 and balanced accuracy measure the actual yes/no decisions at one chosen threshold.
- S14's best threshold is 0.05, which means something about their physiology is very calm (not much change, SCL mean 0.29 to 0.34)
- Averages hide failures: always look at per-subject results.
- High AUROC with low F1 means the model ranks correctly but the threshold is wrong.
- When an optimum lands on the edge of a search grid, the true optimum is probably beyond it.
- MLflow goes in the dev group because it's used for running experiments, not for serving predictions.
- MLflow 3 lists classic runs under 'Training runs'. When a web UI seems empty, check the server log to see whether data was actually requested. http://127.0.0.1:5000/#/experiments/1/runs 
- Write success criteria and predictions down before running an experiment; it protects against
  reinterpreting results afterwards, and makes confirmed predictions meaningful.
- The best mean is not the most reliable choice; check how many individuals improve.
- Compare variants only on identical test windows (here: at the same calibration length).
- A personalisation method rests on its assumptions: if the calibration minutes are not
  representative, normalisation makes things worse for that person.!!
- Calm-only data can describe a person's baseline, but not where their decision threshold
  should be.!!
  
# Questions
- what kinds of personalization for production?
- how to know this threshold for a new subject?

# todos
### Foundation models: swap one modality at a time
- **Why:** Keeping everything else identical (windows, splits, other features, models) means any
  change in performance comes from that foundation model alone. PaPaGei (BVP) first, then UME (EDA).

### Check pretraining data for WESAD before using a foundation model
- **Why:** If WESAD was in the pretraining data, the model has seen our test subjects, and LOSO
  results would be optimistic.