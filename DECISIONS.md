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

# Questions
- what kinds of personalization for production?
- how to know this threshold for a new subject?
