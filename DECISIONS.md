# Decisions

## 2026-09-27: Use uv for environment management
- **Decision:** One isolated Python environment per project, managed with uv.
- **Why:** Keeps this project's packages separate from my research setup, and 
- records exact versions in pyproject.toml / uv.lock so the environment can be
  recreated on any machine (including a cloud server later).
- **Alternatives:** conda, or plain pip + venv.
- **Consequence:** Always run code with the project's .venv, or via `uv run`.

## 2026-09-28
- Using wrist signals only (BVP, EDA, ACC, TEMP) to simulate a wearable product
- Alternative was to add chest ECG and respiration - which would likely improve 
  accuracy but wouldn't match a real deployment.

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
- Overlapping windows are strongly correlated, but harmless as we use LOSO.

- Windows belong to a single condition; "pure" windows. Clean data.

## 2026-09-30
- Neurokit vs. own feature extraction = NeuroKit2
- Why: We are working with simple features and lower frequences, NeuroKit might be an overkill and creates an overhead 
  as a dependency. But since we do not want to change later, we could use NeuroKit2 as it has validated methods.
- Neurokit does not use EDA cleaning at frequencies below 6Hz. We have 4Hz.

- Signal preprocessing needs to be performed on each segment.
- Why: because in production, there will be one window at a time instead of th ewhole recording. 
- This is a problem beecause model would be trained on differently prepared data.
- This is called training skew.






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

# Todo
- Classes are imabalanced (stress class fraction 29%). Balance or choose appropriate metric
- 
