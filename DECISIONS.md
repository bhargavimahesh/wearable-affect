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

# Lessons
- When a package is missing even after installing, first verify which python is running. `import sys; print(sys.executable)`
- Open the project folder itself in VS Code, otherwise it won't find .venv.
- EDA: It rises during the stress condition, which is the classic SNS response. But EDA usually doesn't drop back immediately after the stress condition ends; it recovers slowly. So a window taken just after stress can still look stressed, even though its label says otherwise. Second, the absolute EDA level differs a lot between people. One person's "calm" can be higher than another's "stressed" - strong argument for normalising per person

