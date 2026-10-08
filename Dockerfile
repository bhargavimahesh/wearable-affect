FROM python:3.11-slim

# uv, the same tool as on the laptop, copied from its official image (pinned version)
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /usr/local/bin/uv

# LightGBM needs the OpenMP runtime, which the slim base image doesn't include
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1

# 1) Dependencies only: this layer is reused as long as pyproject.toml and uv.lock don't change
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-default-groups --no-install-project

# 2) Our own code, installed as a normal (non-editable) package
COPY README.md ./
COPY src ./src
RUN uv sync --locked --no-default-groups --no-editable

# 3) The trained model
COPY artifacts/stress_detector.joblib /app/model/stress_detector.joblib
ENV MODEL_PATH=/app/model/stress_detector.joblib \
    PATH="/app/.venv/bin:$PATH"

# Run as an ordinary user, not root
RUN useradd --create-home appuser
USER appuser

EXPOSE 8080
CMD ["uvicorn", "wearable_affect.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]