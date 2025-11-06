# ---------- Base ----------
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# ---------- System deps (needed for some ML wheels and builds) ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev git curl && \
    rm -rf /var/lib/apt/lists/*

# ---------- Poetry ----------
RUN pip install --no-cache-dir poetry==1.8.3
# Copy only dependency files first to leverage Docker layer cache
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root --no-interaction --no-ansi

# ---------- Project ----------
COPY . .

# Prefect/MLflow/Streamlit ports (optional)
EXPOSE 4200 8501

# ---------- Entrypoint ----------
CMD ["poetry", "run", "streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
