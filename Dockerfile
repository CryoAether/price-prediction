# ---------- Base ----------
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# ---------- System deps ----------
RUN apt-get update && apt-get install -y build-essential libpq-dev git curl && rm -rf /var/lib/apt/lists/*

# ---------- Poetry ----------
RUN pip install poetry==1.8.3
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root --no-interaction --no-ansi

# ---------- Project ----------
COPY . .

# Prefect/MLflow ports (optional)
EXPOSE 4200 8501

# ---------- Entrypoint ----------
CMD ["poetry", "run", "streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
