# Price Prediction for eBay Listings

Second-generation machine learning project to **predict the final sale price** and **probability of sale within N days** for eBay listings.  
This version extends the original model with improved data pipelines, feature engineering, automated retraining, and explainability reports.  

📄 See detailed documentation in [`reports/model_cards`](./reports/model_cards) and [`docs/`](./docs/).

---

## Overview

This project implements a full end-to-end ML workflow for price prediction and listing analytics using eBay listing data.  
It integrates data ingestion, feature engineering, model training, evaluation, and interactive visualization in a reproducible environment.

### Key Highlights
- **ETL & Warehousing:** Efficient data ingestion with DuckDB, Polars, and SQLAlchemy.  
- **Feature Engineering:** Text processing, categorical encoding, temporal features, and rolling window statistics.  
- **Modeling:** Multiple regressors and classifiers (Linear, LightGBM, XGBoost, CatBoost) trained via scikit-learn pipelines.  
- **Explainability:** SHAP values, permutation importance, and partial dependence plots generated automatically.  
- **Automation:** Prefect flows orchestrate daily retraining, validation, and logging to MLflow.  
- **Interface:** Streamlit web app for real-time prediction, insights, and model visualization.

---

## Tech Stack

| Layer | Technologies |
|-------|---------------|
| **Language** | Python 3.12 |
| **Core Libraries** | pandas, polars, numpy, scikit-learn |
| **ML/Stats** | LightGBM, XGBoost, CatBoost, statsmodels |
| **Pipelines** | Prefect, Great Expectations |
| **Experiment Tracking** | MLflow |
| **Storage / DB** | DuckDB, PostgreSQL |
| **Interface** | Streamlit |
| **Containerization** | Docker, Poetry |
| **Explainability** | SHAP, permutation importance, ICE plots |

---

## 🧩 Project Structure

```
price-prediction/
├── app/
│   └── streamlit_app.py          # Streamlit interface
├── src/ebay_price/
│   ├── ingest/                   # ETL and raw data handling
│   ├── features/                 # Feature engineering logic
│   ├── modeling/                 # Training, evaluation, MLflow logging
│   ├── explain/                  # SHAP and permutation explainability
│   ├── validation/               # Great Expectations data checks
│   └── flows/                    # Prefect orchestration scripts
├── data/
│   ├── raw/                      # Input data
│   └── artifacts/                # Models, plots, and outputs
├── reports/model_cards/          # Model summaries and evaluation reports
├── docs/                         # Additional documentation and diagrams
├── Makefile                      # Workflow automation
├── pyproject.toml                # Dependencies (Poetry)
└── Dockerfile                    # Containerized environment
```
---

## ⚙️ Setup & Usage

Full setup, configuration, and environment details will be documented soon.  
Refer to the upcoming **[Installation Guide](./docs/setup_guide.md)** for step-by-step instructions once published.
