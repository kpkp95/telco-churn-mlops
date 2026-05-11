# Telco Customer Churn MLOps Pipeline

[![Build and Push Docker Image](https://github.com/kpkp95/telco-churn-mlops/actions/workflows/docker-build-push.yml/badge.svg)](https://github.com/kpkp95/telco-churn-mlops/actions/workflows/docker-build-push.yml)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![Docker Image](https://img.shields.io/docker/v/kunalkp22/telco-churn-api/latest?label=docker%20image)
![Docker Pulls](https://img.shields.io/docker/pulls/kunalkp22/telco-churn-api)

End-to-end MLOps project for predicting telecom customer churn using XGBoost, MLflow, FastAPI, Gradio, Docker, GitHub Actions, and AWS ECS Fargate.

Repository: https://github.com/kpkp95/telco-churn-mlops

---

## Try It In 2 Minutes

1. Clone and enter the repository.

```bash
git clone https://github.com/kpkp95/telco-churn-mlops.git
cd telco-churn-mlops
```

2. Create and activate a virtual environment.

   Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Mac/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

4. Run the API + UI.

```bash
uvicorn src.app.main:app --reload
```

5. Open the app.

- API health: http://127.0.0.1:8000/
- Gradio UI: http://127.0.0.1:8000/ui
- API docs: http://127.0.0.1:8000/docs

6. Live deployed app (AWS ECS).
   - API health: http://23.22.168.233:8000/
   - Gradio UI: http://23.22.168.233:8000/ui/
   - API docs: http://23.22.168.233:8000/docs

---

## Project Overview

Customer churn is a major business problem in telecom. This project predicts whether a customer is likely to churn using demographic, account, service, and billing data.

The pipeline includes:

- Data loading and validation
- Preprocessing and feature engineering
- Model training and hyperparameter tuning
- MLflow experiment tracking
- Inference pipeline
- FastAPI and Gradio serving
- Docker packaging
- CI/CD with GitHub Actions
- Deployment on AWS ECS Fargate

---

## Tech Stack

### Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Optuna
- MLflow
- Joblib

### Serving

- FastAPI
- Gradio
- Pydantic

### DevOps

- Docker
- GitHub Actions
- Docker Hub
- AWS ECS Fargate

---

## Project Architecture

```text
Raw Data
-> Validation + Preprocessing
-> Feature Engineering
-> Model Training + Evaluation
-> Hyperparameter Tuning (Optuna)
-> MLflow Tracking
-> Saved Model + Feature Schema
-> Inference Pipeline
-> FastAPI + Gradio
-> Docker
-> GitHub Actions
-> Docker Hub
-> AWS ECS Fargate
```

---

## Project Structure (Synced)

```text
churML/

- .github/
  - workflows/
    - docker-build-push.yml
- artifacts/
  - feature_columns.json
  - preprocessing.pkl
- configs/
- data/
  - external/
  - processed/
    - telco_churn_processed.csv
  - raw/
    - telecoData.csv
- docker/
- great_expectations/
- mlruns/
- models/
  - xgb_telco_churn_model.pkl
- notebooks/
  - eda.ipynb
- scripts/
  - prepare_processed_data.py
  - run_pipeline.py
  - test_inference.py
  - test_mlflow_tracking.py
  - test_pipeline_phase1_data_features.py
  - test_pipeline_phase2_modeling.py
  - test_validate_data.py
- src/
  - app/
    - app.py
    - main.py
  - data/
    - load_data.py
    - pre_process.py
  - features/
    - build_features.py
  - models/
    - evaluate.py
    - train.py
    - tune.py
  - serving/
    - inference.py
  - utils/
    - experiment_tracking.py
    - validate.py
    - validate_data.py
- .dockerignore
- .gitignore
- dockerfile
- requirements.txt
- README.md
```

---

## Machine Learning Pipeline

1. Data loading and validation

- Loads raw Telco churn CSV data.
- Checks required columns and schema consistency.

2. Preprocessing

- Strips extra whitespace from column names.
- Removes identifiers like customerID.
- Converts TotalCharges to numeric.
- Converts Churn Yes/No labels to 0/1.
- Handles missing numeric values.

3. Feature engineering

- Deterministic binary mappings for yes/no columns.
- One-hot encoding for multi-category columns.
- Column alignment to training schema.

4. Model training

- Trains an XGBoost classifier for churn prediction.

5. Hyperparameter tuning

- Uses Optuna for F1/Recall optimization and threshold-aware experiments.

6. Evaluation

- Accuracy, Precision, Recall, F1, ROC-AUC.

7. Experiment tracking

- Logs params, metrics, and model artifacts to MLflow.

---

## Inference Output Example

```json
{
  "prediction": "Likely to churn",
  "churn_probability": 0.8302,
  "threshold": 0.4,
  "model_output": 1
}
```

---

## Run Training Pipeline

Prepare processed data:

```bash
python scripts/prepare_processed_data.py
```

Run full pipeline:

```bash
python scripts/run_pipeline.py
```

Run Optuna tuning:

```bash
python scripts/run_pipeline.py --tune --n_trials 100 --scoring f1 --threshold 0.4
```

---

## Run Tests

```bash
python scripts/test_pipeline_phase1_data_features.py
python scripts/test_pipeline_phase2_modeling.py
python scripts/test_mlflow_tracking.py
python scripts/test_inference.py
python scripts/test_validate_data.py
```

---

## MLflow

Start UI:

```bash
mlflow ui
```

Open: http://127.0.0.1:5000

---

## Docker

Build image:

```bash
docker build -t telco-churn-api .
```

Run container:

```bash
docker run -p 8000:8000 telco-churn-api
```

Open:

- http://127.0.0.1:8000/
- http://127.0.0.1:8000/ui

---

## CI/CD

Workflow file:
`.github/workflows/docker-build-push.yml`

Current image tag configured in workflow:
`kunalkp22/telco-churn-api:latest`

On push to main:

1. Checkout repository
2. Configure Docker Buildx
3. Login to Docker Hub using repository secrets
4. Build image
5. Push image to Docker Hub

Required GitHub Secrets:

- DOCKERHUB_USERNAME
- DOCKERHUB_TOKEN

---

## Deployment

This project is deployed on AWS ECS Fargate.

Current public deployment endpoint:

- API health: http://23.22.168.233:8000/
- Gradio UI: http://23.22.168.233:8000/ui/
- API docs: http://23.22.168.233:8000/docs

Generic endpoint format:

- API health: http://<ecs-public-ip-or-alb-dns>:8000/
- Gradio UI: http://<ecs-public-ip-or-alb-dns>:8000/ui

ECS settings used:

- Launch type: Fargate
- CPU: 0.25 vCPU
- Memory: 512 MiB
- Container port: 8000
- Network mode: awsvpc

---

## Live Validation (May 11, 2026)

Deployment and prediction flow were validated against the live API.

High-risk sample response:

```json
{
  "prediction": "Likely to churn",
  "churn_probability": 0.8302,
  "threshold": 0.4,
  "model_output": 1
}
```

Low-risk sample response:

```json
{
  "prediction": "Not likely to churn",
  "churn_probability": 0.1813,
  "threshold": 0.4,
  "model_output": 0
}
```

---

## How To Get The Exact Docker Image Name

Use any one of these:

1. From GitHub Actions workflow

- Open `.github/workflows/docker-build-push.yml`
- Read the value under `tags`
- Current value: `kunalkp22/telco-churn-api:latest`

2. From Docker Hub

- Open your Docker Hub repository page
- Copy namespace/repository:tag

3. From local Docker CLI

- Run: `docker images`
- Find repository and tag columns

---

## Future Improvements

- Add automatic ECS redeploy in GitHub Actions
- Add load balancer and custom domain
- Add API endpoint unit tests
- Add model monitoring and request logging
- Add scheduled retraining pipeline

---

## Author

Kunal Pandey
