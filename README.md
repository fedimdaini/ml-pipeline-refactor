# ml-pipeline-refactor

A refactor of a monolithic Flask ML app into a modular, tested training pipeline with MLflow tracking and
Docker.

The original app (kept in [`legacy/old_flask_app.py`](legacy/old_flask_app.py), about 1,500 lines) trained
models from a web form: global variables carried state between routes, and the same split, scale, train,
evaluate and plot code was copied into every model's route. This repository turns that into small
functions with unit tests, one pipeline that runs end to end, and an MLflow record of every run.

## What the pipeline does

Given a CSV or Excel file and a target column:

1. **Load and clean**: median or mode imputation, label encoding of categorical columns.
2. **Detect the problem type**: classification or regression, from the target's dtype and number of
   distinct values.
3. **Split and scale**: train/test split (stratified for classification) and `StandardScaler`.
4. **Train six models**, tuned with 5-fold `GridSearchCV` (plain Linear Regression has nothing to tune):
   - classification: KNN, Decision Tree, Random Forest, Gradient Boosting, Extra Trees, Logistic Regression
   - regression: Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting
5. **Evaluate and plot**: a correlation heatmap, then confusion matrix and feature importance for
   classification, or residuals and actual vs predicted for regression.
6. **Pick a champion**: best F1 for classification, best R² for regression, saved to `outputs/models/`.
7. **Track**: each model is an MLflow run with its parameters, metrics, plots and the model itself.

## Before and after

| Flask app | This pipeline |
|---|---|
| State in global variables shared between routes | Data passed through function arguments |
| Split, scale, train and plot code copied into each model's route | One function per step, reused by every model |
| No tests | 25 tests, including end-to-end runs for both problem types |
| No record of past runs | MLflow tracking (SQLite backend) |
| Runs on the developer's machine | Docker image and Compose file |

## Run it

```bash
pip install -r requirements.txt
PYTHONPATH=. pytest                  # unit tests + a full classification and regression run
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

On your own data:

```python
from src.pipeline.pipeline import MLPipeline

results, champion = MLPipeline(raw_data_path="data/raw/my_data.csv", target_column="label").run()
print(champion["model_name"], champion["metrics"])
```

With Docker:

```bash
docker-compose up --build
```

## Layout

| Path | What it holds |
|---|---|
| `src/preprocessing/` | Loading, cleaning, problem-type detection |
| `src/training/` | Split and scale, grid search, prediction and evaluation |
| `src/metrics/`, `src/plots/` | Metric and plotting functions shared by every model |
| `src/tracking/` | MLflow wrapper |
| `src/pipeline/` | The end-to-end pipeline and champion selection |
| `tests/` | Unit tests per module and end-to-end pipeline tests |
| `legacy/` | The original Flask app, for comparison |
| `docs/` | Architecture notes and future work |

## Next steps

Data versioning with DVC, scheduled retraining with Prefect or Airflow, data validation before training,
and CI running the tests on every push. Details in [`docs/FUTURE_WORK.md`](docs/FUTURE_WORK.md).
