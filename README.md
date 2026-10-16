# ModelGuard

![CI](https://github.com/ons-cherif-dev/ModelGuard/actions/workflows/ci.yml/badge.svg)

An end-to-end MLOps project: an anomaly-detection model for industrial sensor data (temperature and vibration), tracked with **MLflow**, versioned in the **MLflow Model Registry**, served through a **FastAPI** service, packaged with **Docker** and validated by a **GitHub Actions** CI pipeline.

The goal is not only to train a model, but to show the full path from experiment to a tested, containerised service.

## Architecture

```
train.py ──► MLflow (SQLite)  ──► register.py ──► Model Registry (alias: champion)
 train / validation / test         pick best run                 │
                                                                 ▼
                                                        export_model.py
                                                                 │  model/model.joblib
                                                                 ▼
                              GitHub Actions ◄── git push ── FastAPI app ──► Docker image
                              tests + build + smoke test      /health  /predict
```

## What is inside

| Part | Description |
|---|---|
| **Model** | `IsolationForest` (scikit-learn), trained only on normal measurements |
| **Experiment tracking** | MLflow with a local SQLite backend (`mlflow.db`): parameters, metrics, models |
| **Model selection** | The `contamination` parameter is chosen on a **separate validation set**, then the winner is evaluated **once** on a held-out test set |
| **Model Registry** | The selected model is registered as `ModelGuard` and tagged with the alias `champion` |
| **API** | FastAPI service exposing `GET /health` and `POST /predict` |
| **Container** | Docker image based on `python:3.12-slim`, running as a non-root user |
| **CI** | GitHub Actions: unit tests, Docker build, and a smoke test of the running container |

## Model and evaluation

The data is **simulated**: 1000 normal measurements for training, then validation and test sets with 200 normal and 40 anomalous measurements each. Half of the anomalies are obvious (high temperature and vibration), half are subtler, so that the threshold choice actually matters.

Choosing `contamination` on the test set would give optimistic numbers. Instead, `train.py` compares several values on the validation set, keeps those with a recall of at least 0.9 (missing an anomaly is costly), picks the best F1 among them, and only then evaluates on the test set.

| Phase | contamination | Precision | Recall | F1 |
|---|---|---|---|---|
| Validation (selected value) | 0.02 | 0.867 | 0.975 | 0.918 |
| Test (final, seed 42) | 0.02 | 0.974 | 0.950 | 0.962 |

These figures come from simulated data and are only meant to demonstrate the workflow. On real sensor data, expect lower and noisier results.

## Quick start

### 1. Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install numpy scikit-learn mlflow cloudpickle fastapi uvicorn httpx pytest joblib
```

### 2. Train and track experiments

```bash
python train.py
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000 and compare the runs in the `ModelGuard` experiment.

### 3. Register the best model

```bash
python register.py
```

This registers the latest `test_final` run as `ModelGuard` and sets the alias `champion`. It can then be loaded by name:

```python
import mlflow.sklearn
model = mlflow.sklearn.load_model("models:/ModelGuard@champion")
```

### 4. Export the model for the API

```bash
python export_model.py
```

This writes `model/model.joblib` and a pinned `requirements-api.txt`, so the API and the Docker image do not depend on MLflow.

### 5. Run the tests and the API

```bash
python -m pytest -q
uvicorn app.main:app --port 8000
```

Interactive documentation is available at http://127.0.0.1:8000/docs.

### 6. Run with Docker

```bash
docker build -t modelguard-api .
docker run --rm -p 8000:8000 modelguard-api
```

## API

### `GET /health`

```json
{"status": "ok", "model_loaded": true}
```

### `POST /predict`

Request:

```json
{"temperature": 80, "vibration": 22}
```

Response:

```json
{"anomalie": true, "score": -0.1133}
```

`anomalie` is `true` when the measurement is flagged as abnormal. `score` is the model's decision score: the lower it is, the more abnormal the measurement. Invalid input returns HTTP 422.

## CI pipeline

Defined in `.github/workflows/ci.yml`, triggered on every push to `main` and on pull requests:

1. **test**: installs the pinned dependencies and runs the pytest suite.
2. **docker**: builds the image, starts the container, and checks that `/health` reports the model as loaded and that `/predict` flags an obvious anomaly.

## Project structure

```
ModelGuard/
├── app/
│   ├── __init__.py
│   └── main.py              # FastAPI service
├── tests/
│   └── test_api.py          # API tests
├── model/
│   └── model.joblib         # exported champion model (committed for CI and Docker)
├── .github/workflows/ci.yml # CI pipeline
├── train.py                 # data, training, validation, MLflow tracking
├── register.py              # Model Registry + champion alias
├── export_model.py          # export of the champion model
├── Dockerfile
├── .dockerignore
├── requirements-api.txt     # pinned runtime dependencies
└── .gitignore
```

`mlflow.db`, `mlruns/` and `mlartifacts/` are intentionally not versioned.

## Design choices

- **SQLite instead of the file backend**: recent MLflow versions put the file-based tracking store in maintenance mode, so the project uses a SQLite backend.
- **Exported model instead of MLflow inside the container**: the image stays small and independent of local MLflow paths.
- **Pinned versions**: `export_model.py` records the exact scikit-learn version used for training, to avoid loading a model with a different version.

## Limitations and next steps

- The data is simulated. The next step would be to connect real sensor data.
- The model file is committed to the repository for simplicity. A larger project would store it in an artifact registry.
- Possible extensions: publish the Docker image to GitHub Container Registry, add data-drift monitoring, and automate retraining.

## Author

Built by [@ons-cherif-dev](https://github.com/ons-cherif-dev) as part of a DevOps / MLOps learning project.
