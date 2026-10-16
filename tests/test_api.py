from fastapi.testclient import TestClient

from app.main import app


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["model_loaded"] is True


def test_mesure_normale():
    with TestClient(app) as client:
        r = client.post("/predict", json={"temperature": 50, "vibration": 10})
        assert r.status_code == 200
        assert r.json()["anomalie"] is False


def test_mesure_anormale():
    with TestClient(app) as client:
        r = client.post("/predict", json={"temperature": 80, "vibration": 22})
        assert r.status_code == 200
        assert r.json()["anomalie"] is True


def test_entree_invalide():
    with TestClient(app) as client:
        r = client.post("/predict", json={"temperature": "abc"})
        assert r.status_code == 422