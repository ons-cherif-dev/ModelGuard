import os
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "model/model.joblib")
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load(MODEL_PATH)
    yield
    state.clear()


app = FastAPI(title="ModelGuard API", version="1.0.0", lifespan=lifespan)


class Mesure(BaseModel):
    temperature: float = Field(..., examples=[50.0])
    vibration: float = Field(..., examples=[10.0])


class Resultat(BaseModel):
    anomalie: bool
    score: float  # plus il est bas, plus la mesure est anormale


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict", response_model=Resultat)
def predict(mesure: Mesure):
    x = np.array([[mesure.temperature, mesure.vibration]])
    model = state["model"]
    pred = int(model.predict(x)[0])  # 1 = normal, -1 = anomalie
    score = float(model.decision_function(x)[0])
    return Resultat(anomalie=(pred == -1), score=round(score, 4))