"""Exporte le modèle 'champion' du Registry vers ./model (indépendant de MLflow)."""
import os
import joblib
import mlflow
import mlflow.sklearn
import sklearn

mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/ModelGuard@champion")

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.joblib")

# Versions figées pour que l'image Docker reproduise exactement l'environnement
with open("requirements-api.txt", "w") as f:
    f.write(
        "fastapi==0.115.6\n"
        "uvicorn[standard]==0.34.0\n"
        f"scikit-learn=={sklearn.__version__}\n"
        f"joblib=={joblib.__version__}\n"
        "numpy\n"
    )
print("Modèle exporté dans model/model.joblib")
print("Fichier requirements-api.txt généré")