import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# 1. Retrouver le dernier run de test final
exp = client.get_experiment_by_name("ModelGuard")
runs = client.search_runs(
    [exp.experiment_id],
    filter_string="params.phase = 'test_final'",
    order_by=["attributes.start_time DESC"],
    max_results=1,
)
if not runs:
    raise SystemExit("Aucun run 'test_final' trouvé : lance d'abord python train.py")
run = runs[0]
print("Run choisi :", run.info.run_name, "| F1 =", round(run.data.metrics["f1_score"], 3))

# 2. Enregistrer le modèle dans le Registry
mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", "ModelGuard")

# 3. Poser l'alias "champion" sur cette version
client.set_registered_model_alias("ModelGuard", "champion", mv.version)
print(f"ModelGuard version {mv.version} -> alias 'champion'")

# 4. Vérifier : charger le modèle par son nom
model = mlflow.sklearn.load_model("models:/ModelGuard@champion")
print("Chargement OK :", type(model).__name__)