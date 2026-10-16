import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

rng = np.random.default_rng(42)


def normales(n):
    return np.column_stack([rng.normal(50, 3, n), rng.normal(10, 1.5, n)])


def anomalies(n_facile, n_dur):
    # Anomalies évidentes (75 °C) + anomalies plus subtiles (62 °C)
    facile = np.column_stack([rng.normal(75, 5, n_facile), rng.normal(20, 3, n_facile)])
    dur = np.column_stack([rng.normal(62, 5, n_dur), rng.normal(14, 3, n_dur)])
    return np.vstack([facile, dur])


def jeu(n_normal, n_facile, n_dur):
    X = np.vstack([normales(n_normal), anomalies(n_facile, n_dur)])
    y = np.array([1] * n_normal + [-1] * (n_facile + n_dur))
    return X, y


def evalue(model, X, y):
    p = model.predict(X)
    tp, fn, fp, tn = confusion_matrix(y, p, labels=[-1, 1]).ravel()
    return {
        "precision": precision_score(y, p, pos_label=-1),
        "recall": recall_score(y, p, pos_label=-1),
        "f1_score": f1_score(y, p, pos_label=-1),
        "false_alarms": int(fp),
        "missed_anomalies": int(fn),
    }


# 3 jeux séparés
X_train = normales(1000)          # le modèle apprend dessus (normal uniquement)
X_val, y_val = jeu(200, 20, 20)   # sert à CHOISIR contamination
X_test, y_test = jeu(200, 20, 20) # utilisé UNE SEULE FOIS à la fin

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ModelGuard")

best = None  # (contamination, métriques validation, modèle)

for contamination in [0.15, 0.05, 0.02, 0.01, 0.005, 0.001]:
    model = IsolationForest(
        n_estimators=100, contamination=contamination, random_state=42
    ).fit(X_train)
    val = evalue(model, X_val, y_val)

    with mlflow.start_run(run_name=f"val_cont_{contamination}"):
        mlflow.log_param("algorithm", "IsolationForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("contamination", contamination)
        mlflow.log_param("phase", "validation")
        for nom, valeur in val.items():
            mlflow.log_metric(f"val_{nom}", valeur)

    print(f"VAL contamination={contamination} -> {({k: round(v, 3) for k, v in val.items()})}")

    # Règle : garder recall >= 0.9 (rater une anomalie coûte cher), puis meilleur F1
    if val["recall"] >= 0.9 and (best is None or val["f1_score"] > best[1]["f1_score"]):
        best = (contamination, val, model)

best_cont, best_val, best_model = best
print(f"Contamination choisie sur la validation : {best_cont}")

# Évaluation finale sur le test (une seule fois)
test = evalue(best_model, X_test, y_test)

with mlflow.start_run(run_name=f"final_test_cont_{best_cont}"):
    mlflow.log_param("algorithm", "IsolationForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("contamination", best_cont)
    mlflow.log_param("phase", "test_final")
    for nom, valeur in test.items():
        mlflow.log_metric(nom, valeur)
    mlflow.sklearn.log_model(best_model, name="model", serialization_format="cloudpickle")

print("TEST final ->", {k: round(v, 3) for k, v in test.items()})