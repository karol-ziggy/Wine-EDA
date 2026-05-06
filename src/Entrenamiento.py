# src/Entrenamiento.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json

ROOT = Path(__file__).resolve().parent.parent
DATA_CLEAN = ROOT / "data" / "wine_clean.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

def cargar(path):
    return pd.read_csv(path)

def preparar(df):
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    df["target"] = (df["quality"] >= 7).astype(int)
    X = df.drop(columns=["quality", "target"])
    y = df["target"]
    return X, y

if __name__ == "__main__":
    if not DATA_CLEAN.exists():
        raise FileNotFoundError("Ejecuta primero EDA.py")
    
    df = cargar(DATA_CLEAN)
    X, y = preparar(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    
    modelos = {
        "logreg": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]),
        "rf": Pipeline([("clf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))])
    }
    
    resultados = {}
    for name, m in modelos.items():
        m.fit(X_train, y_train)
        ypred = m.predict(X_test)
        resultados[name] = {
            "accuracy": float(accuracy_score(y_test, ypred)),
            "f1": float(f1_score(y_test, ypred, average="weighted"))
        }
        print(f"Modelo {name}: {resultados[name]}")
    
    mejor_nombre = max(resultados.items(), key=lambda x: x[1]["f1"])[0]
    joblib.dump(modelos[mejor_nombre], MODEL_DIR / "best_model.pkl")
    
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(resultados, f, indent=2)
    print(f"Mejor modelo guardado: {mejor_nombre}")