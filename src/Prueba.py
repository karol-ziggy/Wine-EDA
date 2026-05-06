# src/Prueba.py
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_model.pkl"
DATA_CLEAN = ROOT / "data" / "wine_clean.csv"
OUTDIR = ROOT / "outputs"
RANDOM_STATE = 42

if __name__ == "__main__":
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No hay modelo. Ejecuta Entrenamiento.py")
    
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_CLEAN)
    
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    df["target"] = (df["quality"] >= 7).astype(int)
    X = df.drop(columns=["quality", "target"])
    y = df["target"]
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    
    ypred = model.predict(X_test)
    print("Reporte de Prueba:")
    print(classification_report(y_test, ypred, zero_division=0))
    
    cm = confusion_matrix(y_test, ypred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
    plt.xlabel("Predicho")
    plt.ylabel("Verdadero")
    plt.tight_layout()
    plt.savefig(OUTDIR / "confusion_matrix_test.png")
    plt.close()
    print(f"Prueba finalizada. Matriz en {OUTDIR}")