# src/EDA.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "WineQT.csv"
DATA_CLEAN = ROOT / "data" / "wine_clean.csv"
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

def cargar(path):
    # Se eliminó low_memory para evitar el ValueError
    df = pd.read_csv(path, engine="python")
    return df

def limpiar_encabezados_repetidos(df):
    first_col = df.columns[0]
    mask = df[first_col].astype(str).str.contains(first_col, na=False)
    if mask.any():
        df = df.loc[~mask].copy()
    return df

def forzar_numericos(df):
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def resumen(df):
    print("Shape:", df.shape)
    print("\nTipos:\n", df.dtypes)
    print("\nDescripción estadística:\n", df.describe().T)
    print("\nValores faltantes por columna:\n", df.isna().sum())

def limpieza_basica(df):
    antes = len(df)
    df = df.drop_duplicates().copy()
    despues = len(df)
    print(f"Duplicados eliminados: {antes - despues}")
    for c in df.select_dtypes(include="number").columns:
        if df[c].isna().sum() > 0:
            df[c] = df[c].fillna(df[c].median())
    return df

def graficar(df):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if "Id" in num_cols:
        num_cols.remove("Id")
    
    plt.figure(figsize=(12, 8))
    df[num_cols].hist(figsize=(14, 12))
    plt.tight_layout()
    plt.savefig(OUTDIR / "histogramas.png")
    plt.close()

    n = len(num_cols)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(OUTDIR / "boxplots.png")
    plt.close()

    corr = df[num_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.tight_layout()
    plt.savefig(OUTDIR / "correlation_matrix.png")
    plt.close()

if __name__ == "__main__":
    df = cargar(DATA_RAW)
    print("Lectura completada.")
    df = limpiar_encabezados_repetidos(df)
    df = forzar_numericos(df)
    resumen(df)
    df = limpieza_basica(df)
    graficar(df)
    df.to_csv(DATA_CLEAN, index=False)
    print(f"EDA completado. Archivo limpio en: {DATA_CLEAN}")