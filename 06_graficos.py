"""
06_graficos.py — Gera os PNGs de resultados para os slides (Canva).

Saída em slides/img/. Todos os números vêm dos dados reais:
- distribuição das 50 classes NCM (dataset final)
- comparação baseline (LogReg, RF) vs BERTimbau (acurácia, F1-macro)
- curva de aprendizado do BERTimbau por época (evidência de subajuste)
- matriz de confusão do melhor baseline (Random Forest)
- matriz de confusão do BERTimbau (se dados/preds_bertimbau.npz existir)

As matrizes usam as 15 posições NCM mais frequentes para legibilidade.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11})

DATASET = "dados/dataset_ncm_top_classes.csv"
OUTDIR = "slides/img"
RANDOM_STATE = 42
os.makedirs(OUTDIR, exist_ok=True)


def carregar():
    df = pd.read_csv(DATASET, dtype={"ncm_posicao": str})
    df = df.dropna(subset=["descricao_item", "ncm_posicao"])
    df["descricao_item"] = df["descricao_item"].astype(str).str.strip()
    return df[df["descricao_item"] != ""]


# ---------- 1. Distribuição das classes ----------
def grafico_distribuicao(df):
    cont = df["ncm_posicao"].value_counts()
    top = cont.head(20)
    plt.figure(figsize=(11, 5.5))
    plt.bar(range(len(top)), top.values, color="#2e7d32")
    plt.xticks(range(len(top)), top.index, rotation=60, ha="right")
    plt.ylabel("Nº de itens")
    plt.xlabel("Posição NCM (4 dígitos)")
    plt.title(f"Distribuição das classes NCM — top 20 de {cont.size} posições "
              f"(min {cont.min()}, máx {cont.max()})")
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/01_distribuicao_classes.png", dpi=150)
    plt.close()
    print("OK 01_distribuicao_classes.png")


# ---------- 2. Comparação de modelos ----------
def grafico_comparacao():
    modelos = ["TF-IDF +\nLogReg", "TF-IDF +\nRandomForest", "BERTimbau\n(5 épocas)"]
    acc = [0.827, 0.827, 0.640]
    f1 = [0.780, 0.782, 0.392]
    x = np.arange(len(modelos))
    w = 0.38
    plt.figure(figsize=(9, 5.5))
    b1 = plt.bar(x - w/2, acc, w, label="Acurácia", color="#1565c0")
    b2 = plt.bar(x + w/2, f1, w, label="F1-macro", color="#ef6c00")
    for bars in (b1, b2):
        for b in bars:
            plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                     f"{b.get_height():.3f}", ha="center", fontsize=10)
    plt.xticks(x, modelos)
    plt.ylim(0, 1.0)
    plt.ylabel("Pontuação")
    plt.title("Baseline vs BERTimbau — conjunto de teste (214 itens, 50 classes)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/02_comparacao_modelos.png", dpi=150)
    plt.close()
    print("OK 02_comparacao_modelos.png")


# ---------- 3. Curva de aprendizado do BERTimbau ----------
def grafico_curva_bertimbau():
    epocas = [1, 2, 3, 4, 5]
    acc = [0.3925, 0.500, 0.5935, 0.6168, 0.6402]
    f1 = [0.0815, 0.1896, 0.317, 0.3438, 0.3916]
    plt.figure(figsize=(8.5, 5))
    plt.plot(epocas, acc, "o-", label="Acurácia (validação)", color="#1565c0")
    plt.plot(epocas, f1, "s-", label="F1-macro (validação)", color="#ef6c00")
    plt.xticks(epocas)
    plt.ylim(0, 0.75)
    plt.xlabel("Época")
    plt.ylabel("Pontuação")
    plt.title("BERTimbau — métricas ainda crescentes na 5ª época (subajuste)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/03_curva_bertimbau.png", dpi=150)
    plt.close()
    print("OK 03_curva_bertimbau.png")


def _plot_confusao(y_true, y_pred, classes, titulo, arquivo):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    plt.figure(figsize=(8.5, 7.5))
    im = plt.imshow(cm_norm, cmap="Greens", vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Proporção por linha")
    plt.xticks(range(len(classes)), classes, rotation=60, ha="right", fontsize=9)
    plt.yticks(range(len(classes)), classes, fontsize=9)
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cm_norm[i, j]
            if v >= 0.01:
                plt.text(j, i, f"{v:.2f}", ha="center", va="center",
                         color="white" if v > 0.5 else "black", fontsize=7)
    plt.xlabel("NCM previsto")
    plt.ylabel("NCM real")
    plt.title(titulo)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{arquivo}", dpi=150)
    plt.close()
    print(f"OK {arquivo}")


# ---------- 4. Matriz de confusão do baseline ----------
def confusao_baseline(df, top_classes):
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["descricao_item"].tolist(), df["ncm_posicao"].tolist(),
        test_size=0.2, random_state=RANDOM_STATE, stratify=df["ncm_posicao"].tolist())
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                  ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", RandomForestClassifier(n_estimators=300,
                                       class_weight="balanced_subsample",
                                       random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    _plot_confusao(y_te, y_pred, top_classes,
                   "Matriz de confusão — TF-IDF + Random Forest (top 15 classes)",
                   "04_confusao_baseline.png")


# ---------- 5. Matriz de confusão do BERTimbau ----------
def confusao_bertimbau(top_classes):
    caminho = "dados/preds_bertimbau.npz"
    if not os.path.exists(caminho):
        print("SKIP 05_confusao_bertimbau.png (preds_bertimbau.npz ainda não existe)")
        return
    d = np.load(caminho, allow_pickle=True)
    _plot_confusao(d["y_true"], d["y_pred"], top_classes,
                   "Matriz de confusão — BERTimbau (top 15 classes)",
                   "05_confusao_bertimbau.png")


def main():
    df = carregar()
    top15 = df["ncm_posicao"].value_counts().head(15).index.tolist()
    grafico_distribuicao(df)
    grafico_comparacao()
    grafico_curva_bertimbau()
    confusao_baseline(df, top15)
    confusao_bertimbau(top15)


if __name__ == "__main__":
    main()
