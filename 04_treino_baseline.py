"""
04_treino_baseline.py — Baseline TF-IDF + classificadores clássicos.

Consome dados/dataset_ncm_top_classes.csv (saída do estágio 3) e treina o
baseline do artigo: vetorização TF-IDF da descrição do item + Logistic
Regression e Random Forest. Reporta acurácia, F1-macro, as classes NCM mais
confundidas (a partir da matriz de confusão) e exemplos reais de predição
(acertos e erros). Os números impressos alimentam a seção 4 do artigo — nada
é inventado: tudo sai deste run.

Não chama a API; é puramente offline sobre o CSV já coletado.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

DATASET = "dados/dataset_ncm_top_classes.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def carregar_dados():
    df = pd.read_csv(DATASET, dtype={"ncm_posicao": str})
    df = df.dropna(subset=["descricao_item", "ncm_posicao"])
    df["descricao_item"] = df["descricao_item"].astype(str).str.strip()
    df = df[df["descricao_item"] != ""]
    return df


def split_estratificado(df):
    # Estratifica por classe; classes com <2 exemplos quebrariam o split,
    # mas o estágio 3 já garante o top-50 (mínimo 7 exemplos por classe).
    X = df["descricao_item"].values
    y = df["ncm_posicao"].values
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def construir_modelos():
    tfidf = lambda: TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    return {
        "TF-IDF + LogisticRegression": Pipeline([
            ("tfidf", tfidf()),
            ("clf", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
            )),
        ]),
        "TF-IDF + RandomForest": Pipeline([
            ("tfidf", tfidf()),
            ("clf", RandomForestClassifier(
                n_estimators=300, class_weight="balanced_subsample",
                random_state=RANDOM_STATE, n_jobs=-1
            )),
        ]),
    }


def pares_mais_confundidos(y_true, y_pred, labels, top=8):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pares = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                pares.append((labels[i], labels[j], int(cm[i, j])))
    pares.sort(key=lambda t: t[2], reverse=True)
    return pares[:top]


def exemplos_predicao(X_test, y_true, y_pred, n_acertos=3, n_erros=3):
    acertos, erros = [], []
    for desc, real, prev in zip(X_test, y_true, y_pred):
        registro = (desc, real, prev)
        if real == prev and len(acertos) < n_acertos:
            acertos.append(registro)
        elif real != prev and len(erros) < n_erros:
            erros.append(registro)
    return acertos, erros


def main():
    df = carregar_dados()
    print(f"Dataset: {len(df)} itens | {df['ncm_posicao'].nunique()} classes NCM\n")

    X_train, X_test, y_train, y_test = split_estratificado(df)
    print(f"Treino: {len(X_train)} | Teste: {len(X_test)}\n")

    labels = sorted(df["ncm_posicao"].unique())
    resultados = {}

    for nome, modelo in construir_modelos().items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        resultados[nome] = (modelo, y_pred, acc, f1)
        print(f"=== {nome} ===")
        print(f"Acurácia : {acc:.4f}")
        print(f"F1-macro : {f1:.4f}\n")

    # Melhor modelo para análise qualitativa
    melhor_nome = max(resultados, key=lambda k: resultados[k][3])
    _, y_pred, acc, f1 = resultados[melhor_nome]
    print(f"--- Melhor baseline: {melhor_nome} (F1-macro {f1:.4f}) ---\n")

    print("Pares NCM mais confundidos (real -> previsto : ocorrências):")
    for real, prev, n in pares_mais_confundidos(y_test, y_pred, labels):
        print(f"  {real} -> {prev} : {n}")
    print()

    acertos, erros = exemplos_predicao(X_test, y_test, y_pred)
    print("Exemplos de ACERTO (descrição | NCM real | NCM previsto):")
    for desc, real, prev in acertos:
        print(f"  [{real}={prev}] {desc[:70]}")
    print("\nExemplos de ERRO (descrição | NCM real | NCM previsto):")
    for desc, real, prev in erros:
        print(f"  [real {real} != prev {prev}] {desc[:70]}")


if __name__ == "__main__":
    main()
