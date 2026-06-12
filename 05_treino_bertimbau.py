"""
05_treino_bertimbau.py — Fine-tuning do BERTimbau para classificação de NCM.

Modelo profundo do artigo: neuralmind/bert-base-portuguese-cased (BERTimbau base)
ajustado para classificar a descrição do item em uma das 50 posições NCM do
dataset final. Usa EXATAMENTE o mesmo split 80/20 estratificado (random_state=42)
do baseline (04_treino_baseline.py) para que acurácia e F1-macro sejam
comparáveis lado a lado.

Roda em CPU (descrições curtas, max_len=64, dataset pequeno). Os números impressos
no fim alimentam a tabela da seção 4 do artigo — nada é inventado.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

DATASET = "dados/dataset_ncm_top_classes.csv"
MODELO = "neuralmind/bert-base-portuguese-cased"
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_LEN = 64
EPOCHS = 5
BATCH = 16
LR = 3e-5

torch.manual_seed(RANDOM_STATE)


def carregar_dados():
    df = pd.read_csv(DATASET, dtype={"ncm_posicao": str})
    df = df.dropna(subset=["descricao_item", "ncm_posicao"])
    df["descricao_item"] = df["descricao_item"].astype(str).str.strip()
    df = df[df["descricao_item"] != ""]
    return df


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }


def main():
    df = carregar_dados()
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["ncm_posicao"].values)
    num_labels = len(le.classes_)
    print(f"Dataset: {len(df)} itens | {num_labels} classes NCM")

    textos = np.array(df["descricao_item"].tolist(), dtype=object)
    rotulos = np.asarray(df["label"].tolist())
    X_train, X_test, y_train, y_test = train_test_split(
        textos,
        rotulos,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=rotulos,
    )
    print(f"Treino: {len(X_train)} | Teste: {len(X_test)}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODELO)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

    ds_train = Dataset.from_dict({"text": list(X_train), "labels": list(y_train)})
    ds_test = Dataset.from_dict({"text": list(X_test), "labels": list(y_test)})
    ds_train = ds_train.map(tokenize, batched=True, remove_columns=["text"])
    ds_test = ds_test.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODELO, num_labels=num_labels
    )

    args = TrainingArguments(
        output_dir="modelos/bertimbau_ncm",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        learning_rate=LR,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        report_to="none",
        seed=RANDOM_STATE,
        use_cpu=not torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Avaliação final + análise qualitativa
    pred_out = trainer.predict(ds_test)
    y_pred = np.argmax(pred_out.predictions, axis=-1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # Salva predições (em rótulos NCM originais) para plotagem posterior.
    import os
    os.makedirs("dados", exist_ok=True)
    np.savez(
        "dados/preds_bertimbau.npz",
        y_true=le.inverse_transform(y_test),
        y_pred=le.inverse_transform(y_pred),
        acc=acc,
        f1=f1,
    )

    print("\n==================== BERTimbau (teste) ====================")
    print(f"Acurácia : {acc:.4f}")
    print(f"F1-macro : {f1:.4f}\n")

    # Pares mais confundidos (em rótulos NCM originais)
    labels_idx = list(range(num_labels))
    cm = confusion_matrix(y_test, y_pred, labels=labels_idx)
    pares = []
    for i in labels_idx:
        for j in labels_idx:
            if i != j and cm[i, j] > 0:
                pares.append((le.classes_[i], le.classes_[j], int(cm[i, j])))
    pares.sort(key=lambda t: t[2], reverse=True)
    print("Pares NCM mais confundidos (real -> previsto : ocorrências):")
    for real, prev, n in pares[:8]:
        print(f"  {real} -> {prev} : {n}")

    # Exemplos de predição
    ncm_real = le.inverse_transform(y_test)
    ncm_prev = le.inverse_transform(y_pred)
    acertos, erros = [], []
    for desc, r, p in zip(X_test, ncm_real, ncm_prev):
        if r == p and len(acertos) < 3:
            acertos.append((desc, r, p))
        elif r != p and len(erros) < 3:
            erros.append((desc, r, p))
    print("\nExemplos de ACERTO (descrição | real | previsto):")
    for desc, r, p in acertos:
        print(f"  [{r}={p}] {desc[:70]}")
    print("\nExemplos de ERRO (descrição | real | previsto):")
    for desc, r, p in erros:
        print(f"  [real {r} != prev {p}] {desc[:70]}")


if __name__ == "__main__":
    main()
