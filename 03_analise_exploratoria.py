"""Análise exploratória do dataset de itens de Notas Fiscais coletado em
02_coletar_dataset.py.

Objetivos:
  - verificar volume de dados e qualidade das descrições
  - remover duplicatas e linhas inválidas
  - calcular o NCM no nível de "posição" (4 primeiros dígitos)
  - listar as classes (posições NCM) mais frequentes
  - gerar um dataset filtrado (top-N classes) pronto para treino do
    classificador (baseline TF-IDF e fine-tuning do BERTimbau)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ENTRADA = Path("dados/itens_notas_fiscais.csv")
SAIDA_DATASET = Path("dados/dataset_ncm_top_classes.csv")
SAIDA_DISTRIBUICAO = Path("dados/distribuicao_ncm.csv")

# Ajustar conforme volume real de dados coletado: classes com poucos
# exemplos não permitem treino/avaliação confiáveis.
TOP_N_CLASSES = 50


def main() -> None:
    df = pd.read_csv(ENTRADA, dtype={"ncm": str}, encoding="utf-8")
    print(f"Total de linhas carregadas: {len(df)}")

    # Remove duplicatas exatas (podem surgir se a coleta foi retomada)
    df = df.drop_duplicates(subset=["chave_nota", "descricao_item", "ncm"])
    print(f"Após remover duplicatas: {len(df)}")

    # Limpeza básica
    df = df.dropna(subset=["descricao_item", "ncm"])
    df["descricao_item"] = df["descricao_item"].astype(str).str.strip()
    df = df[df["descricao_item"].str.len() > 0]

    # Normaliza o NCM. O código tem 8 dígitos, mas a API devolve
    # `codigoNcmSh` SEM o zero à esquerda nos capítulos 01–09 (animais,
    # carnes, ovos, laticínios — centrais ao agro): ex. "2071400" deveria
    # ser "02071400". Mantém só códigos plausíveis (7–8 dígitos), descarta
    # lixo (ex.: "1") e repõe o zero perdido com zfill antes de cortar a
    # posição — senão a label sairia errada (2071 em vez de 0207).
    df["ncm"] = df["ncm"].astype(str).str.replace(r"\D", "", regex=True)
    df = df[df["ncm"].str.len().isin([7, 8])]
    df["ncm"] = df["ncm"].str.zfill(8)
    df["ncm_posicao"] = df["ncm"].str[:4]

    print(f"Itens válidos após limpeza: {len(df)}")
    if df.empty:
        print("Nenhum item válido após a limpeza — verifique a coleta (estágio 2).")
        return
    print(f"Quantidade de posições NCM distintas: {df['ncm_posicao'].nunique()}")

    # Distribuição das classes
    distribuicao = df["ncm_posicao"].value_counts()
    distribuicao.to_csv(
        SAIDA_DISTRIBUICAO, header=["quantidade"], encoding="utf-8"
    )
    print(f"\nDistribuição completa salva em: {SAIDA_DISTRIBUICAO}")
    print("\nTop 10 posições NCM mais frequentes:")
    print(distribuicao.head(10))

    # Filtra para as TOP_N_CLASSES mais frequentes
    classes_top = distribuicao.head(TOP_N_CLASSES).index
    df_top = df[df["ncm_posicao"].isin(classes_top)].copy()

    print(
        f"\nApós filtrar para top {TOP_N_CLASSES} classes: "
        f"{len(df_top)} itens ({len(df_top) / len(df):.1%} do total válido)"
    )

    if not df_top.empty:
        print("\nDistribuição mínima/máxima por classe (top-N):")
        contagem_top = df_top["ncm_posicao"].value_counts()
        print(f"  menor classe: {contagem_top.min()} exemplos")
        print(f"  maior classe: {contagem_top.max()} exemplos")

        # Estatísticas de tamanho das descrições (útil para decidir
        # max_length do tokenizador na etapa de fine-tuning)
        df_top["tamanho_descricao"] = df_top["descricao_item"].str.len()
        print("\nEstatísticas de tamanho da descrição (caracteres):")
        print(df_top["tamanho_descricao"].describe())

    df_top.to_csv(SAIDA_DATASET, index=False, encoding="utf-8")
    print(f"\nDataset final salvo em: {SAIDA_DATASET}")


if __name__ == "__main__":
    main()
