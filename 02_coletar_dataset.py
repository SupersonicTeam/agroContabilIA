"""Coleta de itens de Notas Fiscais Eletrônicas (Portal da Transparência)
para construir o dataset de treino do classificador de NCM.

IMPORTANTE: antes de rodar este script, execute 01_explorar_api.py e
confirme:
  - o nome correto dos parâmetros do endpoint de listagem de notas
  - o nome do campo com a chave da nota na listagem
  - o nome do campo com a lista de itens no detalhe de cada nota
  - os nomes dos campos de descrição e NCM dentro de cada item

Ajuste as constantes CAMPO_* abaixo de acordo com o que você encontrar.

Grava incrementalmente em dados/itens_notas_fiscais.csv (modo append),
então o script pode ser interrompido e retomado sem perder progresso
(pode gerar linhas duplicadas entre execuções - tratar na etapa de
análise exploratória).
"""

from __future__ import annotations

import csv
import os
import time
from datetime import date, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TRANSPARENCIA_API_KEY")
if not API_KEY:
    raise RuntimeError("Defina TRANSPARENCIA_API_KEY no arquivo .env")

BASE_URL = "https://api.portaldatransparencia.gov.br/api-de-dados"
HEADERS = {"chave-api-dados": API_KEY}
SLEEP_BETWEEN_CALLS = 0.7

# --- ajustar conforme descoberto em 01_explorar_api.py ---
CAMPO_CHAVE_NOTA = "chaveNotaFiscal"  # campo com a chave da NF na listagem
CAMPO_ITENS = "itens"  # campo com a lista de itens no detalhe da nota
CAMPO_DESCRICAO = "descricao"  # campo com a descrição do produto/serviço
CAMPO_NCM = "ncm"  # campo com o código NCM do item
# -----------------------------------------------------------

SAIDA_CSV = Path("dados/itens_notas_fiscais.csv")
SAIDA_CSV.parent.mkdir(parents=True, exist_ok=True)


def listar_notas(data_inicial: str, data_final: str, pagina: int) -> list[dict]:
    resp = requests.get(
        f"{BASE_URL}/notas-fiscais",
        headers=HEADERS,
        params={
            "dataEmissaoInicial": data_inicial,
            "dataEmissaoFinal": data_final,
            "pagina": pagina,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def detalhar_nota(chave: str) -> dict:
    resp = requests.get(
        f"{BASE_URL}/notas-fiscais-por-chave",
        headers=HEADERS,
        params={"chaveUnicaNotaFiscal": chave},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def gerar_intervalos_semanais(inicio: date, fim: date):
    """Gera intervalos de 7 dias entre 'inicio' e 'fim'.

    A API costuma limitar o tamanho do período consultado de uma vez;
    intervalos menores também ajudam a não perder muito progresso se
    a execução for interrompida.
    """
    atual = inicio
    while atual < fim:
        proximo = min(atual + timedelta(days=7), fim)
        yield atual, proximo
        atual = proximo


def main(
    data_inicial: date,
    data_final: date,
    max_paginas_por_semana: int = 5,
) -> None:
    novo_arquivo = not SAIDA_CSV.exists()
    with open(SAIDA_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if novo_arquivo:
            writer.writerow(["chave_nota", "descricao_item", "ncm"])

        for inicio, fim in gerar_intervalos_semanais(data_inicial, data_final):
            fmt_inicio = inicio.strftime("%d/%m/%Y")
            fmt_fim = fim.strftime("%d/%m/%Y")
            print(f"\n--- Período {fmt_inicio} a {fmt_fim} ---")

            for pagina in range(1, max_paginas_por_semana + 1):
                try:
                    notas = listar_notas(fmt_inicio, fmt_fim, pagina)
                except requests.HTTPError as exc:
                    print(f"  erro na listagem (pag {pagina}): {exc}")
                    break

                if not notas:
                    print(f"  pagina {pagina}: sem resultados, próximo período")
                    break

                print(f"  pagina {pagina}: {len(notas)} notas")

                for nota in notas:
                    chave = nota.get(CAMPO_CHAVE_NOTA)
                    if not chave:
                        continue

                    try:
                        detalhe = detalhar_nota(chave)
                    except requests.HTTPError as exc:
                        print(f"    erro ao detalhar {chave}: {exc}")
                        continue

                    itens = detalhe.get(CAMPO_ITENS, []) or []
                    for item in itens:
                        descricao = item.get(CAMPO_DESCRICAO)
                        ncm = item.get(CAMPO_NCM)
                        if descricao and ncm:
                            writer.writerow([chave, descricao, ncm])

                    time.sleep(SLEEP_BETWEEN_CALLS)

                f.flush()
                time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"\nColeta concluída. Dados salvos em: {SAIDA_CSV}")


if __name__ == "__main__":
    # Ajustar o período conforme necessidade de volume de dados.
    main(data_inicial=date(2026, 1, 1), data_final=date(2026, 3, 31))
