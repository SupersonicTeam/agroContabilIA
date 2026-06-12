"""Exploração da API de Dados do Portal da Transparência (Notas Fiscais).

Objetivo: descobrir a estrutura REAL dos endpoints disponíveis antes de
montar o pipeline de coleta definitivo (02_coletar_dataset.py). Roda em
modo "debug" - imprime status code e estrutura da resposta de cada
tentativa, e salva amostras em JSON para inspeção manual.

Pré-requisito: chave de API do Portal da Transparência.
Como obter: https://portaldatransparencia.gov.br/api-de-dados/cadastrar-email
(requer conta gov.br nível Prata/Ouro, ou CPF + autenticação em 2 fatores)

Documentação oficial (Swagger):
https://api.portaldatransparencia.gov.br/swagger-ui/index.html

A documentação pública NÃO confirma com certeza os nomes exatos dos
parâmetros do endpoint de listagem nem dos campos de item (descrição/NCM).
Este script existe justamente para descobrir isso na prática.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TRANSPARENCIA_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "Defina a variável de ambiente TRANSPARENCIA_API_KEY "
        "(crie um arquivo .env com TRANSPARENCIA_API_KEY=sua_chave)"
    )

BASE_URL = "https://api.portaldatransparencia.gov.br/api-de-dados"
HEADERS = {"chave-api-dados": API_KEY}

OUTPUT_DIR = Path("dados/raw_samples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Limite documentado de ~90 requisições/minuto -> ~0.7s entre chamadas
SLEEP_BETWEEN_CALLS = 0.7


def chamar(endpoint: str, params: dict, nome_arquivo: str) -> dict | list | None:
    """Faz uma chamada GET, imprime a estrutura e salva a resposta em JSON."""
    url = f"{BASE_URL}/{endpoint}"
    print(f"\n=== GET {url} ===")
    print(f"params: {params}")

    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    print(f"status: {resp.status_code}")

    if resp.status_code != 200:
        print(f"corpo da resposta (erro): {resp.text[:500]}")
        return None

    data = resp.json()
    print(f"tipo da resposta: {type(data).__name__}")
    if isinstance(data, list):
        print(f"qtd de itens retornados: {len(data)}")
        if data:
            print("chaves do primeiro item:", list(data[0].keys()))
    elif isinstance(data, dict):
        print("chaves do objeto retornado:", list(data.keys()))

    caminho = OUTPUT_DIR / nome_arquivo
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"salvo em: {caminho}")

    time.sleep(SLEEP_BETWEEN_CALLS)
    return data


def main() -> None:
    # 1) Tentativa de listagem de notas fiscais por período.
    #    Padrão observado em outros endpoints da mesma API (ex: licitacoes):
    #    dataInicial/dataFinal no formato DD/MM/AAAA + paginação. Para
    #    notas-fiscais o nome do parâmetro de data pode ser diferente -
    #    se der erro 4xx, o corpo da resposta normalmente diz qual
    #    parâmetro é esperado.
    notas = chamar(
        "notas-fiscais",
        {
            "dataEmissaoInicial": "01/03/2026",
            "dataEmissaoFinal": "31/03/2026",
            "pagina": 1,
        },
        "01_notas_fiscais_lista.json",
    )

    if not notas:
        print(
            "\nTentando nomes alternativos de parâmetros de data "
            "(dataInicial/dataFinal)..."
        )
        notas = chamar(
            "notas-fiscais",
            {
                "dataInicial": "01/03/2026",
                "dataFinal": "31/03/2026",
                "pagina": 1,
            },
            "01b_notas_fiscais_lista_alt.json",
        )

    # 2) Se a listagem funcionou e trouxe uma chave de NF-e, consulta o
    #    detalhe dessa nota - é aqui que esperamos encontrar os itens
    #    (produto/serviço + NCM).
    if notas and isinstance(notas, list) and notas:
        primeira = notas[0]
        chave = (
            primeira.get("chaveNotaFiscal")
            or primeira.get("chaveAcesso")
            or primeira.get("chave")
        )
        if chave:
            print(f"\nChave encontrada na listagem: {chave}")
            chamar(
                "notas-fiscais-por-chave",
                {"chaveUnicaNotaFiscal": chave},
                "02_nota_fiscal_detalhe.json",
            )
        else:
            print("\nAVISO: não encontrei um campo de chave no item retornado.")
            print("Campos disponíveis:", list(primeira.keys()))
            print("Inspecione 01_notas_fiscais_lista.json manualmente.")
    else:
        print(
            "\nAVISO: a listagem de notas-fiscais não retornou itens com "
            "nenhuma combinação de parâmetros testada. Abra o Swagger "
            "(link no topo deste arquivo), procure o endpoint de "
            "notas fiscais e confira o nome correto dos parâmetros, "
            "depois ajuste a função chamar(...) em main()."
        )


if __name__ == "__main__":
    main()
