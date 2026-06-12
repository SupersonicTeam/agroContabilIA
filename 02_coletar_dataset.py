"""Coleta de itens de Notas Fiscais Eletrônicas (Portal da Transparência)
para construir o dataset de treino do classificador de NCM.

IMPORTANTE: antes de rodar este script, execute 01_explorar_api.py e
confirme os nomes dos campos/parâmetros (já refletidos nas constantes
abaixo a partir das amostras em dados/raw_samples/).

ATENÇÃO sobre a varredura: o endpoint `notas-fiscais` NÃO aceita filtro
por data (só `cnpjEmitente`, `codigoOrgao`, `nomeProduto` e `pagina`, esta
obrigatória, e exige pelo menos um dos três primeiros). Por isso a coleta
varre **por órgão** (`codigoOrgao` SIAFI), paginando cada órgão até a
listagem vir vazia — não por janelas de data.

Grava incrementalmente em dados/itens_notas_fiscais.csv (modo append),
então o script pode ser interrompido e retomado sem perder progresso.
Ao retomar, as chaves de nota já coletadas são carregadas e puladas,
evitando redetalhar notas e reduzindo duplicatas (resíduo tratado no
estágio 3).
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TRANSPARENCIA_API_KEY")
if not API_KEY:
    raise RuntimeError("Defina TRANSPARENCIA_API_KEY no arquivo .env")

BASE_URL = "https://api.portaldatransparencia.gov.br/api-de-dados"
HEADERS = {"chave-api-dados": API_KEY}
SLEEP_BETWEEN_CALLS = 0.7  # ~90 req/min documentado -> ~0.7s entre chamadas

# --- nomes confirmados em 01_explorar_api.py (dados/raw_samples/) ---
CAMPO_CHAVE_NOTA = "chaveNotaFiscal"  # chave da NF na listagem
CAMPO_ITENS = "itensNotaFiscal"  # lista de itens no detalhe da nota
CAMPO_DESCRICAO = "descricaoProdutoServico"  # descrição do produto/serviço
CAMPO_NCM = "codigoNcmSh"  # código NCM (8 dígitos) do item
# --------------------------------------------------------------------

# Órgãos (código SIAFI) a varrer, focados no domínio agropecuário.
# Liste mais códigos via endpoint /orgaos-siafi.
CODIGOS_ORGAO = [
    "22000",  # Ministério da Agricultura e Pecuária (vínculo direto)
    "22202",  # Empresa Brasileira de Pesquisa Agropecuária (Embrapa)
    "22211",  # Companhia Nacional de Abastecimento (Conab)
    "22803",  # Secretaria Nacional de Defesa Agropecuária
    "49000",  # Min. do Desenvolvimento Agrário e Agricultura Familiar
]

# Produtos agrícolas a buscar por `nomeProduto` (busca textual na descrição
# do item, sem restrição de órgão). Enriquece as classes NCM de insumos/
# commodities agro, que aparecem pouco na coleta por órgão administrativo.
NOMES_PRODUTO = [
    "soja",
    "milho",
    "trigo",
]

SAIDA_CSV = Path("dados/itens_notas_fiscais.csv")
SAIDA_CSV.parent.mkdir(parents=True, exist_ok=True)


def listar_notas(filtro: dict, pagina: int) -> list[dict]:
    """Lista notas para um filtro (`codigoOrgao` OU `nomeProduto`) + página.

    O endpoint exige exatamente um filtro entre cnpjEmitente/codigoOrgao/
    nomeProduto, além da página obrigatória.
    """
    resp = requests.get(
        f"{BASE_URL}/notas-fiscais",
        headers=HEADERS,
        params={**filtro, "pagina": pagina},
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


def carregar_chaves_existentes() -> set[str]:
    """Lê o CSV de saída e retorna as chaves de nota já coletadas.

    Permite retomar uma execução interrompida sem redetalhar notas que
    já foram processadas.
    """
    if not SAIDA_CSV.exists():
        return set()
    chaves: set[str] = set()
    with open(SAIDA_CSV, newline="", encoding="utf-8") as f:
        leitor = csv.reader(f)
        next(leitor, None)  # cabeçalho
        for linha in leitor:
            if linha:
                chaves.add(linha[0])
    return chaves


def gerar_fontes(
    codigos_orgao: list[str],
    nomes_produto: list[str],
) -> list[tuple[str, dict]]:
    """Monta a lista de fontes a varrer, cada uma com um rótulo e o filtro
    correspondente (`codigoOrgao` ou `nomeProduto`)."""
    fontes: list[tuple[str, dict]] = []
    for codigo in codigos_orgao:
        fontes.append((f"órgão {codigo}", {"codigoOrgao": codigo}))
    for nome in nomes_produto:
        fontes.append((f"produto '{nome}'", {"nomeProduto": nome}))
    return fontes


def main(
    codigos_orgao: list[str],
    nomes_produto: list[str],
    max_paginas_por_fonte: int = 3,
) -> None:
    novo_arquivo = not SAIDA_CSV.exists()
    chaves_vistas = carregar_chaves_existentes()
    if chaves_vistas:
        print(f"Retomando: {len(chaves_vistas)} notas já coletadas serão puladas.")

    fontes = gerar_fontes(codigos_orgao, nomes_produto)

    with open(SAIDA_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if novo_arquivo:
            writer.writerow(["chave_nota", "descricao_item", "ncm"])

        for rotulo, filtro in fontes:
            print(f"\n=== {rotulo} ===")

            for pagina in range(1, max_paginas_por_fonte + 1):
                try:
                    notas = listar_notas(filtro, pagina)
                except requests.HTTPError as exc:
                    print(f"  erro na listagem (pag {pagina}): {exc}")
                    break

                if not notas:
                    print(f"  pagina {pagina}: sem resultados, próxima fonte")
                    break

                print(f"  pagina {pagina}: {len(notas)} notas")
                time.sleep(SLEEP_BETWEEN_CALLS)

                for nota in notas:
                    chave = nota.get(CAMPO_CHAVE_NOTA)
                    if not chave or chave in chaves_vistas:
                        continue
                    chaves_vistas.add(chave)

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

    print(f"\nColeta concluída. Dados salvos em: {SAIDA_CSV}")


if __name__ == "__main__":
    # Ajustar órgãos / produtos / nº de páginas conforme o volume desejado.
    main(codigos_orgao=CODIGOS_ORGAO, nomes_produto=NOMES_PRODUTO)
