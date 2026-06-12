# AgroContábil IA — Coleta e exploração de dados (classificador de NCM)

Scripts para coletar, explorar e preparar o dataset de itens de Notas
Fiscais Eletrônicas (descrição do produto/serviço + código NCM), usado
para treinar o classificador NCM do componente de IA do AgroContábil.

Fonte dos dados: API de Dados do Portal da Transparência do Governo
Federal — conjunto "Notas Fiscais Eletrônicas do Poder Executivo Federal".

## 1. Pré-requisitos

- Python 3.10+
- Chave da API do Portal da Transparência:
  1. Acesse https://portaldatransparencia.gov.br/api-de-dados/cadastrar-email
  2. Cadastre-se com conta gov.br (nível Prata/Ouro, ou CPF + 2FA)
  3. A chave chega por e-mail no cadastrado

Crie um arquivo `.env` na raiz deste diretório com:

```
TRANSPARENCIA_API_KEY=sua_chave_aqui
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

## 2. Ordem de execução

### 2.1 `01_explorar_api.py` — rodar SEMPRE primeiro

A documentação pública da API não confirma com 100% de certeza os nomes
exatos dos campos retornados na listagem de notas fiscais e no detalhe
de itens (descrição/NCM). Este script testa os endpoints documentados,
imprime os campos retornados e salva amostras em `dados/raw_samples/`
para inspeção.

**Antes de seguir para o próximo script, confira no JSON salvo:**

- o nome do campo que identifica a chave da nota na listagem
  (`chaveNotaFiscal`, `chaveAcesso`, etc.)
- o nome do campo que traz a lista de itens no detalhe da nota
- os nomes dos campos de descrição e NCM dentro de cada item

Se algum endpoint retornar erro 4xx, o corpo da resposta geralmente indica
o parâmetro esperado — ajuste o script e rode novamente. Documentação
oficial (Swagger):
https://api.portaldatransparencia.gov.br/swagger-ui/index.html

### 2.2 `02_coletar_dataset.py`

Ajuste as constantes `CAMPO_*` no topo do arquivo de acordo com o que foi
descoberto no passo anterior, e ajuste o período (`data_inicial`,
`data_final`) em `main()`. Coleta os itens de notas fiscais período a
período e grava incrementalmente em `dados/itens_notas_fiscais.csv`
(pode ser interrompido e retomado, pois usa append).

Respeita o limite documentado de ~90 requisições/minuto da API
(intervalo de 0.7s entre chamadas).

### 2.3 `03_analise_exploratoria.py`

Faz a limpeza dos dados, calcula o NCM no nível de "posição" (4 primeiros
dígitos), gera a distribuição de classes e exporta o dataset final
filtrado para as `TOP_N_CLASSES` mais frequentes em
`dados/dataset_ncm_top_classes.csv` — esse é o arquivo que alimenta o
treino do baseline (TF-IDF + LogReg/RandomForest) e do fine-tuning do
BERTimbau.

## 3. Saídas geradas

- `dados/raw_samples/*.json` — amostras brutas da API (depuração)
- `dados/itens_notas_fiscais.csv` — itens coletados
  (chave_nota, descricao_item, ncm)
- `dados/distribuicao_ncm.csv` — contagem de itens por posição NCM
- `dados/dataset_ncm_top_classes.csv` — dataset final filtrado, pronto
  para modelagem

## 4. Limitações conhecidas

- O dataset reflete compras governamentais (Poder Executivo Federal),
  não exclusivamente operações do agronegócio — ver discussão na
  proposta sobre por que isso não invalida a tarefa de classificação
  descrição → NCM.
- `TOP_N_CLASSES` é uma simplificação metodológica necessária dado o
  volume de classes NCM (~10 mil códigos de 8 dígitos); documentar essa
  escolha no artigo.
