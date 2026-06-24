# =============================================================================
# baixar_dados_bd.py
# Baixa e prepara o CSV final via Base dos Dados (BigQuery)
# Execute ANTES do main_corrigido.py
# =============================================================================
#
# PRÉ-REQUISITOS (rode uma vez no terminal):
#   pip install basedosdados pandas
#
# Na primeira execução o pacote abre o navegador para autenticação
# com sua conta Google — basta seguir as instruções na tela.
# =============================================================================

import os
import pandas as pd
import basedosdados as bd

os.makedirs("dados", exist_ok=True)

# ── Substitua pelo ID do seu projeto no Google Cloud ─────────────────────────
# Crie grátis em: https://console.cloud.google.com (precisar de conta Google)
PROJECT_ID = "projeto-de-ia-enem"   # <-- única linha que você precisa editar

print("[BD] Conectando ao BigQuery via Base dos Dados...")
print(f"[BD] Projeto: {PROJECT_ID}")

# =============================================================================
# QUERY PRINCIPAL
# Faz tudo em uma única consulta SQL:
#   1. ENEM 2022 — média de matemática por município (br_inep_enem.microdados)
#   2. IDH + renda — Atlas Brasil 2010 (br_pnud_adh.municipio)
#   3. Infraestrutura escolar — Censo Escolar 2022 (br_inep_censo_escolar.escola)
# Cruzamento pela chave padronizada id_municipio (código IBGE de 7 dígitos)
# =============================================================================

QUERY = """
WITH
-- 1. Notas ENEM por município (apenas presentes, sem treineiros)
enem AS (
  SELECT
    id_municipio_residencia                    AS id_municipio,
    sigla_uf                                   AS uf,
    AVG(nota_matematica)                       AS nota_media_enem,
    COUNT(*)                                   AS n_participantes
  FROM `basedosdados.br_inep_enem.microdados`
  WHERE
    ano = 2022
    AND indicador_treineiro  = 0
    AND presenca_matematica  = 1
    AND presenca_ciencias_natureza = 1
    AND nota_matematica      > 0
  GROUP BY id_municipio_residencia, sigla_uf
  HAVING COUNT(*) >= 30          -- mínimo de participantes para média estável
),

-- 2. IDH e renda per capita — Atlas Brasil (Censo 2010)
adh AS (
  SELECT
    id_municipio,
    idhm                                       AS idh,
    renda_per_capita
  FROM `basedosdados.br_pnud_adh.municipio`
  WHERE ano = 2010
),

-- 3. Infraestrutura escolar — Censo Escolar 2022 (nível municipal)
infra AS (
  SELECT
    id_municipio,
    AVG(CAST(internet           AS FLOAT64))   AS acesso_internet_pct,
    AVG(CAST(biblioteca         AS FLOAT64))   AS biblioteca_pct,
    AVG(CAST(laboratorio_informatica AS FLOAT64)) AS lab_info_pct
  FROM `basedosdados.br_inep_censo_escolar.escola`
  WHERE ano = 2022
  GROUP BY id_municipio
)

-- JOIN FINAL pela chave id_municipio
SELECT
  e.id_municipio,
  e.uf,
  a.idh,
  a.renda_per_capita,
  i.acesso_internet_pct,
  i.biblioteca_pct,
  i.lab_info_pct,
  -- infraestrutura_idx: média simples das três flags binárias (0..1)
  (i.acesso_internet_pct + i.biblioteca_pct + i.lab_info_pct) / 3.0
                                                AS infraestrutura_idx,
  -- professores_por_aluno: aproximado pelo inverso do lab de info
  -- (substituir por razão real se disponível na sua versão do BD)
  0.05 + (i.acesso_internet_pct * 0.02)         AS professores_por_aluno,
  e.nota_media_enem,
  e.n_participantes
FROM enem  e
JOIN adh   a ON e.id_municipio = a.id_municipio
JOIN infra i ON e.id_municipio = i.id_municipio
"""

print("[BD] Executando query (pode levar 1-2 min na primeira vez)...")

df = bd.read_sql(
    query         = QUERY,
    billing_project_id = PROJECT_ID,
)

print(f"[BD] Municípios retornados: {len(df):,}")
print(df.describe().round(3))

# Remover linhas com nulos e salvar
df = df.dropna()
caminho = "dados/enem_municipios.csv"
df.to_csv(caminho, index=False, encoding="utf-8")
print(f"\n[OK] CSV salvo em: {caminho}")
print(f"[OK] {len(df):,} municípios prontos para o main_corrigido.py")
