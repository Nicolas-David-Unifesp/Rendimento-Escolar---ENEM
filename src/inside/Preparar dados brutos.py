# =============================================================================
# preparar_dados_brutos.py
# Tratamento completo dos dados brutos do INEP (ENEM + Censo Escolar + IDH)
# Gera: dados/enem_municipios.csv  — mesmo formato do main_corrigido.py
#
# ARQUIVOS NECESSÁRIOS (baixar manualmente):
#   dados/MICRODADOS_ENEM_2023.csv       → INEP (extraído do ZIP)
#   dados/ESCOLAS_2023.CSV               → Censo Escolar (extraído do ZIP)
#   dados/idh_municipios.csv             → Atlas Brasil (exportado do site)
#
# COMO BAIXAR:
#   ENEM:         https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem
#   Censo Escolar:https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/censo-escolar
#   IDH:          https://www.atlasbrasil.org.br/ranking  → botão "Baixar planilha"
# =============================================================================

import os
import pandas as pd
import numpy as np

os.makedirs("dados", exist_ok=True)

# =============================================================================
# CONFIGURAÇÃO — ajuste os caminhos se necessário
# =============================================================================

CAMINHO_ENEM   = "dados/MICRODADOS_ENEM_2023.csv"
CAMINHO_ESCOLAS = "dados/ESCOLAS_2023.CSV"
CAMINHO_IDH    = "dados/idh_municipios.csv"
CAMINHO_SAIDA  = "dados/enem_municipios.csv"

MIN_PARTICIPANTES = 30   # municípios com menos são descartados (média instável)


# =============================================================================
# ETAPA 1 — ENEM: carregar, limpar e agregar por município
# =============================================================================

def etapa_enem(caminho: str) -> pd.DataFrame:
    print("\n[ETAPA 1] Carregando microdados do ENEM...")
    print("  (arquivo grande ~1 GB, pode levar 1-3 minutos)")

    # Selecionar apenas colunas necessárias para economizar memória
    COLUNAS = [
        "CO_MUNICIPIO_RESIDENCIA",
        "SG_UF_RESIDENCIA",
        "NU_NOTA_MT",
        "NU_NOTA_CN",
        "NU_NOTA_CH",
        "NU_NOTA_LC",
        "NU_NOTA_REDACAO",
        "TP_PRESENCA_CN",
        "TP_PRESENCA_CH",
        "TP_PRESENCA_LC",
        "TP_PRESENCA_MT",
        "IN_TREINEIRO",
    ]

    df = pd.read_csv(
        caminho,
        sep=";",
        encoding="latin-1",   # padrão INEP — não mudar
        usecols=COLUNAS,
        low_memory=False,
    )
    print(f"  Linhas brutas: {len(df):,}")

    # ── Limpeza ──────────────────────────────────────────────────────────────

    # 1. Remover treineiros (fazem a prova sem intenção de usar a nota)
    df = df[df["IN_TREINEIRO"] == 0].copy()

    # 2. Manter apenas quem compareceu em TODAS as provas (presença = 1)
    #    Valor 0 = faltou, 2 = eliminado — ambos devem ser excluídos
    for col in ["TP_PRESENCA_CN", "TP_PRESENCA_CH",
                "TP_PRESENCA_LC", "TP_PRESENCA_MT"]:
        df = df[df[col] == 1]

    # 3. Converter notas para numérico e remover zeros/nulos
    #    No INEP, nota 0 geralmente significa ausência codificada, não nota zero real
    NOTAS = ["NU_NOTA_MT", "NU_NOTA_CN", "NU_NOTA_CH",
             "NU_NOTA_LC", "NU_NOTA_REDACAO"]
    for col in NOTAS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[df[col].notna() & (df[col] > 0)]

    # 4. Remover notas fora do intervalo TRI (300–900 é o esperado)
    for col in NOTAS:
        df = df[(df[col] >= 200) & (df[col] <= 1000)]

    print(f"  Após limpeza: {len(df):,} participantes válidos")

    # ── Agregação por município ───────────────────────────────────────────────
    print("  Agregando por município...")

    df_mun = (
        df.groupby(["CO_MUNICIPIO_RESIDENCIA", "SG_UF_RESIDENCIA"])
        .agg(
            nota_media_enem =("NU_NOTA_MT",      "mean"),
            nota_cn_media   =("NU_NOTA_CN",      "mean"),
            nota_ch_media   =("NU_NOTA_CH",      "mean"),
            nota_lc_media   =("NU_NOTA_LC",      "mean"),
            nota_red_media  =("NU_NOTA_REDACAO", "mean"),
            n_participantes =("NU_NOTA_MT",      "count"),
        )
        .reset_index()
    )

    # Descartar municípios com poucos participantes (média estatisticamente instável)
    antes = len(df_mun)
    df_mun = df_mun[df_mun["n_participantes"] >= MIN_PARTICIPANTES]
    print(f"  Municípios com >= {MIN_PARTICIPANTES} participantes: "
          f"{len(df_mun):,} (removidos {antes - len(df_mun):,})")

    # Renomear coluna de UF para coincidir com o main_corrigido.py
    df_mun = df_mun.rename(columns={"SG_UF_RESIDENCIA": "uf"})

    return df_mun


# =============================================================================
# ETAPA 2 — CENSO ESCOLAR: infraestrutura e professores por município
# =============================================================================

def etapa_censo_escolar(caminho: str) -> pd.DataFrame:
    print("\n[ETAPA 2] Carregando Censo Escolar...")

    # O separador do Censo Escolar é "|", não ";"
    COLUNAS = [
        "CO_MUNICIPIO",        # código IBGE (7 dígitos)
        "IN_INTERNET",         # 1 = tem internet, 0 = não tem
        "IN_BIBLIOTECA",       # 1 = tem biblioteca, 0 = não tem
        "IN_LABORATORIO_INFORMATICA",  # 1 = tem lab de informática
        "QT_DOC_FUND",         # docentes no fundamental
        "QT_DOC_MED",          # docentes no médio
        "QT_MAT_FUND",         # matrículas no fundamental
        "QT_MAT_MED",          # matrículas no médio
    ]

    df = pd.read_csv(
        caminho,
        sep="|",              # separador padrão do Censo Escolar — não mudar
        encoding="latin-1",
        usecols=COLUNAS,
        low_memory=False,
    )
    print(f"  Escolas carregadas: {len(df):,}")

    # Converter tudo para numérico
    for col in COLUNAS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Agregar por município
    df_mun = (
        df.groupby("CO_MUNICIPIO")
        .agg(
            acesso_internet_pct         =("IN_INTERNET",              "mean"),
            biblioteca_pct              =("IN_BIBLIOTECA",            "mean"),
            lab_info_pct                =("IN_LABORATORIO_INFORMATICA","mean"),
            total_docentes              =("QT_DOC_FUND",              "sum"),
            total_docentes_med          =("QT_DOC_MED",               "sum"),
            total_matriculas            =("QT_MAT_FUND",              "sum"),
            total_matriculas_med        =("QT_MAT_MED",               "sum"),
        )
        .reset_index()
    )

    # professores_por_aluno: razão docentes / matrículas (evita divisão por zero)
    total_doc = df_mun["total_docentes"] + df_mun["total_docentes_med"]
    total_mat = df_mun["total_matriculas"] + df_mun["total_matriculas_med"]
    df_mun["professores_por_aluno"] = np.where(
        total_mat > 0,
        (total_doc / total_mat).clip(0, 0.3),   # cap em 0.3 para remover outliers
        np.nan
    )

    # infraestrutura_idx: média simples das 3 flags binárias (resultado entre 0 e 1)
    df_mun["infraestrutura_idx"] = (
        df_mun["acesso_internet_pct"] +
        df_mun["biblioteca_pct"] +
        df_mun["lab_info_pct"]
    ) / 3.0

    # Manter apenas colunas necessárias e renomear chave
    df_mun = df_mun[[
        "CO_MUNICIPIO",
        "acesso_internet_pct",
        "professores_por_aluno",
        "infraestrutura_idx",
    ]].rename(columns={"CO_MUNICIPIO": "CO_MUNICIPIO_RESIDENCIA"})

    print(f"  Municípios com dados de infraestrutura: {len(df_mun):,}")
    return df_mun


# =============================================================================
# ETAPA 3 — IDH: carregar e normalizar o arquivo do Atlas Brasil
# =============================================================================

def etapa_idh(caminho: str) -> pd.DataFrame:
    """
    O Atlas Brasil exporta em vários formatos dependendo da versão do site.
    Esta função tenta detectar automaticamente o separador e as colunas.

    Colunas esperadas no arquivo exportado:
        Código do município IBGE  → renomeado para CO_MUNICIPIO_RESIDENCIA
        IDHM                      → renomeado para idh
        Renda per capita          → renomeado para renda_per_capita
    """
    print("\n[ETAPA 3] Carregando dados de IDH (Atlas Brasil)...")

    # Tentar ler com separador ";" (padrão do Atlas) ou "," (CSV genérico)
    for sep in [";", ","]:
        try:
            df = pd.read_csv(caminho, sep=sep, encoding="latin-1", low_memory=False)
            if len(df.columns) >= 3:
                break
        except Exception:
            continue

    print(f"  Colunas encontradas: {list(df.columns[:8])}")

    # ── Mapear colunas — o Atlas Brasil muda nomes entre versões ─────────────
    # Procura pelo código do município (7 dígitos), IDH e renda
    col_codigo = None
    col_idh    = None
    col_renda  = None

    for col in df.columns:
        col_lower = col.lower().strip()
        if col_codigo is None and any(k in col_lower for k in
                ["codigo", "código", "cod_mun", "codmun", "co_municipio", "id_municipio"]):
            col_codigo = col
        if col_idh is None and any(k in col_lower for k in
                ["idhm", "idh"]) and "renda" not in col_lower and "longe" not in col_lower:
            col_idh = col
        if col_renda is None and any(k in col_lower for k in
                ["renda_per", "renda per", "rendimento", "renda_pc"]):
            col_renda = col

    # Fallback: usar posição se não encontrar pelo nome
    if col_codigo is None:
        col_codigo = df.columns[0]
        print(f"  [AVISO] Coluna de código não identificada — usando '{col_codigo}'")
    if col_idh is None:
        col_idh = df.columns[1]
        print(f"  [AVISO] Coluna de IDH não identificada — usando '{col_idh}'")
    if col_renda is None:
        col_renda = df.columns[2]
        print(f"  [AVISO] Coluna de renda não identificada — usando '{col_renda}'")

    df_idh = df[[col_codigo, col_idh, col_renda]].copy()
    df_idh.columns = ["CO_MUNICIPIO_RESIDENCIA", "idh", "renda_per_capita"]

    # Converter para numérico (o Atlas usa vírgula como decimal em algumas versões)
    for col in ["idh", "renda_per_capita"]:
        df_idh[col] = (
            df_idh[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .pipe(pd.to_numeric, errors="coerce")
        )

    # Garantir que o código do município é inteiro (remover decimais se houver)
    df_idh["CO_MUNICIPIO_RESIDENCIA"] = (
        pd.to_numeric(df_idh["CO_MUNICIPIO_RESIDENCIA"], errors="coerce")
        .dropna()
        .astype(int)
    )
    df_idh = df_idh.dropna()

    print(f"  Municípios com IDH: {len(df_idh):,}")
    print(f"  IDH: min={df_idh['idh'].min():.3f} · max={df_idh['idh'].max():.3f}")
    return df_idh


# =============================================================================
# ETAPA 4 — CRUZAMENTO e NORMALIZAÇÃO
# =============================================================================

def cruzar_e_normalizar(df_enem, df_censo, df_idh) -> pd.DataFrame:
    """
    Une as três tabelas pela chave CO_MUNICIPIO_RESIDENCIA.

    Normalização aplicada:
        acesso_internet_pct  → já está em [0,1] (média de flags binárias)
        infraestrutura_idx   → já está em [0,1]
        professores_por_aluno→ já está em [0, 0.3] após clip
        idh                  → já está em [0,1] (definição do PNUD)
        renda_per_capita     → escala log para reduzir assimetria,
                               depois min-max para [0,1]
        nota_media_enem      → NÃO normalizada — é o target (y),
                               manter na escala original facilita
                               interpretação do RMSE em pontos ENEM
    """
    print("\n[ETAPA 4] Cruzando tabelas...")

    # Garantir tipo inteiro nas chaves para o merge funcionar
    df_enem["CO_MUNICIPIO_RESIDENCIA"]  = df_enem["CO_MUNICIPIO_RESIDENCIA"].astype(int)
    df_censo["CO_MUNICIPIO_RESIDENCIA"] = df_censo["CO_MUNICIPIO_RESIDENCIA"].astype(int)
    df_idh["CO_MUNICIPIO_RESIDENCIA"]   = df_idh["CO_MUNICIPIO_RESIDENCIA"].astype(int)

    df = (
        df_enem
        .merge(df_censo, on="CO_MUNICIPIO_RESIDENCIA", how="inner")
        .merge(df_idh,   on="CO_MUNICIPIO_RESIDENCIA", how="inner")
    )
    print(f"  Municípios após cruzamento: {len(df):,}")

    # ── Normalização da renda (log + min-max) ─────────────────────────────────
    # Renda tem distribuição muito assimétrica — log reduz o efeito de outliers
    # antes de escalar para [0,1]
    renda_log = np.log1p(df["renda_per_capita"])          # log(1 + x)
    df["renda_per_capita"] = (
        (renda_log - renda_log.min()) /
        (renda_log.max() - renda_log.min())
    )

    # ── Remover nulos e selecionar colunas finais ─────────────────────────────
    COLUNAS_FINAIS = [
        "CO_MUNICIPIO_RESIDENCIA",
        "uf",
        "idh",
        "renda_per_capita",
        "acesso_internet_pct",
        "professores_por_aluno",
        "infraestrutura_idx",
        "nota_media_enem",
        "n_participantes",
    ]
    df = df[COLUNAS_FINAIS].dropna()

    print(f"  Municípios válidos finais: {len(df):,}")
    print("\n  Estatísticas das features normalizadas:")
    print(df[[
        "idh", "renda_per_capita", "acesso_internet_pct",
        "professores_por_aluno", "infraestrutura_idx"
    ]].describe().round(4).to_string())

    return df


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

if __name__ == "__main__":

    # Verificar se os arquivos existem antes de começar
    arquivos = {
        "ENEM":          CAMINHO_ENEM,
        "Censo Escolar": CAMINHO_ESCOLAS,
        "IDH":           CAMINHO_IDH,
    }
    print("=" * 60)
    print("VERIFICAÇÃO DOS ARQUIVOS")
    print("=" * 60)
    faltando = []
    for nome, caminho in arquivos.items():
        existe = os.path.isfile(caminho)
        status = "OK" if existe else "FALTANDO"
        print(f"  [{status}] {nome}: {caminho}")
        if not existe:
            faltando.append((nome, caminho))

    if faltando:
        print("\n[ERRO] Arquivos não encontrados:")
        for nome, caminho in faltando:
            print(f"  - {nome}: coloque o arquivo em '{caminho}'")
        print("\nBaixe os arquivos conforme as instruções no topo deste script.")
        exit(1)

    # Executar etapas
    df_enem  = etapa_enem(CAMINHO_ENEM)
    df_censo = etapa_censo_escolar(CAMINHO_ESCOLAS)
    df_idh   = etapa_idh(CAMINHO_IDH)
    df_final = cruzar_e_normalizar(df_enem, df_censo, df_idh)

    # Salvar CSV final
    df_final.to_csv(CAMINHO_SAIDA, index=False, encoding="utf-8")
    print(f"\n[OK] CSV salvo em: {CAMINHO_SAIDA}")
    print(f"[OK] {len(df_final):,} municípios prontos para o main_corrigido.py")
    print("\nPróximo passo:")
    print("  python main_corrigido.py")