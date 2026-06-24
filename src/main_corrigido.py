# =============================================================================
# Projeto Final — Curso de Inteligência Artificial
# Tema: Previsão de Rendimento Escolar com Regressão Linear
# ODS 4 — Educação de Qualidade
# =============================================================================
# CORREÇÕES em relação à versão anterior:
#   [C1] FEATURES e TARGET definidos uma única vez (config central)
#   [C2] Tempo de execução mede fit() + predict() juntos
#   [C3] Alpha de Ridge/Lasso escolhido por CV no treino, não pelo teste
#   [C4] Alinhamento resíduos ↔ municípios feito por index, não por iloc
#   [C5] curva_aprendizado com seed fixo → resultados reproduzíveis
#   [C6] Cores dos gráficos geradas dinamicamente (sem hardcode de quantidade)
#   [C7] Arquivos unificados: um único script autocontido
# =============================================================================

import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
warnings.filterwarnings("ignore")
os.makedirs("graficos", exist_ok=True)

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     KFold, GridSearchCV)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline

# =============================================================================
# [C1] CONFIGURAÇÃO CENTRAL — altere aqui, reflete em todo o código
# =============================================================================

FEATURES = [
    "idh",
    "renda_per_capita",
    "acesso_internet_pct",
    "professores_por_aluno",
    "infraestrutura_idx",
]
TARGET        = "nota_media_enem"
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
PALETTE       = ["#1D9E75", "#0F6E56", "#9FE1CB", "#5DCAA5", "#085041", "#04342C"]


# =============================================================================
# 1. CARREGAMENTO E DESCRIÇÃO DOS DADOS
# =============================================================================

def carregar_dados(caminho_csv: str) -> pd.DataFrame:
    """
    Carrega o dataset municipal (sintético ou real já tratado).
    Colunas esperadas: municipio, uf, + todas em FEATURES + TARGET.
    """
    df = pd.read_csv(caminho_csv, encoding="utf-8", sep=",")
    print(f"[LOAD] {df.shape[0]:,} registros · {df.shape[1]} colunas")
    return df


def descrever_dados(df: pd.DataFrame) -> None:
    """Estatísticas descritivas e visualizações exploratórias."""
    print("\n" + "="*60)
    print("ANÁLISE EXPLORATÓRIA")
    print("="*60)
    print(df[FEATURES + [TARGET]].describe().round(3))
    print(f"\nNulos:\n{df[FEATURES + [TARGET]].isnull().sum()}")

    cols = [TARGET] + FEATURES
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Distribuição das Variáveis — Dataset ENEM/IDEB", fontsize=13)
    for ax, col in zip(axes.flatten(), cols):
        ax.hist(df[col].dropna(), bins=40, color=PALETTE[0], edgecolor="white")
        ax.set_title(col.replace("_", " ").title(), fontsize=10)
        ax.set_ylabel("Frequência")
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("graficos/01_distribuicoes.png", dpi=150, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(df[cols].corr().round(2), annot=True, cmap="YlGn",
                fmt=".2f", linewidths=0.5, square=True)
    plt.title("Mapa de Correlação", fontsize=12)
    plt.tight_layout()
    plt.savefig("graficos/02_correlacao.png", dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# 2. PRÉ-PROCESSAMENTO
# =============================================================================

def preprocessar(df: pd.DataFrame):
    """
    1. Remove nulos nas colunas usadas
    2. Guarda o index original para reuso (corrige [C4])
    3. Split treino/teste com random_state fixo
    4. StandardScaler ajustado APENAS no treino (evita leakage)

    Retorna X_train, X_test, y_train, y_test, scaler, idx_test
    onde idx_test é o index pandas das linhas que foram para o teste.
    """
    df_clean = df.dropna(subset=FEATURES + [TARGET]).copy()

    X = df_clean[FEATURES].values
    y = df_clean[TARGET].values
    idx = df_clean.index.values          # [C4] guarda index original

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, idx,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # fit só no treino
    X_test  = scaler.transform(X_test)        # transform no teste

    print(f"\n[SPLIT] Treino: {len(y_train):,} · Teste: {len(y_test):,}")
    return X_train, X_test, y_train, y_test, scaler, idx_test, df_clean


# =============================================================================
# 3. TREINAMENTO — como funciona a regressão linear
# =============================================================================
#
# A regressão linear encontra os coeficientes w que minimizam o erro
# quadrático médio entre y_real e y_predito = X·w + b.
#
# O scikit-learn usa a solução de mínimos quadrados ordinários (OLS):
#   w* = (XᵀX)⁻¹ Xᵀy
#
# Isso significa que NÃO há iteração nem épocas — o modelo resolve
# o sistema linear diretamente com .fit().  Por isso o tempo de
# treinamento é muito baixo (da ordem de milissegundos).
#
# Ridge adiciona λ‖w‖² à função de custo → encolhe coeficientes.
# Lasso adiciona λ‖w‖₁ → pode zerar coeficientes (seleção de features).
# Polinomial expande as features (x, x², x³...) antes do OLS.
#
# =============================================================================

def _medir_tempo_completo(modelo, X_train, X_test, y_train):
    """
    [C2] Mede fit() + predict() juntos.
    O tempo relevante para comparação inclui os dois.
    """
    t0    = time.perf_counter()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    tempo = time.perf_counter() - t0
    return y_pred, tempo


def _avaliar(nome: str, y_test, y_pred, tempo_s: float) -> dict:
    """Calcula e imprime métricas padronizadas."""
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    print(f"  {'─'*48}")
    print(f"  {nome}")
    print(f"  R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  tempo={tempo_s*1000:.2f}ms")
    return {"modelo": nome, "R2": r2, "RMSE": rmse, "MAE": mae,
            "tempo_ms": tempo_s * 1000}


# ── Algoritmo 1: Regressão Linear Simples ────────────────────────────────────

def regressao_simples(X_train, X_test, y_train, y_test):
    """
    Baseline univariado: apenas IDH (1ª coluna de FEATURES).
    Mostra o limite inferior de desempenho — ponto de partida.
    """
    print("\n[ALG 1] Regressão Linear Simples — IDH")
    modelo = LinearRegression()
    y_pred, t = _medir_tempo_completo(modelo, X_train[:,[0]], X_test[:,[0]], y_train)
    res = _avaliar("Lin. Simples (IDH)", y_test, y_pred, t)
    return res, modelo, y_pred


# ── Algoritmo 2: Regressão Linear Múltipla ───────────────────────────────────

def regressao_multipla(X_train, X_test, y_train, y_test):
    """
    Usa todas as features de FEATURES.
    O ganho em R² em relação ao modelo simples mostra o valor
    de cada variável adicional.
    """
    print("\n[ALG 2] Regressão Linear Múltipla — todas as features")
    modelo = LinearRegression()
    y_pred, t = _medir_tempo_completo(modelo, X_train, X_test, y_train)
    res = _avaliar("Lin. Múltipla", y_test, y_pred, t)
    return res, modelo, y_pred


# ── Algoritmo 3 (bônus): Ridge e Lasso com alpha escolhido por CV ────────────

def regressao_regularizada(X_train, X_test, y_train, y_test):
    """
    [C3] Alpha selecionado por GridSearchCV no TREINO (5-fold interno).
    Nunca usa o conjunto de teste para escolher hiperparâmetros.
    Isso evita o leakage indireto da versão anterior.
    """
    print("\n[ALG 3] Ridge e Lasso — alpha por GridSearchCV (CV interno)")
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    resultados_ridge, resultados_lasso = [], []

    for alpha in alphas:
        for Cls, lista, nome in [
            (Ridge, resultados_ridge, "Ridge"),
            (Lasso, resultados_lasso, "Lasso"),
        ]:
            kwargs = {"alpha": alpha}
            if Cls is Lasso:
                kwargs["max_iter"] = 10_000
            modelo = Cls(**kwargs)
            y_pred, t = _medir_tempo_completo(modelo, X_train, X_test, y_train)
            lista.append({
                "alpha"   : alpha,
                "R2"      : r2_score(y_test, y_pred),
                "RMSE"    : np.sqrt(mean_squared_error(y_test, y_pred)),
                "tempo_ms": t * 1000,
            })

    # Melhor alpha por CV no treino
    gs_ridge = GridSearchCV(Ridge(), {"alpha": alphas}, cv=5, scoring="r2")
    gs_lasso = GridSearchCV(Lasso(max_iter=10_000), {"alpha": alphas}, cv=5, scoring="r2")
    gs_ridge.fit(X_train, y_train)
    gs_lasso.fit(X_train, y_train)
    print(f"  Melhor alpha Ridge (CV): {gs_ridge.best_params_['alpha']}  "
          f"R²_cv={gs_ridge.best_score_:.4f}")
    print(f"  Melhor alpha Lasso (CV): {gs_lasso.best_params_['alpha']}  "
          f"R²_cv={gs_lasso.best_score_:.4f}")

    df_ridge = pd.DataFrame(resultados_ridge)
    df_lasso = pd.DataFrame(resultados_lasso)
    print("\nRidge por alpha:\n", df_ridge.to_string(index=False))
    print("\nLasso por alpha:\n", df_lasso.to_string(index=False))
    return df_ridge, df_lasso, gs_ridge.best_params_["alpha"], gs_lasso.best_params_["alpha"]


# ── Algoritmo 4 (bônus): Regressão Polinomial ────────────────────────────────

def regressao_polinomial(X_train, X_test, y_train, y_test):
    """
    Expande features com termos polinomiais (IDH, IDH², IDH³...).
    Grau 1 reproduz a regressão linear — serve de sanity check.
    Graus altos tendem a overfitting: compare treino vs. teste.
    """
    print("\n[ALG 4] Regressão Polinomial — variação de grau (IDH)")
    resultados = []
    for grau in [1, 2, 3, 4]:
        pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=grau, include_bias=False)),
            ("lr",   LinearRegression()),
        ])
        y_pred, t = _medir_tempo_completo(pipe, X_train[:,[0]], X_test[:,[0]], y_train)
        r2_tr = r2_score(y_train, pipe.predict(X_train[:,[0]]))
        r2_te = r2_score(y_test,  y_pred)
        resultados.append({
            "grau": grau, "R2_treino": r2_tr, "R2_teste": r2_te,
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "tempo_ms": t * 1000,
        })
        print(f"  grau={grau}  R²_treino={r2_tr:.4f}  R²_teste={r2_te:.4f}  "
              f"gap={r2_tr-r2_te:.4f}  RMSE={resultados[-1]['RMSE']:.4f}")
    return pd.DataFrame(resultados)


# =============================================================================
# 4. DIAGNÓSTICOS DE VIÉS E OVERFITTING
# =============================================================================

def curva_aprendizado(X_train, y_train) -> None:
    """[C5] seed fixo → resultados reproduzíveis entre execuções."""
    print("\n[DIAG] Curva de aprendizado...")
    rng = np.random.default_rng(RANDOM_STATE)       # [C5]
    tamanhos = np.linspace(0.05, 1.0, 20)
    r2_treinos, r2_vals = [], []

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for frac in tamanhos:
        n   = max(10, int(frac * len(X_train)))
        idx = rng.choice(len(X_train), n, replace=False)
        Xtr, ytr = X_train[idx], y_train[idx]
        modelo = LinearRegression().fit(Xtr, ytr)
        r2_treinos.append(r2_score(ytr, modelo.predict(Xtr)))
        cv = cross_val_score(LinearRegression(), Xtr, ytr,
                             cv=min(5, n), scoring="r2")
        r2_vals.append(cv.mean())

    ns = (tamanhos * len(X_train)).astype(int)
    plt.figure(figsize=(9, 4))
    plt.plot(ns, r2_treinos, label="R² treino",    color=PALETTE[0], lw=2)
    plt.plot(ns, r2_vals,    label="R² validação", color=PALETTE[1], lw=2, ls="--")
    plt.fill_between(ns, r2_treinos, r2_vals, alpha=0.08, color="#888780")
    plt.xlabel("Amostras de treino"); plt.ylabel("R²")
    plt.title("Curva de Aprendizado — Regressão Linear Múltipla", fontsize=12)
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("graficos/08_curva_aprendizado.png", dpi=150, bbox_inches="tight")
    plt.show()


def validacao_cruzada(X_train, y_train, k: int = 10) -> None:
    """k-fold CV para todos os modelos. Desvio baixo = estabilidade."""
    print(f"\n[DIAG] Validação cruzada {k}-fold...")
    kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    modelos = {
        "Lin. Simples":    (LinearRegression(),          X_train[:,[0]]),
        "Lin. Múltipla":   (LinearRegression(),          X_train),
        "Ridge (α=1)":     (Ridge(alpha=1.0),            X_train),
        "Lasso (α=0.01)":  (Lasso(alpha=0.01),          X_train),
        "Poli. grau 2":    (Pipeline([
            ("poly", PolynomialFeatures(2, include_bias=False)),
            ("lr",   LinearRegression())]), X_train),
    }
    resultados = {}
    for nome, (modelo, Xtr) in modelos.items():
        scores = cross_val_score(modelo, Xtr, y_train, cv=kf, scoring="r2")
        resultados[nome] = scores
        print(f"  {nome:22s}  R²={scores.mean():.4f} ± {scores.std():.4f}")

    plt.figure(figsize=(11, 5))
    plt.boxplot(
        [resultados[n] for n in resultados],
        labels=list(resultados.keys()),
        patch_artist=True,
        boxprops=dict(facecolor=PALETTE[2], color=PALETTE[1]),
        medianprops=dict(color=PALETTE[1], linewidth=2),
        whiskerprops=dict(color=PALETTE[1]),
        capprops=dict(color=PALETTE[1]),
    )
    plt.ylabel("R² por fold")
    plt.title(f"Validação Cruzada {k}-fold", fontsize=12)
    plt.xticks(rotation=15, ha="right"); plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("graficos/09_validacao_cruzada.png", dpi=150, bbox_inches="tight")
    plt.show()


def analise_residuos(X_train, X_test, y_train, y_test,
                     df_clean: pd.DataFrame, idx_test) -> None:
    """
    [C4] Alinhamento correto: usa idx_test (index original) para
    recuperar as linhas exatas do df que foram ao conjunto de teste,
    garantindo que resíduo[i] corresponde ao município correto.
    """
    print("\n[DIAG] Análise de resíduos...")
    modelo = LinearRegression().fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    residuos = y_test - y_pred

    # [C4] alinhamento por index, não por iloc
    df_test_real = df_clean.loc[idx_test].copy()
    df_test_real["residuo"]  = residuos
    df_test_real["y_pred"]   = y_pred

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_pred, residuos, alpha=0.35, color=PALETTE[0], s=18, edgecolors="white")
    ax1.axhline(0, color="red", lw=1.5, ls="--")
    ax1.set(xlabel="Nota predita", ylabel="Resíduo",
            title="Resíduos vs. Predito\n(nuvem plana = sem viés)")
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(residuos, bins=50, color=PALETTE[0], edgecolor="white")
    ax2.axvline(residuos.mean(), color="red", lw=1.5, ls="--",
                label=f"Média: {residuos.mean():.2f}")
    ax2.set(xlabel="Resíduo", ylabel="Frequência",
            title="Distribuição dos Resíduos\n(simétrica ao redor de 0 = sem viés)")
    ax2.legend(); ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(X_test[:, 0], residuos, alpha=0.35, color=PALETTE[1], s=18, edgecolors="white")
    ax3.axhline(0, color="red", lw=1.5, ls="--")
    ax3.set(xlabel="IDH (normalizado)", ylabel="Resíduo",
            title="Resíduos vs. IDH\n(sem tendência = relação bem capturada)")
    ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    col_reg = next((c for c in ["regiao", "SG_UF_RESIDENCIA"] if c in df_test_real.columns), None)
    if col_reg:
        res_reg = df_test_real.groupby(col_reg)["residuo"].mean().sort_values()
        cores   = ["#E24B4A" if v < -5 else ("#1D9E75" if v > 5 else "#888780")
                   for v in res_reg.values]
        ax4.barh(res_reg.index, res_reg.values, color=cores, edgecolor="white")
        ax4.axvline(0, color="red", lw=1.5, ls="--")
        ax4.set(xlabel="Resíduo médio (pontos ENEM)",
                title="Viés por Região\n(barras longas = modelo enviesado)")
        ax4.grid(axis="x", alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "Coluna de região\nnão encontrada",
                 ha="center", va="center", transform=ax4.transAxes,
                 color="#888780", fontsize=11)

    plt.suptitle("Análise de Resíduos — Diagnóstico de Viés", fontsize=13)
    plt.savefig("graficos/10_analise_residuos.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Resíduo médio: {residuos.mean():.4f}  |  Desvio: {residuos.std():.4f}")


def comparar_treino_teste(X_train, X_test, y_train, y_test) -> None:
    """Gap treino−teste > 0.10 sinaliza overfitting."""
    print("\n[DIAG] Gap treino vs. teste por modelo...")
    configs = [
        ("Lin. Simples",    LinearRegression(),              X_train[:,[0]], X_test[:,[0]]),
        ("Lin. Múltipla",   LinearRegression(),              X_train,        X_test),
        ("Ridge α=1",       Ridge(alpha=1.0),                X_train,        X_test),
        ("Lasso α=0.01",    Lasso(alpha=0.01),               X_train,        X_test),
        ("Poli. grau 2",    Pipeline([("p", PolynomialFeatures(2)),
                                     ("lr", LinearRegression())]), X_train, X_test),
        ("Poli. grau 4",    Pipeline([("p", PolynomialFeatures(4)),
                                     ("lr", LinearRegression())]), X_train[:,[0]], X_test[:,[0]]),
    ]
    nomes, r2_tr_list, r2_te_list = [], [], []
    for nome, modelo, Xtr, Xte in configs:
        modelo.fit(Xtr, y_train)
        r_tr = r2_score(y_train, modelo.predict(Xtr))
        r_te = r2_score(y_test,  modelo.predict(Xte))
        gap  = r_tr - r_te
        flag = "⚠  OVERFITTING" if gap > 0.10 else "✓ ok"
        print(f"  {nome:20s}  treino={r_tr:.4f}  teste={r_te:.4f}  gap={gap:.4f}  {flag}")
        nomes.append(nome); r2_tr_list.append(r_tr); r2_te_list.append(r_te)

    x, w = np.arange(len(nomes)), 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x-w/2, r2_tr_list, w, label="R² Treino", color=PALETTE[0], edgecolor="white")
    ax.bar(x+w/2, r2_te_list, w, label="R² Teste",  color=PALETTE[1], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(nomes, rotation=15, ha="right")
    ax.set_ylabel("R²"); ax.set_ylim(0, 1.05)
    ax.set_title("R² Treino vs. Teste — Gap indica Overfitting", fontsize=12)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("graficos/11_treino_vs_teste.png", dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# 5. VISUALIZAÇÕES DE RESULTADOS
# =============================================================================

def _cores(n: int) -> list:
    """[C6] Gera exatamente n cores da paleta (sem hardcode de quantidade)."""
    return [PALETTE[i % len(PALETTE)] for i in range(n)]


def plot_comparacao(resultados: list) -> None:
    df_r = pd.DataFrame(resultados)
    x    = np.arange(len(df_r))
    cores = _cores(len(df_r))                          # [C6]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Comparação entre Modelos", fontsize=13)

    ax1.bar(x, df_r["R2"],   color=cores, edgecolor="white")
    ax2.bar(x, df_r["RMSE"], color=cores, edgecolor="white")
    for ax, titulo, ylabel in zip([ax1, ax2],
                                  ["R² (maior = melhor)", "RMSE (menor = melhor)"],
                                  ["R²", "RMSE"]):
        ax.set_xticks(x)
        ax.set_xticklabels(df_r["modelo"], rotation=12, ha="right")
        ax.set_ylabel(ylabel); ax.set_title(titulo); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("graficos/03_comparacao_modelos.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_alpha(df_ridge, df_lasso) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Influência do Alpha na Regularização", fontsize=13)
    for ax, df, nome, cor in zip(axes, [df_ridge, df_lasso],
                                  ["Ridge (L2)", "Lasso (L1)"],
                                  [PALETTE[0], PALETTE[1]]):
        ax2 = ax.twinx()
        ax.plot(df["alpha"],  df["R2"],   color=cor,       marker="o", label="R²")
        ax2.plot(df["alpha"], df["RMSE"], color="#BA7517", marker="s", ls="--", label="RMSE")
        ax.set_xscale("log"); ax.set_xlabel("Alpha (log)")
        ax.set_ylabel("R²", color=cor); ax2.set_ylabel("RMSE", color="#BA7517")
        ax.set_title(nome)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1+h2, l1+l2, loc="center right")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("graficos/04_alpha_regularizacao.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_polinomial(df_poly) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_poly["grau"], df_poly["R2_treino"], color=PALETTE[0],
            marker="o", label="R² treino")
    ax.plot(df_poly["grau"], df_poly["R2_teste"],  color=PALETTE[1],
            marker="s", ls="--", label="R² teste")
    ax.fill_between(df_poly["grau"],
                    df_poly["R2_treino"], df_poly["R2_teste"],
                    alpha=0.08, color="#888780", label="gap (overfitting)")
    ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel("Grau"); ax.set_ylabel("R²")
    ax.set_title("Regressão Polinomial — Treino vs. Teste por Grau", fontsize=12)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("graficos/05_grau_polinomial.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_real_vs_predito(y_test, yp_simples, yp_multipla) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Valores Reais vs. Preditos", fontsize=13)
    for ax, yp, titulo in zip([ax1, ax2], [yp_simples, yp_multipla],
                               ["Regressão Simples", "Regressão Múltipla"]):
        ax.scatter(y_test, yp, alpha=0.4, color=PALETTE[0], s=20, edgecolors="white")
        lims = [min(y_test.min(), yp.min()), max(y_test.max(), yp.max())]
        ax.plot(lims, lims, "r--", lw=1.5, label="Ideal (y=x)")
        ax.set(xlabel="Nota Real", ylabel="Nota Predita", title=titulo)
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("graficos/06_real_vs_predito.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_tempo(resultados: list) -> None:
    df_r  = pd.DataFrame(resultados)
    cores = _cores(len(df_r))                          # [C6]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(df_r["modelo"], df_r["tempo_ms"], color=cores, edgecolor="white")
    ax.set_xlabel("Tempo (ms)")
    ax.set_title("Tempo de Execução (fit + predict)", fontsize=12)   # [C2]
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("graficos/07_tempo_execucao.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_coeficientes(modelo_multipla) -> None:
    """Importância de cada feature pelos coeficientes padronizados."""
    coefs = pd.Series(modelo_multipla.coef_, index=FEATURES).sort_values()
    cores = [PALETTE[0] if v >= 0 else "#E24B4A" for v in coefs.values]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(coefs.index, coefs.values, color=cores, edgecolor="white")
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("Coeficiente (features normalizadas)")
    ax.set_title("Importância das Features — Regressão Múltipla", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("graficos/12_coeficientes.png", dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# 6. PIPELINE PRINCIPAL
# =============================================================================

if __name__ == "__main__":

    # ── Carregamento ──────────────────────────────────────────────────────────
    # Para dados reais: substitua pelo CSV já tratado pelo tratamento_validacao.py
    df = carregar_dados("dados/enem_municipios.csv")

    # Mapear regiões (necessário para diagnóstico de viés geográfico)
    REGIOES = {
        "SP":"Sudeste","MG":"Sudeste","RJ":"Sudeste","ES":"Sudeste",
        "RS":"Sul",    "PR":"Sul",    "SC":"Sul",
        "BA":"Nordeste","PE":"Nordeste","CE":"Nordeste","MA":"Nordeste",
        "PB":"Nordeste","AL":"Nordeste","RN":"Nordeste","PI":"Nordeste","SE":"Nordeste",
        "PA":"Norte","AM":"Norte","RO":"Norte","TO":"Norte",
        "AC":"Norte","AP":"Norte","RR":"Norte",
        "GO":"Centro-Oeste","MT":"Centro-Oeste","MS":"Centro-Oeste","DF":"Centro-Oeste",
    }
    if "uf" in df.columns:
        df["regiao"] = df["uf"].map(REGIOES)

    # ── EDA ───────────────────────────────────────────────────────────────────
    descrever_dados(df)

    # ── Pré-processamento ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, scaler, idx_test, df_clean = preprocessar(df)

    # ── Algoritmos ───────────────────────────────────────────────────────────
    res_s, mod_s, yp_s = regressao_simples(X_train, X_test, y_train, y_test)
    res_m, mod_m, yp_m = regressao_multipla(X_train, X_test, y_train, y_test)
    df_ridge, df_lasso, alpha_ridge, alpha_lasso = regressao_regularizada(
        X_train, X_test, y_train, y_test
    )
    df_poly = regressao_polinomial(X_train, X_test, y_train, y_test)

    # ── Diagnósticos ──────────────────────────────────────────────────────────
    curva_aprendizado(X_train, y_train)
    validacao_cruzada(X_train, y_train, k=10)
    analise_residuos(X_train, X_test, y_train, y_test, df_clean, idx_test)
    comparar_treino_teste(X_train, X_test, y_train, y_test)

    # ── Visualizações finais ──────────────────────────────────────────────────
    resultados_principais = [res_s, res_m]
    plot_comparacao(resultados_principais)
    plot_alpha(df_ridge, df_lasso)
    plot_polinomial(df_poly)
    plot_real_vs_predito(y_test, yp_s, yp_m)
    plot_tempo(resultados_principais)
    plot_coeficientes(mod_m)

    print("\n[OK] Pipeline concluído. Gráficos em ./graficos/")
