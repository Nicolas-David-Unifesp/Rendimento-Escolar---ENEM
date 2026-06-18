#----
# Projeto Final - IA
# Previsão de Rendimento Escolar com Regressão
# ODS 4
#----


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings("ignore") #Serve para não deixar avisos poluindo a saída do sklearn

#Usei o sklearn para facilitar a implementação dos modelos de regressão e pré-processamento


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline

#----
# 1. CARREGAMENTO E DESCRIÇÃO DOS DADOS
#----

def carregar_dados(caminho_csv: str) -> pd.DataFrame:
    """
    Carrega o dataset do ENEM/IDEB.
    Esperado: colunas com indicadores socioeconômicos por município.

    Colunas esperadas (exemplo):
        - municipio, uf, idh, renda_per_capita, acesso_internet_pct,
          professores_por_aluno, infraestrutura_idx, nota_media_enem
    """
    df = pd.read_csv(caminho_csv, encoding="utf-8", sep=",")
    print(f"[INFO] Dataset carregado: {df.shape[0]} registros, {df.shape[1]} colunas")
    return df


def descrever_dados(df: pd.DataFrame):
    """Estatísticas descritivas e visualizações exploratórias."""
    print("\n" + "="*60)
    print("ANÁLISE EXPLORATÓRIA DOS DADOS")
    print("="*60)
    print(df.describe().round(3))
    print(f"\nValores nulos por coluna:\n{df.isnull().sum()}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Distribuição das Variáveis — Dataset ENEM/IDEB", fontsize=14)

    cols_plot = [
        "nota_media_enem", "idh", "renda_per_capita",
        "acesso_internet_pct", "professores_por_aluno", "infraestrutura_idx"
    ]
    for ax, col in zip(axes.flatten(), cols_plot):
        ax.hist(df[col].dropna(), bins=40, edgecolor="white", color="#1D9E75")
        ax.set_title(col.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("Frequência")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("graficos/01_distribuicoes.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Mapa de correlação
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        df[cols_plot].corr().round(2),
        annot=True, cmap="YlGn", fmt=".2f",
        linewidths=0.5, square=True
    )
    plt.title("Mapa de Correlação entre Variáveis", fontsize=13)
    plt.tight_layout()
    plt.savefig("graficos/02_correlacao.png", dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# 2. PRÉ-PROCESSAMENTO
# =============================================================================

def preprocessar(df: pd.DataFrame):
    """
    Remove nulos, separa features e target,
    aplica normalização e divide em treino/teste.
    """
    df = df.dropna()

    FEATURES = [
        "idh", "renda_per_capita",
        "acesso_internet_pct", "professores_por_aluno", "infraestrutura_idx"
    ]
    TARGET = "nota_media_enem"

    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"\n[PRÉ-PROC] Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")
    return X_train_sc, X_test_sc, y_train, y_test, scaler, FEATURES


# =============================================================================
# 3. ALGORITMOS
# =============================================================================

def avaliar_modelo(nome, modelo, X_test, y_test, y_pred, tempo):
    """Imprime métricas de avaliação de um modelo."""
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print(f"\n{'─'*50}")
    print(f"  {nome}")
    print(f"{'─'*50}")
    print(f"  R²    : {r2:.4f}")
    print(f"  RMSE  : {rmse:.4f}")
    print(f"  MAE   : {mae:.4f}")
    print(f"  Tempo : {tempo*1000:.2f} ms")
    return {"modelo": nome, "R2": r2, "RMSE": rmse, "MAE": mae, "tempo_ms": tempo*1000}


# ── Algoritmo 1: Regressão Linear Simples ────────────────────────────────────

def regressao_linear_simples(X_train, X_test, y_train, y_test):
    """
    Regressão linear usando apenas o IDH (1ª feature).
    Serve como baseline e ilustra o caso univariado.
    """
    print("\n[ALG 1] Regressão Linear Simples (baseline — IDH)")

    X_tr_1d = X_train[:, [0]]
    X_te_1d = X_test[:, [0]]

    t0 = time.time()
    modelo = LinearRegression()
    modelo.fit(X_tr_1d, y_train)
    y_pred = modelo.predict(X_te_1d)
    tempo  = time.time() - t0

    return avaliar_modelo("Regressão Linear Simples (IDH)", modelo,
                          X_te_1d, y_test, y_pred, tempo), modelo, y_pred


# ── Algoritmo 2: Regressão Linear Múltipla ───────────────────────────────────

def regressao_linear_multipla(X_train, X_test, y_train, y_test):
    """
    Regressão linear com todas as features socioeconômicas.
    """
    print("\n[ALG 2] Regressão Linear Múltipla (todas as features)")

    t0 = time.time()
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    tempo  = time.time() - t0

    return avaliar_modelo("Regressão Linear Múltipla", modelo,
                          X_test, y_test, y_pred, tempo), modelo, y_pred


# ── Algoritmo 3 (bônus): Ridge e Lasso com variação de alpha ─────────────────

def regressao_regularizada(X_train, X_test, y_train, y_test):
    """
    Compara Ridge (L2) e Lasso (L1) com diferentes valores de alpha.
    Mostra influência do hiperparâmetro de regularização.
    """
    print("\n[ALG 3] Ridge e Lasso — variação de alpha")

    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    resultados_ridge = []
    resultados_lasso = []

    for alpha in alphas:
        # Ridge
        t0 = time.time()
        r = Ridge(alpha=alpha)
        r.fit(X_train, y_train)
        yp = r.predict(X_test)
        t  = time.time() - t0
        resultados_ridge.append({
            "alpha": alpha,
            "R2": r2_score(y_test, yp),
            "RMSE": np.sqrt(mean_squared_error(y_test, yp)),
            "tempo_ms": (t)*1000
        })

        # Lasso
        t0 = time.time()
        l = Lasso(alpha=alpha, max_iter=5000)
        l.fit(X_train, y_train)
        yp = l.predict(X_test)
        t  = time.time() - t0
        resultados_lasso.append({
            "alpha": alpha,
            "R2": r2_score(y_test, yp),
            "RMSE": np.sqrt(mean_squared_error(y_test, yp)),
            "tempo_ms": (t)*1000
        })

    df_ridge = pd.DataFrame(resultados_ridge)
    df_lasso = pd.DataFrame(resultados_lasso)

    print("\nRidge:")
    print(df_ridge.to_string(index=False))
    print("\nLasso:")
    print(df_lasso.to_string(index=False))

    return df_ridge, df_lasso


# ── Algoritmo 4 (bônus): Regressão Polinomial ────────────────────────────────

def regressao_polinomial(X_train, X_test, y_train, y_test):
    """
    Regressão polinomial variando o grau (1 a 4).
    Usa apenas o IDH para facilitar a visualização.
    """
    print("\n[ALG 4] Regressão Polinomial — variação de grau (IDH)")

    X_tr_1d = X_train[:, [0]]
    X_te_1d = X_test[:, [0]]
    resultados = []

    for grau in [1, 2, 3, 4]:
        t0 = time.time()
        pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=grau, include_bias=False)),
            ("lr",   LinearRegression())
        ])
        pipe.fit(X_tr_1d, y_train)
        yp = pipe.predict(X_te_1d)
        t  = time.time() - t0

        res = {
            "grau": grau,
            "R2": r2_score(y_test, yp),
            "RMSE": np.sqrt(mean_squared_error(y_test, yp)),
            "tempo_ms": t*1000
        }
        resultados.append(res)
        print(f"  Grau {grau} → R²={res['R2']:.4f}  RMSE={res['RMSE']:.4f}  tempo={res['tempo_ms']:.2f}ms")

    return pd.DataFrame(resultados)


# =============================================================================
# 4. VISUALIZAÇÕES DE RESULTADOS
# =============================================================================

def plot_comparacao_modelos(resultados: list):
    """Gráfico comparativo de R² e RMSE entre os modelos principais."""
    df_res = pd.DataFrame(resultados)
    modelos = df_res["modelo"]
    x = np.arange(len(modelos))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Comparação entre Modelos de Regressão", fontsize=14)

    ax1.bar(x, df_res["R2"], color=["#1D9E75", "#0F6E56", "#9FE1CB"], edgecolor="white")
    ax1.set_xticks(x); ax1.set_xticklabels(modelos, rotation=12, ha="right")
    ax1.set_ylabel("R²"); ax1.set_ylim(0, 1)
    ax1.set_title("R² (quanto maior, melhor)")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, df_res["RMSE"], color=["#1D9E75", "#0F6E56", "#9FE1CB"], edgecolor="white")
    ax2.set_xticks(x); ax2.set_xticklabels(modelos, rotation=12, ha="right")
    ax2.set_ylabel("RMSE"); ax2.set_title("RMSE (quanto menor, melhor)")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("graficos/03_comparacao_modelos.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_alpha_regularizacao(df_ridge, df_lasso):
    """Mostra influência do alpha de regularização em Ridge e Lasso."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Influência do Hiperparâmetro Alpha na Regularização", fontsize=14)

    for ax, df, nome, cor in zip(
        axes,
        [df_ridge, df_lasso],
        ["Ridge (L2)", "Lasso (L1)"],
        ["#1D9E75", "#0F6E56"]
    ):
        ax2 = ax.twinx()
        ax.plot(df["alpha"], df["R2"],   color=cor,     marker="o", label="R²")
        ax2.plot(df["alpha"], df["RMSE"], color="#BA7517", marker="s", linestyle="--", label="RMSE")
        ax.set_xscale("log")
        ax.set_xlabel("Alpha (escala log)")
        ax.set_ylabel("R²", color=cor)
        ax2.set_ylabel("RMSE", color="#BA7517")
        ax.set_title(nome)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="center right")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("graficos/04_alpha_regularizacao.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_grau_polinomial(df_poly):
    """Mostra influência do grau polinomial no desempenho."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_poly["grau"], df_poly["R2"],   color="#1D9E75", marker="o", label="R²")
    ax2 = ax.twinx()
    ax2.plot(df_poly["grau"], df_poly["RMSE"], color="#BA7517", marker="s",
             linestyle="--", label="RMSE")
    ax.set_xlabel("Grau do Polinômio")
    ax.set_ylabel("R²", color="#1D9E75")
    ax2.set_ylabel("RMSE", color="#BA7517")
    ax.set_title("Influência do Grau Polinomial (IDH → Nota ENEM)", fontsize=12)
    ax.set_xticks([1, 2, 3, 4])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("graficos/05_grau_polinomial.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_real_vs_predito(y_test, y_pred_simples, y_pred_multipla):
    """Scatter: valores reais vs. preditos para ambos os modelos."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Valores Reais vs. Preditos", fontsize=14)

    for ax, y_pred, titulo in zip(
        [ax1, ax2],
        [y_pred_simples, y_pred_multipla],
        ["Regressão Simples", "Regressão Múltipla"]
    ):
        ax.scatter(y_test, y_pred, alpha=0.4, color="#1D9E75", edgecolors="white", s=25)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Ideal")
        ax.set_xlabel("Nota Real")
        ax.set_ylabel("Nota Predita")
        ax.set_title(titulo)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("graficos/06_real_vs_predito.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_tempo_execucao(resultados: list):
    """Gráfico de barras horizontal com tempo de execução por modelo."""
    df_res = pd.DataFrame(resultados)
    fig, ax = plt.subplots(figsize=(9, 4))
    cores = ["#1D9E75", "#0F6E56", "#9FE1CB"]
    ax.barh(df_res["modelo"], df_res["tempo_ms"], color=cores, edgecolor="white")
    ax.set_xlabel("Tempo (ms)")
    ax.set_title("Tempo de Execução por Modelo", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("graficos/07_tempo_execucao.png", dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# 5. PIPELINE PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    import os
    os.makedirs("graficos", exist_ok=True)

    # ── Carregamento ──────────────────────────────────────────────────────────
    # Substitua pelo caminho real do arquivo CSV baixado do INEP/IBGE.
    # Exemplo: df = carregar_dados("dados/enem_municipios_2022.csv")
    df = carregar_dados("dados/enem_municipios.csv")

    # ── Análise exploratória ──────────────────────────────────────────────────
    descrever_dados(df)

    # ── Pré-processamento ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, scaler, features = preprocessar(df)

    # ── Algoritmos ───────────────────────────────────────────────────────────
    res_simples,  mod_simples,  yp_simples  = regressao_linear_simples(X_train, X_test, y_train, y_test)
    res_multipla, mod_multipla, yp_multipla = regressao_linear_multipla(X_train, X_test, y_train, y_test)
    df_ridge, df_lasso = regressao_regularizada(X_train, X_test, y_train, y_test)
    df_poly            = regressao_polinomial(X_train, X_test, y_train, y_test)

    # ── Comparação e visualizações ────────────────────────────────────────────
    resultados_principais = [res_simples, res_multipla]
    plot_comparacao_modelos(resultados_principais)
    plot_alpha_regularizacao(df_ridge, df_lasso)
    plot_grau_polinomial(df_poly)
    plot_real_vs_predito(y_test, yp_simples, yp_multipla)
    plot_tempo_execucao(resultados_principais)

    print("\n[OK] Análise concluída. Gráficos salvos em ./graficos/")