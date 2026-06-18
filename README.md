# Previsão de Rendimento Escolar com Regressão
## Projeto Final — Curso de Inteligência Artificial
**ODS 4 — Educação de Qualidade
 
---
 
## Descrição do Projeto
Este projeto deseja entender **quais fatores socioeconômicos mais influenciam o rendimento
escolar dos municípios brasileiros**, medido pela nota média do ENEM, utilizando
técnicas de **regressão linear**, **Regressão linear múltipla** e **Regressão polinomial**  como ferramentas essenciais para este tipo de análise.
 
Então, é possível fazer uma pergunta central, que está logo abaixo:
> *É possível prever o desempenho médio de estudantes de um município com base em
> indicadores como IDH, renda per capita, acesso à internet e infraestrutura escolar?*

## Relação com a ODS 4
A ODS 4 busca *"assegurar a educação inclusiva e equitativa e de qualidade, e promover
oportunidades de aprendizagem ao longo da vida para todas e todos"*, o que faz total sentido com o que o projeto visa buscar, isto é, entender de que maneiras é possível fazer com que tal busca possa ser promovida adequadamente. Assim, um modelo que prediz o rendimento escolar pode:
- Identificar municípios em situação de vulnerabilidade educacional
- Servir como ferramenta de monitoramento das metas do IDEB (Índice de Desenvolvimento da Educação Básica)
- Analisar quais são as variáveis com maior influencia sobre o aprendizado

## Fontes de Dados Reais
 
| Dado | Fonte | Link |
|---|---|---|
| Notas por escola/município (ENEM) | INEP | https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem |
| IDEB por escola/município | INEP | https://www.gov.br/inep/pt-br/areas-de-atuacao/pesquisas-estatisticas-e-indicadores/ideb/resultados |
| IDH municipal | PNUD / IPEA | https://www.ipea.gov.br/ipeageo/bases.html |
| Censo Escolar (infraestrutura) | INEP | https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/censo-escolar |
| Renda per capita municipal | IBGE | https://www.ibge.gov.br/estatisticas/economicas/contas-nacionais |

O INEP já disponibiliza microdados do ENEM por escola com n > 100.000.

## Algoritmos que serão implementados
- Regressão Linear Simples
- Regressão Linear Múltipla
- Regressão Polinomial
- Ridge
- Lasso

## Referências
 
- INEP. *Microdados do ENEM*, 2022. Disponível em: https://www.gov.br/inep
- PNUD Brasil. *Atlas do Desenvolvimento Humano no Brasil*, 2013. Disponível em: http://www.atlasbrasil.org.br
- ONU Brasil. *Objetivos de Desenvolvimento Sustentável*, 2015. Disponível em: https://brasil.un.org/pt-br/sdgs
- JAMES, G. et al. *An Introduction to Statistical Learning*. 2ª ed. Springer, 2021.
- PEDREGOSA, F. et al. *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825-2830, 2011.
