# Avaliação Prática 2 Inteligência Artificial e Computacional  

Este projeto consiste na Avaliação Prática 2 da disciplina de Inteligência Artificial e Computacional e tem como objetivo a construção de um modelo de classificador supervisionado e análise.

Três algoritmos de Machine Learning são treinados e comparados:
- Árvore de Decisão
- Random Forest
- Support Vector Machine (SVM com kernel RBF)

## Dataset
O arquivo utilizado é `movies.csv`, que geralmente contém colunas como:
- `name`, `rating`, `genre`, `year`, `released`, `score`, `votes`, `director`, `writer`, `star`, `country`, `budget`, `gross`, `company`, `runtime`

> Fonte do dataset: [Kaggle - Movie Industry](https://www.kaggle.com/datasets/danielgrijalvas/movies)

## Pré-processamento Realizado

1. Remoção da coluna `name` (não relevante para predição)
2. Remoção de linhas sem `score`
3. Criação da coluna alvo `score_class` usando `pd.cut` com os intervalos:
   - `[0, 5.0)` → Baixo
   - `[5.0, 7.5)` → Médio
   - `[7.5, ∞)` → Alto
4. Remoção da coluna original `score`
5. Codificação de variáveis categóricas com `LabelEncoder`:
   - `rating`, `released`, `genre`, `director`, `writer`, `star`, `country`, `company`
6. Remoção de linhas com valores ausentes restantes

## Modelos Utilizados

| Modelo                  | Parâmetros Principais                 |
|------------------------|----------------------------------------|
| DecisionTreeClassifier | `random_state=42`                      |
| RandomForestClassifier | `n_estimators=100`, `random_state=42`  |
| SVC                    | `kernel=`rbf``, `random_state=42`      |

## Métricas Avaliadas

Para cada modelo são exibidos:
- Tempo de treinamento
- Acurácia
- Precisão (weighted)
- Recall (weighted)
- F1-Score (weighted)
- Relatório completo de classificação (`classification_report`)

## Requisitos

```bash
pip install pandas scikit-learn numpy
