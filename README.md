# 📈 Previsão de Preço de Bitcoin com LSTM e Otimização de Hiperparâmetros (Optuna)

Repositório dedicado ao **Mini-projeto com LSTMs** da disciplina de Redes Neurais (**IF702 - CIn/UFPE**), desenvolvido pelos integrantes **Caio Ferreira** e **José Janailson**. O objetivo principal é implementar, treinar e otimizar redes neurais recorrentes baseadas em LSTM para prever a cotação de fechamento (`close`) do Bitcoin utilizando dados históricos (2017–2023).

---

## 📋 Sumário
- [Visão Geral do Projeto](#-visão-geral-do-projeto)
- [Diretrizes Oficiais do Mini-Projeto](#-diretrizes-oficiais-do-mini-projeto)
- [Estrutura e Divisão dos Dados](#-estrutura-e-divisão-dos-dados)
- [Métricas Utilizadas](#-métricas-utilizadas)
- [Evolução dos Experimentos e Modelos](#-evolução-dos-experimentos-e-modelos)
  - [1. Modelo Base (Univariado + StandardScaler)](#1-modelo-base-univariado--standardscaler)
  - [2. Mudança de Scaler (MinMaxScaler)](#2-mudança-de-scaler-minmaxscaler)
  - [3. Abordagem Multivariada Inicial](#3-abordagem-multivariada-inicial)
  - [4. Implementação Keras LSTM com EarlyStopping](#4-implementação-keras-lstm-com-earlystopping)
  - [5. Otimização Avançada com Optuna (Modelo Final)](#5-otimização-avançada-com-optuna-modelo-final)
- [Resultados das Métricas (Conjunto de Teste Final)](#-resultados-das-métricas-conjunto-de-teste-final)
- [Principais Conclusões](#-principais-conclusões)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)

---

## 🚀 Visão Geral do Projeto
A predição de séries temporais financeiras apresenta alta complexidade e volatilidade. Seguindo as diretrizes do mini-projeto, este trabalho explora desde implementações univariadas baseadas em tutoriais de referência até abordagens multivariadas avançadas com sintonia de hiperparâmetros auxiliada pelo **Optuna**, garantindo o respeito estrito à ordem cronológica dos dados sem vazamento de informação futura.

---

## 📐 Diretrizes Oficiais do Mini-Projeto
- **Janela de Dados:** Embora o histórico geral do Bitcoin abarque períodos anteriores, o escopo oficial do mini-projeto utiliza os dados compreendidos estritamente entre **2017 e 2023**.
- **Restrição Temporal:** Apenas os dados até o dia atual ($D$) devem ser empregados para prever o fechamento do dia seguinte ($D+1$), evitando o uso indevido de variáveis futuras do mesmo dia (como o preço de abertura do dia seguinte).

---

## 📊 Estrutura e Divisão dos Dados
- **Dataset:** `data-bitcoin_timedata_v2.csv` (`date`, `open`, `high`, `low`, `close`, `number_of_trades`).
- **Divisão Temporal:** 
  - **Treino:** Primeiros 80% dos registros.
  - **Teste:** 20% restantes (mais recentes).
- **Janela Temporal (`window_size`):** Sequência de dias passados utilizada para inferir o próximo ponto da série.

---

## 📐 Métricas de Avaliação
- **MSE / RMSE / MAE:** Avaliam a magnitude dos erros de previsão de preço em valores absolutos ou escalados.
- **$$R^2$$ (Coeficiente de Determinação):** Mede a aderência do modelo à variabilidade dos dados reais.
- **POCID (Percentage of Correct Direction):** Métrica fundamental exigida no projeto que computa a assertividade do modelo em prever a direção correta da variação diária (subida ou queda da cotação).

---

## 🔬 Evolução dos Experimentos e Modelos

### 1. Modelo Base (Univariado + StandardScaler)
- **Configuração:** Uso exclusivo da coluna `close`, normalizada via `StandardScaler` ajustado estritamente sobre o conjunto de treino.
- **Desempenho:** Apresentou um $$R^2$$ elevado (~0.9784) devido à forte autocorrelação da série, mas com **POCID de 47.54%** (próximo a escolhas aleatórias de direção).

### 2. Mudança de Scaler (`MinMaxScaler`)
- **Configuração:** Substituição do `StandardScaler` por `MinMaxScaler(feature_range=(-1, 1))`.
- **Motivo:** Alinhar os dados ao intervalo da função de ativação interna `tanh` da camada LSTM, beneficiando a convergência.
- **Desempenho:** MSE de 0.0010, $$R^2$$ de 0.9503 e **POCID de 49.18%**.

### 3. Abordagem Multivariada Inicial
- **Configuração:** Inclusão de variáveis exógenas (`open`, `high`, `low`, `close`, `number_of_trades`) para fornecer mais contexto de mercado à rede.

### 4. Implementação Keras LSTM com Dropout e EarlyStopping
- Estruturação robusta em Keras utilizando camadas densas, dropout para controle de regularização contra *overfitting* e *callbacks* de parada antecipada.

### 5. Otimização Avançada com Optuna (Modelo Final)
Busca automatizada de hiperparâmetros ótimos para a arquitetura multivariada:
* **`window_size`**: 24 dias
* **`units`**: 60 neurônios
* **`n_layers`**: 1 camada LSTM
* **`dropout_rate`**: ~0.2445
* **`learning_rate`**: ~0.0026
* **`batch_size`**: 16

---

## 📈 Resultados das Métricas (Set de Teste Normalizado - Modelo Otimizado)
- **MSE (Teste):** 0.001096
- **RMSE (Teste):** 0.033107
- **MAE (Teste):** 0.026437
- **$$R^2$$ (Teste):** 0.941953
- **POCID (Teste):** **48.05%**

---

## 💡 Principais Conclusões
1. **$$R^2$$ Alto vs. POCID Neutro:** O modelo obteve excelente desempenho em rastrear a escala do preço absoluto ($R^2 > 0.94$), mas manteve o POCID próximo a 48–50%. Isso evidencia a forte autocorrelação do ativo (o preço de amanhã tende a ser muito parecido com o de hoje), demonstrando que prever a **direção exata** da mudança diária permanece como o principal desafio em modelos puramente supervisionados de regressão temporal.
2. **Impacto da Abordagem Multivariada + Optuna:** A sintonia de parâmetros via Optuna evitou que o modelo gerasse previsões ativamente distorcidas ($\text{POCID} < 45\%$), mantendo-o estável e alinhado ao comportamento de persistência do mercado.

---

## 🛠️ Tecnologias Utilizadas
- **Python**
- **TensorFlow / Keras** (Redes Neurais LSTM)
- **Scikit-Learn** (`MinMaxScaler`, métricas estatísticas)
- **Optuna** (Otimização de Hiperparâmetros)
- **Pandas & NumPy** (Manipulação de Séries Temporais)
- **Matplotlib** (Visualização gráfica dos resultados de treino e teste)
