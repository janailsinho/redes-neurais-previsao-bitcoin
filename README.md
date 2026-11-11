# 📈 Previsão de Preço de Bitcoin com LSTM e Otimização de Hiperparâmetros (Optuna)

Projeto desenvolvido para a disciplina de Redes Neurais (**IF702 - CIn/UFPE**) pelos integrantes **Caio Ferreira** e **José Janailson**. O objetivo principal é desenvolver um modelo LSTM capaz de prever o preço de fechamento (`close`) do Bitcoin utilizando abordagens univariadas e multivariadas, otimização de hiperparâmetros via Optuna e validação temporal rigorosa.

---

## 📋 Sumário
- [Visão Geral do Projeto](#-visão-geral-do-projeto)
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
A predição de preços de criptomoedas apresenta grande complexidade devido à alta volatilidade e forte autocorrelação temporal. Este repositório documenta a evolução experimental de redes neurais recorrentes (LSTM) aplicadas ao histórico diário du Bitcoin entre os anos de 2017 e 2023.

---

## 📊 Estrutura e Divisão dos Dados
- **Dataset:** `data-bitcoin_timedata_v2.csv` (contendo colunas como `date`, `open`, `high`, `low`, `close` e `number_of_trades`).
- **Divisão dos Dados:** 
  - **Treino:** 80% do histórico.
  - **Teste:** 20% do histórico.
- **Janela Temporal (`window_size`):** O modelo consome uma sequência de dias $D$ para prever o fechamento do dia seguinte $D+1$.

---

## 📐 Métricas de Avaliação
- **MSE (Mean Squared Error):** Média dos quadrados dos erros, penalizando erros maiores.
- **RMSE (Root Mean Squared Error):** Raiz quadrada du MSE, expressando o erro médio na mesma unidade da variável.
- **MAE (Mean Absolute Error):** Média dos erros absolutos (magnitude do erro sem considerar direções).
- **$R^2$ (Coeficiente de Determinação):** Mede o quanto da variabilidade e comportamento dos dados reais o modelo conseguiu capturar.
- **POCID (Percentage of Correct Direction):** Percentual de acertos na direção da variação (subida/queda) de um dia para o outro.

---

## 🔬 Evolução dos Experimentos e Modelos

### 1. Modelo Base (Univariado + StandardScaler)
- **Configuração:** Uso exclusivo da coluna `close`, normalizada com `StandardScaler` (centralizada em zero, ajuste feito estritamente sobre os dados de treino).
- **Desempenho:** Apresentou $R^2$ elevado (~0.9784) devido à autocorrelação, mas com tempo de treinamento estendido e **POCID de 47.54%**.

### 2. Mudança de Scaler (`MinMaxScaler`)
- **Configuração:** Substituição do `StandardScaler` por `MinMaxScaler(feature_range=(-1, 1))`.
- **Motivo:** Alinhar a escala dos dados com o intervalo da função de ativação interna `tanh` da LSTM, facilitando a convergência e agilizando o treinamento.
- **Desempenho:** MSE de 0.0010, $R^2$ de 0.9503 e **POCID de 49.18%**.

### 3. Abordagem Multivariada Inicial
- **Configuração:** Inclusão de mais features além do fechamento: `open`, `high`, `low`, `close` e `number_of_trades`.
- **Desempenho:** Permitiu que a rede analise mais contexto sobre o comportamento do mercado.

### 4. Implementação Keras LSTM com Dropout e EarlyStopping
- Criação de uma arquitetura Keras otimizada, integrando `EarlyStopping` para evitar overfitting, camadas de `Dropout`, e ajustes no tamanho da janela temporal e taxa de aprendizado.

### 5. Otimização Avançada com Optuna (Modelo Final)
O **Optuna** foi utilizado para varrer o espaço de hiperparâmetros e encontrar a configuração ideal para o modelo multivariado:
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
- **$R^2$ (Teste):** 0.941953
- **POCID (Teste):** **48.05%**

---

## 💡 Principais Conclusões
1. **$R^2$ Alto vs. POCID Neutro:** O modelo final tornou-se excelente em rastrear o nível do preço ($R^2 > 0.94$), mas manteve um POCID em torno de ~48-50%. Isso ocorre porque o preço de amanhã possui forte correlação com o preço de hoje (persistência temporal).
2. **Impacto da Abordagem Multivariada + Optuna:** A estratégia multivariada aliada à busca de hiperparâmetros evitou que o modelo caísse em um comportamento ativamente errado ($\text{POCID} < 45\%$), mantendo-o estável, embora ainda precise de novas features preditivas para antecipar reversões bruscas de tendência com precisão.

---

## 🛠️ Tecnologias Utilizadas
- **Python**
- **TensorFlow / Keras** (Redes Neurais LSTM)
- **Scikit-Learn** (`MinMaxScaler`, métricas de erro)
- **Optuna** (Otimização de Hiperparâmetros)
- **Pandas & NumPy** (Manipulação de Séries Temporais)
- **Matplotlib** (Visualização de Dados e Gráficos de Previsão)
