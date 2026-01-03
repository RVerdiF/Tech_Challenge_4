# 📈 Tech Challenge - Fase 4: Previsão de Preços de Ações com LSTM

Uma solução completa de Machine Learning para prever o preço de fechamento de ações, utilizando redes neurais LSTM e servindo previsões através de uma API RESTful.

---

## 📋 Índice

1. [Visão Geral](#-visão-geral)
2. [Arquitetura da Solução](#-arquitetura-da-solução)
3. [Métricas do Modelo](#-métricas-do-modelo)
4. [Estrutura do Projeto](#-estrutura-do-projeto)
5. [Como Executar](#-como-executar)
6. [Endpoints da API](#-endpoints-da-api)
7. [Exemplos de Uso](#-exemplos-de-uso)
8. [Docker](#-docker)
9. [Monitoramento](#-monitoramento)

---

## 🎯 Visão Geral

Este projeto implementa um modelo de **Deep Learning (LSTM)** para prever o valor de fechamento de ações da bolsa de valores brasileira. A ação utilizada para treinamento é a **PETR4.SA (Petrobras PN)**.

### Fluxo do Sistema

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Coleta de      │     │  Pré-processamento│     │  Treinamento   │
│  Dados          │ ──► │  e Normalização   │ ──► │  do Modelo     │
│  (yfinance)     │     │  (MinMaxScaler)   │     │  LSTM          │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Cliente        │     │  API FastAPI     │     │  Modelo Salvo  │
│  (Requisição)   │ ◄─► │  + Prometheus    │ ◄── │  (.h5 + .joblib)│
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### Principais Funcionalidades

- ✅ **Coleta automática** de dados históricos via `yfinance`
- ✅ **Pré-processamento** com normalização Min-Max
- ✅ **Modelo LSTM** com arquitetura de 2 camadas (50 unidades cada)
- ✅ **API RESTful** com FastAPI para servir previsões
- ✅ **Monitoramento** com Prometheus e métricas expostas
- ✅ **Containerização** com Docker para deploy fácil

---

## 🏗️ Arquitetura da Solução

### Modelo LSTM

O modelo utiliza uma arquitetura de rede neural recorrente do tipo **Long Short-Term Memory (LSTM)**, ideal para séries temporais:

```
┌───────────────────────────────────────────────────────────────┐
│                      Arquitetura do Modelo                    │
├───────────────────────────────────────────────────────────────┤
│  Input Layer   │  Shape: (60, 1) - 60 dias de histórico       │
├────────────────┼──────────────────────────────────────────────┤
│  LSTM Layer 1  │  50 unidades, return_sequences=True          │
│  Dropout       │  20% dropout para regularização              │
├────────────────┼──────────────────────────────────────────────┤
│  LSTM Layer 2  │  50 unidades, return_sequences=False         │
│  Dropout       │  20% dropout para regularização              │
├────────────────┼──────────────────────────────────────────────┤
│  Dense Layer   │  25 unidades (camada intermediária)          │
├────────────────┼──────────────────────────────────────────────┤
│  Output Layer  │  1 unidade (previsão do preço)               │
└────────────────┴──────────────────────────────────────────────┘
```

### Parâmetros de Treinamento

| Parâmetro       | Valor                |
|-----------------|----------------------|
| **Ação**        | PETR4.SA (Petrobras) |
| **Período**     | 2015-01-01 a 2023-12-31 |
| **Time Steps**  | 60 dias              |
| **Train/Test**  | 80% / 20%            |
| **Epochs**      | 25                   |
| **Batch Size**  | 32                   |
| **Optimizer**   | Adam                 |
| **Loss**        | Mean Squared Error   |

---

## 📊 Métricas do Modelo

O modelo foi avaliado no conjunto de teste (20% dos dados) e obteve os seguintes resultados:

| Métrica | Valor | Descrição |
|---------|-------|-----------|
| **MAE** (Mean Absolute Error) | **1.1661** | Erro médio absoluto em R$ |
| **RMSE** (Root Mean Squared Error) | **1.4382** | Raiz do erro quadrático médio em R$ |
| **MAPE** (Mean Absolute Percentage Error) | **6.02%** | Erro percentual médio |

### Interpretação das Métricas

- **MAE = R$ 1.17**: Em média, a previsão erra por aproximadamente R$ 1,17 do valor real.
- **RMSE = R$ 1.44**: Penaliza erros maiores; valores próximos ao MAE indicam erros consistentes.
- **MAPE = 6.02%**: O modelo erra, em média, cerca de 6% do valor real da ação.

> 💡 **Nota:** Considerando a volatilidade do mercado de ações brasileiro, um MAPE de ~6% representa um desempenho razoável para um modelo base utilizando apenas preços históricos.

---

## 📂 Estrutura do Projeto

```
Tech_Challenge_4/
│
├── main.py                    # API FastAPI principal
├── requirements.txt           # Dependências Python
├── Dockerfile                 # Container Docker
├── test_api.py               # Script de teste da API
├── README.md                 # Documentação (este arquivo)
│
└── source/
    ├── data/
    │   ├── PETR4.SA_raw.csv             # Dados brutos baixados
    │   └── PETR4.SA_processed.npz       # Dados pré-processados
    │
    ├── models/
    │   ├── lstm_model.h5                # Modelo LSTM treinado
    │   ├── scaler.joblib                # MinMaxScaler para normalização
    │   └── metrics.json                 # Métricas de avaliação
    │
    └── src/
        ├── data_processing.py           # Script de coleta e processamento
        └── train.py                     # Script de treinamento do modelo
```

---

## 🚀 Como Executar

### Pré-requisitos

- Python 3.9+
- `pip` para instalação de dependências

### 1. Configuração do Ambiente

```bash
# Clone o repositório
git clone <url-do-repositorio>
cd Tech_Challenge_4

# Crie e ative o ambiente virtual
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

# Instale as dependências
pip install -r requirements.txt
```

### 2. Coleta de Dados e Treinamento

Execute os scripts em ordem para preparar os dados e treinar o modelo:

```bash
# Passo 1: Baixar e processar os dados históricos
python source/src/data_processing.py

# Passo 2: Treinar o modelo LSTM
python source/src/train.py
```

**Saída esperada do treinamento:**
```
Carregando dados processados...
Construindo modelo LSTM...
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 lstm (LSTM)                 (None, 60, 50)            10400
 dropout (Dropout)           (None, 60, 50)            0
 lstm_1 (LSTM)               (None, 50)                20200
 dropout_1 (Dropout)         (None, 50)                0
 dense (Dense)               (None, 25)                1275
 dense_1 (Dense)             (None, 1)                 26
=================================================================
Total params: 31,901
Trainable params: 31,901
Non-trainable params: 0
_________________________________________________________________
Iniciando o treinamento do modelo...
Epoch 1/25
...
Modelo salvo em: source/models/lstm_model.h5
Métricas de Avaliação no Conjunto de Teste:
  MAE:  1.1661
  RMSE: 1.4382
  MAPE: 6.02%
```

### 3. Executando a API

```bash
python main.py
```

A API estará disponível em:
- **Aplicação:** http://127.0.0.1:8000
- **Documentação Swagger:** http://127.0.0.1:8000/docs
- **Métricas Prometheus:** http://127.0.0.1:8000/metrics

---

## 🔌 Endpoints da API

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `GET`  | `/` | Verifica o status da API |
| `GET`  | `/model/info` | Retorna informações e métricas do modelo |
| `POST` | `/predict` | Realiza uma previsão de preço |
| `GET`  | `/metrics` | Métricas Prometheus para monitoramento |

### Detalhamento dos Endpoints

#### `GET /` - Health Check
Verifica se a API está funcionando corretamente.

**Resposta:**
```json
{
  "message": "API de Previsão de Ações está no ar!"
}
```

#### `GET /model/info` - Informações do Modelo
Retorna detalhes sobre o modelo treinado, incluindo métricas.

**Resposta:**
```json
{
  "ticker": "PETR4.SA",
  "company": "Petrobras PN",
  "training_period": {
    "start": "2015-01-01",
    "end": "2023-12-31"
  },
  "model_architecture": "LSTM (2 layers, 50 units each)",
  "time_steps": 60,
  "metrics": {
    "mae": 1.1661,
    "rmse": 1.4382,
    "mape": 6.02
  }
}
```

#### `POST /predict` - Previsão de Preço
Recebe um histórico de preços e retorna a previsão para o próximo dia.

**Requisição:**
```json
{
  "history": [20.5, 21.2, 20.8, ...] // mínimo 60 valores
}
```

**Resposta:**
```json
{
  "prediction": 21.35
}
```

---

## 📝 Exemplos de Uso

### Usando Python (requests)

```python
import requests

# Histórico de 60 dias de preços de fechamento
history = [
    20.5, 21.2, 20.8, 22.1, 21.9, 20.5, 21.2, 20.8, 22.1, 21.9,
    20.5, 21.2, 20.8, 22.1, 21.9, 20.5, 21.2, 20.8, 22.1, 21.9,
    20.5, 21.2, 20.8, 22.1, 21.9, 20.5, 21.2, 20.8, 22.1, 21.9,
    20.5, 21.2, 20.8, 22.1, 21.9, 20.5, 21.2, 20.8, 22.1, 21.9,
    20.5, 21.2, 20.8, 22.1, 21.9, 20.5, 21.2, 20.8, 22.1, 21.9,
    20.5, 21.2, 20.8, 22.1, 21.9, 20.5, 21.2, 20.8, 22.1, 21.9
]

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"history": history}
)

print(f"Previsão: R$ {response.json()['prediction']:.2f}")
```

### Usando cURL

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "history": [
      20.5, 21.2, 20.8, 22.1, 21.9, 20.5, 21.2, 20.8, 22.1, 21.9,
      20.5, 21.2, 20.8, 22.1, 21.9, 20.5, 21.2, 20.8, 22.1, 21.9,
      20.5, 21.2, 20.8, 22.1, 21.9, 20.5, 21.2, 20.8, 22.1, 21.9,
      20.5, 21.2, 20.8, 22.1, 21.9, 20.5, 21.2, 20.8, 22.1, 21.9,
      20.5, 21.2, 20.8, 22.1, 21.9, 20.5, 21.2, 20.8, 22.1, 21.9,
      20.5, 21.2, 20.8, 22.1, 21.9, 20.5, 21.2, 20.8, 22.1, 21.9
    ]
  }'
```

### Script de Teste Automatizado

Use o script de teste incluído no projeto:

```bash
python test_api.py
```

**Saída esperada:**
```
=== Iniciando Teste da API ===
[SUCCESS] API Root Check: {'message': 'API de Previsão de Ações está no ar!'}

[INFO] Enviando payload com 60 dias de histórico...
[SUCCESS] Predição recebida em 0.1234s
Previsão de preço: R$ 21.35
```

---

## 🐳 Docker

### Construir e Executar

```bash
# Construir a imagem
docker build -t stock-predictor-api .

# Executar o contêiner
docker run -d -p 8000:8000 --name stock-api stock-predictor-api
```

### Verificar Logs

```bash
docker logs stock-api
```

A API estará disponível em `http://localhost:8000`.

---

## 📡 Monitoramento

A API expõe métricas para o **Prometheus** no endpoint `/metrics`.

### Métricas Disponíveis

| Métrica | Tipo | Descrição |
|---------|------|-----------|
| `model_mae` | Gauge | MAE do modelo treinado |
| `model_rmse` | Gauge | RMSE do modelo treinado |
| `model_mape` | Gauge | MAPE do modelo treinado |
| `http_requests_total` | Counter | Total de requisições HTTP |
| `http_request_duration_seconds` | Histogram | Latência das requisições |
| `process_cpu_seconds_total` | Counter | Uso de CPU do processo |
| `process_resident_memory_bytes` | Gauge | Uso de memória RAM |

### Integração com Grafana

Para visualização das métricas, configure o Prometheus para fazer scrape do endpoint `/metrics` e conecte ao Grafana para criar dashboards de:

- 📈 Latência de requisições
- 📊 Taxa de requisições por minuto (RPM)
- ⚠️ Taxa de erros (4xx/5xx)
- 💾 Uso de CPU e Memória

---

## 📄 Licença

Este projeto foi desenvolvido como parte do **Tech Challenge - Fase 4** do curso de Machine Learning Engineering da **FIAP**.

---

<p align="center">
  <strong>Desenvolvido com ❤️ para o Tech Challenge FIAP</strong>
</p>
