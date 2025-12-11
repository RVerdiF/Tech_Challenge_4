# Tech Challenge - Fase 4: Previsão de Preços de Ações com LSTM

Este projeto é uma implementação de um modelo de Deep Learning (LSTM) para prever o preço de fechamento de ações, com uma API para servir as previsões.

## 📝 Descrição

O objetivo deste desafio é construir uma solução completa de Machine Learning, desde a coleta de dados até o deploy de uma API. O modelo utiliza uma rede neural recorrente do tipo LSTM (Long Short-Term Memory) para prever o valor de fechamento de uma ação com base em seu histórico de preços.

A solução inclui:
- Coleta de dados históricos da ação PETR4.SA (Petrobras) via `yfinance`.
- Pré-processamento e normalização dos dados.
- Treinamento de um modelo LSTM com TensorFlow/Keras.
- Uma API RESTful construída com FastAPI para servir as previsões do modelo.
- Dockerfile para facilitar o deploy da aplicação.

## 📂 Estrutura do Projeto

```
.
├── Dockerfile
├── payload.json
├── source
│   ├── data
│   │   ├── PETR4.SA_processed.npz
│   │   └── PETR4.SA_raw.csv
│   ├── main.py
│   ├── models
│   │   ├── lstm_model.h5
│   │   └── scaler.joblib
│   ├── plan.md
│   ├── requirements.txt
│   └── src
│       ├── data_processing.py
│       └── train.py
└── test_api.py
```

## 🚀 Como Executar

### Pré-requisitos

- Python 3.9+
- `pip` e `venv`

### 1. Configuração do Ambiente

Clone o repositório e configure o ambiente virtual:

```bash
git clone <url-do-repositorio>
cd <nome-do-repositorio>
python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate
pip install -r source/requirements.txt
```

### 2. Coleta e Treinamento

Execute os scripts em ordem para baixar os dados, pré-processá-los e treinar o modelo.

```bash
# 1. Baixar e processar os dados
python source/src/data_processing.py

# 2. Treinar o modelo LSTM
python source/src/train.py
```
Após a execução, o modelo (`lstm_model.h5`) e o normalizador (`scaler.joblib`) estarão salvos na pasta `source/models`.

### 3. Executando a API Localmente

Com o modelo treinado, inicie a API FastAPI:

```bash
python source/main.py
```
A API estará disponível em `http://127.0.0.1:8000`. Você pode acessar a documentação interativa em `http://127.0.0.1:8000/docs`.

### 4. Testando a API

Para fazer uma previsão, envie uma requisição POST para o endpoint `/predict` com um histórico de 60 preços de fechamento.

Você pode usar o script `test_api.py` para isso:
```bash
python test_api.py
```

Ou usar uma ferramenta como o `curl`:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d 
  "history": [
    20.5, 21.2, 20.8, 22.1, 21.9, 20.5, 21.2, 20.8, 22.1, 21.9,
    20.5, 21.2, 20.8, 22.1, 21.9, 20.5, 21.2, 20.8, 22.1, 21.9,
    20.5, 21.2, 20.8, 22.1, 21.9, 20.5, 21.2, 20.8, 22.1, 21.9,
    20.5, 21.2, 20.8, 22.1, 21.9, 20.5, 21.2, 20.8, 22.1, 21.9,
    20.5, 21.2, 20.8, 22.1, 21.9, 20.5, 21.2, 20.8, 22.1, 21.9,
    20.5, 21.2, 20.8, 22.1, 21.9, 20.5, 21.2, 20.8, 22.1, 21.9
  ]
}"
```

## 🐳 Docker

Para construir e executar a aplicação com Docker:

```bash
# 1. Construir a imagem Docker
docker build -t stock-predictor-api .

# 2. Executar o contêiner
docker run -d -p 8000:8000 --name stock-api stock-predictor-api
```
A API estará acessível da mesma forma, em `http://localhost:8000`.

## 7. Monitoramento e Observabilidade

Para garantir a confiabilidade e o desempenho da API em produção, foi planejado um sistema de monitoramento baseado em métricas e logs.

### Logging

A aplicação implementa logs básicos via saída padrão (stdout), o que é ideal para ambientes containerizados.
- **O que é logado:** Erros críticos de carregamento de modelo (`RuntimeError`), exceções durante a predição e erros de validação.
- **Coleta:** Em produção, esses logs devem ser capturados pelo driver de log do Docker e encaminhados para um agregador como ELK Stack (Elasticsearch, Logstash, Kibana) ou AWS CloudWatch.

### Plano de Métricas (Prometheus + Grafana)

Para monitoramento de performance e saúde da aplicação, o plano de arquitetura recomenda:

1.  **Instrumentação da API:**
    Utilizar a biblioteca `prometheus-fastapi-instrumentator` para expor métricas automáticas.
    *   *Alteração necessária no `main.py`:*
        ```python
        from prometheus_fastapi_instrumentator import Instrumentator
        
        # Após criar a app
        Instrumentator().instrument(app).expose(app)
        ```

2.  **Coleta de Métricas (Prometheus):**
    Configurar um serviço Prometheus para realizar o *scrape* do endpoint `/metrics` da API a cada 15-30 segundos.

3.  **Visualização (Grafana):**
    Criar dashboards no Grafana conectados ao Prometheus para monitorar:
    *   **Latência:** Tempo de resposta do endpoint `/predict`.
    *   **Tráfego:** Número de requisições por minuto (RPM).
    *   **Erros:** Taxa de respostas 4xx/5xx.
    *   **Recursos:** Uso de CPU e Memória do container.
