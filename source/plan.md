# Tech Challenge - Fase 4: Previsão de Preços de Ações com LSTM

Este documento descreve o plano para atender aos requisitos do Desafio Técnico da Fase 4.

## 1. Configuração do Projeto

*   Inicializar um repositório Git.
*   Configurar um ambiente virtual Python.
*   Criar um arquivo `requirements.txt` para gerenciar as dependências (`yfinance`, `pandas`, `scikit-learn`, `tensorflow`, `fastapi`, `uvicorn`).
*   Estruturar o projeto em pastas para código-fonte, dados, modelos, etc.

## 2. Coleta e Pré-processamento de Dados

*   Utilizar a biblioteca `yfinance` para baixar dados históricos de uma ação à escolha.
*   Limpar e pré-processar os dados:
    *   Tratar valores ausentes.
    *   Normalizar os dados (ex: `MinMaxScaler`).
*   Criar sequências de dados para alimentar o modelo LSTM.
*   Dividir os dados em conjuntos de treino e teste.

## 3. Desenvolvimento do Modelo LSTM

*   Construir um modelo LSTM usando TensorFlow/Keras.
*   Compilar o modelo com um otimizador (ex: Adam) e uma função de perda (ex: MSE).
*   Treinar o modelo com os dados de treino.
*   Ajustar hiperparâmetros (nº de camadas, neurônios, épocas, etc.) para otimizar o desempenho.
*   Avaliar o modelo com os dados de teste usando as métricas MAE, RMSE e MAPE.

## 4. Salvamento do Modelo

*   Salvar o modelo treinado em um arquivo (formato H5 ou SavedModel do Keras).

## 5. Desenvolvimento da API

*   Utilizar FastAPI para criar uma API RESTful.
*   Criar um endpoint que:
    *   Receba dados históricos de preços como entrada.
    *   Carregue o modelo salvo.
    *   Pré-processe os dados de entrada.
    *   Faça previsões usando o modelo.
    *   Retorne as previsões em formato JSON.

## 6. Dockerização

*   Criar um `Dockerfile` para containerizar a aplicação da API.
*   Incluir todas as dependências e configurações necessárias.
*   Construir e testar a imagem Docker localmente.

## 7. Monitoramento

*   Implementar logging básico na API para registrar requisições, respostas e possíveis erros.
*   Pesquisar e planejar a integração com ferramentas de monitoramento como Prometheus e Grafana para observar o tempo de resposta e o uso de recursos.

## 8. Documentação e Entregáveis

*   Criar um `README.md` detalhado com instruções de configuração e uso do projeto.
*   Gravar um vídeo demonstrando o funcionamento da API.
*   Subir todo o código, documentação e o `Dockerfile` para o repositório Git.
*   (Opcional) Fazer o deploy da aplicação em um serviço de nuvem e disponibilizar o link.
