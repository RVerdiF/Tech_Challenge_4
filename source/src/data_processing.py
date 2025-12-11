import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

# Define o ticker da ação e o período
TICKER = "PETR4.SA"
START_DATE = "2015-01-01"
END_DATE = "2023-12-31"
TIME_STEPS = 60
TRAIN_SIZE = 0.8

# Caminhos
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
RAW_DATA_PATH = os.path.join(DATA_DIR, f'{TICKER}_raw.csv')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, f'{TICKER}_processed.npz')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.joblib')

def download_data():
    """Baixa os dados históricos da ação usando o yfinance."""
    print(f"Baixando dados para {TICKER} de {START_DATE} a {END_DATE}...")
    data = yf.download(TICKER, start=START_DATE, end=END_DATE)
    if data.empty:
        raise ValueError("Nenhum dado baixado. Verifique o ticker e o período.")
    data.to_csv(RAW_DATA_PATH)
    print(f"Dados brutos salvos em: {RAW_DATA_PATH}")
    return data

def process_data(data):
    """Pré-processa os dados para o modelo LSTM."""
    print("Iniciando pré-processamento dos dados...")
    
    # Usar apenas o preço de fechamento
    data_close = data['Close'].values.reshape(-1, 1)
    
    # Normalizar os dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_close)
    
    # Criar sequências
    X, y = [], []
    for i in range(TIME_STEPS, len(scaled_data)):
        X.append(scaled_data[i-TIME_STEPS:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    # Reshape para o formato do LSTM [samples, time_steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Dividir em treino e teste
    training_data_len = int(len(X) * TRAIN_SIZE)
    
    X_train, X_test = X[:training_data_len], X[training_data_len:]
    y_train, y_test = y[:training_data_len], y[training_data_len:]
    
    # Salvar dados processados e o scaler
    np.savez(PROCESSED_DATA_PATH, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    joblib.dump(scaler, SCALER_PATH)
    
    print(f"Dados processados e salvos em: {PROCESSED_DATA_PATH}")
    print(f"Scaler salvo em: {SCALER_PATH}")
    print("Formato dos dados de treino (X_train):", X_train.shape)
    print("Formato dos dados de teste (X_test):", X_test.shape)

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    try:
        stock_data = download_data()
        process_data(stock_data)
        print("Processo de coleta e pré-processamento de dados concluído com sucesso!")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
