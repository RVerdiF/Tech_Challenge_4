import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import joblib

# Caminhos
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'PETR4.SA_processed.npz')
MODEL_PATH = os.path.join(MODELS_DIR, 'lstm_model.h5')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.joblib')

def mean_absolute_percentage_error(y_true, y_pred):
    """Calcula o MAPE."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def build_model(input_shape):
    """Constrói o modelo LSTM."""
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    return model

def train_model(model, X_train, y_train, epochs=25, batch_size=32):
    """Treina o modelo LSTM."""
    print("Iniciando o treinamento do modelo...")
    history = model.fit(
        X_train, 
        y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_split=0.1,
        verbose=1
    )
    print("Treinamento concluído.")
    return history

def evaluate_model(model, X_test, y_test, scaler):
    """Avalia o modelo com os dados de teste."""
    print("Iniciando a avaliação do modelo...")
    predictions = model.predict(X_test)
    
    # Desnormalizar os dados para avaliação
    predictions_rescaled = scaler.inverse_transform(predictions)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calcular métricas
    mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled))
    mape = mean_absolute_percentage_error(y_test_rescaled, predictions_rescaled)
    
    print("\nMétricas de Avaliação no Conjunto de Teste:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return mae, rmse, mape

if __name__ == "__main__":
    # Carregar dados
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"Arquivo de dados processados não encontrado em {PROCESSED_DATA_PATH}. Execute o script data_processing.py primeiro.")
    else:
        print("Carregando dados processados...")
        data = np.load(PROCESSED_DATA_PATH)
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        
        # Carregar scaler
        scaler = joblib.load(SCALER_PATH)

        # Construir o modelo
        lstm_model = build_model(input_shape=(X_train.shape[1], 1))
        
        # Treinar o modelo
        train_model(lstm_model, X_train, y_train)
        
        # Salvar o modelo
        lstm_model.save(MODEL_PATH)
        print(f"\nModelo salvo em: {MODEL_PATH}")
        
        # Avaliar o modelo
        evaluate_model(lstm_model, X_test, y_test, scaler)
