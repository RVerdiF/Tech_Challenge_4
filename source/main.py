from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import tensorflow as tf
import joblib
import os

# --- Configuração de Caminhos e Constantes ---
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'lstm_model.h5')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.joblib')
TIME_STEPS = 60  # Deve ser o mesmo valor usado no treinamento

# --- Carregamento de Modelo e Scaler ---
# Validação de existência dos arquivos
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise RuntimeError(
        "Modelo ou scaler não encontrado. "
        "Certifique-se de que os arquivos 'lstm_model.h5' e 'scaler.joblib' "
        "existam no diretório 'source/models'."
    )

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Modelo e scaler carregados com sucesso.")
except Exception as e:
    raise RuntimeError(f"Erro ao carregar o modelo ou scaler: {e}")


# --- Definição da Aplicação FastAPI ---
app = FastAPI(
    title="API de Previsão de Preços de Ações",
    description="Uma API para prever o próximo preço de fechamento de uma ação usando um modelo LSTM.",
    version="1.0.0"
)

# --- Modelos de Dados (Pydantic) ---
class StockInput(BaseModel):
    """Modelo de entrada para a API, esperando uma lista de preços de fechamento."""
    history: List[float]

    class Config:
        schema_extra = {
            "example": {
                "history": [20.5, 21.2, 20.8, 22.1, 21.9] * 12 # Exemplo com 60 valores
            }
        }

class PredictionOutput(BaseModel):
    """Modelo de saída para a API, retornando a previsão."""
    prediction: float


# --- Endpoints da API ---
@app.get("/", tags=["Root"])
def read_root():
    """Endpoint raiz para verificar o status da API."""
    return {"message": "API de Previsão de Ações está no ar!"}

@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(data: StockInput):
    """
    Recebe um histórico de preços de fechamento e retorna a previsão para o próximo dia.
    
    - **history**: Uma lista de floats contendo os preços de fechamento dos últimos N dias.
                   O número de dias deve ser igual ou maior que a janela de tempo do modelo (60 dias).
    """
    # Validação da entrada
    if len(data.history) < TIME_STEPS:
        raise HTTPException(
            status_code=400,
            detail=f"A entrada de dados históricos deve conter pelo menos {TIME_STEPS} valores."
        )

    try:
        # Pré-processamento dos dados de entrada
        last_60_days = np.array(data.history[-TIME_STEPS:]).reshape(-1, 1)
        scaled_data = scaler.transform(last_60_days)
        
        # Reshape para o formato do modelo: [1, time_steps, features]
        X_pred = np.reshape(scaled_data, (1, TIME_STEPS, 1))

        # Realizar a predição
        predicted_price_scaled = model.predict(X_pred)
        
        # Desnormalizar o resultado
        predicted_price = scaler.inverse_transform(predicted_price_scaled)

        return {"prediction": predicted_price[0][0]}

    except Exception as e:
        # Log do erro no servidor para depuração
        print(f"Erro durante a predição: {e}")
        raise HTTPException(
            status_code=500,
            detail="Ocorreu um erro interno ao processar a predição."
        )

# --- Execução da Aplicação (para desenvolvimento local) ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
