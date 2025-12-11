import requests
import json
import random
import time

# Configurações
API_URL = "http://127.0.0.1:8000"
PREDICT_ENDPOINT = f"{API_URL}/predict"

def generate_dummy_data(days=60):
    """Gera uma lista de preços fictícios para teste."""
    # Gera uma sequência aleatória em torno de 20.0
    return [20.0 + random.uniform(-2, 2) for _ in range(days)]

def test_root():
    """Testa o endpoint raiz para verificar se a API está no ar."""
    try:
        response = requests.get(API_URL)
        if response.status_code == 200:
            print(f"[SUCCESS] API Root Check: {response.json()}")
            return True
        else:
            print(f"[ERROR] API Root Check Failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Não foi possível conectar a {API_URL}. A API está rodando?")
        return False

def test_prediction():
    """Testa o endpoint de predição com dados dummy."""
    history = generate_dummy_data(60)
    payload = {"history": history}
    
    print(f"\n[INFO] Enviando payload com {len(history)} dias de histórico...")
    
    try:
        start_time = time.time()
        response = requests.post(PREDICT_ENDPOINT, json=payload)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"[SUCCESS] Predição recebida em {duration:.4f}s")
            print(f"Previsão de preço: R$ {result['prediction']:.2f}")
        else:
            print(f"[ERROR] Falha na predição: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"[ERROR] Erro na requisição: {e}")

if __name__ == "__main__":
    print("=== Iniciando Teste da API ===")
    if test_root():
        test_prediction()
    else:
        print("\nAbortando testes pois a API parece estar offline.")
        print("Dica: Execute 'python source/main.py' em outro terminal.")
