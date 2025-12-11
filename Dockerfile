# Estágio 1: Base com Python
FROM python:3.11-slim

# Definir o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copiar a pasta 'source' para o diretório de trabalho /app
# Isso fará com que o conteúdo de 'source' seja a raiz em /app
COPY source/ .

# Instalar as dependências do projeto
# O pip usará o requirements.txt que está agora em /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expor a porta em que o Uvicorn estará rodando
EXPOSE 8000

# Comando para iniciar a aplicação quando o contêiner for executado
# O uvicorn irá procurar pelo objeto 'app' no arquivo 'main.py'
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
