# Imagem base Python
FROM python:3.11-slim

# Define o diretório de trabalho no container
WORKDIR /app

# Copia os arquivos de requisitos
COPY requirements.txt .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copia o resto do código da aplicação
COPY . .

# Expõe a porta que o FastAPI vai usar
EXPOSE 8000

# Comando para rodar a aplicação
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 