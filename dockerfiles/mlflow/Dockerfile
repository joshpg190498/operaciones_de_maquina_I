FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements y actualizar pip
COPY requirements.txt .
RUN apt-get update && apt-get install -y wget

# Instalar dependencias básicas de MLflow y conexión con Postgres/S3
RUN pip install --upgrade pip && \
    pip install mlflow[extras] psycopg2-binary boto3 cryptography pymysql && \
    pip install --no-cache-dir -r requirements.txt

# Exponer el puerto por defecto
EXPOSE 5000

# Comando por defecto para servir el tracking server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "/mlruns"]
