FROM python:3.11-slim

# ≡ 1. Configuración básica
#ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ≡ 2. Dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ≡ 3. Copiar el código (¡sin modelo local!)
COPY app.py .

# ≡ 4. Puerto coherente con docker-compose.yml
EXPOSE 8800

# ≡ 5. CMD
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8800"]