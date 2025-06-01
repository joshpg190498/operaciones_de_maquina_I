import os
import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Verificar que las variables necesarias estén en el entorno
for var in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "MLFLOW_S3_ENDPOINT_URL", "MLFLOW_TRACKING_URI"]:
    if not os.getenv(var):
        raise EnvironmentError(f"Falta la variable de entorno requerida: {var}")

# Configurar MLflow con tracking y endpoint MinIO
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# Nombre del modelo y alias
MODEL_URI = "models:/Census_Income_Prediction@Champion"

# Cargar modelo desde MLflow
try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
except Exception as e:
    raise RuntimeError(f"Error al cargar el modelo desde MLflow: {e}")

# Inicializar FastAPI
app = FastAPI(title="API de Predicción de Ingreso Census")

# Esquema de entrada para la predicción
class CensusFeatures(BaseModel):
    age: int
    workclass: str
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    gender: str
    hours_per_week: int
    native_country: str

# Endpoint raíz
@app.get("/")
def root():
    return {"message": "API de predicción de ingresos funcionando - Modelo Champion activo"}

# Endpoint de predicción
@app.post("/predict")
def predict(data: CensusFeatures):
    try:
        input_df = pd.DataFrame([data.dict()])
        prediction = model.predict(input_df)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {e}")