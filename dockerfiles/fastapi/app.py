import os
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --------------------------------------------------------------------
# 1. Configuración MLflow → leer de variables de entorno ó defaults
# --------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_S3_ENDPOINT  = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://s3:9000")

os.environ["MLFLOW_TRACKING_URI"]      = MLFLOW_TRACKING_URI
os.environ["MLFLOW_S3_ENDPOINT_URL"]   = MLFLOW_S3_ENDPOINT
os.environ["AWS_ACCESS_KEY_ID"]        = os.getenv("AWS_ACCESS_KEY_ID", "minio")
os.environ["AWS_SECRET_ACCESS_KEY"]    = os.getenv("AWS_SECRET_ACCESS_KEY", "minio123")

MODEL_URI = "models:/Census_Income_Prediction@Champion"

try:
    pipeline = mlflow.sklearn.load_model(MODEL_URI)
except Exception as e:
    raise RuntimeError(f"No se pudo cargar el modelo desde MLflow → {e}")


# ----------------------------------------------------------------------
# 2. Definir esquema Pydantic para la entrada
# ----------------------------------------------------------------------
class CensusFeatures(BaseModel):
    age: int = 32
    workclass: str = "Private"
    educationnum: int = 13
    marital_status: str = "Married-civ-spouse"
    occupation: str = "Tech-support"
    relationship: str = "Husband"
    race: str = "White"
    gender: str = "Male"
    hours_per_week: int = 40
    native_country: str = "United-States"

# ----------------------------------------------------------------------
# 3. Mapeo snake_case  →  nombres originales con espacio
# ----------------------------------------------------------------------
RAW_TO_MODEL = {
    "age":              "age",
    "workclass":        "workclass",
    "educationnum":     "educationnum",
    "marital_status":   "marital status",
    "occupation":       "occupation",
    "relationship":     "relationship",
    "race":             "race",
    "gender":           "gender",
    "hours_per_week":   "hours per week",
    "native_country":   "native country",
}

def payload_to_dataframe(payload: dict) -> pd.DataFrame:
    """Convierte el dict validado por Pydantic a DataFrame 1×N."""
    renamed = {RAW_TO_MODEL[k]: v for k, v in payload.items()}
    return pd.DataFrame([renamed])

# ----------------------------------------------------------------------
# 4. Instanciar FastAPI
# ----------------------------------------------------------------------
app = FastAPI(
    title="API de Predicción de Ingresos (Census)",
    description="Predicción de ingresos (>50K) usando modelo Champion de MLflow",
    version="1.1.0"
)

# ----------------------------------------------------------------------
# 5. Endpoints
# ----------------------------------------------------------------------

# Endpoint raíz   
@app.get("/")
def root():
    return {"message": "API de predicción de ingresos funcionando – modelo local activo"}

@app.post("/predict")
def predict(data: CensusFeatures):
    try:
        X = payload_to_dataframe(data.model_dump())
        pred_class = int(pipeline.predict(X)[0])
        proba_1    = float(pipeline.predict_proba(X)[0, 1])  # prob >50K
        return {"prediction": pred_class, "proba": round(proba_1, 4)}
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Error durante la predicción: {e}")