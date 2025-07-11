from typing import Any, Dict, Union

import mlflow
import mlflow.sklearn
from mlflow.exceptions import RestException
import pandas as pd
from api.models.schemas import CensusFeaturesInput
from fastapi import HTTPException
from api.services.history_service import log_prediction


# --- URI fija al modelo en producción ---
MLFLOW_MODEL_URI = "models:/Census_Income_Prediction@Champion"

mlflow.set_tracking_uri("http://mlflow:5000")

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


def predict_income(
    input_data: CensusFeaturesInput
) -> float:
    """Obtiene la predicción de ganancia a partir de la entrada."""
    try:
        model = mlflow.sklearn.load_model(MLFLOW_MODEL_URI)
    except RestException:
        raise HTTPException(status_code=503, detail="Modelo no disponible en MLflow todavía.")
    
    X = payload_to_dataframe(input_data.model_dump())
    pred_class = int(model.predict(X)[0])
    proba_1 = float(model.predict_proba(X)[0][1])  # Probabilidad de clase >50K
    proba_rounded = round(proba_1, 4)
    
    try:
        log_prediction(input_data, pred_class, proba_rounded)
    except Exception as e:
        print(f"Error loggeando log: {e}")

    return {"prediction": pred_class, "proba": proba_rounded}


