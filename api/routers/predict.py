from http import HTTPStatus

from fastapi import APIRouter, HTTPException

from api.models.schemas import CensusFeaturesInput, CensusFeaturesOutput
from api.services.prediction_service import predict_income


router = APIRouter(prefix="/predict", tags=["Prediction"])

@router.post("/", response_model=CensusFeaturesOutput, status_code=HTTPStatus.OK, description="Recibe datos del individuo y devuelve la predicción")
def predict(input_data: CensusFeaturesInput) -> CensusFeaturesOutput:
    """Genera una predicción de si supera o no los $50.000 anualesy la registra en el historial."""
    try:
        pred = predict_income(input_data)
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Error al generar la predicción: {e}"
        )
    
    return CensusFeaturesOutput(**pred)