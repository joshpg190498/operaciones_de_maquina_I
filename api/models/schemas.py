from datetime import datetime
from typing import Any, Dict

from pydantic import BaseModel, Field

class CensusFeaturesInput(BaseModel):
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

class CensusFeaturesOutput(BaseModel):
    """Esquema de salida de la predicción."""
    prediction: int = Field(..., description="Clase predicha: 0 (<=50K), 1 (>50K)")
    proba: float = Field(..., description="Probabilidad de que income > 50K")

class HistoryEntry(BaseModel):
    """Esquema para una entrada del historial de predicciones."""
    id: int
    input_data: Dict[str, Any]
    prediction: float
    timestamp: datetime

    class Config:
        orm_mode = True
        #from_attributes = True
