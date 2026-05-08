from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PatientData(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Patient age in years")
    sex: int = Field(..., ge=0, le=1, description="0 = female, 1 = male")
    chest: int = Field(..., ge=1, le=4, description="Chest pain type")
    resting_blood_pressure: int = Field(..., ge=50, le=250, description="Resting blood pressure")
    serum_cholestoral: int = Field(..., ge=100, le=700, description="Serum cholesterol")
    fasting_blood_sugar: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    resting_electrocardiographic_results: int = Field(..., ge=0, le=2, description="Resting ECG results")
    maximum_heart_rate_achieved: int = Field(..., ge=60, le=230, description="Maximum heart rate")
    exercise_induced_angina: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=1, le=3, description="Slope of the peak exercise ST segment")
    number_of_major_vessels: int = Field(..., ge=0, le=3, description="Number of major vessels")
    thal: int = Field(..., ge=3, le=7, description="Thalassemia category")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63,
                "sex": 1,
                "chest": 3,
                "resting_blood_pressure": 145,
                "serum_cholestoral": 233,
                "fasting_blood_sugar": 1,
                "resting_electrocardiographic_results": 0,
                "maximum_heart_rate_achieved": 150,
                "exercise_induced_angina": 0,
                "oldpeak": 2.3,
                "slope": 1,
                "number_of_major_vessels": 0,
                "thal": 6,
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability: Optional[float]
    risk_level: str
    model_version: str
    inference_time_ms: float
    probabilities: Optional[Dict[str, float]] = None


class BatchPredictionRequest(BaseModel):
    patients: List[PatientData] = Field(..., min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int
    total_inference_time_ms: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str]
    timestamp: str


class VersionResponse(BaseModel):
    app_name: str
    app_version: str
    model_version: str
    model_name: str
    environment: str
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    algorithm: str
    accuracy: Optional[float]
    created_at: Optional[str]
    input_features: List[str]
    model_path: str
