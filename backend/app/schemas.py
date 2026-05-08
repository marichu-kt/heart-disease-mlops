from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PatientData(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Edad del paciente en años.", examples=[63])
    sex: int = Field(..., ge=0, le=1, description="Sexo codificado: 0 = mujer, 1 = hombre.", examples=[1])
    chest: int = Field(
        ...,
        ge=1,
        le=4,
        description="Tipo de dolor torácico: 1 angina típica, 2 angina atípica, 3 dolor no anginoso, 4 asintomático.",
        examples=[3],
    )
    resting_blood_pressure: int = Field(
        ...,
        ge=50,
        le=250,
        description="Presión arterial en reposo, medida en mmHg.",
        examples=[145],
    )
    serum_cholestoral: int = Field(
        ...,
        ge=100,
        le=700,
        description="Colesterol sérico, medido en mg/dl.",
        examples=[233],
    )
    fasting_blood_sugar: int = Field(
        ...,
        ge=0,
        le=1,
        description="Glucosa en ayunas superior a 120 mg/dl: 0 = no, 1 = sí.",
        examples=[1],
    )
    resting_electrocardiographic_results: int = Field(
        ...,
        ge=0,
        le=2,
        description="Resultados del ECG en reposo: 0 normal, 1 anomalía ST-T, 2 hipertrofia ventricular probable.",
        examples=[0],
    )
    maximum_heart_rate_achieved: int = Field(
        ...,
        ge=60,
        le=230,
        description="Frecuencia cardíaca máxima alcanzada, en latidos por minuto.",
        examples=[150],
    )
    exercise_induced_angina: int = Field(
        ...,
        ge=0,
        le=1,
        description="Angina inducida por ejercicio: 0 = no, 1 = sí.",
        examples=[0],
    )
    oldpeak: float = Field(
        ...,
        ge=0,
        le=10,
        description="Depresión ST inducida por ejercicio respecto al reposo.",
        examples=[2.3],
    )
    slope: int = Field(
        ...,
        ge=1,
        le=3,
        description="Pendiente del segmento ST: 1 ascendente, 2 plana, 3 descendente.",
        examples=[1],
    )
    number_of_major_vessels: int = Field(
        ...,
        ge=0,
        le=3,
        description="Número de vasos principales coloreados por fluoroscopia.",
        examples=[0],
    )
    thal: int = Field(
        ...,
        ge=3,
        le=7,
        description="Categoría thal: 3 normal, 6 defecto fijo, 7 defecto reversible.",
        examples=[6],
    )

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
    prediction: int = Field(..., description="Clase predicha por el modelo: 0 = No Disease, 1 = Disease.")
    label: str = Field(..., description="Etiqueta legible asociada a la predicción.")
    probability: Optional[float] = Field(None, description="Probabilidad estimada de la clase Disease.")
    risk_level: str = Field(..., description="Nivel de riesgo orientativo: Low, Medium o High.")
    model_version: str = Field(..., description="Versión del modelo usada para la inferencia.")
    inference_time_ms: float = Field(..., description="Tiempo de inferencia aproximado en milisegundos.")
    probabilities: Optional[Dict[str, float]] = Field(
        None,
        description="Distribución de probabilidad por clase cuando el modelo la ofrece.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction": 1,
                "label": "Disease",
                "probability": 0.82,
                "risk_level": "High",
                "model_version": "v3.0.0",
                "inference_time_ms": 4.7,
                "probabilities": {
                    "no_disease": 0.18,
                    "disease": 0.82,
                },
            }
        }
    }


class BatchPredictionRequest(BaseModel):
    patients: List[PatientData] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Lista de pacientes a evaluar. Se aceptan entre 1 y 100 registros por petición.",
    )


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse] = Field(..., description="Predicciones generadas para cada paciente.")
    count: int = Field(..., description="Número de registros procesados.")
    total_inference_time_ms: float = Field(..., description="Tiempo total aproximado de inferencia en milisegundos.")
    model_version: str = Field(..., description="Versión del modelo usada en todo el batch.")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Estado de salud de la API.")
    model_loaded: bool = Field(..., description="Indica si el modelo está cargado en memoria.")
    model_version: Optional[str] = Field(None, description="Versión del modelo cargado, si existe.")
    timestamp: str = Field(..., description="Fecha y hora UTC de la comprobación.")


class VersionResponse(BaseModel):
    app_name: str = Field(..., description="Nombre público de la API.")
    app_version: str = Field(..., description="Versión de la aplicación.")
    model_version: str = Field(..., description="Versión del modelo activo.")
    model_name: str = Field(..., description="Nombre lógico del modelo activo.")
    environment: str = Field(..., description="Entorno de ejecución declarado.")
    timestamp: str = Field(..., description="Fecha y hora UTC de la respuesta.")


class ModelInfoResponse(BaseModel):
    model_name: str = Field(..., description="Nombre lógico del modelo cargado.")
    version: str = Field(..., description="Versión del modelo cargado.")
    algorithm: str = Field(..., description="Pipeline o algoritmo usado por el modelo.")
    accuracy: Optional[float] = Field(None, description="Exactitud registrada durante la evaluación del modelo.")
    precision: Optional[float] = Field(None, description="Precisión registrada durante la evaluación del modelo.")
    recall: Optional[float] = Field(None, description="Recall registrado durante la evaluación del modelo.")
    f1_score: Optional[float] = Field(None, description="F1-score registrado durante la evaluación del modelo.")
    roc_auc: Optional[float] = Field(None, description="ROC-AUC registrado cuando el modelo permite estimarlo.")
    selected_metric: Optional[str] = Field(None, description="Métrica principal usada para seleccionar el modelo activo.")
    created_at: Optional[str] = Field(None, description="Fecha de creación del artefacto, si está disponible.")
    input_features: List[str] = Field(..., description="Variables de entrada esperadas por el modelo.")
    model_path: str = Field(..., description="Ruta del artefacto joblib cargado.")
    dataset_source: Optional[str] = Field(None, description="Fuente del dataset usado durante el entrenamiento.")
    evaluation_report_path: Optional[str] = Field(None, description="Ruta del reporte de evaluación asociado.")
