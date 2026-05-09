import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from .schemas import PatientData


logger = logging.getLogger(__name__)

INPUT_FEATURES = [
    "age",
    "sex",
    "chest",
    "resting_blood_pressure",
    "serum_cholestoral",
    "fasting_blood_sugar",
    "resting_electrocardiographic_results",
    "maximum_heart_rate_achieved",
    "exercise_induced_angina",
    "oldpeak",
    "slope",
    "number_of_major_vessels",
    "thal",
]


@dataclass
class PredictionResult:
    prediction: int
    label: str
    probability: Optional[float]
    risk_level: str
    model_version: str
    inference_time_ms: float
    probabilities: Optional[Dict[str, float]]
    decision_threshold: Optional[float]


class ModelService:
    def __init__(self) -> None:
        self.model: Any = None
        self.metadata: Dict[str, Any] = {}
        self.model_path: Optional[Path] = None
        self.loaded = False

    @property
    def input_features(self) -> List[str]:
        return self.metadata.get("input_features") or INPUT_FEATURES

    @property
    def version(self) -> str:
        return str(self.metadata.get("version", "unknown"))

    @property
    def decision_threshold(self) -> Optional[float]:
        raw_threshold = self.metadata.get("decision_threshold")
        if raw_threshold is None:
            return None
        return float(raw_threshold)

    def load_latest_model(self) -> None:
        model_dir = self._model_dir()
        metadata_path = model_dir / "model_metadata.json"
        logger.info("Loading model metadata from %s", metadata_path)

        if metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            model_path = self._resolve_model_path(model_dir, self.metadata.get("model_path"))
        else:
            model_path = self._find_latest_model(model_dir)
            self.metadata = {
                "model_name": model_path.stem,
                "version": self._version_from_filename(model_path.name),
                "algorithm": "unknown",
                "accuracy": None,
                "created_at": None,
                "input_features": INPUT_FEATURES,
                "model_path": str(model_path),
            }

        self.model_path = model_path
        self.model = joblib.load(model_path)
        self.loaded = True
        logger.info(
            "Model loaded successfully: model_name=%s model_version=%s model_path=%s",
            self.metadata.get("model_name"),
            self.version,
            self.model_path,
        )

    def model_info(self) -> Dict[str, Any]:
        if not self.loaded:
            raise RuntimeError("Model is not loaded")
        return {
            "model_name": self.metadata.get("model_name", "heart_disease_model"),
            "version": self.version,
            "algorithm": self.metadata.get("algorithm", type(self.model).__name__),
            "accuracy": self.metadata.get("accuracy"),
            "precision": self.metadata.get("precision"),
            "recall": self.metadata.get("recall"),
            "f1_score": self.metadata.get("f1_score"),
            "f2_score": self.metadata.get("f2_score"),
            "roc_auc": self.metadata.get("roc_auc"),
            "selected_metric": self.metadata.get("selected_metric"),
            "decision_threshold": self.decision_threshold,
            "created_at": self.metadata.get("created_at"),
            "input_features": self.input_features,
            "model_path": str(self.model_path),
            "dataset_source": self.metadata.get("dataset_source"),
            "evaluation_report_path": self.metadata.get("evaluation_report_path"),
        }

    def predict(self, patient: PatientData) -> PredictionResult:
        if not self.loaded:
            raise RuntimeError("Model is not loaded")

        start = time.perf_counter()
        row = patient.model_dump()
        frame = self._to_frame([row])
        model_prediction = int(self.model.predict(frame)[0])
        probabilities = self._predict_probabilities(frame)
        disease_probability = self._disease_probability(probabilities, model_prediction)
        prediction = self._prediction_with_threshold(disease_probability, model_prediction)
        inference_time_ms = round((time.perf_counter() - start) * 1000, 3)

        result = PredictionResult(
            prediction=prediction,
            label="Disease" if prediction == 1 else "No Disease",
            probability=round(disease_probability, 4) if disease_probability is not None else None,
            risk_level=self._risk_level(disease_probability, prediction),
            model_version=self.version,
            inference_time_ms=inference_time_ms,
            probabilities=probabilities,
            decision_threshold=self.decision_threshold,
        )
        logger.info(
            "Prediction completed: prediction=%s risk_level=%s model_version=%s inference_time_ms=%s",
            result.prediction,
            result.risk_level,
            result.model_version,
            result.inference_time_ms,
        )
        return result

    def predict_batch(self, patients: List[PatientData]) -> List[PredictionResult]:
        return [self.predict(patient) for patient in patients]

    def _to_frame(self, rows: List[Dict[str, Any]]) -> pd.DataFrame:
        frame = pd.DataFrame(rows)
        missing = set(self.input_features) - set(frame.columns)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Missing required features: {missing_list}")
        return frame[self.input_features].apply(pd.to_numeric, errors="raise")

    def _predict_probabilities(self, frame: pd.DataFrame) -> Optional[Dict[str, float]]:
        if not hasattr(self.model, "predict_proba"):
            return None

        raw = self.model.predict_proba(frame)[0]
        classes = list(getattr(self.model, "classes_", [0, 1]))
        result: Dict[str, float] = {}
        for label, value in zip(classes, raw):
            key = "disease" if int(label) == 1 else "no_disease"
            result[key] = round(float(value), 4)
        return result

    def _disease_probability(self, probabilities: Optional[Dict[str, float]], prediction: int) -> Optional[float]:
        if not probabilities:
            return None
        if "disease" in probabilities:
            return float(probabilities["disease"])
        return float(probabilities.get(str(prediction), 0.0))

    def _prediction_with_threshold(self, disease_probability: Optional[float], model_prediction: int) -> int:
        threshold = self.decision_threshold
        if disease_probability is None or threshold is None:
            return model_prediction
        return 1 if disease_probability >= threshold else 0

    def _risk_level(self, disease_probability: Optional[float], prediction: int) -> str:
        if disease_probability is None:
            return "High" if prediction == 1 else "Low"
        if disease_probability >= 0.7:
            return "High"
        if disease_probability >= 0.35:
            return "Medium"
        return "Low"

    def _model_dir(self) -> Path:
        env_dir = os.getenv("MODEL_DIR")
        if env_dir:
            return Path(env_dir).resolve()
        return Path(__file__).resolve().parents[2] / "models"

    def _resolve_model_path(self, model_dir: Path, raw_path: Optional[str]) -> Path:
        if raw_path:
            candidates = [
                Path(raw_path),
                model_dir / Path(raw_path).name,
                Path(__file__).resolve().parents[2] / raw_path,
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate.resolve()
        return self._find_latest_model(model_dir)

    def _find_latest_model(self, model_dir: Path) -> Path:
        candidates = sorted(model_dir.glob("heart_model_*.joblib"), reverse=True)
        if not candidates:
            raise FileNotFoundError(f"No model files found in {model_dir}")
        return candidates[0].resolve()

    def _version_from_filename(self, filename: str) -> str:
        for part in filename.split("_"):
            if part.startswith("v"):
                return part
        return "unknown"
