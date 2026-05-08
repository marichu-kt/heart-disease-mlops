import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import joblib
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
DEFAULT_DATASET_PATH = DATA_DIR / "heart.csv"
MODEL_VERSION = "v2.0.0"
MODEL_NAME = "heart_disease_mlp"
MODEL_FILENAME = "heart_model_v2_mlp.joblib"

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

COLUMN_ALIASES = {
    "chest_pain": "chest",
    "cp": "chest",
    "rest_blood_pressure": "resting_blood_pressure",
    "trestbps": "resting_blood_pressure",
    "cholesterol": "serum_cholestoral",
    "chol": "serum_cholestoral",
    "rest_ecg": "resting_electrocardiographic_results",
    "restecg": "resting_electrocardiographic_results",
    "max_heart_rate": "maximum_heart_rate_achieved",
    "thalach": "maximum_heart_rate_achieved",
    "exercise_angina": "exercise_induced_angina",
    "exang": "exercise_induced_angina",
    "vessels": "number_of_major_vessels",
    "ca": "number_of_major_vessels",
    "target": "class",
    "num": "class",
    "diagnosis": "class",
}


def load_dataset(dataset_path: Optional[Path]) -> pd.DataFrame:
    if dataset_path and dataset_path.exists():
        print(f"Loading local dataset from {dataset_path}")
        return pd.read_csv(dataset_path)

    if DEFAULT_DATASET_PATH.exists():
        print(f"Loading local dataset from {DEFAULT_DATASET_PATH}")
        return pd.read_csv(DEFAULT_DATASET_PATH)

    print("No local dataset found. Downloading OpenML heart-statlog used by the workshop notebook.")
    try:
        data = fetch_openml("heart-statlog", version=1, as_frame=True, parser="auto")
    except TypeError:
        data = fetch_openml("heart-statlog", version=1, as_frame=True)
    except Exception as exc:
        raise RuntimeError(
            "Dataset not found. Place a CSV file at data/heart.csv or run with "
            "--data-path /path/to/file.csv. OpenML download also failed."
        ) from exc

    frame = data.frame
    if "class" not in frame.columns and data.target is not None:
        frame["class"] = data.target
    return frame


def normalize_dataset(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    frame = frame.rename(columns={column: COLUMN_ALIASES.get(column, column) for column in frame.columns})
    if "class" not in frame.columns:
        raise ValueError("Dataset must include a target column named class, target, num or diagnosis.")

    missing = set(INPUT_FEATURES) - set(frame.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Dataset is missing required features: {missing_list}")

    X = frame[INPUT_FEATURES].apply(pd.to_numeric, errors="raise")
    y = normalize_target(frame["class"])
    return X, y


def normalize_target(target: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(target):
        numeric = pd.to_numeric(target)
        if set(numeric.dropna().unique()) <= {1, 2}:
            return numeric.map({1: 0, 2: 1}).astype(int)
        return (numeric > 0).astype(int)

    lowered = target.astype(str).str.lower().str.strip()
    mapping = {
        "absent": 0,
        "no": 0,
        "none": 0,
        "0": 0,
        "false": 0,
        "present": 1,
        "yes": 1,
        "1": 1,
        "true": 1,
        "disease": 1,
    }
    mapped = lowered.map(mapping)
    if mapped.isna().any():
        unknown = sorted(lowered[mapped.isna()].unique())
        raise ValueError(f"Unsupported target labels: {unknown}")
    return mapped.astype(int)


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=(16, 8),
                    activation="relu",
                    solver="adam",
                    alpha=0.001,
                    learning_rate_init=0.001,
                    max_iter=1200,
                    early_stopping=True,
                    n_iter_no_change=30,
                    random_state=42,
                ),
            ),
        ]
    )


def train(dataset_path: Optional[Path] = None) -> dict:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_dataset(dataset_path)
    X, y = normalize_dataset(frame)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions, target_names=["No Disease", "Disease"]))
    print(f"Accuracy: {accuracy:.4f}")

    model_path = MODEL_DIR / MODEL_FILENAME
    joblib.dump(pipeline, model_path)

    metadata = {
        "model_name": MODEL_NAME,
        "version": MODEL_VERSION,
        "algorithm": "StandardScaler + MLPClassifier",
        "accuracy": round(accuracy, 4),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_features": INPUT_FEATURES,
        "model_path": f"models/{MODEL_FILENAME}",
        "dataset_source": str(dataset_path or DEFAULT_DATASET_PATH if (dataset_path or DEFAULT_DATASET_PATH).exists() else "OpenML heart-statlog"),
    }
    metadata_path = MODEL_DIR / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model to {model_path}")
    print(f"Saved metadata to {metadata_path}")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Heart Disease MLP model.")
    parser.add_argument("--data-path", type=Path, default=None, help="Optional CSV path. Defaults to data/heart.csv or OpenML.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data_path)
