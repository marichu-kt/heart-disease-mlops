import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
DOCS_IMAGES_DIR = ROOT_DIR / "docs" / "images"
DEFAULT_DATASET_PATH = DATA_DIR / "heart.csv"
MODEL_VERSION = "v3.0.0"
MODEL_NAME = "heart_disease_best_model"
MODEL_FILENAME = "heart_model_v3_best.joblib"
EVALUATION_REPORT_FILENAME = "evaluation_report.json"
CONFUSION_MATRIX_FILENAME = "confusion_matrix.png"
SELECTED_METRIC = "recall"
SELECTION_CRITERION = (
    "maximize recall to reduce false negatives in an academic clinical-risk context; "
    "ties are resolved by f1_score, roc_auc and accuracy"
)

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


def dataset_source(dataset_path: Optional[Path]) -> str:
    if dataset_path and dataset_path.exists():
        return str(dataset_path)
    if DEFAULT_DATASET_PATH.exists():
        return str(DEFAULT_DATASET_PATH)
    return "OpenML heart-statlog"


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


def build_model_candidates() -> Dict[str, Any]:
    return {
        "LogisticRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
        "MLPClassifier": Pipeline(
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
        ),
        "SVC": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def model_description(name: str, model: Any) -> str:
    if isinstance(model, Pipeline):
        step_names = [type(step).__name__ for _, step in model.steps]
        return " + ".join(step_names)
    return type(model).__name__


def positive_class_scores(model: Any, X_test: pd.DataFrame) -> Optional[pd.Series]:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        classes = list(getattr(model, "classes_", [0, 1]))
        positive_index = classes.index(1) if 1 in classes else -1
        return pd.Series(probabilities[:, positive_index])

    if hasattr(model, "decision_function"):
        return pd.Series(model.decision_function(X_test))

    return None


def evaluate_model(name: str, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    predictions = model.predict(X_test)
    scores = positive_class_scores(model, X_test)
    roc_auc = None
    if scores is not None and len(set(y_test)) == 2:
        roc_auc = float(roc_auc_score(y_test, scores))

    matrix = confusion_matrix(y_test, predictions).tolist()
    return {
        "name": name,
        "algorithm": model_description(name, model),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "confusion_matrix": matrix,
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=["No Disease", "Disease"],
            output_dict=True,
            zero_division=0,
        ),
    }


def selection_score(metrics: Dict[str, Any]) -> Tuple[float, float, float, float]:
    roc_auc = metrics["roc_auc"] if metrics["roc_auc"] is not None else -1.0
    return (
        metrics["recall"],
        metrics["f1_score"],
        roc_auc,
        metrics["accuracy"],
    )


def save_confusion_matrix_image(matrix: list, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    labels = ["No Disease", "Disease"]
    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Valor real",
        xlabel="Predicción",
        title=title,
    )

    threshold = max(max(row) for row in matrix) / 2
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            color = "white" if value > threshold else "#1f2937"
            ax.text(column_index, row_index, value, ha="center", va="center", color=color, fontsize=12)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


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

    candidates = build_model_candidates()
    trained_models: Dict[str, Any] = {}
    comparison: list[Dict[str, Any]] = []

    for name, model in candidates.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        metrics = evaluate_model(name, model, X_test, y_test)
        comparison.append(metrics)
        print(
            f"{name}: accuracy={metrics['accuracy']:.4f} "
            f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
            f"f1={metrics['f1_score']:.4f} roc_auc={metrics['roc_auc']}"
        )

    best_metrics = max(comparison, key=selection_score)
    best_name = best_metrics["name"]
    best_model = trained_models[best_name]
    created_at = datetime.now(timezone.utc).isoformat()
    source = dataset_source(dataset_path)

    print("\nSelected best model")
    print(f"Model: {best_name}")
    print(f"Criterion: {SELECTION_CRITERION}")
    print(
        f"Accuracy: {best_metrics['accuracy']:.4f} | "
        f"Precision: {best_metrics['precision']:.4f} | "
        f"Recall: {best_metrics['recall']:.4f} | "
        f"F1-score: {best_metrics['f1_score']:.4f} | "
        f"ROC-AUC: {best_metrics['roc_auc']}"
    )

    model_path = MODEL_DIR / MODEL_FILENAME
    joblib.dump(best_model, model_path)

    confusion_matrix_path = DOCS_IMAGES_DIR / CONFUSION_MATRIX_FILENAME
    save_confusion_matrix_image(
        best_metrics["confusion_matrix"],
        confusion_matrix_path,
        f"Matriz de confusión - {best_name}",
    )

    metadata = {
        "model_name": MODEL_NAME,
        "version": MODEL_VERSION,
        "algorithm": best_metrics["algorithm"],
        "accuracy": best_metrics["accuracy"],
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
        "f1_score": best_metrics["f1_score"],
        "roc_auc": best_metrics["roc_auc"],
        "selected_metric": SELECTED_METRIC,
        "created_at": created_at,
        "input_features": INPUT_FEATURES,
        "model_path": f"models/{MODEL_FILENAME}",
        "dataset_source": source,
        "evaluation_report_path": f"models/{EVALUATION_REPORT_FILENAME}",
        "confusion_matrix_image": f"docs/images/{CONFUSION_MATRIX_FILENAME}",
    }
    metadata_path = MODEL_DIR / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    evaluation_report = {
        "version": MODEL_VERSION,
        "selected_metric": SELECTED_METRIC,
        "selection_criterion": SELECTION_CRITERION,
        "best_model": best_metrics,
        "models_compared": comparison,
        "confusion_matrix": best_metrics["confusion_matrix"],
        "dataset_source": source,
        "created_at": created_at,
        "input_features": INPUT_FEATURES,
        "model_path": f"models/{MODEL_FILENAME}",
        "confusion_matrix_image": f"docs/images/{CONFUSION_MATRIX_FILENAME}",
    }
    evaluation_report_path = MODEL_DIR / EVALUATION_REPORT_FILENAME
    evaluation_report_path.write_text(json.dumps(evaluation_report, indent=2), encoding="utf-8")

    print(f"Saved model to {model_path}")
    print(f"Saved metadata to {metadata_path}")
    print(f"Saved evaluation report to {evaluation_report_path}")
    print(f"Saved confusion matrix image to {confusion_matrix_path}")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare Heart Disease ML models.")
    parser.add_argument("--data-path", type=Path, default=None, help="Optional CSV path. Defaults to data/heart.csv or OpenML.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data_path)
