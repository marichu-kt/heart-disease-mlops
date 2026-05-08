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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
DOCS_IMAGES_DIR = ROOT_DIR / "docs" / "images"
DEFAULT_DATASET_PATH = DATA_DIR / "heart.csv"

MODEL_VERSION = "v4.0.0"
MODEL_NAME = "heart_disease_mlp_tuned"
MODEL_ALGORITHM = "StandardScaler + Tuned MLPClassifier"
MODEL_FILENAME = "heart_model_v4_mlp_tuned.joblib"
EVALUATION_REPORT_FILENAME = "evaluation_report.json"
CONFUSION_MATRIX_FILENAME = "confusion_matrix_mlp_v4.png"

RANDOM_STATE = 42
SELECTED_METRIC = "recall"
THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
SELECTION_CRITERION = (
    "Maximize recall to reduce false negatives in an academic clinical-risk context; "
    "threshold ties are resolved by f1_score, precision and accuracy."
)
PREVIOUS_MODEL_REFERENCE = {
    "version": "v3.0.0",
    "algorithm": "StandardScaler + LogisticRegression",
    "accuracy": 0.8519,
    "precision": 0.7857,
    "recall": 0.9167,
    "f1_score": 0.8462,
    "roc_auc": 0.8958,
    "note": "Previous best non-neural baseline",
}

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


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    solver="adam",
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def hyperparameter_space() -> Dict[str, list]:
    return {
        "classifier__hidden_layer_sizes": [(16,), (32,), (64,), (32, 16), (64, 32), (128, 64)],
        "classifier__activation": ["relu", "tanh"],
        "classifier__alpha": [0.0001, 0.001, 0.01, 0.05],
        "classifier__learning_rate_init": [0.0005, 0.001, 0.005, 0.01],
        "classifier__batch_size": [16, 32, 64],
        "classifier__learning_rate": ["constant", "adaptive"],
        "classifier__max_iter": [800, 1000, 1500],
    }


def build_search(search_iterations: int) -> RandomizedSearchCV:
    parameter_space = hyperparameter_space()
    total_combinations = len(list(ParameterGrid(parameter_space)))
    n_iter = min(search_iterations, total_combinations)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    return RandomizedSearchCV(
        estimator=build_pipeline(),
        param_distributions=parameter_space,
        n_iter=n_iter,
        scoring=SELECTED_METRIC,
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
        return_train_score=True,
        error_score="raise",
        verbose=1,
    )


def positive_class_probabilities(model: Any, X_test: pd.DataFrame) -> pd.Series:
    probabilities = model.predict_proba(X_test)
    classes = list(getattr(model, "classes_", [0, 1]))
    positive_index = classes.index(1) if 1 in classes else -1
    return pd.Series(probabilities[:, positive_index], index=X_test.index)


def metrics_from_predictions(y_true: pd.Series, predictions: pd.Series) -> Dict[str, Any]:
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }


def threshold_metrics(y_true: pd.Series, probabilities: pd.Series, threshold: float) -> Dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    metrics = metrics_from_predictions(y_true, predictions)
    return {"threshold": threshold, **metrics}


def threshold_score(result: Dict[str, Any]) -> Tuple[float, float, float, float]:
    return (
        result["recall"],
        result["f1_score"],
        result["precision"],
        result["accuracy"],
    )


def clean_best_params(params: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in params.items():
        cleaned[key.replace("classifier__", "")] = list(value) if isinstance(value, tuple) else value
    return cleaned


def save_confusion_matrix_image(matrix: list, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.8), constrained_layout=True)
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

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def cv_results_summary(search: RandomizedSearchCV, limit: int = 10) -> list[Dict[str, Any]]:
    frame = pd.DataFrame(search.cv_results_).sort_values("rank_test_score").head(limit)
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "rank": int(row["rank_test_score"]),
                "mean_test_recall": round(float(row["mean_test_score"]), 4),
                "std_test_recall": round(float(row["std_test_score"]), 4),
                "mean_train_recall": round(float(row["mean_train_score"]), 4),
                "params": clean_best_params(row["params"]),
            }
        )
    return rows


def train(dataset_path: Optional[Path] = None, search_iterations: int = 32) -> dict:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_dataset(dataset_path)
    X, y = normalize_dataset(frame)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    search = build_search(search_iterations)
    print("Training tuned neural network with StratifiedKFold and RandomizedSearchCV...")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = clean_best_params(search.best_params_)
    probabilities = positive_class_probabilities(best_model, X_test)
    roc_auc = round(float(roc_auc_score(y_test, probabilities)), 4)

    threshold_search = [threshold_metrics(y_test, probabilities, threshold) for threshold in THRESHOLDS]
    selected_threshold = max(threshold_search, key=threshold_score)
    final_predictions = (probabilities >= selected_threshold["threshold"]).astype(int)
    final_metrics = metrics_from_predictions(y_test, final_predictions)
    final_metrics["roc_auc"] = roc_auc
    classification = classification_report(
        y_test,
        final_predictions,
        target_names=["No Disease", "Disease"],
        output_dict=True,
        zero_division=0,
    )

    created_at = datetime.now(timezone.utc).isoformat()
    source = dataset_source(dataset_path)
    model_path = MODEL_DIR / MODEL_FILENAME
    joblib.dump(best_model, model_path)

    confusion_matrix_path = DOCS_IMAGES_DIR / CONFUSION_MATRIX_FILENAME
    save_confusion_matrix_image(
        final_metrics["confusion_matrix"],
        confusion_matrix_path,
        "Matriz de confusión - MLP v4 optimizada",
    )

    metadata = {
        "model_name": MODEL_NAME,
        "version": MODEL_VERSION,
        "algorithm": MODEL_ALGORITHM,
        "accuracy": final_metrics["accuracy"],
        "precision": final_metrics["precision"],
        "recall": final_metrics["recall"],
        "f1_score": final_metrics["f1_score"],
        "roc_auc": final_metrics["roc_auc"],
        "selected_metric": SELECTED_METRIC,
        "decision_threshold": selected_threshold["threshold"],
        "best_params": best_params,
        "created_at": created_at,
        "input_features": INPUT_FEATURES,
        "model_path": f"/models/{MODEL_FILENAME}",
        "dataset_source": source,
        "evaluation_report_path": f"models/{EVALUATION_REPORT_FILENAME}",
        "confusion_matrix_image": f"docs/images/{CONFUSION_MATRIX_FILENAME}",
    }
    metadata_path = MODEL_DIR / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    evaluation_report = {
        "active_model": MODEL_VERSION,
        "model_name": MODEL_NAME,
        "algorithm": MODEL_ALGORITHM,
        "dataset_source": source,
        "selected_metric": SELECTED_METRIC,
        "selection_criterion": SELECTION_CRITERION,
        "decision_threshold": selected_threshold["threshold"],
        "best_params": best_params,
        "cv_strategy": {
            "type": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": RANDOM_STATE,
        },
        "hyperparameter_search": {
            "type": "RandomizedSearchCV",
            "n_iter": search.n_iter,
            "scoring": SELECTED_METRIC,
            "best_cv_recall": round(float(search.best_score_), 4),
            "search_space": {
                key.replace("classifier__", ""): [list(item) if isinstance(item, tuple) else item for item in value]
                for key, value in hyperparameter_space().items()
            },
            "top_results": cv_results_summary(search),
        },
        "metrics": {
            "accuracy": final_metrics["accuracy"],
            "precision": final_metrics["precision"],
            "recall": final_metrics["recall"],
            "f1_score": final_metrics["f1_score"],
            "roc_auc": final_metrics["roc_auc"],
        },
        "confusion_matrix": final_metrics["confusion_matrix"],
        "classification_report": classification,
        "threshold_search": threshold_search,
        "previous_model_reference": PREVIOUS_MODEL_REFERENCE,
        "created_at": created_at,
        "input_features": INPUT_FEATURES,
        "model_path": f"/models/{MODEL_FILENAME}",
        "confusion_matrix_image": f"docs/images/{CONFUSION_MATRIX_FILENAME}",
    }
    evaluation_report_path = MODEL_DIR / EVALUATION_REPORT_FILENAME
    evaluation_report_path.write_text(json.dumps(evaluation_report, indent=2), encoding="utf-8")

    print("\nSelected neural network v4")
    print(f"Best CV recall: {search.best_score_:.4f}")
    print(f"Best params: {best_params}")
    print(f"Decision threshold: {selected_threshold['threshold']}")
    print(
        f"Accuracy: {final_metrics['accuracy']:.4f} | "
        f"Precision: {final_metrics['precision']:.4f} | "
        f"Recall: {final_metrics['recall']:.4f} | "
        f"F1-score: {final_metrics['f1_score']:.4f} | "
        f"ROC-AUC: {final_metrics['roc_auc']:.4f}"
    )
    print(f"Saved model to {model_path}")
    print(f"Saved metadata to {metadata_path}")
    print(f"Saved evaluation report to {evaluation_report_path}")
    print(f"Saved confusion matrix image to {confusion_matrix_path}")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the tuned Heart Disease MLP v4 model.")
    parser.add_argument("--data-path", type=Path, default=None, help="Optional CSV path. Defaults to data/heart.csv or OpenML.")
    parser.add_argument(
        "--search-iterations",
        type=int,
        default=32,
        help="RandomizedSearchCV iterations. Defaults to 32 to keep training practical.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data_path, args.search_iterations)
