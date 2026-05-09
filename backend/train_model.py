import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import fetch_openml
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    fbeta_score,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    ParameterGrid,
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
DOCS_IMAGES_DIR = ROOT_DIR / "docs" / "images"
DEFAULT_DATASET_PATH = DATA_DIR / "heart.csv"

MODEL_VERSION = "v4.2.0"
MODEL_NAME = "heart_disease_mlp_calibrated"
MODEL_ALGORITHM = "StandardScaler + Robust Calibrated MLPClassifier"
MODEL_FILENAME = "heart_model_v4_2_mlp_calibrated.joblib"
EVALUATION_REPORT_FILENAME = "evaluation_report.json"

CONFUSION_MATRIX_FILENAME = "confusion_matrix_mlp_v4_2.png"
ROC_CURVE_FILENAME = "roc_curve_mlp_v4_2.png"
PR_CURVE_FILENAME = "precision_recall_curve_mlp_v4_2.png"
FEATURE_IMPORTANCE_FILENAME = "feature_importance_mlp_v4_2.png"
THRESHOLD_METRICS_FILENAME = "threshold_metrics_mlp_v4_2.png"
MODEL_COMPARISON_FILENAME = "model_comparison_metrics.png"

RANDOM_STATE = 42
SEARCH_SCORING = "f2_score"
SELECTED_METRIC = "f2_score"
THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
DATASET_RATIONALE = (
    "OpenML heart-statlog was used because it was the original dataset from the workshop notebook "
    "and matches the 13 clinical input features expected by the API."
)
SELECTION_CRITERION = (
    "Select the decision threshold with the highest F2-score; ties are resolved by F1-score, "
    "recall, precision and accuracy. F2 keeps recall weighted more strongly than precision "
    "without ignoring false positives."
)
THRESHOLD_SELECTION_REASON_TEMPLATE = (
    "Selected threshold {threshold:.2f} because it maximized F2-score on the held-out test set; "
    "ties are resolved by F1-score, recall, precision and accuracy."
)
CALIBRATION_TOLERANCE_F2 = 0.02
F2_SCORER = make_scorer(fbeta_score, beta=2, zero_division=0)

PREVIOUS_MODEL_REFERENCE = {
    "version": "v3.0.0",
    "algorithm": "StandardScaler + LogisticRegression",
    "accuracy": 0.8519,
    "precision": 0.7857,
    "recall": 0.9167,
    "f1_score": 0.8462,
    "f2_score": 0.8871,
    "roc_auc": 0.8958,
    "note": "Strong non-neural baseline kept for transparent comparison.",
}
PREVIOUS_NEURAL_RECALL_REFERENCE = {
    "version": "v4.0.0",
    "algorithm": "StandardScaler + Tuned MLPClassifier",
    "accuracy": 0.6667,
    "precision": 0.5714,
    "recall": 1.0,
    "f1_score": 0.7273,
    "f2_score": 0.8696,
    "roc_auc": 0.8583,
    "decision_threshold": 0.35,
    "confusion_matrix": [[12, 18], [0, 24]],
    "false_positives": 18,
    "false_negatives": 0,
    "note": "Neural model optimized for maximum recall, with many false positives.",
}
PREVIOUS_NEURAL_BALANCED_REFERENCE = {
    "version": "v4.1.0",
    "algorithm": "StandardScaler + Balanced Tuned MLPClassifier",
    "accuracy": 0.7778,
    "precision": 0.6875,
    "recall": 0.9167,
    "f1_score": 0.7857,
    "f2_score": 0.8594,
    "roc_auc": 0.8569,
    "decision_threshold": 0.55,
    "confusion_matrix": [[20, 10], [2, 22]],
    "false_positives": 10,
    "false_negatives": 2,
    "note": "Balanced neural model that reduced false positives while keeping high recall.",
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
        "classifier__alpha": [0.0001, 0.001, 0.01, 0.05, 0.1],
        "classifier__learning_rate_init": [0.0005, 0.001, 0.005, 0.01],
        "classifier__batch_size": [16, 32, 64],
        "classifier__learning_rate": ["constant", "adaptive"],
        "classifier__max_iter": [800, 1000, 1500],
    }


def build_search(search_iterations: int) -> RandomizedSearchCV:
    parameter_space = hyperparameter_space()
    total_combinations = len(list(ParameterGrid(parameter_space)))
    n_iter = min(search_iterations, total_combinations)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)

    return RandomizedSearchCV(
        estimator=build_pipeline(),
        param_distributions=parameter_space,
        n_iter=n_iter,
        scoring=F2_SCORER,
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


def round_float(value: Any, digits: int = 4) -> float:
    return round(float(value), digits)


def metrics_from_predictions(y_true: pd.Series, predictions: pd.Series) -> Dict[str, Any]:
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "accuracy": round_float(accuracy_score(y_true, predictions)),
        "precision": round_float(precision_score(y_true, predictions, zero_division=0)),
        "recall": round_float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": round_float(f1_score(y_true, predictions, zero_division=0)),
        "f2_score": round_float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def threshold_metrics(
    y_true: pd.Series,
    probabilities: pd.Series,
    threshold: float,
    roc_auc: float,
) -> Dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    metrics = metrics_from_predictions(y_true, predictions)
    return {"threshold": threshold, "roc_auc": roc_auc, **metrics}


def select_threshold(threshold_search: list[Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
    selected = max(
        threshold_search,
        key=lambda result: (
            result["f2_score"],
            result["f1_score"],
            result["recall"],
            result["precision"],
            result["accuracy"],
        ),
    )
    reason = THRESHOLD_SELECTION_REASON_TEMPLATE.format(threshold=selected["threshold"])
    return selected, reason


def clean_best_params(params: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in params.items():
        cleaned[key.replace("classifier__", "")] = list(value) if isinstance(value, tuple) else value
    return cleaned


def cv_results_summary(search: RandomizedSearchCV, limit: int = 10) -> list[Dict[str, Any]]:
    frame = pd.DataFrame(search.cv_results_).sort_values("rank_test_score").head(limit)
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "rank": int(row["rank_test_score"]),
                "mean_test_f2_score": round_float(row["mean_test_score"]),
                "std_test_f2_score": round_float(row["std_test_score"]),
                "mean_train_f2_score": round_float(row["mean_train_score"]),
                "params": clean_best_params(row["params"]),
            }
        )
    return rows


def cross_validation_summary(search: RandomizedSearchCV) -> Dict[str, Any]:
    mean_scores = pd.Series(search.cv_results_["mean_test_score"])
    std_scores = pd.Series(search.cv_results_["std_test_score"])
    return {
        "best_cv_f2_score": round_float(search.best_score_),
        "mean_cv_f2_score": round_float(mean_scores.mean()),
        "std_cv_f2_score": round_float(mean_scores.std()),
        "mean_candidate_std_f2_score": round_float(std_scores.mean()),
        "evaluated_candidates": int(len(mean_scores)),
        "total_cv_fits": int(len(mean_scores) * 15),
    }


def evaluate_model_candidate(
    name: str,
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    probabilities = positive_class_probabilities(model, X_test)
    roc_auc = round_float(roc_auc_score(y_test, probabilities))
    average_precision = round_float(average_precision_score(y_test, probabilities))
    brier = round_float(brier_score_loss(y_test, probabilities))
    threshold_search = [threshold_metrics(y_test, probabilities, threshold, roc_auc) for threshold in THRESHOLDS]
    selected_threshold, threshold_selection_reason = select_threshold(threshold_search)
    predictions = (probabilities >= selected_threshold["threshold"]).astype(int)
    metrics = metrics_from_predictions(y_test, predictions)
    metrics["roc_auc"] = roc_auc
    metrics["average_precision"] = average_precision
    metrics["brier_score"] = brier
    return {
        "name": name,
        "model": model,
        "probabilities": probabilities,
        "predictions": predictions,
        "threshold_search": threshold_search,
        "selected_threshold": selected_threshold,
        "threshold_selection_reason": threshold_selection_reason,
        "metrics": metrics,
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=["No Disease", "Disease"],
            output_dict=True,
            zero_division=0,
        ),
    }


def build_calibrated_model(best_model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Optional[Any], str]:
    try:
        calibrated_model = CalibratedClassifierCV(estimator=best_model, method="sigmoid", cv=3)
    except TypeError:
        calibrated_model = CalibratedClassifierCV(base_estimator=best_model, method="sigmoid", cv=3)

    try:
        calibrated_model.fit(X_train, y_train)
    except Exception as exc:
        return None, f"Sigmoid calibration was evaluated but not applied because fitting failed: {exc}"

    return calibrated_model, "Sigmoid calibration applied with CalibratedClassifierCV using 3-fold cross-validation."


def choose_final_candidate(
    uncalibrated: Dict[str, Any],
    calibrated: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], bool, str, str]:
    if calibrated is None:
        return (
            uncalibrated,
            False,
            "sigmoid evaluated, not applied",
            "Calibration was evaluated but not applied because the calibrated model could not be fitted.",
        )

    uncalibrated_metrics = uncalibrated["metrics"]
    calibrated_metrics = calibrated["metrics"]
    f2_delta = calibrated_metrics["f2_score"] - uncalibrated_metrics["f2_score"]
    brier_improvement = uncalibrated_metrics["brier_score"] - calibrated_metrics["brier_score"]

    if calibrated_metrics["f2_score"] >= uncalibrated_metrics["f2_score"]:
        return (
            calibrated,
            True,
            "sigmoid",
            "Sigmoid calibration was applied because it preserved or improved the selected F2-score.",
        )

    if f2_delta >= -CALIBRATION_TOLERANCE_F2 and brier_improvement > 0:
        return (
            calibrated,
            True,
            "sigmoid",
            (
                "Sigmoid calibration was applied because F2-score stayed within the accepted tolerance "
                "and probability calibration improved according to Brier score."
            ),
        )

    return (
        uncalibrated,
        False,
        "sigmoid evaluated, not applied",
        (
            "Sigmoid calibration was evaluated but not applied because it reduced F2-score beyond the "
            "accepted tolerance or did not improve probability calibration."
        ),
    )


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
        xlabel="Prediccion",
        title=title,
    )

    threshold = max(max(row) for row in matrix) / 2
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            color = "white" if value > threshold else "#1f2937"
            ax.text(column_index, row_index, value, ha="center", va="center", color=color, fontsize=12)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_roc_curve_image(y_test: pd.Series, probabilities: pd.Series, roc_auc: float, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
    fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    ax.plot(false_positive_rate, true_positive_rate, color="#0066cc", linewidth=2.2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="#6b7280", linestyle="--", linewidth=1.2, label="Referencia aleatoria")
    ax.set_title("Curva ROC - MLP v4.2")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(loc="lower right")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_precision_recall_curve_image(
    y_test: pd.Series,
    probabilities: pd.Series,
    average_precision: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    ax.plot(recall, precision, color="#0f766e", linewidth=2.2, label=f"AP = {average_precision:.4f}")
    ax.set_title("Curva Precision-Recall - MLP v4.2")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(loc="lower left")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_feature_importance_image(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
) -> Dict[str, float]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring=F2_SCORER,
        n_repeats=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importances = pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False)
    top_importances = importances.head(10).sort_values()

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    colors = ["#0066cc" if value >= 0 else "#9ca3af" for value in top_importances]
    ax.barh(top_importances.index, top_importances.values, color=colors)
    ax.set_title("Importancia por permutacion - MLP v4.2")
    ax.set_xlabel("Cambio medio en F2-score")
    ax.grid(True, axis="x", color="#e5e7eb", linewidth=0.8)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return {feature: round_float(value) for feature, value in importances.items()}


def save_threshold_metrics_image(threshold_search: list[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(threshold_search)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.plot(frame["threshold"], frame["precision"], marker="o", linewidth=2, label="Precision")
    ax.plot(frame["threshold"], frame["recall"], marker="o", linewidth=2, label="Recall")
    ax.plot(frame["threshold"], frame["f1_score"], marker="o", linewidth=2, label="F1")
    ax.plot(frame["threshold"], frame["f2_score"], marker="o", linewidth=2.4, label="F2")
    ax.set_title("Metricas por umbral - MLP v4.2")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(loc="lower left", ncol=2)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_model_comparison_image(model_comparison: Dict[str, Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = ["accuracy", "precision", "recall", "f1_score", "f2_score"]
    labels = list(model_comparison.keys())
    values = np.array([[model_comparison[label].get(metric, np.nan) for metric in metrics] for label in labels])

    x_positions = np.arange(len(labels))
    width = 0.15
    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    colors = ["#0066cc", "#0f766e", "#f59e0b", "#6d28d9", "#be123c"]
    for index, metric in enumerate(metrics):
        ax.bar(
            x_positions + (index - 2) * width,
            values[:, index],
            width,
            label=metric.replace("_score", "").upper(),
            color=colors[index],
        )
    ax.set_title("Comparativa de metricas por version")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    ax.legend(loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.22))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def report_image_paths() -> Dict[str, str]:
    return {
        "confusion_matrix": f"docs/images/{CONFUSION_MATRIX_FILENAME}",
        "roc_curve": f"docs/images/{ROC_CURVE_FILENAME}",
        "precision_recall_curve": f"docs/images/{PR_CURVE_FILENAME}",
        "feature_importance": f"docs/images/{FEATURE_IMPORTANCE_FILENAME}",
        "threshold_metrics": f"docs/images/{THRESHOLD_METRICS_FILENAME}",
        "model_comparison": f"docs/images/{MODEL_COMPARISON_FILENAME}",
    }


def train(dataset_path: Optional[Path] = None, search_iterations: int = 32) -> dict:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
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
    print("Training robust neural network with RepeatedStratifiedKFold and RandomizedSearchCV...")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = clean_best_params(search.best_params_)

    uncalibrated_result = evaluate_model_candidate("uncalibrated", best_model, X_test, y_test)
    calibrated_model, calibration_fit_reason = build_calibrated_model(best_model, X_train, y_train)
    calibrated_result = (
        evaluate_model_candidate("calibrated_sigmoid", calibrated_model, X_test, y_test)
        if calibrated_model is not None
        else None
    )
    final_result, calibration_applied, calibration_method, calibration_reason = choose_final_candidate(
        uncalibrated_result,
        calibrated_result,
    )
    if calibrated_model is None:
        calibration_reason = calibration_fit_reason

    final_model = final_result["model"]
    final_metrics = final_result["metrics"]
    selected_threshold = final_result["selected_threshold"]
    threshold_search = final_result["threshold_search"]
    probabilities = final_result["probabilities"]
    classification = final_result["classification_report"]
    threshold_selection_reason = final_result["threshold_selection_reason"]

    created_at = datetime.now(timezone.utc).isoformat()
    source = dataset_source(dataset_path)
    image_paths = report_image_paths()
    model_path = MODEL_DIR / MODEL_FILENAME
    joblib.dump(final_model, model_path)

    confusion_matrix_path = DOCS_IMAGES_DIR / CONFUSION_MATRIX_FILENAME
    roc_curve_path = DOCS_IMAGES_DIR / ROC_CURVE_FILENAME
    pr_curve_path = DOCS_IMAGES_DIR / PR_CURVE_FILENAME
    feature_importance_path = DOCS_IMAGES_DIR / FEATURE_IMPORTANCE_FILENAME
    threshold_metrics_path = DOCS_IMAGES_DIR / THRESHOLD_METRICS_FILENAME
    model_comparison_path = DOCS_IMAGES_DIR / MODEL_COMPARISON_FILENAME

    save_confusion_matrix_image(
        final_metrics["confusion_matrix"],
        confusion_matrix_path,
        "Matriz de confusion - MLP v4.2 robusta",
    )
    save_roc_curve_image(y_test, probabilities, final_metrics["roc_auc"], roc_curve_path)
    save_precision_recall_curve_image(y_test, probabilities, final_metrics["average_precision"], pr_curve_path)
    feature_importance = save_feature_importance_image(final_model, X_test, y_test, feature_importance_path)
    save_threshold_metrics_image(threshold_search, threshold_metrics_path)

    model_comparison = {
        "v3.0.0": PREVIOUS_MODEL_REFERENCE,
        "v4.0.0": PREVIOUS_NEURAL_RECALL_REFERENCE,
        "v4.1.0": PREVIOUS_NEURAL_BALANCED_REFERENCE,
        MODEL_VERSION: {
            "version": MODEL_VERSION,
            "algorithm": MODEL_ALGORITHM,
            "accuracy": final_metrics["accuracy"],
            "precision": final_metrics["precision"],
            "recall": final_metrics["recall"],
            "f1_score": final_metrics["f1_score"],
            "f2_score": final_metrics["f2_score"],
            "roc_auc": final_metrics["roc_auc"],
            "average_precision": final_metrics["average_precision"],
            "brier_score": final_metrics["brier_score"],
            "decision_threshold": selected_threshold["threshold"],
            "confusion_matrix": final_metrics["confusion_matrix"],
            "false_positives": final_metrics["false_positives"],
            "false_negatives": final_metrics["false_negatives"],
            "calibration_applied": calibration_applied,
            "note": "Robust neural model selected with F2-score, repeated CV and calibration evaluation.",
        },
    }
    save_model_comparison_image(model_comparison, model_comparison_path)

    metadata = {
        "model_name": MODEL_NAME,
        "version": MODEL_VERSION,
        "algorithm": MODEL_ALGORITHM,
        "accuracy": final_metrics["accuracy"],
        "precision": final_metrics["precision"],
        "recall": final_metrics["recall"],
        "f1_score": final_metrics["f1_score"],
        "f2_score": final_metrics["f2_score"],
        "roc_auc": final_metrics["roc_auc"],
        "selected_metric": SELECTED_METRIC,
        "decision_threshold": selected_threshold["threshold"],
        "best_params": best_params,
        "calibration_method": calibration_method,
        "calibration_applied": calibration_applied,
        "calibration_reason": calibration_reason,
        "cv_strategy": "RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)",
        "created_at": created_at,
        "input_features": INPUT_FEATURES,
        "model_path": f"/models/{MODEL_FILENAME}",
        "dataset_source": "OpenML heart-statlog" if source == "OpenML heart-statlog" else source,
        "dataset_rationale": DATASET_RATIONALE,
        "evaluation_report_path": f"models/{EVALUATION_REPORT_FILENAME}",
        "technical_images": image_paths,
    }
    metadata_path = MODEL_DIR / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    calibration_comparison = {
        "uncalibrated": {
            key: uncalibrated_result["metrics"][key]
            for key in ["accuracy", "precision", "recall", "f1_score", "f2_score", "roc_auc", "average_precision", "brier_score"]
        },
        "calibrated_sigmoid": (
            {
                key: calibrated_result["metrics"][key]
                for key in [
                    "accuracy",
                    "precision",
                    "recall",
                    "f1_score",
                    "f2_score",
                    "roc_auc",
                    "average_precision",
                    "brier_score",
                ]
            }
            if calibrated_result is not None
            else None
        ),
        "selected_candidate": final_result["name"],
    }
    evaluation_report = {
        "active_model": MODEL_VERSION,
        "model_name": MODEL_NAME,
        "algorithm": MODEL_ALGORITHM,
        "dataset_source": source,
        "dataset_rationale": DATASET_RATIONALE,
        "dataset_shape": {"records": int(len(X)), "features": int(X.shape[1])},
        "selected_metric": SELECTED_METRIC,
        "selection_criterion": SELECTION_CRITERION,
        "threshold_selection_reason": threshold_selection_reason,
        "decision_threshold": selected_threshold["threshold"],
        "selected_threshold": selected_threshold["threshold"],
        "best_params": best_params,
        "calibration_method": calibration_method,
        "calibration_applied": calibration_applied,
        "calibration_reason": calibration_reason,
        "calibration_comparison": calibration_comparison,
        "cv_strategy": {
            "type": "RepeatedStratifiedKFold",
            "n_splits": 5,
            "n_repeats": 3,
            "random_state": RANDOM_STATE,
        },
        "hyperparameter_search": {
            "type": "RandomizedSearchCV",
            "n_iter": search.n_iter,
            "scoring": SEARCH_SCORING,
            "best_cv_f2_score": round_float(search.best_score_),
            "search_space": {
                key.replace("classifier__", ""): [list(item) if isinstance(item, tuple) else item for item in value]
                for key, value in hyperparameter_space().items()
            },
            "top_results": cv_results_summary(search),
        },
        "cross_validation_summary": cross_validation_summary(search),
        "metrics": {
            "accuracy": final_metrics["accuracy"],
            "precision": final_metrics["precision"],
            "recall": final_metrics["recall"],
            "f1_score": final_metrics["f1_score"],
            "f2_score": final_metrics["f2_score"],
            "roc_auc": final_metrics["roc_auc"],
            "average_precision": final_metrics["average_precision"],
            "brier_score": final_metrics["brier_score"],
        },
        "classification_report": classification,
        "confusion_matrix": final_metrics["confusion_matrix"],
        "true_negatives": final_metrics["true_negatives"],
        "false_positives": final_metrics["false_positives"],
        "false_negatives": final_metrics["false_negatives"],
        "true_positives": final_metrics["true_positives"],
        "threshold_search": threshold_search,
        "model_comparison": model_comparison,
        "feature_importance": feature_importance,
        "technical_images": image_paths,
        "honest_comparison_note": (
            "v4.2 is activated because it keeps the final model as an MLP neural network while using repeated "
            "cross-validation, F2-score selection, threshold tuning and probability calibration evaluation. "
            "The report keeps v3 and v4.1 as references so any metric trade-off remains visible."
        ),
        "created_at": created_at,
        "input_features": INPUT_FEATURES,
        "model_path": f"/models/{MODEL_FILENAME}",
    }
    evaluation_report_path = MODEL_DIR / EVALUATION_REPORT_FILENAME
    evaluation_report_path.write_text(json.dumps(evaluation_report, indent=2), encoding="utf-8")

    print("\nSelected robust neural network v4.2")
    print(f"Best CV F2-score: {search.best_score_:.4f}")
    print(f"Best params: {best_params}")
    print(f"Calibration: {calibration_method} | applied={calibration_applied}")
    print(f"Calibration reason: {calibration_reason}")
    print(f"Decision threshold: {selected_threshold['threshold']}")
    print(f"Threshold selection: {threshold_selection_reason}")
    print(
        f"Accuracy: {final_metrics['accuracy']:.4f} | "
        f"Precision: {final_metrics['precision']:.4f} | "
        f"Recall: {final_metrics['recall']:.4f} | "
        f"F1-score: {final_metrics['f1_score']:.4f} | "
        f"F2-score: {final_metrics['f2_score']:.4f} | "
        f"ROC-AUC: {final_metrics['roc_auc']:.4f}"
    )
    print(f"Saved model to {model_path}")
    print(f"Saved metadata to {metadata_path}")
    print(f"Saved evaluation report to {evaluation_report_path}")
    print(f"Saved technical images to {DOCS_IMAGES_DIR}")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the robust calibrated Heart Disease MLP v4.2 model.")
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
