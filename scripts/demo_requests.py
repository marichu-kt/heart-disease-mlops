"""
SCRIPT DE DEMO PARA GENERAR TRÁFICO CONTRA LA API FastAPI

Este archivo envía peticiones automáticas al endpoint POST /predict para:
- probar la API con varios pacientes de ejemplo
- generar pacientes sintéticos realistas para una demo
- poblar métricas de Prometheus y Grafana
- medir tiempos de respuesta y latencia
- contar predicciones Disease / No Disease
- contar niveles de riesgo Low / Medium / High
- guardar un resumen opcional en JSON
"""

import argparse
import json
import os
import random
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REQUIRED_FEATURES = [
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


# Casos fijos conservados como referencia/control.
PATIENT_TEMPLATES = [
    {
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
    },
    {
        "age": 42,
        "sex": 0,
        "chest": 2,
        "resting_blood_pressure": 120,
        "serum_cholestoral": 204,
        "fasting_blood_sugar": 0,
        "resting_electrocardiographic_results": 0,
        "maximum_heart_rate_achieved": 172,
        "exercise_induced_angina": 0,
        "oldpeak": 0.1,
        "slope": 1,
        "number_of_major_vessels": 0,
        "thal": 3,
    },
    {
        "age": 57,
        "sex": 1,
        "chest": 4,
        "resting_blood_pressure": 150,
        "serum_cholestoral": 276,
        "fasting_blood_sugar": 0,
        "resting_electrocardiographic_results": 2,
        "maximum_heart_rate_achieved": 112,
        "exercise_induced_angina": 1,
        "oldpeak": 1.6,
        "slope": 2,
        "number_of_major_vessels": 1,
        "thal": 6,
    },
    {
        "age": 51,
        "sex": 0,
        "chest": 3,
        "resting_blood_pressure": 130,
        "serum_cholestoral": 256,
        "fasting_blood_sugar": 0,
        "resting_electrocardiographic_results": 0,
        "maximum_heart_rate_achieved": 149,
        "exercise_induced_angina": 0,
        "oldpeak": 0.5,
        "slope": 1,
        "number_of_major_vessels": 0,
        "thal": 3,
    },
    {
        "age": 68,
        "sex": 1,
        "chest": 4,
        "resting_blood_pressure": 144,
        "serum_cholestoral": 193,
        "fasting_blood_sugar": 1,
        "resting_electrocardiographic_results": 0,
        "maximum_heart_rate_achieved": 141,
        "exercise_induced_angina": 0,
        "oldpeak": 3.4,
        "slope": 2,
        "number_of_major_vessels": 2,
        "thal": 7,
    },
]


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def weighted_choice(rng: random.Random, choices: list[tuple[Any, float]]) -> Any:
    total = sum(weight for _, weight in choices)
    point = rng.uniform(0, total)
    current = 0.0
    for value, weight in choices:
        current += weight
        if current >= point:
            return value
    return choices[-1][0]


def generate_realistic_patient(rng: random.Random) -> dict[str, int | float]:
    """
    Genera un paciente sintético de demo.

    Importante:
    - No se usa para entrenar.
    - No representa un paciente real.
    - Solo sirve para generar tráfico variado y razonable contra /predict.
    """

    profile = weighted_choice(
        rng,
        [
            ("low", 0.35),
            ("medium", 0.35),
            ("high", 0.30),
        ],
    )

    if profile == "low":
        age = int(clamp(rng.gauss(45, 8), 29, 65))
        chest = weighted_choice(rng, [(1, 0.20), (2, 0.35), (3, 0.30), (4, 0.15)])
        bp = int(clamp(rng.gauss(122, 12), 95, 160))
        chol = int(clamp(rng.gauss(210, 35), 126, 310))
        oldpeak = round(clamp(rng.expovariate(2.8), 0.0, 2.0), 1)
        vessels = weighted_choice(rng, [(0, 0.75), (1, 0.18), (2, 0.05), (3, 0.02)])
        thal = weighted_choice(rng, [(3, 0.78), (6, 0.12), (7, 0.10)])
        exang_prob = 0.10
    elif profile == "medium":
        age = int(clamp(rng.gauss(56, 8), 38, 74))
        chest = weighted_choice(rng, [(1, 0.08), (2, 0.20), (3, 0.25), (4, 0.47)])
        bp = int(clamp(rng.gauss(136, 15), 105, 180))
        chol = int(clamp(rng.gauss(245, 45), 150, 390))
        oldpeak = round(clamp(rng.gauss(1.2, 0.8), 0.0, 4.0), 1)
        vessels = weighted_choice(rng, [(0, 0.45), (1, 0.32), (2, 0.16), (3, 0.07)])
        thal = weighted_choice(rng, [(3, 0.45), (6, 0.25), (7, 0.30)])
        exang_prob = 0.35
    else:
        age = int(clamp(rng.gauss(64, 7), 45, 79))
        chest = weighted_choice(rng, [(1, 0.03), (2, 0.10), (3, 0.17), (4, 0.70)])
        bp = int(clamp(rng.gauss(148, 18), 115, 200))
        chol = int(clamp(rng.gauss(270, 55), 160, 520))
        oldpeak = round(clamp(rng.gauss(2.4, 1.1), 0.2, 6.2), 1)
        vessels = weighted_choice(rng, [(0, 0.18), (1, 0.32), (2, 0.30), (3, 0.20)])
        thal = weighted_choice(rng, [(3, 0.20), (6, 0.25), (7, 0.55)])
        exang_prob = 0.62

    sex = weighted_choice(rng, [(0, 0.42), (1, 0.58)])
    fasting_blood_sugar = 1 if rng.random() < (0.12 if profile == "low" else 0.20 if profile == "medium" else 0.30) else 0

    resting_ecg = weighted_choice(
        rng,
        [
            (0, 0.60 if profile == "low" else 0.45 if profile == "medium" else 0.35),
            (1, 0.12 if profile == "low" else 0.18 if profile == "medium" else 0.20),
            (2, 0.28 if profile == "low" else 0.37 if profile == "medium" else 0.45),
        ],
    )

    exercise_induced_angina = 1 if rng.random() < exang_prob else 0

    # Frecuencia máxima: relacionada con edad y perfil.
    theoretical_max = 220 - age
    if profile == "low":
        max_hr = int(clamp(rng.gauss(theoretical_max - 8, 14), 105, 202))
    elif profile == "medium":
        max_hr = int(clamp(rng.gauss(theoretical_max - 22, 16), 85, 190))
    else:
        max_hr = int(clamp(rng.gauss(theoretical_max - 38, 18), 70, 175))

    # Pendiente ST aproximada según oldpeak/perfil.
    if oldpeak < 0.8 and profile == "low":
        slope = weighted_choice(rng, [(1, 0.75), (2, 0.22), (3, 0.03)])
    elif oldpeak < 1.8:
        slope = weighted_choice(rng, [(1, 0.35), (2, 0.55), (3, 0.10)])
    else:
        slope = weighted_choice(rng, [(1, 0.10), (2, 0.65), (3, 0.25)])

    return {
        "age": age,
        "sex": sex,
        "chest": chest,
        "resting_blood_pressure": bp,
        "serum_cholestoral": chol,
        "fasting_blood_sugar": fasting_blood_sugar,
        "resting_electrocardiographic_results": resting_ecg,
        "maximum_heart_rate_achieved": max_hr,
        "exercise_induced_angina": exercise_induced_angina,
        "oldpeak": oldpeak,
        "slope": slope,
        "number_of_major_vessels": vessels,
        "thal": thal,
    }


def validate_patient(patient: dict[str, Any]) -> None:
    missing = [feature for feature in REQUIRED_FEATURES if feature not in patient]
    if missing:
        raise ValueError(f"Paciente incompleto. Faltan campos: {', '.join(missing)}")


def load_patients_file(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        patients = json.load(file)

    if not isinstance(patients, list) or not patients:
        raise ValueError("--patients-file debe contener una lista JSON no vacía.")

    for patient in patients:
        if not isinstance(patient, dict):
            raise ValueError("Cada paciente del archivo debe ser un objeto JSON.")
        validate_patient(patient)

    return patients


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera tráfico de demo contra POST /predict para poblar métricas de Prometheus y Grafana."
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("API_URL", "http://localhost:8000"),
        help="URL base de la API. También puede configurarse con API_URL.",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=20,
        help="Número de peticiones de predicción que se enviarán.",
    )
    parser.add_argument("--timeout", type=float, default=5.0, help="Timeout por petición en segundos.")
    parser.add_argument("--sleep", type=float, default=0.05, help="Pausa entre peticiones en segundos.")
    parser.add_argument("--seed", type=int, default=None, help="Semilla para hacer la demo reproducible.")
    parser.add_argument(
        "--mode",
        choices=["synthetic", "templates", "mixed"],
        default="synthetic",
        help=(
            "Modo de generación: synthetic crea pacientes sintéticos realistas; "
            "templates reutiliza casos fijos; mixed combina ambos."
        ),
    )
    parser.add_argument(
        "--patients-file",
        type=Path,
        default=None,
        help="Archivo JSON opcional con pacientes personalizados.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="No imprime una línea por petición. Útil para 500/1000 peticiones.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="En modo quiet, muestra progreso cada N peticiones.",
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="No comprueba /health antes de lanzar la demo.",
    )
    parser.add_argument("--retries", type=int, default=1, help="Reintentos por petición en errores recuperables.")
    parser.add_argument("--retry-sleep", type=float, default=0.2, help="Pausa entre reintentos.")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Guarda un resumen JSON de la demo.",
    )
    return parser.parse_args()


def get_json(api_url: str, path: str, timeout: float) -> dict[str, Any]:
    endpoint = f"{api_url.rstrip('/')}{path}"
    request = Request(endpoint, headers={"Accept": "application/json"}, method="GET")

    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def check_health(api_url: str, timeout: float) -> bool:
    try:
        get_json(api_url, "/health", timeout)
        return True
    except Exception as exc:
        print(f"ERROR: la API no responde correctamente en /health: {exc}")
        return False


def read_model_info(api_url: str, timeout: float) -> dict[str, Any]:
    try:
        info = get_json(api_url, "/info", timeout)
        model_name = info.get("model_name", "unknown")
        version = info.get("version", "unknown")
        algorithm = info.get("algorithm", "unknown")
        print(f"Modelo activo: {model_name} | {version} | {algorithm}")

        if version != "v4.2.0":
            print(f"AVISO: la API no está usando v4.2.0, sino {version}.")

        return info
    except Exception as exc:
        print(f"AVISO: no se pudo leer /info: {exc}")
        return {}


def post_prediction(api_url: str, payload: dict[str, Any], timeout: float) -> tuple[dict[str, Any] | None, float, str | None]:
    endpoint = f"{api_url.rstrip('/')}/predict"
    data = json.dumps(payload).encode("utf-8")
    request = Request(endpoint, data=data, headers={"Content-Type": "application/json"}, method="POST")

    start = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            elapsed_ms = (time.perf_counter() - start) * 1000
            body = json.loads(response.read().decode("utf-8"))
            return body, elapsed_ms, None
    except HTTPError as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return None, elapsed_ms, f"HTTP {exc.code}: {exc.reason}"
    except URLError as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return None, elapsed_ms, f"Conexión: {exc.reason}"
    except TimeoutError:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return None, elapsed_ms, "Timeout"
    except json.JSONDecodeError as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return None, elapsed_ms, f"JSON inválido: {exc}"


def post_prediction_with_retries(
    api_url: str,
    payload: dict[str, Any],
    timeout: float,
    retries: int,
    retry_sleep: float,
) -> tuple[dict[str, Any] | None, float, str | None]:
    attempts = max(1, retries + 1)
    total_elapsed = 0.0
    last_error: str | None = None

    for attempt in range(1, attempts + 1):
        result, elapsed_ms, error = post_prediction(api_url, payload, timeout)
        total_elapsed += elapsed_ms

        if not error:
            return result, total_elapsed, None

        last_error = error

        # No reintentar errores 4xx: suelen ser payload inválido.
        if error.startswith("HTTP 4"):
            return None, total_elapsed, error

        if attempt < attempts:
            time.sleep(retry_sleep)

    return None, total_elapsed, last_error


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percent
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower

    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }

    return {
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "mean": round(statistics.mean(values), 2),
        "p50": round(percentile(values, 0.50), 2),
        "p95": round(percentile(values, 0.95), 2),
        "p99": round(percentile(values, 0.99), 2),
    }


def choose_patient(
    rng: random.Random,
    mode: str,
    external_patients: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if external_patients:
        return dict(rng.choice(external_patients))

    if mode == "templates":
        return dict(rng.choice(PATIENT_TEMPLATES))

    if mode == "mixed":
        if rng.random() < 0.35:
            return dict(rng.choice(PATIENT_TEMPLATES))
        return generate_realistic_patient(rng)

    return generate_realistic_patient(rng)


def validate_prediction_response(result: dict[str, Any]) -> list[str]:
    required = ["label", "risk_level", "probability", "model_version", "inference_time_ms"]
    return [field for field in required if field not in result]


def main() -> int:
    args = parse_args()

    if args.requests < 1:
        raise SystemExit("--requests debe ser mayor que 0")

    if args.progress_every < 1:
        raise SystemExit("--progress-every debe ser mayor que 0")

    if args.retries < 0:
        raise SystemExit("--retries no puede ser negativo")

    rng = random.Random(args.seed)

    external_patients = None
    if args.patients_file:
        external_patients = load_patients_file(args.patients_file)

    api_url = args.api_url.rstrip("/")

    if not args.skip_health_check:
        if not check_health(api_url, args.timeout):
            return 1

    model_info = read_model_info(api_url, args.timeout)

    counts = Counter()
    risk_levels = Counter()
    errors: list[str] = []
    warnings: list[str] = []
    durations: list[float] = []

    print(f"Enviando {args.requests} peticiones a {api_url}/predict")
    print(f"Modo de pacientes: {'external-file' if external_patients else args.mode}")

    if args.seed is not None:
        print(f"Seed: {args.seed}")

    for index in range(1, args.requests + 1):
        patient = choose_patient(rng, args.mode, external_patients)
        validate_patient(patient)

        result, elapsed_ms, error = post_prediction_with_retries(
            api_url=api_url,
            payload=patient,
            timeout=args.timeout,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
        )

        durations.append(elapsed_ms)

        if error:
            errors.append(error)
            if not args.quiet:
                print(f"[{index:04d}] ERROR {error} ({elapsed_ms:.1f} ms)")
        else:
            assert result is not None
            missing_fields = validate_prediction_response(result)
            if missing_fields:
                warnings.append(f"Respuesta incompleta: {', '.join(missing_fields)}")

            label = result.get("label", "Unknown")
            risk = result.get("risk_level", "Unknown")
            probability = result.get("probability")
            counts[label] += 1
            risk_levels[risk] += 1

            probability_text = f"{probability:.2f}" if isinstance(probability, (float, int)) else "n/a"

            if not args.quiet:
                print(f"[{index:04d}] {label} | riesgo={risk} | prob={probability_text} | {elapsed_ms:.1f} ms")

        if args.quiet and index % args.progress_every == 0:
            print(f"[{index}/{args.requests}] completadas...")

        if args.sleep:
            time.sleep(args.sleep)

    latency = latency_summary(durations)

    print("\nResumen de demo")
    print(f"- Peticiones realizadas: {args.requests}")
    print(f"- Correctas: {args.requests - len(errors)}")
    print(f"- Disease: {counts.get('Disease', 0)}")
    print(f"- No Disease: {counts.get('No Disease', 0)}")
    print(f"- Riesgo Low: {risk_levels.get('Low', 0)}")
    print(f"- Riesgo Medium: {risk_levels.get('Medium', 0)}")
    print(f"- Riesgo High: {risk_levels.get('High', 0)}")
    print(f"- Errores: {len(errors)}")
    print(f"- Warnings: {len(warnings)}")
    print(f"- Latencia media: {latency['mean']:.1f} ms")
    print(f"- Latencia p50: {latency['p50']:.1f} ms")
    print(f"- Latencia p95: {latency['p95']:.1f} ms")
    print(f"- Latencia p99: {latency['p99']:.1f} ms")
    print(f"- Latencia min/max: {latency['min']:.1f} / {latency['max']:.1f} ms")

    if errors:
        grouped_errors = Counter(errors)
        print("- Detalle de errores:")
        for message, count in grouped_errors.items():
            print(f"  {count}x {message}")

    if warnings:
        grouped_warnings = Counter(warnings)
        print("- Detalle de warnings:")
        for message, count in grouped_warnings.items():
            print(f"  {count}x {message}")

    summary = {
        "api_url": api_url,
        "requests": args.requests,
        "success": args.requests - len(errors),
        "errors": len(errors),
        "warnings": len(warnings),
        "mode": "external-file" if external_patients else args.mode,
        "seed": args.seed,
        "counts": {
            "Disease": counts.get("Disease", 0),
            "No Disease": counts.get("No Disease", 0),
        },
        "risk_levels": {
            "Low": risk_levels.get("Low", 0),
            "Medium": risk_levels.get("Medium", 0),
            "High": risk_levels.get("High", 0),
        },
        "latency_ms": latency,
        "model": {
            "model_name": model_info.get("model_name"),
            "version": model_info.get("version"),
            "algorithm": model_info.get("algorithm"),
        },
    }

    if args.json_output:
        args.json_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nResumen JSON guardado en: {args.json_output}")

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
