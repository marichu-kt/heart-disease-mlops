import argparse
import json
import os
import random
import time
from collections import Counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PATIENTS = [
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
    return parser.parse_args()


def post_prediction(api_url: str, payload: dict, timeout: float) -> tuple[dict | None, float, str | None]:
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


def main() -> int:
    args = parse_args()
    if args.requests < 1:
        raise SystemExit("--requests debe ser mayor que 0")

    counts = Counter()
    errors: list[str] = []
    durations: list[float] = []

    print(f"Enviando {args.requests} peticiones a {args.api_url.rstrip('/')}/predict")

    for index in range(1, args.requests + 1):
        patient = random.choice(PATIENTS)
        result, elapsed_ms, error = post_prediction(args.api_url, patient, args.timeout)
        durations.append(elapsed_ms)

        if error:
            errors.append(error)
            print(f"[{index:02d}] ERROR {error} ({elapsed_ms:.1f} ms)")
        else:
            label = result.get("label", "Unknown")
            risk = result.get("risk_level", "Unknown")
            probability = result.get("probability")
            counts[label] += 1
            probability_text = f"{probability:.2f}" if isinstance(probability, (float, int)) else "n/a"
            print(f"[{index:02d}] {label} | riesgo={risk} | prob={probability_text} | {elapsed_ms:.1f} ms")

        if args.sleep:
            time.sleep(args.sleep)

    average_ms = sum(durations) / len(durations)
    print("\nResumen de demo")
    print(f"- Peticiones realizadas: {args.requests}")
    print(f"- Disease: {counts.get('Disease', 0)}")
    print(f"- No Disease: {counts.get('No Disease', 0)}")
    print(f"- Errores: {len(errors)}")
    print(f"- Tiempo medio aproximado: {average_ms:.1f} ms")

    if errors:
        grouped_errors = Counter(errors)
        print("- Detalle de errores:")
        for message, count in grouped_errors.items():
            print(f"  {count}x {message}")

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
