# Guía De Queries Prometheus

Esta guía resume las consultas más útiles para validar la observabilidad de la API FastAPI y del modelo de predicción cardíaca.

## Antes De Consultar

1. Levanta los servicios:

```bash
docker compose up --build
```

2. Genera tráfico real:

```bash
python scripts/demo_requests.py --requests 500 --sleep 0.01
```

3. Abre Prometheus:

- Targets: http://localhost:9090/targets
- Graph: http://localhost:9090/graph

El target `heart-api` debe aparecer como `UP`.

## Queries Recomendadas

| Query | Qué Mide | Cuándo Usarla | Qué Esperar |
|---|---|---|---|
| `heart_api_up` | Estado de la API. | Comprobar disponibilidad del servicio. | `1` si la API está operativa. |
| `rate(api_requests_total[5m])` | Tasa de requests por segundo. | Ver si la API recibe tráfico. | Valores mayores que 0 tras usar frontend o demo script. |
| `heart_predictions_total` | Predicciones acumuladas. | Confirmar que `/predict` y `/predict/batch` registran inferencias. | Contador creciente por endpoint. |
| `sum by (label) (heart_predictions_by_class_total)` | Predicciones agrupadas por clase. | Ver distribución `Disease` / `No Disease`. | Series por etiqueta real devuelta por el modelo. |
| `histogram_quantile(0.95, sum(rate(heart_prediction_duration_seconds_bucket[5m])) by (le, endpoint))` | Percentil 95 de latencia de inferencia. | Revisar rendimiento del modelo. | Latencias bajas en segundos para demo local. |
| `sum by (endpoint) (rate(api_request_duration_seconds_sum[5m])) / sum by (endpoint) (rate(api_request_duration_seconds_count[5m]))` | Latencia HTTP media por endpoint. | Detectar endpoints lentos. | Valores bajos en entorno local. |
| `api_request_errors_total` | Errores HTTP 5xx acumulados. | Buscar fallos de servidor. | 0 en una demo correcta. |
| `heart_prediction_errors_total` | Errores de inferencia acumulados. | Revisar problemas de modelo o entrada. | 0 si el modelo está cargado y las entradas son válidas. |
| `heart_model_info` | Metadata del modelo cargado. | Ver versión y algoritmo activos. | Gauge con valor `1` y labels `model_name`, `version`, `algorithm`. |

## Interpretación Rápida

- Si `heart_api_up` es `0`, revisa `docker compose ps` y `docker compose logs api`.
- Si no hay datos de predicción, ejecuta `scripts/demo_requests.py` o usa el formulario del frontend.
- Si Prometheus no muestra el target, revisa `monitoring/prometheus.yml` y confirma que el servicio Docker se llama `api`.
- Si hay latencia alta, revisa carga local, logs de API y tiempos de inferencia devueltos por `/predict`.
