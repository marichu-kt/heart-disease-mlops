# Guía De Grafana

Grafana se usa para visualizar las métricas que Prometheus scrapea desde `GET /metrics` de FastAPI.

## Acceso

URL local:

```text
http://localhost:3001/login
```

Credenciales de demo local:

```text
usuario: admin
password: change-me-local-demo
```

Estas credenciales son solo para entorno local. En un despliegue real deben cambiarse con `GRAFANA_ADMIN_USER` y `GRAFANA_ADMIN_PASSWORD` en un `.env` privado no versionado.

## Provisionamiento

El proyecto provisiona automáticamente:

| Recurso | Archivo |
|---|---|
| Datasource Prometheus | `monitoring/grafana/provisioning/datasources/prometheus.yml` |
| Dashboard | `monitoring/grafana-dashboard.json` |
| Provider de dashboards | `monitoring/grafana/provisioning/dashboards/dashboards.yml` |

El datasource apunta a:

```text
http://prometheus:9090
```

Ese hostname funciona dentro de la red de Docker Compose.

## Dashboard

Nombre del dashboard:

```text
Heart Disease MLOps Observability
```

El dashboard está organizado por filas:

| Fila | Paneles | Propósito |
|---|---|---|
| Overview | API status, Total predictions, Request rate, Errors | Entender el estado general de un vistazo. |
| Predictions | Predictions by class, Predictions by risk level | Ver distribución de predicciones reales. |
| Latency | Prediction latency p95, HTTP latency average | Evaluar rendimiento de inferencia y API. |
| Errors | API errors by endpoint, Prediction errors by endpoint | Detectar fallos operativos. |
| Model | Model info | Confirmar modelo, versión y algoritmo activos. |

## Cómo Ver Datos

1. Levanta la pila:

```bash
docker compose up --build
```

2. Genera tráfico:

```bash
python scripts/demo_requests.py --requests 25 --sleep 0.05
```

3. Entra en Grafana.
4. Abre la carpeta `MLOps`.
5. Abre `Heart Disease MLOps Observability`.
6. Comprueba que el rango temporal está en `Last 1 hour` y el refresh en `10s`.

## Si Aparece Sin Datos

- Comprueba que Prometheus ve el target `heart-api` en http://localhost:9090/targets.
- Ejecuta de nuevo `python scripts/demo_requests.py --requests 25`.
- Revisa que el dashboard usa el datasource `Prometheus` con uid `prometheus`.
- Comprueba `docker compose logs api prometheus grafana`.
- Recuerda que algunas series solo se mueven tras generar predicciones reales.
