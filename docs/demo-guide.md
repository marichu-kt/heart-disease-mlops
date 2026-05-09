# Guía De Demo Profesional

Esta guía describe una demo completa para enseñar el proyecto como una aplicación MLOps con frontend, API, métricas y dashboard.

## 1. Levantar Servicios

```bash
docker compose up --build
```

Servicios esperados:

| Servicio | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Swagger | http://localhost:8000/docs |
| Metrics | http://localhost:8000/metrics |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3001/login |

## 2. Enseñar El Frontend

1. Abre http://localhost:3000.
2. Revisa la fila de estado: API, modelo, última inferencia y resultado.
3. Muestra el formulario agrupado por secciones clínicas.
4. Ejecuta `Evaluar riesgo`.
5. Explica el resultado, el nivel de riesgo, las probabilidades y el tiempo de inferencia.
6. Cambia entre modo claro y oscuro para mostrar accesibilidad visual.

## 3. Probar Swagger

1. Abre http://localhost:8000/docs.
2. Revisa los tags `System`, `Model`, `Prediction` y `Monitoring`.
3. Ejecuta `GET /health`.
4. Ejecuta `GET /info`.
5. Ejecuta `POST /predict` con el ejemplo del schema.

## 4. Generar Métricas

```bash
python scripts/demo_requests.py --requests 500 --sleep 0.01
```

Para una demo breve pueden usarse 25 peticiones, pero para capturas finales de Prometheus y Grafana se recomienda generar 500 peticiones. En la validación final de esta entrega se usaron 500 peticiones con 0 errores.

El script muestra:

- peticiones realizadas;
- predicciones `Disease` y `No Disease`;
- errores si aparecen;
- tiempo medio aproximado.

## 5. Enseñar Prometheus

1. Abre http://localhost:9090/targets.
2. Confirma que `heart-api` está `UP`.
3. Abre http://localhost:9090/graph.
4. Ejecuta:

```promql
heart_api_up
heart_predictions_total
sum by (label) (heart_predictions_by_class_total)
```

## 6. Enseñar Grafana

1. Abre http://localhost:3001/login.
2. Entra con las credenciales de demo local.
3. Abre `Heart Disease MLOps Observability`.
4. Revisa las filas `Overview`, `Predictions`, `Latency`, `Errors` y `Model`.
5. Confirma que aparecen datos en `Total predictions`, `Predictions by class`, latencia y errores.

## 7. Cierre De La Demo

Mensaje recomendado:

> El proyecto no solo sirve un modelo, sino que lo empaqueta como una aplicación MLOps observable: frontend, API documentada, modelo versionado, métricas Prometheus, dashboard Grafana, Docker Compose, tests y CI.
